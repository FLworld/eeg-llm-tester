"""Cohort / batch execution: run one approved pipeline across every subject in a BIDS dataset.

Mirrors `sweep.py` in spirit -- a deterministic engine that reuses the single-subject machinery,
with NO model in the loop. Per subject: reset SESSION (clean slate, no cross-subject leakage),
scope the recording (structure + its own BIDS codebook), rebind the pipeline's load_eeg to that
subject, run it, scrape QC + the endpoint, and capture the ERP for a grand average. A subject that
fails is flagged and the cohort continues (one bad file must not halt 40 subjects).

Outputs: a per-subject QC table, an endpoint distribution with outlier flags, a group
grand-average ERP, and per-subject saved derivatives (mirrors save_eeg).
"""
from __future__ import annotations

import copy
import datetime
import json
import os

import numpy as np
import mne

from pipeline import run_pipeline
from recipes import rebind_to_current
from tools import SESSION, save_eeg, scope_eeg

# Honors STATE_DIR (a mounted volume in a container) so batch outputs survive restarts;
# defaults to the module dir for a plain local checkout.
BATCHES_DIR = os.path.join(
    os.environ.get("STATE_DIR", os.path.dirname(os.path.abspath(__file__))), "batches")
_REC_EXTS = (".set", ".fif", ".vhdr", ".edf", ".bdf")


def _reset_session() -> None:
    """Clear every SESSION slot so each subject starts from a clean reload (no leakage). SCOPE is
    separate and is refreshed per subject by scope_eeg."""
    for k in list(SESSION.keys()):
        SESSION[k] = None


def discover_subjects(dataset_dir: str) -> list[tuple[str, str | None]]:
    """[(sub_id, recording_path|None)] for every sub-*/ under a BIDS dataset dir, in order.
    Prefers `<sub>/eeg/*_eeg.<ext>`, then any recording; None if the subject has no recording."""
    out: list[tuple[str, str | None]] = []
    if not os.path.isdir(dataset_dir):
        return out
    for d in sorted(os.listdir(dataset_dir)):
        if not d.startswith("sub-") or not os.path.isdir(os.path.join(dataset_dir, d)):
            continue
        eegdir = os.path.join(dataset_dir, d, "eeg")
        rec = None
        if os.path.isdir(eegdir):
            files = sorted(os.listdir(eegdir))
            for ext in _REC_EXTS:                       # prefer the BIDS *_eeg.<ext>
                hit = [f for f in files if f.endswith("_eeg" + ext)]
                if hit:
                    rec = os.path.join(eegdir, hit[0])
                    break
            if rec is None:                             # else any recording file
                for ext in _REC_EXTS:
                    hit = [f for f in files if f.endswith(ext)]
                    if hit:
                        rec = os.path.join(eegdir, hit[0])
                        break
        out.append((d, rec))
    return out


# result-dict keys → per-subject QC metric. Scraped from real tool outputs, never inferred.
def _scrape_qc(results: list[dict]) -> dict:
    """Pull per-subject QC + the endpoint from a pipeline's step results.

    endpoint = the LAST successful measure_component's amplitude (+latency). QC is collected from
    whichever preprocessing tools ran."""
    qc: dict = {}
    endpoint_uv = latency_ms = None
    for step in results:
        res = step.get("result")
        if not isinstance(res, dict) or res.get("ok") is False:
            continue
        tool = step.get("tool")
        if tool == "measure_component" and res.get("amplitude_uv") is not None:
            endpoint_uv = res.get("amplitude_uv")       # last one wins
            latency_ms = res.get("latency_ms")
        if "n_kept" in res:
            qc["n_kept"] = res["n_kept"]
        elif "n_epochs_after" in res:                   # run_autoreject
            qc["n_kept"] = res["n_epochs_after"]
        if "n_interpolated" in res:
            qc["n_interpolated"] = res["n_interpolated"]
        if "bad_channels" in res:
            qc["n_bad_channels"] = len(res["bad_channels"] or [])
        if tool == "run_ica" and "n_components" in res:
            qc["n_components"] = res["n_components"]
        if tool in ("create_epochs", "compute_erp") and "n_epochs" in res:
            qc.setdefault("n_epochs", res["n_epochs"])
    return {"qc": qc, "endpoint_uv": endpoint_uv, "latency_ms": latency_ms}


def _endpoint_step(spec: dict) -> dict | None:
    """The pipeline's measure_component step (its args define the endpoint window/channels), or None."""
    for step in spec.get("pipeline", []) or []:
        if isinstance(step, dict) and step.get("tool") == "measure_component":
            return step
    return None


def _grand_average(evokeds: list):
    """mne.grand_average over subjects, aligned on the channels COMMON to all (so differing
    bad/interpolated sets don't break it). Returns an Evoked or None (need >= 2)."""
    evs = [e for e in evokeds if e is not None]
    if len(evs) < 2:
        return None
    common = set(evs[0].ch_names)
    for e in evs[1:]:
        common &= set(e.ch_names)
    if not common:
        return None
    picked = [e.copy().pick(sorted(common)) for e in evs]
    return mne.grand_average(picked)


def _plot_grand_average(evoked) -> str | None:
    """Butterfly plot of the grand-average ERP -> base64 PNG."""
    import base64
    import io
    try:
        fig = evoked.plot(spatial_colors=True, show=False, verbose="ERROR")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _aggregate(rows: list[dict], evokeds: list, spec: dict) -> dict:
    """Endpoint mean/sd + outlier flags; group grand-average ERP (+ its endpoint)."""
    ok_rows = [r for r in rows if r.get("ok") and isinstance(r.get("endpoint_uv"), (int, float))]
    eps = np.array([r["endpoint_uv"] for r in ok_rows], float)
    agg: dict = {"n_ok": len(ok_rows), "n_total": len(rows)}
    if eps.size:
        mean, sd = float(eps.mean()), float(eps.std(ddof=0))
        agg["endpoint"] = {"mean_uv": round(mean, 4), "sd_uv": round(sd, 4), "n": int(eps.size)}
        # outliers: |z| > 3 on endpoint; and low-trial (< half the median surviving trials)
        kept = [r["qc"].get("n_kept") for r in ok_rows if r["qc"].get("n_kept") is not None]
        med_kept = float(np.median(kept)) if kept else None
        for r in ok_rows:
            flags = []
            if sd > 0 and abs(r["endpoint_uv"] - mean) / sd > 3:
                flags.append("endpoint |z|>3")
            nk = r["qc"].get("n_kept")
            if med_kept and nk is not None and nk < 0.5 * med_kept:
                flags.append(f"low trials ({nk})")
            r["outlier_flags"] = flags
    ga = _grand_average(evokeds)
    if ga is not None:
        agg["grand_average"] = {"n_subjects": int(getattr(ga, "nave", len(evokeds))),
                                "n_channels": len(ga.ch_names)}
        ep_step = _endpoint_step(spec)
        if ep_step:                                     # measure the group ERP the same way
            from tools import measure_component
            SESSION["evoked"] = ga
            gm = measure_component(**(ep_step.get("args", {}) or {}))
            if isinstance(gm, dict) and gm.get("amplitude_uv") is not None:
                agg["grand_average"]["endpoint_uv"] = gm["amplitude_uv"]
                if gm.get("latency_ms") is not None:
                    agg["grand_average"]["latency_ms"] = gm["latency_ms"]
        agg["_grand_average_evoked"] = ga               # popped by the caller for plotting
    return agg


async def run_batch(spec: dict, dataset_dir: str, run_step, save_derivatives: bool = True,
                    run_name: str | None = None) -> dict:
    """Run a plan `spec` ({"pipeline":[...]}) on every subject under `dataset_dir`.

    `run_step(name, args)->(result, image)` is the same dispatcher run_pipeline uses. Returns
    {ok, run_name, rows, aggregate, grand_average_png?} or {ok: False, error}.
    """
    if not isinstance(spec, dict) or "pipeline" not in spec:
        return {"ok": False, "error": "batch needs a PLAN (a saved /plan recipe or a staged plan), "
                "not a sweep. Draft/select a plan whose last step is measure_component."}
    subjects = discover_subjects(dataset_dir)
    if not subjects:
        return {"ok": False, "error": f"No sub-*/ subjects found under {dataset_dir!r}."}

    run_name = run_name or f"batch_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(BATCHES_DIR, run_name)
    rows: list[dict] = []
    evokeds: list = []

    for sub_id, filepath in subjects:
        _reset_session()
        if filepath is None:
            rows.append({"sub": sub_id, "ok": False, "error": "no recording found", "qc": {}})
            continue
        sc = scope_eeg(filepath)                         # per-subject structure + BIDS codebook
        warnings = sc.get("warnings") or [] if isinstance(sc, dict) else []
        cfg = rebind_to_current(copy.deepcopy(spec), filepath)
        results, _ = await run_pipeline({"pipeline": cfg["pipeline"]}, run_step)
        failed = next((r["result"] for r in results
                       if isinstance(r["result"], dict) and r["result"].get("ok") is False), None)
        scraped = _scrape_qc(results)
        row = {"sub": sub_id, "ok": failed is None,
               "endpoint_uv": scraped["endpoint_uv"], "latency_ms": scraped["latency_ms"],
               "qc": scraped["qc"], "warnings": warnings}
        if failed is not None:
            row["error"] = failed.get("error", "a pipeline step failed")
        rows.append(row)
        if row["ok"]:
            ev = SESSION.get("evoked")
            if ev is not None:
                evokeds.append(ev.copy())
            if save_derivatives:
                row["derivative"] = _save_derivative(out_dir, sub_id, cfg)

    aggregate = _aggregate(rows, evokeds, spec)
    ga = aggregate.pop("_grand_average_evoked", None)
    ga_png = _plot_grand_average(ga) if ga is not None else None
    _write_summary(out_dir, run_name, spec, rows, aggregate)
    return {"ok": True, "run_name": run_name, "out_dir": out_dir,
            "rows": rows, "aggregate": aggregate, "grand_average_png": ga_png,
            "n_subjects": len(subjects)}


def _save_derivative(out_dir: str, sub_id: str, cfg: dict) -> str | None:
    """Persist one subject's processed recording + its pipeline config (mirrors /save)."""
    sub_dir = os.path.join(out_dir, sub_id)
    os.makedirs(sub_dir, exist_ok=True)
    try:
        res = save_eeg(os.path.join(sub_dir, "state.fif"))
        with open(os.path.join(sub_dir, "pipeline.json"), "w") as fh:
            json.dump({"pipeline": cfg.get("pipeline", [])}, fh, indent=2, default=str)
        return res.get("saved") if isinstance(res, dict) else None
    except Exception:
        return None


def _write_summary(out_dir: str, run_name: str, spec: dict, rows: list[dict],
                   aggregate: dict) -> None:
    """Write batches/<run>/summary.json so a run is reproducible / re-aggregatable."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"run_name": run_name,
                   "created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "spec": {k: v for k, v in spec.items() if k != "_planner_meta"},
                   "rows": rows, "aggregate": aggregate}, fh, indent=2, default=str)
