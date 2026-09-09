"""Deterministic parameter-sweep engine for eeg-llm ("iterative planning").

Front-end agnostic: this module consumes a SWEEP SPEC and does everything the model is
NOT trusted to do. The model (or a human) only *proposes* the spec; the expansion into
concrete pipeline variants, the re-validation of every variant through the same
structural gate as a normal `/plan`, the execution, and the measurement all happen here,
in plain deterministic Python with no model in the loop.

Because the spec is separate from how it was produced, both front-ends -- the NL planner
(`propose_sweep` in pipeline.py) and a future deterministic `/sweep` over an approved
plan -- feed this same engine unchanged.

SWEEP SPEC shape:
    {
      "base_pipeline": [ {"tool": "load_eeg", "args": {...}}, ... ],   # a normal pipeline
      "axes": [
        {"tool": "filter_eeg", "param": "l_freq",     "values": [0.1, 0.5, 1.0]},
        {"tool": "filter_eeg", "param": "fir_window", "values": ["hamming", "kaiser"]},
        {"tool": "run_ica",    "param": "random_state", "repeat": 8}   # stochastic
      ],
      "endpoint": {"tool": "measure_component", "args": {...}},        # one scalar/variant
      "compare_checkpoint": "hpfilt",                                  # optional grading
      "notes": [...]
    }

Axis kinds:
  - values grid : {"tool","param","values":[...]}  -> one variant per value
  - stochastic  : {"tool","param","repeat":N}      -> N variants with distinct seeds
An axis targets the first step whose `tool` matches; add "index" to pick a later one.
Multiple `values` axes multiply (cartesian product), guarded by MAX_VARIANTS.
"""
from __future__ import annotations

import copy
import itertools
import re
import time

import numpy as np

from pipeline import validate_pipeline, run_pipeline, _TOOL_PARAMS

MAX_VARIANTS = 24        # hard cap so a grid can never silently explode
SEED_BASE = 97           # matches the project's ICA seed convention
SEED_STEP = 13           # distinct, reproducible seeds for a `repeat` axis


def _workflow_errors(steps: list[dict]) -> list[str]:
    errors = []
    if steps[0].get("tool") != "load_eeg":
        errors.append("Each sweep variant starts from an empty session; base_pipeline must start with load_eeg.")
    have_erp = False
    for i, step in enumerate(steps):
        name = step.get("tool")
        if name in ("load_eeg", "create_epochs", "create_bins"):
            have_erp = False
        elif name in ("compute_erp", "compute_difference_erp"):
            have_erp = True
        elif name == "measure_component" and not have_erp:
            errors.append(f"Step {i} (measure_component): no ERP producer precedes the measurement. "
                          "Add compute_erp for one condition or compute_difference_erp for a contrast "
                          "after epoching and before the endpoint; create_epochs alone does not average an ERP.")
    return errors


def validate_sweep_request(spec: dict, request: str, scope: dict | None = None) -> list[str]:
    """Guard explicit 'epoch A vs B' contrasts without rewriting scientific choices."""
    contrast = re.search(r"\bepoch\s+([\w-]+)\s+(?:vs\.?|versus|minus)\s+([\w-]+)\b", request, re.I)
    if not contrast:
        return []
    names = [name.lower() for name in contrast.groups()]
    steps = spec.get("base_pipeline") or []
    if not isinstance(steps, list) or any(not isinstance(s, dict) or not isinstance(s.get("args", {}), dict) for s in steps):
        return []  # structural validation reports these malformed steps
    final_steps = steps + ([spec["endpoint"]] if isinstance(spec.get("endpoint"), dict) else [])
    if not any(s.get("tool") in ("compute_erp", "compute_difference_erp", "measure_component") for s in final_steps):
        return []  # epoch-only comparisons need not average or measure an ERP
    differences = [s for s in steps if isinstance(s, dict) and s.get("tool") == "compute_difference_erp"]
    errors = []
    if len(differences) != 1:
        errors.append(f"'epoch {names[0]} vs {names[1]}' requires one compute_difference_erp "
                      "(first condition minus second), not separate condition variants.")
    if any(a.get("param") in ("event_id", "event_id_a", "event_id_b")
           for a in spec.get("axes", []) if isinstance(a, dict)):
        errors.append("The requested contrast is fixed; do not add an event_id/condition sweep axis.")
    if scope is not None:
        conditions = {k.lower(): v for k, v in ((scope.get("codebook") or {}).get("conditions") or {}).items()}
        if any(name not in conditions for name in names):
            errors.append("Scope the recording and supply a codebook defining both contrast conditions; do not invent event codes.")
        elif len(differences) == 1:
            from tools import _iter_codebook_codes
            bins = {b.get("label"): b.get("codes") for s in steps if s.get("tool") == "create_bins"
                    for b in s.get("args", {}).get("bins", []) if isinstance(b, dict)}
            for key, name in zip(("event_id_a", "event_id_b"), names):
                selected = differences[0].get("args", {}).get(key)
                try:
                    expected = set(_iter_codebook_codes({"conditions": {name: conditions[name]}}))
                    actual = set()
                    labels = [selected] if isinstance(selected, str) else (selected or [])
                    for label in labels:
                        if not isinstance(label, str):
                            raise ValueError("ERP event labels must be strings")
                        if label in bins:
                            actual.update(_iter_codebook_codes({"conditions": {name: bins[label]}}))
                        else:
                            actual.add(int(label))
                except (TypeError, ValueError):
                    actual, expected = set(), {None}
                if actual != expected:
                    errors.append(f"compute_difference_erp.{key} must select the scoped '{name}' "
                                  f"codes {conditions[name]!r}; set its create_bins bin codes to "
                                  f"{conditions[name]!r}. A range requires nested brackets: "
                                  "codes=[[lo, hi]], not codes=[lo, hi] (which selects only two codes).")
    return errors


def _axis_value_list(axis: dict) -> list:
    """Concrete values an axis contributes: an explicit list, or fanned-out seeds."""
    if axis.get("values"):
        return list(axis["values"])
    if axis.get("repeat") is not None:
        n = int(axis["repeat"])
        return [SEED_BASE + i * SEED_STEP for i in range(n)]
    return []


def expand_sweep(spec: dict) -> tuple[list[dict], list[str]]:
    """Expand a spec into concrete, validated pipeline variants.

    Returns (variants, errors). If errors is non-empty, variants is []. Each variant is
    {"label", "assignments": {"tool.param": value}, "pipeline": [steps]}.
    """
    if not isinstance(spec, dict):
        return [], ["Sweep spec is not an object."]
    base = spec.get("base_pipeline")
    axes = spec.get("axes") or []
    if not isinstance(base, list) or not base:
        return [], ["base_pipeline is empty or not a list."]
    if not isinstance(axes, list) or not axes:
        return [], ["No sweep axes were given (need at least one)."]
    if any(not isinstance(step, dict) for step in base):
        return [], ["Every base_pipeline step must be an object."]
    endpoint = spec.get("endpoint")
    if endpoint is not None:
        if not isinstance(endpoint, dict) or endpoint.get("tool") not in _TOOL_PARAMS:
            return [], ["endpoint must name a real tool, or be omitted to compare the final base_pipeline step."]
        if endpoint in base:
            return [], ["The endpoint duplicates a base_pipeline step. Execute the final operation only once: "
                        "keep it in endpoint or in base_pipeline, not both."]

    errors: list[str] = []
    resolved: list[dict] = []
    for ai, axis in enumerate(axes):
        if not isinstance(axis, dict):
            errors.append(f"Axis {ai}: not an object.")
            continue
        tool = axis.get("tool")
        param = axis.get("param")
        if tool not in _TOOL_PARAMS:
            errors.append(f"Axis {ai}: unknown tool '{tool}'.")
            continue
        if param not in _TOOL_PARAMS.get(tool, set()):
            errors.append(f"Axis {ai} ({tool}): '{param}' is not a parameter of {tool}.")
            continue
        values = _axis_value_list(axis)
        if not values or (axis.get("repeat") is not None and int(axis["repeat"]) < 2):
            errors.append(
                f"Axis {ai} ({tool}.{param}): needs a non-empty 'values' list or 'repeat' >= 2.")
            continue
        occ = int(axis.get("index", 0))
        step_idxs = [i for i, s in enumerate(base) if s.get("tool") == tool]
        if len(step_idxs) <= occ:
            errors.append(
                f"Axis {ai}: base_pipeline has no '{tool}' step at occurrence {occ}.")
            continue
        resolved.append({"tool": tool, "param": param,
                         "step": step_idxs[occ], "values": values})
    if errors:
        return [], errors

    combos = list(itertools.product(*[a["values"] for a in resolved]))
    if len(combos) > MAX_VARIANTS:
        return [], [f"Sweep would produce {len(combos)} variants (cap {MAX_VARIANTS}). "
                    f"Reduce the number of values or axes."]

    variants: list[dict] = []
    for combo in combos:
        pipe = copy.deepcopy(base)
        assignments: dict = {}
        for a, val in zip(resolved, combo):
            pipe[a["step"]].setdefault("args", {})[a["param"]] = val
            assignments[f"{a['tool']}.{a['param']}"] = val
        label = ", ".join(f"{k}={v}" for k, v in assignments.items())
        complete = pipe + ([endpoint] if endpoint is not None else [])
        verrs = validate_pipeline({"pipeline": complete})
        if not verrs:
            verrs = _workflow_errors(complete)
        if verrs:
            return [], [f"Variant [{label}] failed validation: " + "; ".join(verrs)]
        variants.append({"label": label, "assignments": assignments, "pipeline": pipe})
    return variants, []


def validate_sweep(spec: dict) -> list[str]:
    """Structural gate for a sweep spec: valid iff expand_sweep produces no errors.

    Checks the full base-plus-optional-endpoint schema and any ERP prerequisite in
    each clean-session variant. Returns a list of problems ([] == valid).
    """
    return expand_sweep(spec)[1]


def _reset_session() -> None:
    """Clear every SESSION slot so a variant starts from a clean reload (no leakage)."""
    from tools import SESSION
    for k in list(SESSION.keys()):
        SESSION[k] = None


async def _call(run_step, name: str, args: dict) -> dict:
    """run_step returns (result, image_b64); we only need the result dict here."""
    result, _ = await run_step(name, args or {})
    return result if isinstance(result, dict) else {"ok": False, "error": "no result"}


def _aggregate(spec: dict, rows: list[dict]) -> dict:
    """For any `repeat` (stochastic) axis, report endpoint mean/sd/spread per group.

    A group = the set of non-stochastic assignments, so a grid crossed with a repeat
    axis yields one mean/sd/spread per grid cell.
    """
    axes = spec.get("axes") or []
    repeat_params = {f"{a.get('tool')}.{a.get('param')}"
                     for a in axes if isinstance(a, dict) and a.get("repeat") is not None}
    if not repeat_params:
        return {}
    groups: dict = {}
    for r in rows:
        if not r.get("ok") or r.get("endpoint_uv") is None:
            continue
        key = tuple(sorted((k, v) for k, v in r["assignments"].items()
                           if k not in repeat_params))
        groups.setdefault(key, []).append(float(r["endpoint_uv"]))
    stochastic = []
    for key, eps in groups.items():
        arr = np.asarray(eps, float)
        stochastic.append({
            "group": dict(key) or "(all seeds)",
            "n": int(arr.size),
            "mean_uv": float(arr.mean()),
            "sd_uv": float(arr.std(ddof=0)),
            "spread_uv": float(arr.max() - arr.min()),
        })
    return {"stochastic": stochastic}


async def run_sweep(spec: dict, run_step) -> dict:
    """Execute each variant and compare its final tool result, scalar or otherwise.

    `run_step(name, args) -> (result, image_b64)` is the same dispatcher run_pipeline
    uses (in app.py that is `_run_tool`; headless tests pass a thin async wrapper around
    call_tool). No model is consulted. Each variant starts from a cleared SESSION and
    re-runs its own load_eeg, so variants cannot contaminate each other.

    Returns {"ok", "rows", "summary", "n_variants"} (or {"ok": False, "errors"}).
    """
    variants, errors = expand_sweep(spec)
    if errors:
        return {"ok": False, "errors": errors}

    endpoint = spec.get("endpoint") or {}
    checkpoint = spec.get("compare_checkpoint")
    rows: list[dict] = []
    images = []
    for vi, v in enumerate(variants, 1):
        _reset_session()
        timings = []
        final_image = None
        async def checked_step(name, args):
            nonlocal final_image
            started = time.perf_counter()
            value, image = await run_step(name, args)
            timings.append(time.perf_counter() - started)
            if not isinstance(value, dict):
                value = {"ok": False, "error": "Tool returned no result object."}
            else:
                value = dict(value)
                image = image or value.pop("image", None)
                if value.get("error"):
                    value["ok"] = False
            final_image = image
            return value, image
        complete = v["pipeline"] + ([endpoint] if endpoint else [])
        results, _images = await run_pipeline({"pipeline": complete}, checked_step)
        failed = next(
            (r["result"] for r in results
             if isinstance(r["result"], dict) and r["result"].get("ok") is False), None)
        final = results[-1]
        output = final["result"]
        row = {"label": v["label"], "assignments": v["assignments"], "ok": failed is None,
               "output_tool": final["tool"], "output": output, "steps": results,
               "elapsed_s": round(sum(timings), 3), "output_elapsed_s": round(timings[-1], 3),
               "metrics": _result_metrics(output)}
        if final_image:
            figure_name = f"variant-{vi}-{final['tool']}"
            images.append((figure_name, final_image))
            row["figure"] = figure_name
        if failed is not None:
            row["error"] = failed.get("error", "a pipeline step failed")
            rows.append(row)
            continue
        if final["tool"] == "measure_component":
            amplitude = output.get("amplitude_uv")
            if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)) or not np.isfinite(amplitude):
                row["ok"] = False
                row["error"] = "Endpoint returned no finite amplitude_uv measurement."
            else:
                row["endpoint_uv"] = amplitude
            if output.get("latency_ms") is not None:
                row["latency_ms"] = output["latency_ms"]
        if checkpoint:
            cp = await _call(run_step, "compare_to_checkpoint", {"checkpoint": checkpoint})
            row["vs_ref_r"] = cp.get("min_r")
        rows.append(row)

    return {"ok": True, "rows": rows, "images": images,
            "summary": _aggregate(spec, rows), "n_variants": len(variants)}


def _result_metrics(result: dict) -> dict:
    """Compact reported values only; do not invent a universal quality score."""
    metrics = {}
    for key, value in result.items():
        if key in ("ok", "error", "traceback", "note", "image", "filepath", "saved"):
            continue
        if isinstance(value, (int, float, bool)) or (isinstance(value, str) and len(value) < 80):
            metrics[key] = value
        elif isinstance(value, dict):
            for child, number in value.items():
                if isinstance(number, (int, float)) and not isinstance(number, bool):
                    metrics[f"{key}.{child}"] = number
    return metrics


def format_sweep_table(rows: list[dict], success: str = "ok", failure: str = "failed") -> str:
    """Shared UI/export comparison: ERP amplitudes only for ERP measurement outputs."""
    erp = any(r.get("output_tool") == "measure_component" or "endpoint_uv" in r for r in rows)
    keys = []
    if not erp:
        for row in rows:
            for key in row.get("metrics", {}):
                if key not in keys:
                    keys.append(key)
        keys = keys[:6]
    has_lat = any(r.get("latency_ms") is not None for r in rows)
    has_ref = any(r.get("vs_ref_r") is not None for r in rows)
    head = ["variant"] + (["endpoint (uV)"] if erp else ["final tool"] + keys + ["final step (s)"])
    if has_lat:
        head.append("latency (ms)")
    if has_ref:
        head.append("vs-ref r")
    head.append("status")
    def cell(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for row in rows:
        values = [row.get("label", "(base)")]
        values += ([row.get("endpoint_uv")] if erp else [row.get("output_tool")] +
                   [row.get("metrics", {}).get(k) for k in keys] + [row.get("output_elapsed_s")])
        if has_lat:
            values.append(row.get("latency_ms"))
        if has_ref:
            values.append(row.get("vs_ref_r"))
        values.append(success if row.get("ok") else f"{failure} {row.get('error', '')}")
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    return "\n".join(lines)
