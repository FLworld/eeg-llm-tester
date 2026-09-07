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

import numpy as np

from pipeline import validate_pipeline, run_pipeline, _TOOL_PARAMS

MAX_VARIANTS = 24        # hard cap so a grid can never silently explode
SEED_BASE = 97           # matches the project's ICA seed convention
SEED_STEP = 13           # distinct, reproducible seeds for a `repeat` axis


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
        verrs = validate_pipeline({"pipeline": pipe})
        if verrs:
            return [], [f"Variant [{label}] failed validation: " + "; ".join(verrs)]
        variants.append({"label": label, "assignments": assignments, "pipeline": pipe})
    return variants, []


def validate_sweep(spec: dict) -> list[str]:
    """Structural gate for a sweep spec: valid iff expand_sweep produces no errors.

    Also requires an endpoint tool that exists, so every variant yields a comparable
    scalar. Returns a list of problems ([] == valid).
    """
    _, errors = expand_sweep(spec)
    endpoint = spec.get("endpoint") if isinstance(spec, dict) else None
    if not isinstance(endpoint, dict) or endpoint.get("tool") not in _TOOL_PARAMS:
        errors = list(errors) + [
            "endpoint must name a real measurement tool (e.g. measure_component)."]
    return errors


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
    """Execute a sweep: one full pipeline per variant, then the endpoint measurement.

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
    for v in variants:
        _reset_session()
        results, _images = await run_pipeline({"pipeline": v["pipeline"]}, run_step)
        failed = next(
            (r["result"] for r in results
             if isinstance(r["result"], dict) and r["result"].get("ok") is False), None)
        row = {"label": v["label"], "assignments": v["assignments"], "ok": failed is None}
        if failed is not None:
            row["error"] = failed.get("error", "a pipeline step failed")
            rows.append(row)
            continue
        if endpoint.get("tool"):
            ep = await _call(run_step, endpoint["tool"], endpoint.get("args", {}))
            row["endpoint_uv"] = ep.get("amplitude_uv")
            if ep.get("latency_ms") is not None:
                row["latency_ms"] = ep.get("latency_ms")
            if ep.get("ok") is False:
                row["error"] = ep.get("error", "endpoint measurement failed")
        if checkpoint:
            cp = await _call(run_step, "compare_to_checkpoint", {"checkpoint": checkpoint})
            row["vs_ref_r"] = cp.get("min_r")
        rows.append(row)

    return {"ok": True, "rows": rows,
            "summary": _aggregate(spec, rows), "n_variants": len(variants)}
