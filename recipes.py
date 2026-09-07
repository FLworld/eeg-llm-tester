"""Named, reusable plan/sweep recipes (templates).

A recipe is a saved planner artifact -- a `/plan` pipeline config or a `/sweep` spec -- stored
under a human name so it can be re-invoked without the model. The classic case: build a "filter
sweep" once, name it, then run it on any recording later.

Data-portable by design: a recipe stores the processing + axes, NOT a fixed input file. On run,
`rebind_to_current` points its `load_eeg` at whatever recording the user currently has loaded/
scoped, so one "filter sweep" applies to every subject.

Stored as one JSON file per recipe in `recipes/`:
    {"name": <str>, "kind": "sweep"|"plan", "created": <iso>, "spec": {...}}
filename = "<slug>.<kind>.json".
"""
from __future__ import annotations

import datetime
import json
import os
import re

# Honors STATE_DIR (a mounted volume in a container) so saved recipes survive restarts;
# defaults to the module dir for a plain local checkout.
RECIPES_DIR = os.path.join(
    os.environ.get("STATE_DIR", os.path.dirname(os.path.abspath(__file__))), "recipes")


def slugify(name: str) -> str:
    """'Filter Sweep' -> 'filter-sweep'. Matching is slug-based so spaces/case don't matter."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")


def _steps_key(spec: dict) -> str:
    """The list of steps: 'base_pipeline' for a sweep spec, 'pipeline' for a plan config."""
    return "base_pipeline" if "base_pipeline" in spec else "pipeline"


def save_recipe(name: str, kind: str, spec: dict) -> dict:
    """Persist a spec under `name`. kind is 'sweep' or 'plan'. Returns {ok, path, slug}."""
    slug = slugify(name)
    if not slug:
        return {"ok": False, "error": "a recipe needs a non-empty name."}
    if kind not in ("sweep", "plan"):
        return {"ok": False, "error": f"kind must be 'sweep' or 'plan', got {kind!r}."}
    spec = {k: v for k, v in (spec or {}).items() if k != "_planner_meta"}  # drop telemetry
    os.makedirs(RECIPES_DIR, exist_ok=True)
    path = os.path.join(RECIPES_DIR, f"{slug}.{kind}.json")
    record = {"name": name, "kind": kind,
              "created": datetime.datetime.now().isoformat(timespec="seconds"), "spec": spec}
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    return {"ok": True, "path": path, "slug": slug}


def _summary(record: dict) -> str:
    """One-line description of a recipe for listing."""
    spec = record.get("spec", {})
    if record["kind"] == "sweep":
        axes = spec.get("axes") or []
        parts = []
        for a in axes:
            v = a.get("values")
            v = v if v is not None else f"repeat={a.get('repeat')}"
            parts.append(f"{a.get('tool')}.{a.get('param')}={v}")
        return "sweep: " + ("; ".join(parts) or "(no axes)")
    steps = spec.get("pipeline") or []
    return "plan: " + " -> ".join(s.get("tool", "?") for s in steps)


def list_recipes() -> list[dict]:
    """All saved recipes: [{name, kind, slug, summary, created}], newest first."""
    if not os.path.isdir(RECIPES_DIR):
        return []
    out = []
    for fn in os.listdir(RECIPES_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(RECIPES_DIR, fn)) as fh:
                rec = json.load(fh)
            out.append({"name": rec["name"], "kind": rec["kind"],
                        "slug": slugify(rec["name"]), "created": rec.get("created", ""),
                        "summary": _summary(rec)})
        except Exception:
            continue
    return sorted(out, key=lambda r: r["created"], reverse=True)


def load_recipe(name: str, kind: str | None = None) -> dict | None:
    """Load a recipe's spec by name (slug match). If kind is given, restrict to it.
    Returns {"kind","name","spec"} or None."""
    want = slugify(name)
    if not os.path.isdir(RECIPES_DIR):
        return None
    for fn in os.listdir(RECIPES_DIR):
        if not fn.endswith(".json"):
            continue
        k = "sweep" if fn.endswith(".sweep.json") else "plan" if fn.endswith(".plan.json") else None
        if k is None or (kind and k != kind):
            continue
        base = fn[: -(len(k) + 6)]  # strip ".<kind>.json"
        if base == want:
            try:
                with open(os.path.join(RECIPES_DIR, fn)) as fh:
                    rec = json.load(fh)
                return {"kind": k, "name": rec.get("name", name), "spec": rec.get("spec", {})}
            except Exception:
                return None
    return None


def rebind_to_current(spec: dict, filepath: str | None) -> dict:
    """Point the recipe's load_eeg at the CURRENT recording (portability). If the spec has a
    load_eeg step, overwrite its filepath; if not, prepend one. No-op if filepath is None
    (caller should warn: a recipe needs a loaded/scoped recording to run)."""
    if not filepath:
        return spec
    key = _steps_key(spec)
    steps = spec.get(key) or []
    loads = [s for s in steps if isinstance(s, dict) and s.get("tool") == "load_eeg"]
    if loads:
        for s in loads:
            s.setdefault("args", {})["filepath"] = filepath
    else:
        steps = [{"tool": "load_eeg", "args": {"filepath": filepath}}] + steps
        spec[key] = steps
    return spec
