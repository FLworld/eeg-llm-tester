"""QC linter: flag SILENT mechanical mistakes in a drafted plan/sweep before the human approves.

Two rule sources, both DETERMINISTIC (no model), surfaced as NON-BLOCKING review-card warnings:

  1. Built-in universal traps -- constructions that run without error but are near-universally
     wrong: filtering after epoching, re-referencing after ICA, resampling after epoching, and
     (using the scoped recording) measuring a channel or binning an event code that isn't present.
     These enforce the standard pipeline ORDER the project already prescribes (Modelfile SYSTEM /
     prompt.py: "filter -> re-reference -> artifact handling -> epoch; filter+reref before ICA")
     and the tool docstrings ("resample before epoching"); they are field consensus, not a
     debatable parameter choice.

  2. Lab rules -- a lab's own conventions, opted into via a structured lab_rules.(json|yaml) file
     (require_steps / forbid_steps / order / require_before / param_equals). The lab writes them,
     so enforcing them -- even value conventions like "reference must be mastoids" -- is faithful,
     not the model second-guessing the science.

What it deliberately does NOT do: judge scientific PARAMETER choices (a cutoff, a threshold, an
epoch length). Those are the expert's call. It only checks order / presence / window-containment.
Everything is a warning; the expert overrides by just running.
"""
from __future__ import annotations

import json
import os

# --------------------------------------------------------------------------- #
# helpers over an ordered step list ({"tool","args"})
# --------------------------------------------------------------------------- #
def _index(steps: list, tool: str) -> int | None:
    for i, s in enumerate(steps):
        if isinstance(s, dict) and s.get("tool") == tool:
            return i
    return None


def _indices(steps: list, tool: str) -> list[int]:
    return [i for i, s in enumerate(steps) if isinstance(s, dict) and s.get("tool") == tool]


def _args(step) -> dict:
    return step.get("args", {}) or {} if isinstance(step, dict) else {}


def _codes_of_bin(b) -> set[int]:
    """Concrete integer codes referenced by ONE create_bins entry."""
    out: set[int] = set()
    for c in (b.get("codes") if isinstance(b, dict) else None) or []:
        if isinstance(c, list) and len(c) == 2 and all(isinstance(x, int) for x in c):
            out.update(range(c[0], c[1] + 1))
        elif isinstance(c, int):
            out.add(c)
    return out


# --------------------------------------------------------------------------- #
# built-in universal traps
# --------------------------------------------------------------------------- #
def _builtin_traps(steps: list, scope: dict | None) -> list[str]:
    w: list[str] = []
    i_filt = _index(steps, "filter_eeg")
    i_epoch = _index(steps, "create_epochs")
    i_reref = _index(steps, "set_reference")
    i_ica = _index(steps, "run_ica")
    i_applyica = _index(steps, "apply_ica")
    i_resample = _index(steps, "resample")

    # order traps
    if i_filt is not None and i_epoch is not None and i_filt > i_epoch:
        w.append("filter_eeg runs AFTER create_epochs: filtering epoched data distorts epoch "
                 "edges. Filter before epoching.")
    ica_first = min([x for x in (i_ica, i_applyica) if x is not None], default=None)
    if i_reref is not None and ica_first is not None and i_reref > ica_first:
        w.append("set_reference runs AFTER ICA: ICA is bound to the reference it was fit on, so "
                 "re-referencing afterward invalidates it. Re-reference before run_ica.")
    if i_resample is not None and i_epoch is not None and i_resample > i_epoch:
        w.append("resample runs AFTER create_epochs: resampling epoched data jitters event "
                 "timing. Resample before epoching.")

    # existence traps (need the scoped recording)
    chans = set((scope or {}).get("ch_names") or [])
    if chans:
        for idx in _indices(steps, "measure_component"):
            for ch in _args(steps[idx]).get("channels") or []:
                if ch not in chans:
                    w.append(f"measure_component channel '{ch}' is not in this recording "
                             f"(scoped channels do not include it) -> it would measure nothing.")
    codes_present = set(int(c) for c in ((scope or {}).get("codes") or {})
                        if str(c).lstrip("-").isdigit())
    if codes_present:
        for idx in _indices(steps, "create_bins"):
            for b in _args(steps[idx]).get("bins") or []:
                referenced = _codes_of_bin(b)
                # A range like [1,40] is fine if ANY code hits; the trap is a bin with NONE
                # present -> it would be empty. (Partial coverage of a range is legitimate.)
                if referenced and not (referenced & codes_present):
                    label = b.get("label", "?") if isinstance(b, dict) else "?"
                    w.append(f"create_bins bin '{label}' references only event codes absent from "
                             f"this recording -> it would be empty.")

    # window-containment trap: measure window inside the epoch window
    if i_epoch is not None:
        ea = _args(steps[i_epoch])
        etmin, etmax = ea.get("tmin"), ea.get("tmax")
        if isinstance(etmin, (int, float)) and isinstance(etmax, (int, float)):
            for idx in _indices(steps, "measure_component"):
                ma = _args(steps[idx])
                mtmin, mtmax = ma.get("tmin"), ma.get("tmax")
                if (isinstance(mtmin, (int, float)) and isinstance(mtmax, (int, float))
                        and (mtmin < etmin - 1e-9 or mtmax > etmax + 1e-9)):
                    w.append(f"measure_component window [{mtmin}, {mtmax}] s falls outside the "
                             f"epoch window [{etmin}, {etmax}] s -> it would measure off the edge.")
    return w


# --------------------------------------------------------------------------- #
# structured lab rules
# --------------------------------------------------------------------------- #
def _lab_rule_checks(steps: list, lab_rules: dict) -> list[str]:
    w: list[str] = []
    present = [s.get("tool") for s in steps if isinstance(s, dict)]
    present_set = set(present)

    for tool in lab_rules.get("require_steps") or []:
        if tool not in present_set:
            w.append(f"lab rule: plan is missing a required step '{tool}'.")
    for tool in lab_rules.get("forbid_steps") or []:
        if tool in present_set:
            w.append(f"lab rule: step '{tool}' is not allowed by this lab's conventions.")

    # relative order of tools that ARE present must follow the listed sequence
    order = lab_rules.get("order") or []
    order_present = [t for t in order if t in present_set]
    positions = {t: _index(steps, t) for t in order_present}
    for a, b in zip(order_present, order_present[1:]):
        if positions[a] is not None and positions[b] is not None and positions[a] > positions[b]:
            w.append(f"lab rule: '{a}' must come before '{b}' (lab pipeline order).")

    for a, b in (lab_rules.get("require_before") or {}).items():
        ia, ib = _index(steps, a), _index(steps, b)
        if ia is not None and ib is not None and ia > ib:
            w.append(f"lab rule: '{a}' must come before '{b}'.")

    for tool, wants in (lab_rules.get("param_equals") or {}).items():
        for idx in _indices(steps, tool):
            args = _args(steps[idx])
            for param, val in (wants or {}).items():
                if param in args and args[param] != val:
                    w.append(f"lab rule: {tool}.{param} should be {val!r} "
                             f"(lab convention), plan has {args[param]!r}.")
    return w


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def lint_plan(steps: list, scope: dict | None = None,
              lab_rules: dict | None = None) -> list[str]:
    """Return QC warnings for an ordered step list ({"tool","args"}). Empty == clean.
    Non-blocking: the caller shows these for review; the human decides."""
    if not isinstance(steps, list):
        return []
    warnings = _builtin_traps(steps, scope)
    if lab_rules:
        warnings += _lab_rule_checks(steps, lab_rules)
    return warnings


def lint_config(config: dict, scope: dict | None = None,
                lab_rules: dict | None = None) -> list[str]:
    """lint_plan for a plan config ({"pipeline"}) or a sweep spec ({"base_pipeline"})."""
    if not isinstance(config, dict):
        return []
    steps = config.get("pipeline")
    if steps is None:
        steps = config.get("base_pipeline")
    return lint_plan(steps or [], scope, lab_rules)


def load_lab_rules(path: str) -> dict | None:
    """Load a lab_rules.(json|yaml|yml) file. Returns the rules dict, or None if absent/bad."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            if path.endswith((".yaml", ".yml")):
                import yaml
                return yaml.safe_load(fh) or {}
            return json.load(fh)
    except Exception:
        return None


def find_lab_rules(*dirs: str) -> tuple[dict | None, str | None]:
    """Look for lab_rules.(json|yaml|yml) in the given dirs (first hit wins).
    Returns (rules, path) or (None, None)."""
    for d in dirs:
        if not d:
            continue
        for name in ("lab_rules.json", "lab_rules.yaml", "lab_rules.yml"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                r = load_lab_rules(p)
                if r is not None:
                    return r, p
    return None, None
