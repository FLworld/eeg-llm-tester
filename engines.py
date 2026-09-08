"""Session toolbox defaults and explicit natural-language per-tool overrides."""
import importlib.util
import os
from pathlib import Path
import re
import shutil

from tools import TOOL_SCHEMAS


ENGINE_CHOICES = {"mne", "eeglab"}
ENGINE_VALUES = {
    schema["function"]["name"]: tuple(
        schema["function"]["parameters"]["properties"]["engine"].get("enum", ("mne", "erplab")))
    for schema in TOOL_SCHEMAS
    if "engine" in schema["function"]["parameters"].get("properties", {})
}
_TOOL_WORDS = {
    "run_ica": r"\b(?:run_ica|ica|infomax|picard|fastica|independent components?)\b",
    "filter_eeg": r"\b(?:filter_eeg|filter(?:ing|s)?|(?:band|high|low)[ -]?pass)\b",
    "create_bins": r"\b(?:create_bins|bin(?:s|ning)?)\b",
    "create_epochs": r"\b(?:create_epochs|epoch(?:s|ing)?)\b",
    "measure_component": r"\b(?:measure_component|measure|measurement|measuring)\b",
}
_ENGINE_WORD = re.compile(r"\b(?:eeglab-python|eeglab|erplab|mne)\b", re.I)


def engine_for_tool(profile, tool):
    if tool not in ENGINE_VALUES or profile is None:
        return None
    if profile == "mne":
        return "mne"
    if profile == "eeglab":
        return "eeglab" if tool == "run_ica" else "erplab"
    raise ValueError(f"Unknown session engine {profile!r}; choose mne or eeglab.")


def engine_request(request, profile):
    """Resolve explicit toolbox mentions by clause; unclear conflicts require clarification.

    A named step binds an override only to that tool. 'Use MNE' without a step is a request-wide
    override. The session value itself is never changed by these one-request overrides.
    """
    if profile not in ENGINE_CHOICES and profile is not None:
        return {}, [f"Session engine {profile!r} is no longer a profile. Set /engine mne or /engine eeglab."]
    overrides, broad, errors = {}, {}, []
    request = re.sub(r"\beeglab(?:[- ]compatible)?\s+(?:pure[- ]?python|python)\b",
                     "eeglab-python", request, flags=re.I)
    clauses = re.split(r"[,;\n]|(?<!\d)\.(?!\d)|\b(?:then|and|but)\b", request, flags=re.I)
    for clause in clauses:
        engines = {m.group().lower() for m in _ENGINE_WORD.finditer(clause)}
        if not engines:
            continue
        if len(engines) > 1 or re.search(r"\b(?:not|never|avoid|without|except|instead of|rather than)\b", clause, re.I):
            errors.append("Ambiguous toolbox override. Name one engine for each step, e.g. 'use MNE for ICA'.")
            continue
        value = engines.pop()
        targets = [tool for tool, pattern in _TOOL_WORDS.items() if re.search(pattern, clause, re.I)]
        if not targets and not re.search(r"\b(?:use|using|engine|backend|toolbox)\b", clause, re.I):
            continue
        dest = overrides if targets else broad
        for tool in targets or ENGINE_VALUES:
            mapped = engine_for_tool(value, tool) if value in ENGINE_CHOICES else value
            if tool in dest and dest[tool] != mapped:
                errors.append(f"Conflicting engine overrides for {tool}; name one engine for that step.")
            dest[tool] = mapped
    resolved = {tool: (value, "session default") for tool in ENGINE_VALUES
                if (value := engine_for_tool(profile, tool)) is not None}
    resolved.update({tool: (value, "request override") for tool, value in broad.items()})
    resolved.update({tool: (value, "request override") for tool, value in overrides.items()})
    return resolved, errors


def config_steps(config):
    if not isinstance(config, dict):
        return []
    steps = config.get("pipeline", config.get("base_pipeline", []))
    steps = list(steps) if isinstance(steps, list) else []
    if isinstance(config.get("endpoint"), dict):
        steps.append(config["endpoint"])
    return steps


def apply_engine_defaults(config, resolved):
    """Materialize the user's choices when omitted; never replace a conflicting model value."""
    errors = []
    for step in config_steps(config):
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        if tool not in resolved:
            continue
        args = step.setdefault("args", {})
        if not isinstance(args, dict):
            continue
        expected, source = resolved[tool]
        if args.get("engine") is None:
            args["engine"] = expected
        elif args["engine"] != expected:
            errors.append(f"{tool}.engine must be '{expected}' ({source}), got {args['engine']!r}. "
                          "Preserve the session default unless the request explicitly overrides this step.")
    return errors


def validate_engine_values(config):
    errors = []
    for step in config_steps(config):
        if not isinstance(step, dict) or not isinstance(step.get("args"), dict):
            continue
        tool, args = step.get("tool"), step["args"]
        if tool not in ENGINE_VALUES:
            continue
        value = args.get("engine")
        if value is not None and value not in ENGINE_VALUES[tool]:
            errors.append(f"{tool}.engine must be one of {', '.join(ENGINE_VALUES[tool])}; got {value!r}.")
        if tool == "run_ica" and value in ("eeglab", "eeglab-python"):
            algorithm = re.sub(r"[\s_-]+", "", str(args.get("algorithm", "")).lower())
            if algorithm not in ("infomax", "extendedinfomax", "extinfomax"):
                errors.append(f"{value} ICA supports Infomax and extended Infomax only; "
                              "explicitly request MNE for Picard or FastICA. Do not change the algorithm.")
    return errors


def engine_review(config, profile=None):
    rows = []
    for index, step in enumerate(config_steps(config), 1):
        if not isinstance(step, dict) or step.get("tool") not in ENGINE_VALUES:
            continue
        tool, args = step["tool"], step.get("args", {})
        actual = args.get("engine") or "mne"
        session_value = engine_for_tool(profile, tool) if profile in ENGINE_CHOICES else None
        source = " (tool default)" if not args.get("engine") else ""
        if session_value and actual != session_value:
            source = f" (overrides session default `{session_value}`)"
        rows.append(f"{index}. `{tool}`: `{actual}`{source}")
    if not rows:
        return ""
    label = profile.upper() if profile in ENGINE_CHOICES else "tool defaults"
    return f"\n\n**Engines in this plan** (session default: {label}):\n" + "\n".join(rows)


def eeglab_ica_availability():
    """Check prerequisites without starting Octave or changing the selected implementation."""
    missing = []
    for module in ("ica_eeglab", "oct2py"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if not (shutil.which(os.environ.get("OCTAVE_EXECUTABLE", "octave")) or shutil.which("octave-cli")):
        missing.append("Octave")
    roots = [os.environ.get("EEGLAB_PATH"), str(Path.home() / "eeglab"),
             str(Path.home() / "Documents/MATLAB/eeglab")]
    if not any(root and (Path(root) / "functions/sigprocfunc/runica.m").is_file() for root in roots):
        missing.append("EEGLAB runica.m")
    return missing


def engine_runtime_errors(config):
    errors = []
    for step in config_steps(config):
        if not isinstance(step, dict) or step.get("tool") != "run_ica":
            continue
        value = (step.get("args") or {}).get("engine")
        missing = eeglab_ica_availability() if value == "eeglab" else []
        if value == "eeglab-python" and importlib.util.find_spec("ica_eeglab") is None:
            missing = ["ica_eeglab"]
        if missing:
            errors.append(f"ICA engine '{value}' is unavailable in this runtime (missing: "
                          f"{', '.join(missing)}). Install that backend or explicitly request MNE for ICA. "
                          "No engine was substituted.")
    return errors
