"""Deterministic, config-driven pipeline execution.

The agentic loop in app.py lets the model trigger tool calls live: whatever the
model emits as a tool_call runs immediately, with no gate. This module is the
alternative role. The model only PROPOSES a pipeline config (inert JSON, nothing
runs); a human reviews and approves it; then deterministic code executes the
exact steps through the same `call_tool` dispatcher.

Because no model turn sits between approval and execution, the run does exactly
what was approved -- the model can no longer substitute a default, reorder, or
add an unrequested step at runtime. Faithfulness is a property of *who* feeds
the executor: `message["tool_calls"]` (model, live) vs. an approved
`config["pipeline"]` (you, reviewed).
"""

from __future__ import annotations

import inspect
import json
import re

import ollama

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

# Allowed parameter names (and their JSON-schema types) per tool, derived from the
# Ollama schemas so this stays in sync with tools.py automatically.
_TOOL_PARAMS: dict[str, set] = {
    s["function"]["name"]: set(s["function"]["parameters"].get("properties", {}))
    for s in TOOL_SCHEMAS
}
_TOOL_PARAM_TYPES: dict[str, dict] = {
    s["function"]["name"]: {
        pname: prop.get("type")
        for pname, prop in s["function"]["parameters"].get("properties", {}).items()
    }
    for s in TOOL_SCHEMAS
}
_TOOL_REQUIRED: dict[str, set] = {
    s["function"]["name"]: set(s["function"]["parameters"].get("required", []))
    for s in TOOL_SCHEMAS
}
_JSON_PY_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _tool_catalog() -> str:
    """One line per tool: name(params) -- description. '*' marks a required param."""
    lines = []
    for s in TOOL_SCHEMAS:
        f = s["function"]
        props = f["parameters"].get("properties", {})
        required = set(f["parameters"].get("required", []))
        sig = ", ".join(f"{k}*" if k in required else k for k in props)
        lines.append(f"- {f['name']}({sig}): {f['description']}")
    return "\n".join(lines)


_SENTINEL = inspect.Parameter.empty


def list_tool_names() -> list[str]:
    """Every planner-visible tool name (from the schema registry)."""
    return [s["function"]["name"] for s in TOOL_SCHEMAS]


def describe_tool(name: str) -> dict:
    """Deterministic per-tool parameter reference: for each parameter, whether it is REQUIRED
    and, if optional, its DEFAULT value.

    Reliability by construction: `required` + `type` + `description` come from the schema
    (`TOOL_SCHEMAS`, what the model reads), and the DEFAULT comes from the Python function
    signature (`TOOL_FUNCTIONS`). Defaults are NOT in the schema -- which is exactly why the
    model cannot state them reliably and this must be a lookup, not a generation.

    Returns {ok, name, description, params:[{name, required, default, has_default, type,
    description}]} or {ok: False, error, suggestions} for an unknown tool.
    """
    schema = next((s for s in TOOL_SCHEMAS if s["function"]["name"] == name), None)
    fn = TOOL_FUNCTIONS.get(name)
    if schema is None or fn is None:
        near = [n for n in list_tool_names() if name and name.lower() in n.lower()]
        return {"ok": False, "error": f"Unknown tool {name!r}.",
                "suggestions": near[:8] or list_tool_names()}
    f = schema["function"]
    props = f["parameters"].get("properties", {})
    required = set(f["parameters"].get("required", []))
    sig = inspect.signature(fn).parameters
    params = []
    for pname, prop in props.items():
        p = sig.get(pname)
        has_default = p is not None and p.default is not _SENTINEL
        params.append({
            "name": pname,
            "required": pname in required,
            "has_default": has_default and pname not in required,
            "default": p.default if has_default else None,
            "type": prop.get("type"),
            "description": prop.get("description", ""),
        })
    # required first, then optional, preserving schema order within each group
    params.sort(key=lambda x: (not x["required"],))
    return {"ok": True, "name": name, "description": f.get("description", ""), "params": params}


PLANNER_PROMPT = f"""You are the planning component of an EEG analysis toolkit. You convert a \
user's request into an explicit, ordered pipeline of tool calls. You do NOT run anything: you \
emit a plan that a human reviews and approves before it executes.

Output ONLY a single JSON object, no prose and no markdown, of this exact shape:
{{"pipeline": [{{"tool": "<tool_name>", "args": {{<arguments>}}}}], "notes": ["<assumption>"]}}

Use ONLY these tools and ONLY their listed parameters ('*' = required):
{_tool_catalog()}

Rules -- follow them exactly; they are what make the plan trustworthy:
- Include ONLY the steps the user asked for. Do NOT add preprocessing (filtering, \
referencing, artifact removal, epoching) that the user did not request.
- For any parameter the user specified, use that EXACT value.
- For any parameter the user did NOT specify, OMIT the key entirely so the tool's own \
default applies. Do NOT guess a value, and do NOT include a key set to null, false, or an \
empty list to signal "unused" -- leave it out.
- If a recording must be loaded and none is loaded yet, the first step is load_eeg.
- Put every assumption you made, and every required parameter you had to leave to the tool \
default, into "notes" so the human can catch it at review.
- If a REQUIRED parameter has no default and the user did NOT specify it (e.g. run_ica's \
`algorithm`), you must choose a valid value to keep the plan runnable -- but that is a scientific \
choice the user did not make, so you MUST record it in "notes" explicitly, e.g. "run_ica algorithm \
not specified; chose 'fastica'." Never let a scientific choice you made go unstated.
- If the request cannot be expressed with these tools, return \
{{"pipeline": [], "notes": ["<why>"]}}.

Argument derivation -- when the request describes these, translate to the exact argument:
- Time windows for measure_component (tmin/tmax) are in SECONDS. If the request gives a window \
in milliseconds, CONVERT: 110-150 ms -> tmin 0.11, tmax 0.15; 130-200 ms -> 0.13, 0.20. Never \
pass the raw ms number. (measure_component's `baseline`, however, IS in ms, e.g. [-200, 0].)
- Butterworth roll-off/slope in dB/octave -> filter order, by this table (never guess or leave \
it unset): 6 dB/oct=order 1, 12=2, 18=3, 24=4, 30=5, 36=6, 48=8. The dB/octave value is already \
final; do NOT double it for zero-phase/non-causal (two-pass) filtering.
- A high-pass-only filter (a low/high-pass cutoff given with no upper edge) -> set h_freq to null \
explicitly. Both l_freq and h_freq default to null; at least one must be set or filter_eeg will \
error.
- "reproduce ERPLAB / MATLAB / pop_basicfilter exactly" (or match a MATLAB filter bit-for-bit) -> \
filter_eeg engine="erplab".
- A CONTRAST between two conditions ("A vs B", "A minus B", e.g. "faces vs cars") is NOT one \
epoching call with a list of conditions. create_epochs.event_id and compute_erp.event_id are a \
SINGLE label (string), never a list, and create_epochs.baseline is a boolean (true/false), never a \
window. Build a contrast as three steps: (1) create_bins with one bin per condition -- \
bins=[{{"label": "B1", "codes": [[lo,hi], ...]}}, {{"label": "B2", "codes": [...]}}], taking the \
code ranges for each named condition from the scoped codebook in the context (do NOT invent codes; \
if no codebook is present, say so in notes); (2) create_epochs with tmin/tmax and NO event_id (it \
cuts one epoch set per bin); (3) compute_difference_erp with event_id_a="B1", event_id_b="B2" (the two bin labels) AND tmin/tmax \
set to the SAME epoch window in seconds as create_epochs (compute_difference_erp REQUIRES tmin and \
tmax -- never omit them). Then measure_component scores that difference wave. Use compute_erp (single \
event_id string) only for a single-condition ERP, never for a contrast. Reminder: create_epochs.baseline \
is a boolean -- pass true or false, never a [lo, hi] window.
"""


PLANNER_SWEEP_PROMPT = f"""You are the planning component of an EEG analysis toolkit. You \
convert a user's request into a PARAMETER SWEEP: one base pipeline plus one or more axes that \
vary a parameter, so the user can compare how a choice (e.g. a filter cutoff, filter window, or \
an ICA seed) changes a downstream measurement. You do NOT run anything: you emit a spec that a \
human reviews and approves before it executes.

Output ONLY a single JSON object, no prose and no markdown, of this exact shape:
{{"base_pipeline": [{{"tool": "<tool_name>", "args": {{<arguments>}}}}],
 "axes": [{{"tool": "<tool_name>", "param": "<param>", "values": [<v1>, <v2>]}}],
 "endpoint": {{"tool": "measure_component", "args": {{<arguments>}}}},
 "notes": ["<assumption>"]}}

Use ONLY these tools and ONLY their listed parameters ('*' = required):
{_tool_catalog()}

Rules -- follow them exactly; they are what make the sweep trustworthy:
- The "base_pipeline" is an ordinary ordered pipeline; it obeys the same rules as a normal plan:
  include ONLY the steps the user asked for, use EXACT specified values, OMIT any parameter the
  user did not specify so the tool default applies, and start with load_eeg if a recording must
  be loaded. Do NOT put the swept values in base_pipeline -- put a sensible single value (or omit
  the key), and let the axes vary it.
- Each entry in "axes" varies ONE real parameter of ONE tool that appears in base_pipeline:
  * a grid axis has "values": [...] (the list of values to try), OR
  * a stochastic axis (for a random step like ICA seeds) has "repeat": N (an integer >= 2); the
    engine supplies N distinct seeds, so use "repeat" with param "random_state", not "values".
  * "param" MUST be a real parameter of "tool" (from the catalog above). If several steps use the
    same tool, add "index": k to pick the k-th (0-based).
- "endpoint" is the SINGLE measurement compared across variants -- normally measure_component with
  the time window and channels the user named. Exactly one endpoint.
- Multiple grid axes multiply (a 3-value and a 2-value axis = 6 variants). Keep the total small
  (<= 24). Put the variant count and every assumption in "notes".
- If the request is not a sweep (no parameter to vary), return {{"base_pipeline": [], "axes": [], \
"notes": ["not a sweep: <why>"]}}.

Argument derivation -- same as normal planning:
- measure_component tmin/tmax are in SECONDS: convert ms -> s (110-150 ms -> 0.11, 0.15). Its \
`baseline` is the exception and stays in ms (e.g. [-200, 0]).
- Butterworth roll-off in dB/octave -> iir_order (6=1,12=2,18=3,24=4,30=5,36=6,48=8).
- A high-pass-only filter -> set h_freq to null explicitly.
- "reproduce ERPLAB/MATLAB exactly" -> filter_eeg engine="erplab".
- Every axis tool MUST also appear as a step in base_pipeline (an axis varies a parameter of a \
step that is actually in the pipeline). If you sweep resample.sfreq, include a resample step; \
if you sweep run_ica.algorithm, include a run_ica step. Put a placeholder value there; the axis \
overrides it.
"""


def _engine_directive(engine: str | None) -> str:
    """A session-level toolbox-convention directive appended to the planner system prompt.

    The human sets it once (`/engine`); it is explicit and reviewable, never auto-detected.
    Tells the model which `engine` to pass on engine-capable tools when the request itself
    does not name a toolbox -- so the per-call override still wins.
    """
    if not engine:
        return ""
    return (
        f"\n\nSESSION ENGINE DEFAULT: '{engine}'. For every tool that accepts an `engine` "
        f"parameter (filter_eeg, resample, create_bins, create_epochs, measure_component, "
        f"run_ica, ...), set engine='{engine}' UNLESS the request explicitly names a different "
        f"toolbox for that step. This is the user's chosen convention for the session."
    )


def _planner_messages(system: str, request: str, ctx_text: str | None) -> list[dict]:
    user = request
    if ctx_text:
        user = (
            "Context from the EEG knowledge base (reference only):\n"
            f"{ctx_text}\n\n----\nRequest: {request}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# Tools that yield the single scalar an endpoint compares across variants.
_MEASUREMENT_TOOLS = sorted(
    n for n in _TOOL_PARAMS if "measure" in n) or ["measure_component"]
_ALL_PARAM_NAMES = sorted({p for params in _TOOL_PARAMS.values() for p in params})


def _sweep_schema() -> dict:
    """JSON Schema handed to Ollama as `format=` so the sweep spec is grammar-CONSTRAINED at
    decode time (structured outputs), not merely validated afterward.

    Derived from the twin registry (`_TOOL_PARAMS`) so it never drifts. It makes whole error
    classes impossible to EMIT: an axis tool that isn't a real tool, an axis param that isn't a
    real parameter of ANY tool, an endpoint that isn't a measurement tool, a scalar 'values'
    item, a non-integer 'repeat'. Cross-tool checks (param real but not on THAT tool; axis tool
    absent from base_pipeline) can't be expressed in a static schema and are left to
    validate_sweep + the repair loop. Args objects stay unconstrained (per-tool arg grammars
    would be huge and fight legitimate params)."""
    tool_names = sorted(_TOOL_PARAMS)
    step = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": tool_names},
            "args": {"type": "object"},
        },
        "required": ["tool", "args"],
    }
    axis = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": tool_names},
            "param": {"type": "string", "enum": _ALL_PARAM_NAMES},
            "values": {"type": "array", "items": {"type": ["number", "string", "boolean"]}},
            "repeat": {"type": "integer"},
            "index": {"type": "integer"},
        },
        "required": ["tool", "param"],
    }
    endpoint = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": _MEASUREMENT_TOOLS},
            "args": {"type": "object"},
        },
        "required": ["tool", "args"],
    }
    # endpoint is NOT required: a legitimate decline ("not a sweep") returns empty axes and no
    # endpoint, and must not be forced to fabricate one.
    return {
        "type": "object",
        "properties": {
            "base_pipeline": {"type": "array", "items": step},
            "axes": {"type": "array", "items": axis},
            "endpoint": endpoint,
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["base_pipeline", "axes"],
    }


def strip_unknown_args(steps) -> list[str]:
    """Remove arg KEYS that are not real parameters of the step's tool (mutates in place).
    Returns the "tool.key" names dropped.

    The schema can constrain arg *values* but leaves each step's `args` object open, so the model
    can still emit a bogus arg key (e.g. an instruction-to-itself as a key). That key is provably
    invalid -- `validate_pipeline` would reject the whole plan and the repair loop might not fix
    it. Dropping it deterministically (and noting it) is more reliable than repairing it, and is
    faithful: the human sees the dropped keys in the review card.
    """
    dropped: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        name = step.get("tool")
        args = step.get("args")
        if name not in _TOOL_PARAMS or not isinstance(args, dict):
            continue
        for key in list(args):
            if key not in _TOOL_PARAMS[name]:
                dropped.append(f"{name}.{key}")
                del args[key]
    return dropped


def _sanitize_sweep(spec: dict) -> list[str]:
    """Drop bogus arg keys from a sweep spec's base_pipeline + endpoint. Returns dropped keys
    and records them in the spec's notes (visible, not silent)."""
    if not isinstance(spec, dict):
        return []
    dropped = strip_unknown_args(spec.get("base_pipeline"))
    ep = spec.get("endpoint")
    if isinstance(ep, dict):
        dropped += strip_unknown_args([ep])
    if dropped:
        spec.setdefault("notes", []).append(
            "dropped unrecognized args: " + ", ".join(sorted(set(dropped))))
    return dropped


def propose_sweep(model: str, request: str, ctx_text: str | None = None,
                  engine: str | None = None, max_repairs: int = 2) -> dict:
    """Translate a request into a sweep spec. Runs nothing.

    (a) Schema-CONSTRAINED decoding: `_sweep_schema()` is passed as `format=`, so the model can
    only emit a structurally valid spec (real tools, real param names, typed values).
    (b) Validate-and-REPAIR: the semantic checks the grammar can't express (cross-tool params,
    axis tool missing from base_pipeline, endpoint sanity, variant cap) run through
    `validate_sweep`; on failure the errors are fed back and the model regenerates, up to
    `max_repairs` times. Deterministic-gated self-correction -- no reliance on first-shot wording.

    Returns the spec; when repair is exhausted the last (still-invalid) spec is returned so the
    caller surfaces the errors. A `_planner_meta` key records attempts + whether it validated.
    """
    from sweep import validate_sweep  # lazy: sweep imports pipeline (avoid import cycle)

    schema = _sweep_schema()
    messages = _planner_messages(PLANNER_SWEEP_PROMPT + _engine_directive(engine),
                                 request, ctx_text)
    spec, errors = {}, ["no attempt"]
    for attempt in range(max_repairs + 1):
        resp = ollama.chat(model=model, messages=messages, format=schema)
        content = resp["message"].get("content") or ""
        spec = _parse_config(content)
        # Empty axes == "nothing to sweep" == an intentional decline. Return as-is and do NOT
        # repair: pushing the model to add an axis it didn't produce is fabrication pressure,
        # which violates faithfulness. (Cost: a rare genuine whiff -- axes dropped by mistake --
        # is not auto-recovered; the user re-runs. Protecting against fabrication wins.)
        if isinstance(spec, dict) and not (spec.get("axes") or []):
            spec.setdefault("_planner_meta", {}).update(
                {"attempts": attempt + 1, "valid": True, "declined": True})
            return spec
        _sanitize_sweep(spec)  # drop bogus arg KEYS before validating (deterministic, noted)
        errors = validate_sweep(spec)
        if not errors:
            spec.setdefault("_planner_meta", {})["attempts"] = attempt + 1
            spec["_planner_meta"]["valid"] = True
            return spec
        if attempt < max_repairs:  # feed the failures back and let it fix them
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                "That sweep spec is invalid:\n- " + "\n- ".join(errors)
                + "\nReturn a corrected spec as JSON only, fixing exactly these problems and "
                "changing nothing else.")})
    if isinstance(spec, dict):
        spec.setdefault("_planner_meta", {})["attempts"] = max_repairs + 1
        spec["_planner_meta"]["valid"] = False
        spec["_planner_meta"]["errors"] = errors
    return spec


def _parse_config(raw: str) -> dict:
    """Parse the planner's JSON. Falls back to brace extraction if needed."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"pipeline": [], "notes": ["Planner did not return valid JSON."]}


def propose_pipeline(model: str, request: str, ctx_text: str | None = None,
                     engine: str | None = None, max_repairs: int = 2) -> dict:
    """Ask the model to translate a request into a pipeline config. Runs nothing.

    The JSON constraint guarantees syntax, not that an argument belongs to the selected tool or
    has its required type. Validate each proposal and give a bounded correction turn when the
    model emits a structurally invalid plan. This mirrors ``propose_sweep``: no model call occurs
    after approval, and an exhausted repair still returns an invalid plan for the UI to reject.
    """
    messages = _planner_messages(PLANNER_PROMPT + _engine_directive(engine), request, ctx_text)
    config, errors, attempts = {}, ["no attempt"], 0
    for attempt in range(max_repairs + 1):
        attempts = attempt + 1
        resp = ollama.chat(model=model, messages=messages, format="json")
        content = resp["message"].get("content") or ""
        config = _parse_config(content)
        errors = validate_pipeline(config)
        if not errors:
            if isinstance(config, dict):
                config.setdefault("_planner_meta", {}).update(
                    {"attempts": attempt + 1, "valid": True})
            return config
        # An empty plan is an intentional refusal, not a malformed proposal to pressure into a
        # fabricated analysis. The caller keeps presenting its existing explanatory error.
        if not isinstance(config, dict) or not config.get("pipeline"):
            break
        if attempt < max_repairs:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                "That pipeline is invalid:\n- " + "\n- ".join(errors)
                + "\nReturn a corrected pipeline JSON only, fixing exactly these problems and "
                "changing nothing else.")})
    if isinstance(config, dict):
        config.setdefault("_planner_meta", {}).update(
            {"attempts": attempts, "valid": False, "errors": errors})
    return config


def validate_pipeline(config: dict) -> list[str]:
    """Return a list of problems with a proposed config. Empty list == valid.

    This is the structural gate: every tool name must exist and every argument
    key must be a real parameter of that tool, so `run_pipeline` can never be
    handed a call that tools.py would reject.
    """
    if not isinstance(config, dict):
        return ["Planner did not return a JSON object."]
    steps = config.get("pipeline")
    if not isinstance(steps, list) or not steps:
        return ["No pipeline steps were produced."]

    errors: list[str] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step {i}: not an object.")
            continue
        name = step.get("tool")
        args = step.get("args", {})
        if name not in TOOL_FUNCTIONS:
            errors.append(f"Step {i}: unknown tool '{name}'.")
            continue
        if not isinstance(args, dict):
            errors.append(f"Step {i} ({name}): 'args' must be an object.")
            continue
        unknown = set(args) - _TOOL_PARAMS.get(name, set())
        if unknown:
            errors.append(f"Step {i} ({name}): unknown args {sorted(unknown)}.")
        # Required args must be present and non-null (null == omit, so counts as missing).
        supplied = {k for k, v in args.items() if v is not None}
        missing = _TOOL_REQUIRED.get(name, set()) - supplied
        if missing:
            errors.append(f"Step {i} ({name}): missing required args {sorted(missing)}.")
        # Type-check the known args against the schema (null == omit, allowed).
        # A schema "type" may be a single string or a list of accepted types (e.g.
        # ["string", "array"]); the value is OK if it matches ANY listed, checkable type.
        for key, value in args.items():
            if key in unknown or value is None:
                continue
            expected = _TOOL_PARAM_TYPES.get(name, {}).get(key)
            allowed = expected if isinstance(expected, list) else [expected]
            ok_type, checkable = False, False
            for exp in allowed:
                pytypes = _JSON_PY_TYPES.get(exp)
                if pytypes is None:
                    continue
                checkable = True
                # For numbers: reject bool AND non-numbers ("400"); ints/floats ok.
                if exp in ("number", "integer"):
                    if not isinstance(value, bool) and isinstance(value, (int, float)):
                        ok_type = True
                elif isinstance(value, pytypes):
                    ok_type = True
            if checkable and not ok_type:
                errors.append(
                    f"Step {i} ({name}): arg '{key}' should be {expected}, "
                    f"got {type(value).__name__}."
                )
    return errors


async def run_pipeline(config: dict, run_step) -> tuple[list[dict], list[tuple[str, str]]]:
    """Execute an approved config deterministically, step by step.

    `run_step(name, args)` must be an async callable returning (result, image_b64)
    -- in app.py that is `_run_tool`, the same dispatcher the agentic loop uses.
    No model is consulted here: the steps and their arguments come entirely from
    `config`. Stops at the first failing step, since later steps depend on it.

    Returns (results, images) where results is a list of
    {"tool", "args", "result"} and images is a list of (tool_name, base64_png).
    """
    results: list[dict] = []
    images: list[tuple[str, str]] = []
    for step in config["pipeline"]:
        name = step["tool"]
        args = step.get("args", {}) or {}
        result, image_b64 = await run_step(name, args)
        results.append({"tool": name, "args": args, "result": result})
        if image_b64:
            images.append((name, image_b64))
        if isinstance(result, dict) and result.get("ok") is False:
            break  # hard stop: downstream steps depend on this one
    return results, images


def try_parse_config(text: str) -> dict | None:
    """If `text` is a JSON pipeline config the user wrote by hand, return it.

    This is the fully model-free path: when you know exactly which steps you want,
    paste the config and no planner is consulted at all. Returns None if `text` is
    not a JSON object with a "pipeline" key (i.e. treat it as natural language).
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "pipeline" in obj else None


def extract_inline_tool_calls(content: str) -> list[dict]:
    """Recover a tool call a model wrote as text instead of via native tool_calls.

    Coder-tuned models (e.g. qwen2.5-coder) tend to ignore the <tool_call>
    protocol and emit the call as a ```json fenced block in their reply, so
    Ollama returns an empty tool_calls and the call never runs. This pulls the
    FIRST valid call back out and returns it shaped like an Ollama tool_calls
    entry: [{"function": {"name": ..., "arguments": {...}}}]. Returns [] if none.
    """
    if not content:
        return []
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidates += re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", content)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            name = obj.get("name")
            args = obj.get("arguments", obj.get("parameters"))
            if name in TOOL_FUNCTIONS and isinstance(args, dict):
                return [{"function": {"name": name, "arguments": args}}]
    return []


def format_config(config: dict) -> str:
    """Pretty-print a config for the approval message."""
    return json.dumps(config, indent=2, default=str)
