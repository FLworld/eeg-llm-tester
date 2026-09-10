"""Chainlit chat app wiring the EEG model, RAG, and MNE tools together.

Run with:  chainlit run app.py --port 8000
"""

from __future__ import annotations

import base64
import datetime
import inspect
import json
import os
import re
import shutil

import chainlit as cl
import ollama

import sys as _sys

# Guarantee this directory is on sys.path before importing project modules, so their lazy
# sibling imports (e.g. artifact_break_removal) resolve no matter how chainlit launches us
# or what the cwd is at tool-call time.
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (
    describe_tool,
    extract_inline_tool_calls,
    format_config,
    list_tool_names,
    propose_pipeline,
    propose_sweep,
    run_pipeline,
    try_parse_config,
    validate_pipeline,
)
from prompt import EEG_SYSTEM_PROMPT
from rag import retrieve_context
from batch import BATCHES_DIR, run_batch
from exports import export_analysis
from engines import (ENGINE_CHOICES, eeglab_ica_availability, engine_review, engine_runtime_errors)
from qc import (LAB_RULE_KEYS, discover_lab_rules, find_lab_rules, lint_config,
                load_lab_rules, looks_like_lab_rules)
from recipes import list_recipes, load_recipe, rebind_to_current, save_recipe
from sweep import expand_sweep, format_sweep_table, run_sweep, validate_sweep
from tools import (
    DATA_DIR,
    SCOPE,
    SESSION,
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
    call_tool,
    compare_to_checkpoint,
    scope_context,
    scope_eeg,
    set_codebook,
)

MODEL = "eeg-qwen"
MAX_TOOL_ROUNDS = 6
_HERE = os.path.dirname(os.path.abspath(__file__))
# Runtime state root. Defaults to the app dir; set STATE_DIR (e.g. a mounted volume) so
# sessions/recipes/batches persist across container restarts.
STATE_DIR = os.environ.get("STATE_DIR", _HERE)
SESSIONS_DIR = os.path.join(STATE_DIR, "sessions")
# Lab conventions supervised by the QC linter. Auto-load a lab_rules.(json|yaml) from the mounted
# data-in (DATA_DIR) first, then the persistent STATE_DIR, then a baked default beside app.py.
LAB_RULES, LAB_RULES_PATH = find_lab_rules(DATA_DIR, STATE_DIR, _HERE)

# Optional password gate for a shared machine. Unset APP_PASSWORD => no auth (the default;
# the app is bound to loopback anyway). When set, any username + this password logs in.
if os.environ.get("APP_PASSWORD"):
    import hashlib
    os.environ.setdefault(
        "CHAINLIT_AUTH_SECRET",
        hashlib.sha256(("eeg-llm:" + os.environ["APP_PASSWORD"]).encode()).hexdigest(),
    )

    @cl.password_auth_callback
    def _auth(username: str, password: str):
        if password == os.environ["APP_PASSWORD"]:
            return cl.User(identifier=username or "tester")
        return None


def _qc_block(config: dict) -> str:
    """QC-linter warnings for a drafted plan/sweep, as a non-blocking review-card section (or '')."""
    warnings = lint_config(config, SCOPE if SCOPE.get("filepath") else None, LAB_RULES)
    if not warnings:
        return ""
    return "\n\n**⚠ QC checks (review — not blocking):**\n- " + "\n- ".join(warnings)


def _png_element(b64: str, name: str) -> cl.Image:
    return cl.Image(content=base64.b64decode(b64), name=name, display="inline")


@cl.on_chat_start
async def start():
    cl.user_session.set(
        "history",
        [{"role": "system", "content": EEG_SYSTEM_PROMPT}],
    )
    cl.user_session.set("applied_steps", [])
    cl.user_session.set("last_export", None)

    data_dir = os.environ.get("EEG_DATA_DIR", "/data")
    have_sample = os.path.exists(os.path.join(data_dir, "sub-002.set"))
    if have_sample:
        sample_block = (
            "**1 · Try the bundled sample** (ERP CORE N170):\n"
            "```\n"
            "/scope sub-002.set\n"
            "/plan load it, band-pass 0.1-30 Hz, average reference, epoch faces vs cars, "
            "measure the N170 at PO8\n"
            "```\n"
            "Review the drafted plan, then `/run` to execute exactly that (or `/cancel`).\n\n"
        )
    else:
        sample_block = (
            "**1 · Get a sample to try** (optional): run `make fetch-sample` on your machine, "
            "then `/scope sub-002.set`.\n\n"
        )

    await cl.Message(
        content=(
            "**EEG assistant ready.** I load recordings and run filtering, ICA, ERP, PSD, "
            "AutoReject and more via MNE-Python — you drive, I draft; nothing runs until you "
            "approve it.\n\n"
            + sample_block +
            "**2 · Use your own data:** drop recordings into the **`data-in/`** folder on your "
            "computer, then reference them by name — e.g. `/scope my-recording.set`. If your "
            "event codes aren't in a BIDS `events.tsv`, define what they mean with "
            "`/codebook {\"conditions\": {\"target\": [[1,40]], ...}}`.\n\n"
            "**Handy commands:** `/plan <pipeline>` · `/sweep <param sweep>` · "
            "`/batch <dataset-dir>` (whole cohort) · `/params <tool>` (what a tool takes) · "
            "`/engine mne|eeglab` · `/export <name>` (analysis deliverable) · "
            "`/save-plan <name>` & `/recipes` (reuse). "
            "Or just chat: *\"load my-recording.set and compute its band power.\"*"
        )
    ).send()


def _effective_params(name: str, args: dict) -> tuple[list[str], list[str]]:
    """Split a call into (model-supplied, silently-defaulted) 'key=value' strings.

    Reads the tool function's signature so defaults the model DID NOT pass are made
    explicit -- the point being that the model's own prose is an unreliable audit of
    what actually ran. Works for any tool with no per-tool wiring.
    """
    fn = TOOL_FUNCTIONS.get(name)
    supplied, defaulted = [], []
    if fn is None:
        return [f"{k}={v!r}" for k, v in (args or {}).items()], []
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if pname in (args or {}):
            supplied.append(f"{pname}={args[pname]!r}")
        elif p.default is not inspect.Parameter.empty:
            defaulted.append(f"{pname}={p.default!r}")
    return supplied, defaulted


def _applied_line(result: dict) -> str:
    """A compact line of runtime-computed effective values worth surfacing."""
    if not isinstance(result, dict):
        return ""
    bits = []
    aa = result.get("aa_filter")
    if isinstance(aa, dict):
        bits.append(f"aa_filter: {aa.get('num_taps')} taps, phase={aa.get('phase')!r}, "
                    f"cutoff={aa.get('cutoff_hz')} Hz")
    if result.get("applied_shift_samples") is not None:
        bits.append(f"applied shift: {result['applied_shift_samples']} samples "
                    f"({result.get('applied_shift_ms')} ms, rounding={result.get('rounding')!r})")
    if result.get("dc_window_samples") is not None:
        bits.append(f"remove_dc: {result['dc_window_samples']}-sample window "
                    f"over {result.get('dc_n_segments')} segment(s)")
    if result.get("engine") == "erplab" and result.get("filt_n_segments") is not None:
        bits.append(f"engine=erplab: b,a filtfilt ({result.get('filt_band')}, "
                    f"design order {result.get('design_order')} = effective "
                    f"{result.get('iir_order')}), {result.get('filt_pad_samples')}-sample pad "
                    f"over {result.get('filt_n_segments')} segment(s)")
    return "  ·  ".join(bits)


async def _run_tool(name: str, args: dict) -> dict:
    """Execute a tool in a worker thread and surface its figure (if any)."""
    async with cl.Step(name=f"tool: {name}", type="tool") as step:
        step.input = args
        result = await cl.make_async(call_tool)(name, args)

        # Detach any image so it isn't shipped back to the model as raw base64.
        image_b64 = result.pop("image", None)
        step.output = json.dumps(result, indent=2, default=str)[:4000]
        if image_b64:
            step.elements = [_png_element(image_b64, f"{name}.png")]
            await step.update()

    # Always-visible effective-params summary (not a collapsed step): the complete
    # parameterization, with silently-applied defaults flagged, independent of the model.
    supplied, defaulted = _effective_params(name, args)
    lines = [f"🔧 **{name}**(" + ", ".join(supplied) + ")"]
    if defaulted:
        lines.append("· defaults applied: " + ", ".join(defaulted))
    applied = _applied_line(result)
    if applied:
        lines.append("· " + applied)
    await cl.Message(content="\n".join(lines), author="params").send()

    # Record successful, state-affecting calls as a replayable config for /save (provenance).
    # Both the agentic loop and the deterministic /run path funnel through here.
    if name != "save_eeg" and not (isinstance(result, dict) and result.get("ok") is False):
        steps = cl.user_session.get("applied_steps") or []
        steps.append({"tool": name, "args": args})
        cl.user_session.set("applied_steps", steps)

    return result, image_b64


def _narrate_results(results: list[dict]) -> str:
    """Plain-language interpretation of a deterministic run's numeric results.

    Read-only: the model sees the results (minus images) and explains them. It
    cannot change what ran -- this is narration after the fact, not a tool loop.
    """
    payload = json.dumps(
        [
            {
                "tool": r["tool"],
                "args": r["args"],
                "result": (
                    {k: v for k, v in r["result"].items() if k != "image"}
                    if isinstance(r["result"], dict) else r["result"]
                ),
            }
            for r in results
        ],
        default=str,
    )[:6000]
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": EEG_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "These EEG tools were just executed deterministically from an "
                "approved plan. Interpret the numeric results for the user in plain "
                "language. Do NOT propose new steps.\n\n" + payload
            )},
        ],
    )
    return (resp["message"].get("content") or "").strip()


async def _handle_plan(request: str):
    """Draft a pipeline config from the request, validate it, and stage it for /run."""
    if not request:
        await cl.Message(content=(
            "Usage: `/plan <what to run>` — e.g. `/plan load /data/sub-01.edf, "
            "band-pass 1-40 Hz, notch 50, average reference, then PSD`\n\n"
            "Or paste a JSON config directly (no model involved): "
            "`/plan {\"pipeline\": [{\"tool\": \"load_eeg\", \"args\": "
            "{\"filepath\": \"/data/sub-01.edf\"}}]}`"
        )).send()
        return

    # Model-free path: if the user pasted a JSON config, use it verbatim.
    config = try_parse_config(request)
    source = "your config (verbatim, no model)"
    if config is None:
        rec = load_recipe(request, kind="plan")  # a saved recipe name? (slug match, no model)
        if rec is not None:
            config = rebind_to_current(rec["spec"], SESSION.get("filepath"))
            source = f"saved recipe '{rec['name']}' (rebound to the current recording)"
            if not SESSION.get("filepath"):
                await cl.Message(content="⚠ No recording loaded — `/scope <file>` first so the "
                                 "recipe runs on your data.").send()
        else:
            source = "drafted by the assistant — check it for missing or extra steps"
            ctx = await cl.make_async(retrieve_context)(request)
            async with cl.Step(name="planning", type="llm"):
                config = await cl.make_async(propose_pipeline)(
                    MODEL, request, _with_scope(ctx["text"]), cl.user_session.get("engine"))

    # Planner telemetry is useful for the review heading, but must never become part of the
    # runnable config displayed to or approved by the user.
    meta = config.pop("_planner_meta", {}) if isinstance(config, dict) else {}
    errors = validate_pipeline(config) + engine_runtime_errors(config)
    if meta.get("valid") is False:
        errors.extend(e for e in (meta.get("errors") or ["Planner validation failed."])
                      if e not in errors)
    if errors:
        cl.user_session.set("pending_pipeline", None)
        cl.user_session.set("pending_sweep", None)
        await cl.Message(content=(
            "**Could not build a valid pipeline:**\n- " + "\n- ".join(errors)
            + "\n\nRefine the request and `/plan` again."
        )).send()
        return

    cl.user_session.set("pending_pipeline", config)
    notes = config.get("notes") or []
    note_md = ("\n\n**Notes / assumptions:**\n- " + "\n- ".join(notes)) if notes else ""
    repaired = (f" · self-corrected in {meta['attempts']} attempts"
                if meta.get("attempts", 1) > 1 else "")
    await cl.Message(content=(
        f"**Proposed pipeline** ({source}{repaired}). Review it, then `/run` to execute "
        "*exactly this* (no model in the loop), or `/cancel`:\n"
        f"```json\n{format_config(config)}\n```"
        f"{engine_review(config, cl.user_session.get('engine'))}{note_md}{_qc_block(config)}"
    )).send()


async def _handle_compare(arg: str):
    """Deterministic, out-of-band verification: score the CURRENT session data against an
    ERP CORE checkpoint. The model is NOT involved and never sees ERP CORE specifics -- this
    is the experimenter checking the data the model produced from natural language alone.

    Usage: /compare <checkpoint> [mean] [bipolar] [chan=PO8]
      checkpoint : shifted_ds | reref_ucbip | hpfilt | <path to .set>
      mean       : use the mean-correlation gate (looser; for filtered data)
      bipolar    : compare the bipolar EOG channels instead of the scalp channels
      chan=NAME  : channel to draw in the overlay plot (default PO8)
    """
    toks = arg.split()
    if not toks:
        await cl.Message(content=(
            "Usage: `/compare <checkpoint> [mean] [bipolar] [chan=PO8]`\n"
            "- checkpoint: `shifted_ds`, `reref_ucbip`, `hpfilt`, or a path to a .set file\n"
            "- `mean`: looser correlation-only gate (for filtered data)\n"
            "- `bipolar`: compare the bipolar EOG channels\n"
            "- `chan=NAME`: channel for the overlay plot (default PO8)\n\n"
            "This scores the data currently in the session against the checkpoint — the "
            "model is not involved."
        )).send()
        return

    checkpoint = toks[0]
    gate = "mean" if any(t in ("mean", "gate=mean") for t in toks[1:]) else "tight"
    bipolar = any(t == "bipolar" for t in toks[1:])
    plot_channel = next((t.split("=", 1)[1] for t in toks[1:] if t.startswith("chan=")), "PO8")

    async with cl.Step(name=f"compare vs {checkpoint}", type="tool") as step:
        step.input = {"checkpoint": checkpoint, "gate": gate,
                      "bipolar": bipolar, "plot_channel": plot_channel}
        res = await cl.make_async(compare_to_checkpoint)(
            checkpoint=checkpoint, gate=gate, bipolar=bipolar, plot_channel=plot_channel)
        image_b64 = res.pop("image", None)
        step.output = json.dumps(res, indent=2, default=str)[:4000]

    if not res.get("ok"):
        await cl.Message(content=f"**Compare failed:** {res.get('error')}").send()
        return

    lines = [
        f"**Compared current data vs `{res['checkpoint']}` — verdict: {res['verdict']}** "
        f"_(gate: {res['gate']})_",
        f"- channels: {res['n_channels']}  |  samples: {res['n_samples']}",
        f"- min r = {res['min_r']}  |  mean r = {res['mean_r']}  |  "
        f"worst max|Δ| = {res['worst_max_abs_diff_uV']} µV",
        "",
        "| channel | Pearson r | max|Δ| (µV) |",
        "|---|---:|---:|",
    ]
    lines += [f"| {r['channel']} | {r['pearson_r']:.6f} | {r['max_abs_diff_uV']:.4f} |"
              for r in res["rows"]]
    elements = [_png_element(image_b64, "compare.png")] if image_b64 else []
    await cl.Message(content="\n".join(lines), elements=elements).send()


async def _handle_tune(arg: str):
    """Interactive artifact-threshold tuning, in-chat (Plotly). Runs the ERPLAB
    peak-to-peak detector on the CURRENTLY LOADED continuous recording at the given
    threshold, renders the p2p envelope vs ampth with flagged spans shaded, and lists
    the triggering channels. Works on any data; shows a Kappenman-overlap verdict only
    when the ERP CORE sub-002 prep1 checkpoint is loaded."""
    import mne
    import plotly.graph_objects as go

    from artifact_continuous_detect import (
        continuous_artifact_detect,
        reconstruct_deleted_spans,
        span_overlap,
        windowed_p2p_envelope,
    )

    raw = SESSION.get("raw")
    if raw is None or not isinstance(raw, mne.io.BaseRaw):
        await cl.Message(content=(
            "`/tune` needs a continuous recording loaded. Load one first "
            "(`load_eeg`), then `/tune <ampth_uV> [winms=500] [stepms=50]`."
        )).send()
        return
    toks = arg.split()
    if not toks:
        await cl.Message(content=(
            "Usage: `/tune <ampth_uV> [winms=500] [stepms=50]`\n"
            "Flags continuous segments whose peak-to-peak amplitude exceeds `ampth` "
            "(moving window) on the loaded recording. Lower `ampth` = more aggressive."
        )).send()
        return
    try:
        ampth = float(toks[0])
        winms = float(toks[1]) if len(toks) > 1 else 500.0
        stepms = float(toks[2]) if len(toks) > 2 else 50.0
    except ValueError:
        await cl.Message(content="`ampth/winms/stepms` must be numbers, e.g. `/tune 300 500 50`.").send()
        return

    # scan real signal channels only (avoid stim/misc blowing up peak-to-peak)
    picks = mne.pick_types(raw.info, eeg=True, eog=True, exclude=[])
    chan_array = list(picks) if len(picks) else list(range(len(raw.ch_names)))
    fs = float(raw.info["sfreq"])

    async with cl.Step(name=f"tune ampth={ampth}", type="tool") as step:
        step.input = {"ampth": ampth, "winms": winms, "stepms": stepms,
                      "n_channels_scanned": len(chan_array)}
        res = await cl.make_async(continuous_artifact_detect)(
            raw, ampth=ampth, winms=winms, stepms=stepms, chan_array=chan_array)
        centres, env = await cl.make_async(windowed_p2p_envelope)(
            raw, winms, stepms, chan_array)
        step.output = {"n_segments": res["n_segments"], "pct_removed": res["pct_removed"]}

    spans = res["deleted_spans"]

    # conditional Kappenman verdict (only when ERP CORE sub-002 prep1 is loaded)
    verdict_line = None
    fp = SESSION.get("filepath") or ""
    if "erpcore_n170" in fp and "ica_prep1" in os.path.basename(fp):
        prep2 = fp.replace("ica_prep1", "ica_prep2")
        if os.path.exists(prep2):
            gt_spans, _ = await cl.make_async(reconstruct_deleted_spans)(prep2, fp)
            ov = span_overlap(spans, gt_spans, res["n_times"])
            if ov["a_samples"] == 0 and ov["b_samples"] == 0:
                verdict = "matches ERP CORE (both cut nothing)"
            elif ov["iou"] >= 0.5:
                verdict = "about right"
            elif ov["a_samples"] > ov["b_samples"]:
                verdict = "over-cutting vs Kappenman"
            else:
                verdict = "under-cutting vs Kappenman"
            verdict_line = (f"- **vs Kappenman:** IoU={ov['iou']}, you cut "
                            f"{ov['a_samples']} vs their {ov['b_samples']} samples "
                            f"→ **{verdict}**")

    # Plotly: p2p envelope vs ampth, flagged spans shaded
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centres, y=env, mode="lines",
                             name="max peak-to-peak (µV)", line=dict(width=1)))
    fig.add_hline(y=ampth, line_dash="dash", line_color="orange",
                  annotation_text=f"ampth = {ampth} µV")
    for s, e in spans:
        fig.add_vrect(x0=s / fs, x1=e / fs, fillcolor="red", opacity=0.25, line_width=0)
    fig.update_layout(template="plotly_dark", height=420,
                      title=f"/tune  ampth={ampth} winms={winms} stepms={stepms} "
                            f"— {res['pct_removed']}% removed",
                      xaxis_title="Time (s)", yaxis_title="peak-to-peak (µV)",
                      margin=dict(l=60, r=20, t=50, b=45))

    lines = [
        f"**/tune** ampth={ampth} µV, winms={winms}, stepms={stepms} "
        f"(scanned {len(chan_array)} channels)",
        f"- **{res['n_segments']} segment(s)** flagged, {res['total_ms_removed']} ms "
        f"= **{res['pct_removed']}% removed**",
    ]
    for i, (s, e) in enumerate(spans[:8]):
        trig = res["triggers_per_span"][i][:4]
        trig_txt = ", ".join(f"{n} ({v:.0f}µV)" for n, v in trig) or "—"
        lines.append(f"  - {s / fs:.2f}–{e / fs:.2f}s ← {trig_txt}")
    if len(spans) > 8:
        lines.append(f"  … and {len(spans) - 8} more span(s)")
    if verdict_line:
        lines.append(verdict_line)
    lines.append("_Re-run `/tune` with a new ampth to iterate. Cut gross muscle/offsets; "
                 "keep blinks & eye movements (ICA models those)._")

    await cl.Message(content="\n".join(lines),
                     elements=[cl.Plotly(name="tune", figure=fig, display="inline")]).send()


def _with_scope(ctx_text: str | None) -> str | None:
    """Prepend the scoped-recording summary (channels, sfreq, codebook) to the planner context
    so it names real channels and builds create_bins from the real code->condition map."""
    scope = scope_context()
    if not scope:
        return ctx_text
    return f"{scope}\n\n----\n{ctx_text}" if ctx_text else scope


async def _handle_scope(arg: str):
    """Scope a recording: read structure + resolve the event code->condition map (onboarding)."""
    path = arg.strip()
    if not path:
        await cl.Message(content=(
            "Usage: `/scope /path/to/recording.set` — reads channels, sampling rate, and the "
            "event-code inventory, and resolves a codebook (BIDS events.tsv, or a saved "
            "`<name>.codebook.json`). Then `/plan` and `/sweep` know your real channels and "
            "condition codes."
        )).send()
        return
    try:
        async with cl.Step(name="scope", type="tool"):
            res = await cl.make_async(scope_eeg)(path)
    except Exception as exc:
        await cl.Message(content=f"**Scope failed:** {exc}").send()
        return False
    if not res.get("ok"):
        await cl.Message(content=f"**Scope failed:** {res.get('error', 'unknown')}").send()
        return False
    lines = [f"**Scoped** `{os.path.basename(path)}` ({res['kind']}).",
             f"- {res['n_channels']} channels @ {res['sfreq']} Hz",
             f"- {res['n_event_codes']} distinct event codes"]
    cb, src = res.get("codebook"), res.get("codebook_source")
    if cb and cb.get("conditions"):
        conds = ", ".join(cb["conditions"].keys())
        lines.append(f"- **codebook** ({src}): conditions = {conds}")
        if src == "bids":
            lines.append("  ⚠ from BIDS `trial_type` — may be coarse (e.g. 'stimulus'). Refine "
                         "with `/codebook {\"conditions\": {...}}` if you need finer bins.")
    else:
        lines.append("- **no codebook** — I can read the codes but not what they MEAN. Provide "
                     "one with `/codebook {\"conditions\": {\"face\": [[1,40]], ...}, "
                     "\"responses\": {\"correct\": 201}}` (saved for reuse), or drop a "
                     "`<name>.codebook.json` beside the file.")
    for w in res.get("warnings") or []:
        lines.append(f"- ⚠ {w}")
    await cl.Message(content="\n".join(lines)).send()
    pending = cl.user_session.get("pending_codebook_uploads") or []
    if pending:
        if not await _apply_dropped_codebook(pending):
            return False
        cl.user_session.set("pending_codebook_uploads", None)
    return True


async def _handle_codebook(arg: str):
    """Set / show the event code->condition map for the scoped recording (Option A / B)."""
    text = arg.strip()
    if not text:
        cur = scope_context()
        await cl.Message(content=(
            "**Current scope / codebook:**\n```\n" + (cur or "(nothing scoped — /scope first)")
            + "\n```\nSet one by pasting JSON: `/codebook {\"conditions\": {\"face\": [[1,40]], "
            "\"car\": [[41,80]]}, \"responses\": {\"correct\": 201}}`"
        )).send()
        return
    try:
        cb = json.loads(text)
    except json.JSONDecodeError:
        await cl.Message(content="Could not parse the codebook JSON. Expected "
                         "`{\"conditions\": {...}, \"responses\": {...}}`.").send()
        return
    res = await cl.make_async(set_codebook)(cb.get("conditions"), cb.get("responses"), True)
    if not res.get("ok"):
        await cl.Message(content=f"**Codebook rejected:** {res.get('error')}").send()
        return
    msg = "**Codebook set and saved** (reused next session)."
    if res.get("warnings"):
        msg += "\n" + "\n".join(f"- ⚠ {w}" for w in res["warnings"])
    await cl.Message(content=msg + "\n\n```\n" + scope_context() + "\n```").send()


_LAB_RULES_KEYS_HELP = (
    "Supported keys: `require_steps` (steps that must appear), `forbid_steps` (not allowed), "
    "`order` (required relative order), `require_before` ({A: B} — A before B), "
    "`param_equals` ({tool: {param: value}}). All are deterministic, non-blocking review-card "
    "warnings on every `/plan` and `/sweep`, on top of the built-in universal traps "
    "(filter-after-epoch, reref-after-ICA, absent channels/codes)."
)


def _resolve_lab_rules_path(arg: str) -> str | None:
    """Resolve a lab-rules argument to a file: an explicit path, else a bare name looked up in
    data-in (DATA_DIR) then STATE_DIR. Lets a user assign an arbitrarily-named file."""
    if os.path.exists(arg):
        return arg
    for base in (DATA_DIR, STATE_DIR):
        cand = os.path.join(base, arg)
        if os.path.exists(cand):
            return cand
    return None


async def _handle_lab_rules(arg: str):
    """Show, assign, or clear the lab conventions the QC linter enforces.

    `/lab-rules`                  show the active rules + supported keys + other conventions found
    `/lab-rules <path-or-name>`   assign ANY JSON/YAML file (arbitrary name) as the active rules;
                                  a bare name is resolved against data-in then STATE_DIR
    `/lab-rules off`              deactivate lab rules (built-in traps still run)
    """
    global LAB_RULES, LAB_RULES_PATH
    arg = arg.strip()

    if arg.lower() in ("off", "none", "clear", "disable"):
        LAB_RULES, LAB_RULES_PATH = None, None
        await cl.Message(content="Lab rules cleared. The built-in universal traps still run on "
                                 "every plan/sweep.").send()
        return

    if arg:  # assign a specific file (arbitrary name), resolving bare names against data-in
        path = _resolve_lab_rules_path(arg)
        if path is None:
            await cl.Message(content=(
                f"Could not find `{arg}`. Give a path, or a filename present in `data-in/`. "
                f"Drop your rules file there (any name), then `/lab-rules <name>`.")).send()
            return
        rules = await cl.make_async(load_lab_rules)(path)
        if rules is None:
            await cl.Message(content=f"Could not parse `{path}` (expected JSON/YAML).").send()
            return
        if not looks_like_lab_rules(rules):
            await cl.Message(content=(
                f"`{os.path.basename(path)}` has none of the lab-rule keys, so it isn't a rules "
                f"file. {_LAB_RULES_KEYS_HELP}")).send()
            return
        LAB_RULES, LAB_RULES_PATH = rules, path

    # Discover other convention files the user could switch to (any name, in data-in/STATE_DIR).
    discovered = await cl.make_async(discover_lab_rules)(DATA_DIR, STATE_DIR, _HERE)
    others = [p for p, _ in discovered if p != LAB_RULES_PATH]

    if not LAB_RULES:
        lines = ["No lab rules active — only the built-in universal traps run.", "",
                 "Assign one with `/lab-rules <path-or-name>` (drop a JSON/YAML file with any name "
                 "into `data-in/`, then `/lab-rules <name>`).", "", _LAB_RULES_KEYS_HELP]
        if others:
            lines += ["", "**Convention files found** (assign with `/lab-rules <name>`):"]
            lines += [f"- `{os.path.basename(p)}`" for p in others]
        await cl.Message(content="\n".join(lines)).send()
        return

    body = json.dumps(LAB_RULES, indent=2, default=str)
    src = f" (from `{LAB_RULES_PATH}`)" if LAB_RULES_PATH else ""
    lines = [f"**Active lab rules**{src} — enforced on every `/plan` and `/sweep`:",
             f"```json\n{body}\n```", "", _LAB_RULES_KEYS_HELP]
    if others:
        lines += ["", "**Other convention files** (switch with `/lab-rules <name>`):"]
        lines += [f"- `{os.path.basename(p)}`" for p in others]
    await cl.Message(content="\n".join(lines)).send()


def _fmt_default(v) -> str:
    """Render a default value for display."""
    if v is None:
        return "null"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


async def _handle_params(arg: str):
    """Deterministic tool reference: which parameters a tool needs and their defaults.

    `/params` lists tools; `/params <tool>` shows each parameter, whether it is REQUIRED (you
    must state it) or optional (with its DEFAULT, applied if you omit it). Read straight from the
    registry + function signature -- reliable, no model. Lets a user know exactly what to specify
    in a `/plan`, and that omitting a parameter (or saying 'use the defaults') applies the default.
    """
    name = arg.strip()
    if not name:
        names = ", ".join(sorted(list_tool_names()))
        await cl.Message(content=(
            "**`/params <tool>`** — see a tool's parameters, which are required, and their "
            f"defaults.\n\nTools: {names}"
        )).send()
        return
    info = await cl.make_async(describe_tool)(name)
    if not info.get("ok"):
        sugg = ", ".join(info.get("suggestions") or [])
        await cl.Message(content=f"{info.get('error')} Did you mean: {sugg}?").send()
        return

    req = [p for p in info["params"] if p["required"]]
    opt = [p for p in info["params"] if not p["required"]]
    out = [f"**`{info['name']}`** — {info['description']}", ""]
    if req:
        out.append("**Required** (you must specify these):")
        for p in req:
            t = f" _{p['type']}_" if p.get("type") else ""
            out.append(f"- **`{p['name']}`**{t} — {p['description']}")
    else:
        out.append("**Required:** none.")
    out.append("")
    out.append("**Optional** (default applies if you omit it — or just say \"use the defaults\"):")
    for p in opt:
        t = f" _{p['type']}_" if p.get("type") else ""
        dflt = _fmt_default(p["default"]) if p["has_default"] else "—"
        out.append(f"- `{p['name']}`{t} — default `{dflt}` — {p['description']}")
    out.append("")
    out.append("_Specify only what you need in `/plan`; anything omitted uses the default above._")
    await cl.Message(content="\n".join(out)).send()


# Friendly aliases → tool names, so a free-text param question can name a tool casually.
# ORDER MATTERS: specific phrases before general ones (checked in insertion order), so
# "apply ica" resolves to apply_ica rather than run_ica, "difference wave" to the diff-ERP, etc.
# Only fires once a param-question trigger is present, so broad words here are low-risk.
_TOOL_ALIASES = {
    # --- I/O ---
    "load recording": "load_eeg", "load the recording": "load_eeg", "load data": "load_eeg",
    "load file": "load_eeg", "import recording": "load_eeg",
    "save recording": "save_eeg", "export recording": "save_eeg", "save to disk": "save_eeg",
    # --- filtering / resample / line noise ---
    "band-pass": "filter_eeg", "bandpass": "filter_eeg", "band pass": "filter_eeg",
    "high-pass": "filter_eeg", "highpass": "filter_eeg", "low-pass": "filter_eeg",
    "lowpass": "filter_eeg", "notch filter": "filter_eeg", "filter": "filter_eeg",
    "downsample": "resample", "down-sample": "resample", "sampling rate": "resample",
    "sample rate": "resample", "resampling": "resample",
    "line noise": "remove_line_noise", "zapline": "remove_line_noise",
    "powerline": "remove_line_noise", "power-line": "remove_line_noise",
    # --- referencing / events / channels ---
    "re-reference": "set_reference", "rereference": "set_reference", "reref": "set_reference",
    "average reference": "set_reference", "linked mastoid": "set_reference",
    "mastoid": "set_reference", "reference": "set_reference",
    "event shift": "shift_events", "shift events": "shift_events", "shift the events": "shift_events",
    "channel type": "set_channel_types", "channel types": "set_channel_types",
    "set channel": "set_channel_types", "mark eog": "set_channel_types",
    "bipolar eog": "derive_bipolar_eog", "bipolar": "derive_bipolar_eog",
    "heog": "derive_bipolar_eog", "veog": "derive_bipolar_eog",
    # --- bad channels / interpolation / robust ref ---
    "interpolate": "interpolate_bads", "interpolation": "interpolate_bads",
    "interp": "interpolate_bads", "faster": "run_faster",
    "pyprep": "run_prep", "robust reference": "run_prep", "prep pipeline": "run_prep",
    "prep": "run_prep",
    # --- artifact detection / rejection / ASR / autoreject ---
    "extreme value": "detect_artifacts_extreme_value", "voltage threshold": "detect_artifacts_extreme_value",
    "moving window": "detect_artifacts_moving_window", "peak-to-peak": "detect_artifacts_moving_window",
    "peak to peak": "detect_artifacts_moving_window", "crap": "detect_artifacts_moving_window",
    "step-like": "detect_artifacts_step", "step detection": "detect_artifacts_step",
    "step artifact": "detect_artifacts_step", "saccade": "detect_artifacts_step",
    "reject flagged": "reject_flagged_epochs", "reject epochs": "reject_flagged_epochs",
    "drop epochs": "reject_flagged_epochs",
    "auto-reject": "run_autoreject", "autoreject": "run_autoreject", "auto reject": "run_autoreject",
    "artifact subspace": "run_asr", "clean_rawdata": "run_asr", "clean rawdata": "run_asr",
    "asr": "run_asr",
    # --- break / continuous cleaning ---
    "break segment": "delete_break_segments", "remove break": "delete_break_segments",
    "delete break": "delete_break_segments", "breaks": "delete_break_segments",
    "continuous artifact": "continuous_artifact_detect",
    # --- binning / epoching ---
    "binlister": "create_bins", "binning": "create_bins", "bins": "create_bins",
    "epoching": "create_epochs", "epoch": "create_epochs", "epochs": "create_epochs",
    # --- ICA ---
    "apply ica": "apply_ica", "remove components": "apply_ica", "ocular correction": "apply_ica",
    "component removal": "apply_ica",
    "load ica": "load_ica_eeglab", "precomputed ica": "load_ica_eeglab", "eeglab ica": "load_ica_eeglab",
    "independent component": "run_ica", "ica": "run_ica",
    # --- spectral / ERP / analysis ---
    "band power": "compute_psd", "power spectral": "compute_psd", "power spectrum": "compute_psd",
    "spectral density": "compute_psd", "psd": "compute_psd",
    "difference wave": "compute_difference_erp", "difference erp": "compute_difference_erp",
    "diff wave": "compute_difference_erp",
    "evoked response": "compute_erp", "event-related potential": "compute_erp",
    "event related potential": "compute_erp", "erp": "compute_erp",
    "measure component": "measure_component", "peak amplitude": "measure_component",
    "mean amplitude": "measure_component", "measure": "measure_component",
    "aperiodic": "run_fooof", "specparam": "run_fooof", "fooof": "run_fooof", "1/f": "run_fooof",
    "connectivity": "compute_connectivity", "coherence": "compute_connectivity",
    "wpli": "compute_connectivity", "sift": "compute_connectivity",
    "microstate": "compute_microstates", "microstates": "compute_microstates",
    "pycrostates": "compute_microstates",
}

# Strong param-question wording: rarely appears in an imperative run request, so it may also
# trigger the "which tool did you mean?" list when no tool is named.
_PARAM_STRONG = ("parameter", "argument", " params ", " args ",
                 "defaults for", "defaults of", "default value", "default for", "default of")
# Weaker / more adjacent wording: only fires when a real tool is ALSO named (else it would be
# too easy to hijack ordinary chat). Never asks on its own.
_PARAM_WEAK = ("options for", "option for", "settings for", "setting for",
               "how do i use", "how to use", "how do you use", " inputs for ",
               "help with", "help for", "how do i configure", "how to configure")
_VERB_LEAD = ("what does", "what can", "what do")
_VERB_OBJ = (" take", " takes", " need", " needs", " accept", " accepts",
             " require", " requires", " expect", " expects")


def _find_tool_in(t: str) -> str | None:
    """First tool referenced in `t` (space-padded, lowercased): exact name/spaced form, then
    a casual alias (specific→general by insertion order)."""
    for name in list_tool_names():
        if name in t or name.replace("_", " ") in t:
            return name
    for alias, name in _TOOL_ALIASES.items():
        if alias in t:
            return name
    return None


def _detect_params_query(text: str) -> str | None:
    """Detect a free-text 'what parameters/defaults does <tool> take?' question, so the answer
    comes from the DETERMINISTIC `/params` reference instead of the model's memory.

    Returns a tool name, "ASK" (a strong param question with no tool named → list tools), or
    None (not a param question → normal chat). Tiered so it will not hijack a real command:
    STRONG wording may ASK; WEAK/adjacent wording only fires if a tool is also named.
    """
    t = f" {text.lower().strip()} "
    strong = any(k in t for k in _PARAM_STRONG)
    weak = any(k in t for k in _PARAM_WEAK)
    verby = any(k in t for k in _VERB_LEAD) and any(k in t for k in _VERB_OBJ)
    if not (strong or weak or verby):
        return None
    tool = _find_tool_in(t)
    if tool:
        return tool
    return "ASK" if strong else None   # weak/verb wording without a tool -> normal chat


def _inspect_components_request(text: str) -> str | None:
    """Recognize standalone inspection requests, not plans or compound actions.

    Also accepts the verbs review/overview WHEN specific component numbers follow: "review the
    ICA components 1 and 8" names components, so it is an inspect-those, not the whole-line-up scan
    view (routing sends a numberless "review ... components" to _review_ica_request instead).
    """
    command = re.fullmatch(r"/inspect-ica(?:\s+(.*))?", text.strip(), re.I)
    if command:
        return command.group(1) or ""
    match = re.fullmatch(
        r"(?:please\s+|can you\s+)?(?:inspect|show|view|review|overview)\s+"
        r"(?:the\s+)?(?:ICA\s+)?"
        r"components?(?:\s+([\d\s,&.\[\]-]+(?:and[\d\s,&.\[\]-]+)*))?"
        r"[.!?]?", text.strip(), re.I,
    )
    return (match.group(1) or "").rstrip(".!?").strip() if match else None


async def _handle_inspect_components(arg: str, request: str):
    values = arg.strip()
    if values.startswith("[") and values.endswith("]"):
        values = values[1:-1].strip()
    if not re.fullmatch(r"\d+(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s+)\d+)*",
                        values, re.I):
        await cl.Message(content=(
            "Specify 1-based ICA component numbers, for example `/inspect-ica 1 8` "
            "or `inspect components 1 and 8`. Inspection removes nothing."
        )).send()
        return
    components = list(dict.fromkeys(int(n) for n in re.findall(r"\d+", values)))
    if len(components) > 6:
        await cl.Message(content="Inspect at most six ICA components at a time.").send()
        return
    out = cl.Message(content="Inspecting ICA components " + ", ".join(map(str, components)) + "...")
    await out.send()
    result, image_b64 = await _run_tool("inspect_ica_component", {"components": components})
    if result.get("ok") is False:
        out.content = "Could not inspect ICA components: " + result.get("error", "Unknown tool error.")
    elif not image_b64:
        out.content = "ICA inspection returned no plot. No components were removed."
    else:
        shown = result.get("components", components)
        out.content = ("**ICA components " + ", ".join(map(str, shown))
                       + " (1-based)**\n\nReview only. No components were removed.")
        out.elements = [_png_element(image_b64, "ica-components.png")]
    await out.update()
    history = cl.user_session.get("history")
    if history is not None:
        history.extend([{"role": "user", "content": request},
                        {"role": "assistant", "content": out.content}])


def _review_ica_request(text: str) -> bool:
    """Recognize a standalone ICA overview/line-up request (the tune_ica-style scan view).

    Distinct from the single-component inspect route: this uses the verbs review/overview (not
    inspect/show/view) so the two never collide, and takes no component numbers.
    """
    t = text.strip()
    if re.fullmatch(r"/review-ica\b.*", t, re.I):
        return True
    # verb-first: "review/overview [the] {ICA [noun] | noun}" — require an ICA-related token
    # (ICA, or components/decomposition/line-up) so bare "review"/"overview" does NOT match.
    # ICA needs no trailing whitespace, so "review ica" (ICA as the final token) also routes.
    if re.fullmatch(
        r"(?:please\s+|can you\s+)?(?:review|overview)\s+(?:the\s+)?"
        r"(?:ICA(?:\s+(?:components?|overview|line[-\s]?up|decomposition))?"
        r"|components?|line[-\s]?up|decomposition)[.!?]?", t, re.I):
        return True
    # noun-first: "ICA/component review|overview|line-up"
    return bool(re.fullmatch(
        r"(?:the\s+)?(?:ICA|components?)\s+(?:review|overview|line[-\s]?up)[.!?]?", t, re.I))


async def _handle_review_ica():
    """Render the ICA scan view: topography grid + EOG-correlation table (review_ica). Removes nothing."""
    out = cl.Message(content="Reviewing the ICA component line-up…")
    await out.send()
    result, image_b64 = await _run_tool("review_ica", {})
    if result.get("ok") is False:
        out.content = "Could not review ICA: " + result.get("error", "Unknown tool error.")
    elif not image_b64:
        out.content = "ICA review returned no plot. No components were removed."
    else:
        rows = result.get("eog_correlation_table") or []
        n = result.get("n_components") or len(rows)
        lines = [f"**ICA component review** — all {n} components: the topography grid is shown "
                 "inline below, with the EOG-correlation table for every component. Review only; "
                 "nothing removed."]
        if rows:
            lines += ["", "| comp | r_max (EOG) | ICLabel |", "|---:|---:|---|"]
            for r in rows:  # every component, not just the top few
                rmax = r.get("r_max")
                rmax_s = f"{rmax:.2f}" if isinstance(rmax, (int, float)) else "-"
                lines.append(f"| {r.get('component')} | {rmax_s} | {r.get('iclabel') or '-'} |")
            lines.append("\n(Sorted by EOG correlation.) High r_max + frontal topography = "
                         "eye-artifact candidate. Inspect one with `/inspect-ica <n>`, then remove it "
                         "via the plan's apply_ica step.")
        out.content = "\n".join(lines)
        out.elements = [_png_element(image_b64, "ica-review.png")]
    await out.update()
    history = cl.user_session.get("history")
    if history is not None:
        history.extend([{"role": "user", "content": "review ICA components"},
                        {"role": "assistant", "content": out.content}])


# --------------------------------------------------------------------------- #
# Deterministic in-session "remove ICA components -> [epoch A vs B] -> [measure]" executor.
# The compound apply-and-analyze one-liner MUST run in-session (it uses the ICA fitted earlier;
# /plan's load_eeg would wipe it), and the local model is unreliable at emitting the chained tool
# calls. So we parse and execute it deterministically. Faithful: nothing is invented -- if the
# epoch window, a codebook condition, the channel, or the measurement window is missing, we fail
# loud and say what to add, never substitute a task-specific default (e.g. ERP CORE's -200..800 ms).
# --------------------------------------------------------------------------- #
_ELECTRODE_RE = r"[A-Za-z]{1,3}\d{0,2}[zZ]?"


def _apply_ica_compound_request(text: str) -> bool:
    """True for "remove/drop/reject/exclude [the] ICA component(s) <number> ...">."""
    return bool(re.match(
        r"\s*(?:remove|drop|reject|exclude)\s+(?:the\s+)?ICA\s+components?\s+\d",
        text, re.I))


def _ms_window(m) -> tuple[float, float]:
    """(lo, hi) match groups in ms/s -> seconds."""
    lo, hi, unit = float(m.group(1)), float(m.group(2)), (m.group(3) or "ms").lower()
    return (lo, hi) if unit == "s" else (lo / 1000.0, hi / 1000.0)


def _resolve_condition_codes(name: str, conditions: dict) -> list | None:
    """Resolve a spoken condition ('faces') to the codebook's code list, tolerant of plural/case."""
    if not name or not conditions:
        return None
    want = name.strip().lower()
    for key, codes in conditions.items():
        k = key.lower()
        if k == want or k == want.rstrip("s") or k.rstrip("s") == want.rstrip("s"):
            return codes
    return None


def _parse_apply_ica_compound(text: str) -> dict:
    """Parse the compound request into fields + a list of blocking errors (faithful: no defaults)."""
    p: dict = {"errors": []}
    # components to remove: the number list right after "components", before any "then"/"epoch"
    m = re.search(r"components?\s+(.*)", text, re.I)
    head = re.split(r"\bthen\b|\bepoch\w*\b|\bmeasure\b|;", m.group(1), maxsplit=1, flags=re.I)[0] if m else ""
    p["components"] = [int(n) for n in re.findall(r"\d+", head)]
    if not p["components"]:
        p["errors"].append("no component numbers to remove (e.g. 'remove ICA components 1 and 7').")

    # optional contrast: "epoch A vs/versus/minus B"
    mc = re.search(r"epoch\w*\s+(?:the\s+)?(\w+)\s+(?:vs\.?|versus|minus|against|v\.?)\s+(\w+)",
                   text, re.I)
    p["cond_a"] = mc.group(1) if mc else None
    p["cond_b"] = mc.group(2) if mc else None

    # Split at the measurement keyword: the EPOCH window is sought only BEFORE it, the MEASUREMENT
    # window only after it. This prevents the epoch parse from greedily swallowing the measurement
    # window when no epoch window was given (which would silently epoch the wrong span).
    _range = r"(-?\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(-?\d+(?:\.\d+)?)\s*(ms|s)\b"
    mkw = re.search(r"\b(?:measure|mean|peak|amplitude)\b", text, re.I)
    pre = text[:mkw.start()] if mkw else text
    post = text[mkw.start():] if mkw else ""
    mm = re.search(_range, post, re.I)
    if mm:
        p["meas_tmin"], p["meas_tmax"] = _ms_window(mm)
    me = re.search(_range, pre, re.I)  # epoch window must appear in the pre-measurement clause
    if me:
        p["epoch_tmin"], p["epoch_tmax"] = _ms_window(me)

    # channel: "at [channel] PO8"
    mch = re.search(rf"\bat\s+(?:channel\s+)?({_ELECTRODE_RE})\b", text, re.I)
    p["channel"] = mch.group(1) if mch else None
    p["mode"] = "peak" if re.search(r"\bpeak\b", text, re.I) else "mean"
    p["wants_measure"] = bool(re.search(r"\b(measure|mean|peak|amplitude)\b", text, re.I))
    p["wants_epoch"] = bool(mc)
    return p


async def _handle_apply_ica_compound(text: str):
    """Execute remove-components (-> epoch contrast -> measure) deterministically, in session."""
    p = _parse_apply_ica_compound(text)
    if p["errors"]:
        await cl.Message(content="Cannot run that: " + " ".join(p["errors"])).send()
        return

    # 1) apply_ica -- the must-be-in-session action (uses the ICA fitted earlier this session)
    result, image_b64 = await _run_tool("apply_ica", {"components": p["components"]})
    if result.get("ok") is False:
        await cl.Message(content="Could not remove components: "
                         + result.get("error", "Unknown tool error.")).send()
        return
    msg = cl.Message(content=f"Removed ICA components {p['components']} (in-session).")
    if image_b64:
        msg.elements = [_png_element(image_b64, "apply-ica.png")]
    await msg.send()

    if not (p["wants_epoch"] or p["wants_measure"]):
        return  # remove-only request: done

    # 2) contrast: resolve codes from the scoped codebook, bin -> epoch -> difference wave
    if p["wants_epoch"]:
        conditions = (SCOPE.get("codebook") or {}).get("conditions") or {}
        if not conditions:
            await cl.Message(content=(
                "Removed the components, but cannot epoch the contrast: no event codebook is "
                "scoped. Run `/scope <file>` (or drop the recording) so 'faces vs cars' resolves "
                "to codes, then re-issue the epoch/measure part.")).send()
            return
        codes_a = _resolve_condition_codes(p["cond_a"], conditions)
        codes_b = _resolve_condition_codes(p["cond_b"], conditions)
        missing = [n for n, c in ((p["cond_a"], codes_a), (p["cond_b"], codes_b)) if c is None]
        if missing:
            await cl.Message(content=(
                f"Removed the components, but these conditions are not in the codebook: {missing}. "
                f"Known conditions: {sorted(conditions)}.")).send()
            return
        if "epoch_tmin" not in p:
            await cl.Message(content=(
                "Removed the components, but no epoch window was given, and I won't assume one "
                "(that would bake in a task-specific value). Add it explicitly, e.g. "
                "'epoch faces vs cars from -200 to 800 ms'.")).send()
            return
        et0, et1 = p["epoch_tmin"], p["epoch_tmax"]
        r, _ = await _run_tool("create_bins", {"bins": [
            {"label": "B1", "codes": codes_a}, {"label": "B2", "codes": codes_b}]})
        if r.get("ok") is False:
            await cl.Message(content="create_bins failed: " + r.get("error", "?")).send()
            return
        r, _ = await _run_tool("create_epochs", {"tmin": et0, "tmax": et1})
        if r.get("ok") is False:
            await cl.Message(content="create_epochs failed: " + r.get("error", "?")).send()
            return
        chans = [p["channel"]] if p["channel"] else None
        r, image_b64 = await _run_tool("compute_difference_erp", {
            "event_id_a": "B1", "event_id_b": "B2", "tmin": et0, "tmax": et1,
            "channels": chans, "plot_style": "channels"})
        if r.get("ok") is False:
            await cl.Message(content="compute_difference_erp failed: " + r.get("error", "?")).send()
            return
        dm = cl.Message(content=(f"Difference wave {p['cond_a']} minus {p['cond_b']} "
                                 f"({r.get('n_epochs_a')} vs {r.get('n_epochs_b')} epochs)."))
        if image_b64:
            dm.elements = [_png_element(image_b64, "difference-erp.png")]
        await dm.send()

    # 3) measure the component on the stored difference wave
    if p["wants_measure"]:
        if "meas_tmin" not in p or not p["channel"]:
            need = []
            if "meas_tmin" not in p:
                need.append("a measurement window (e.g. '110-150 ms')")
            if not p["channel"]:
                need.append("a channel (e.g. 'at PO8')")
            await cl.Message(content="Skipped the measurement: missing " + " and ".join(need)
                             + ".").send()
            return
        r, image_b64 = await _run_tool("measure_component", {
            "tmin": p["meas_tmin"], "tmax": p["meas_tmax"],
            "channels": [p["channel"]], "mode": p["mode"]})
        if r.get("ok") is False:
            await cl.Message(content="measure_component failed: " + r.get("error", "?")).send()
            return
        val = r.get("amplitude_uv")
        val_s = f"{val:.3f} µV" if isinstance(val, (int, float)) else str(val)
        lat = r.get("latency_ms")
        extra = f" at {lat} ms" if p["mode"] == "peak" and isinstance(lat, (int, float)) else ""
        mm = cl.Message(content=(f"**{p['mode'].title()} amplitude {p['meas_tmin']*1000:.0f}–"
                                 f"{p['meas_tmax']*1000:.0f} ms at {p['channel']}: {val_s}{extra}** "
                                 "(eye-corrected, this pipeline)."))
        if image_b64:
            mm.elements = [_png_element(image_b64, "measure.png")]
        await mm.send()


async def _handle_engine(arg: str):
    """Set a session default, with explicit natural-language per-step overrides."""
    val = arg.strip().lower()
    if val in ("none", "clear", "off"):
        cl.user_session.set("engine", None)
        val = ""
    elif val and val not in ENGINE_CHOICES:
        await cl.Message(content=(
            f"Unknown session engine '{val}'. Choose `/engine mne` or `/engine eeglab`. "
            "`erplab` is the ERP backend within the EEGLAB profile; `eeglab-python` is an "
            "advanced ICA backend that can be requested for that step in natural language. "
            "The session default has not changed."
        )).send()
        return
    elif val:
        cl.user_session.set("engine", val)
    current = cl.user_session.get("engine")
    label = current.upper() if current in ENGINE_CHOICES else "not set (tool defaults: MNE)"
    lines = [f"**Session engine default: {label}**"]
    if current == "eeglab":
        lines.append("ICA: **EEGLAB** (`eeglab`, genuine runica via Octave). "
                     "Filtering, binning, epoching and component measurement: **ERPLAB** (`erplab`).")
        missing = eeglab_ica_availability()
        if missing:
            lines.append(f"EEGLAB ICA is unavailable in this runtime (missing: {', '.join(missing)}). "
                         "It will not fall back to MNE.")
    else:
        lines.append("Engine-capable tools use **MNE** (`mne`) unless explicitly overridden.")
    lines.append("Applies to new natural-language `/plan` and `/sweep` requests. "
                 "A step-specific instruction such as 'use MNE for ICA' overrides it for that "
                 "request without changing the session default. Pasted JSON and saved recipes retain their engines.")
    lines.append("`/engine` shows the default; `/engine none` clears it. "
                 "Tools without an engine option keep their existing implementation.")
    if cl.user_session.get("pending_pipeline") or cl.user_session.get("pending_sweep"):
        lines.append("The already staged plan is unchanged; draft a new plan to use the new default.")
    await cl.Message(content="\n\n".join(lines)).send()


def _try_parse_sweep(text: str) -> dict | None:
    """If `text` is a JSON sweep spec the user pasted, return it (model-free path)."""
    t = text.strip()
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "base_pipeline" in obj else None


async def _handle_save_recipe(kind: str, name: str):
    """Save the currently-staged sweep/plan under a name, for reuse via `/sweep <name>` or
    `/plan <name>`. Usage: `/save-sweep <name>` (after `/sweep`), `/save-plan <name>`."""
    if not name:
        await cl.Message(content=f"Usage: `/save-{kind} <name>` — first `/{kind} ...` to draft "
                         f"one, then name it. e.g. `/save-{kind} filter sweep`.").send()
        return
    staged = cl.user_session.get("pending_sweep" if kind == "sweep" else "pending_pipeline")
    if not staged:
        await cl.Message(content=f"Nothing to save — `/{kind} ...` first, review it, then "
                         f"`/save-{kind} {name}`.").send()
        return
    res = await cl.make_async(save_recipe)(name, kind, staged)
    if not res.get("ok"):
        await cl.Message(content=f"**Could not save:** {res.get('error')}").send()
        return
    await cl.Message(content=(
        f"**Saved {kind} recipe** `{res['slug']}`. Reuse it anytime with "
        f"`/{kind} {res['slug']}` (it rebinds to whatever recording you have loaded). "
        "See all with `/recipes`."
    )).send()


async def _handle_recipes():
    """List saved recipes."""
    recs = await cl.make_async(list_recipes)()
    if not recs:
        await cl.Message(content="No saved recipes yet. Draft a `/sweep` or `/plan`, then "
                         "`/save-sweep <name>` / `/save-plan <name>`.").send()
        return
    lines = ["**Saved recipes** (run with `/sweep <name>` or `/plan <name>`):"]
    for r in recs:
        lines.append(f"- **{r['slug']}** ({r['kind']}) — {r['summary']}")
    await cl.Message(content="\n".join(lines)).send()


def _format_batch_table(rows: list[dict]) -> str:
    """Per-subject QC table: sub | trials | interp | comps | endpoint uV | status."""
    has_lat = any(r.get("latency_ms") is not None for r in rows)
    head = ["subject", "trials", "interp", "comps", "endpoint µV"]
    if has_lat:
        head.append("latency ms")
    head.append("status")
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for r in rows:
        qc = r.get("qc", {})
        trials = qc.get("n_kept", qc.get("n_epochs"))
        cells = [r["sub"],
                 "—" if trials is None else str(trials),
                 str(qc.get("n_interpolated", "—")),
                 str(qc.get("n_components", "—")),
                 "—" if r.get("endpoint_uv") is None else f"{r['endpoint_uv']:.3f}"]
        if has_lat:
            cells.append("—" if r.get("latency_ms") is None else f"{r['latency_ms']:.0f}")
        if r.get("ok"):
            flags = r.get("outlier_flags") or []
            cells.append("✅" if not flags else "⚠ " + "; ".join(flags))
        else:
            cells.append(f"❌ {str(r.get('error', ''))[:44]}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _resolve_dataset_dir(path: str) -> str | None:
    """A dataset dir as typed, or resolved against DATA_DIR. None if neither is a directory."""
    for cand in (path, os.path.join(DATA_DIR, path)):
        if cand and os.path.isdir(cand):
            return cand
    return None


def _split_batch_arg(arg: str) -> tuple[str | None, str, str | None]:
    """Parse `[recipe-name] <dataset-dir>` robustly. The dataset dir may contain spaces (e.g.
    'erpcore_n170 copy') and may be quoted, so a plain last-token split is wrong. Returns
    (recipe_name_or_None, resolved_dataset_dir_or_raw, resolved_or_None). We disambiguate by
    checking which candidate actually exists (as typed or under DATA_DIR), never by whitespace."""
    arg = arg.strip()
    # 1) quoted dataset dir, optionally preceded by a recipe name: n170 "erpcore_n170 copy"
    m = re.match(r'^(?:(.*?)\s+)?["\'](.+?)["\']\s*$', arg)
    if m:
        raw = m.group(2)
        return ((m.group(1) or "").strip() or None), raw, _resolve_dataset_dir(raw)
    # 2) the WHOLE arg is a directory (no recipe) -- covers a bare spaced name like 'erpcore_n170 copy'
    resolved = _resolve_dataset_dir(arg)
    if resolved:
        return None, arg, resolved
    # 3) recipe + dir: peel the FIRST token as the recipe, the rest (spaces allowed) as the dir
    toks = arg.split()
    if len(toks) >= 2:
        rest = " ".join(toks[1:])
        resolved = _resolve_dataset_dir(rest)
        if resolved:
            return toks[0], rest, resolved
    # 4) nothing resolves: report the WHOLE arg as the attempted dataset (spaced dirs are the common
    #    case), so the error names what the user typed and can suggest quoting — not a bare last token.
    return None, arg, None


async def _handle_batch(arg: str):
    """Run a plan across every subject in a BIDS dataset. Usage:
    `/batch [recipe-name] <dataset-dir>` — a saved plan recipe, or the staged /plan if omitted."""
    if not arg.strip():
        await cl.Message(content=(
            "Usage: `/batch [recipe-name] <dataset-dir>` — run a plan on every `sub-*/` in a "
            "BIDS dataset.\n- `/batch n170 erpcore_n170` uses a saved recipe (`/save-plan n170`)\n"
            "- `/batch erpcore_n170` uses the plan you just `/plan`-ned (staged).\n"
            "- Paths with spaces work too, e.g. `/batch \"erpcore_n170 copy\"`.\n"
            "The plan should end in `measure_component` (that's the per-subject endpoint)."
        )).send()
        return

    recipe_name, dataset_raw, dataset_dir = _split_batch_arg(arg)
    if dataset_dir is None:
        listing = [d for d in sorted(os.listdir(DATA_DIR))
                   if os.path.isdir(os.path.join(DATA_DIR, d))] if os.path.isdir(DATA_DIR) else []
        hint = ("\n\nDirectories in your data folder: "
                + ", ".join(f"`{d}`" for d in listing[:20])) if listing else ""
        await cl.Message(content=(
            f"No dataset directory `{dataset_raw}` (looked as-is and under your data folder). "
            f"If the name has spaces, quote it: `/batch \"{dataset_raw}\"`.{hint}")).send()
        return

    spec = None
    source = ""
    if recipe_name:
        rec = load_recipe(recipe_name, kind="plan")
        if rec is None:
            await cl.Message(content=f"No saved plan recipe named `{recipe_name}`. See "
                             "`/recipes`, or omit the name to use your staged `/plan`.").send()
            return
        spec, source = {"pipeline": rec["spec"].get("pipeline", [])}, f"recipe '{rec['name']}'"
    else:
        staged = cl.user_session.get("pending_pipeline")
        if not staged:
            await cl.Message(content="No staged plan and no recipe named. `/plan ...` first (then "
                             "optionally `/save-plan <name>`), or pass a recipe name.").send()
            return
        spec, source = staged, "your staged plan"

    if not _endpoint_in_plan(spec):
        await cl.Message(content="⚠ The plan has no `measure_component` step, so there's no "
                         "per-subject endpoint to tabulate. Add one, or expect an empty endpoint "
                         "column.").send()

    await cl.Message(content=f"**Running batch** ({source}) over `{dataset_dir}` — this runs the "
                     "pipeline on every subject and may take a while…").send()
    result = await cl.make_async(_run_batch_sync)(spec, dataset_dir)
    if not result.get("ok"):
        await cl.Message(content=f"**Batch could not run:** {result.get('error')}").send()
        return

    rows, agg = result["rows"], result["aggregate"]
    content = [f"**Batch complete** — {result['n_subjects']} subjects "
               f"({agg.get('n_ok', 0)} ok). Saved to `{result['out_dir']}`.", "",
               _format_batch_table(rows)]
    ep = agg.get("endpoint")
    if ep:
        content += ["", f"**Endpoint across subjects:** mean {ep['mean_uv']:.3f} µV, "
                    f"sd {ep['sd_uv']:.3f} µV (n={ep['n']})."]
        outliers = [f"{r['sub']} ({'; '.join(r['outlier_flags'])})"
                    for r in rows if r.get("outlier_flags")]
        content.append("**Outliers:** " + (", ".join(outliers) if outliers else "none."))
    ga = agg.get("grand_average")
    if ga:
        line = f"**Grand average:** {ga['n_subjects']} subjects, {ga['n_channels']} channels"
        if ga.get("endpoint_uv") is not None:
            line += f"; group endpoint {ga['endpoint_uv']:.3f} µV"
        content.append(line)
    ok_subs = [r["sub"] for r in rows if r.get("ok")]
    if ok_subs:
        content += ["", f"Inspect any subject's own ERP + endpoint with "
                    f"`/batch-inspect {ok_subs[0]}` (loads that subject for further commands). "
                    f"Name this run with `/name-batch <name>` to refer to it later."]
    cl.user_session.set("last_batch_dir", result["out_dir"])
    elements = ([_png_element(result["grand_average_png"], "grand_average.png")]
                if result.get("grand_average_png") else [])
    await cl.Message(content="\n".join(content), elements=elements).send()


def _endpoint_in_plan(spec: dict) -> bool:
    return any(isinstance(s, dict) and s.get("tool") == "measure_component"
              for s in (spec.get("pipeline") or []))


# --------------------------------------------------------------------------- #
# Inspect one subject after a batch. The batch saves each subject's fully-processed recording
# (batches/<run>/<sub>/state_*.fif). Re-averaging those saved epochs is DETERMINISTIC, so we
# faithfully reproduce that subject's ERP + endpoint (no re-running the stochastic ICA fit), then
# leave the subject loaded so the expert can drill in with any normal command.
# --------------------------------------------------------------------------- #
def _batch_inspect_request(text: str) -> str | None:
    """Return the argument string for a batch-inspect request, else None."""
    t = text.strip()
    # accept both orderings (/batch-inspect and the easily-transposed /inspect-batch)
    m = re.match(r"/(?:batch-inspect|inspect-batch)\b(.*)", t, re.I)
    if m:
        return m.group(1).strip()
    m = re.fullmatch(r"(?:please\s+|can you\s+)?inspect\s+(?:batch\s+)?subject\s+"
                     r"(\S+)(?:\s+(?:from|in)\s+(?:the\s+)?(?:last\s+)?batch)?[.!?]?", t, re.I)
    return m.group(1) if m else None


def _resolve_batch_run(token: str | None) -> str | None:
    """Resolve a run name/path to a batch run dir. None token -> the session's last run, else the
    most recent under BATCHES_DIR."""
    if token:
        for cand in (token, os.path.join(BATCHES_DIR, token)):
            if os.path.isdir(cand):
                return cand
        return None
    last = cl.user_session.get("last_batch_dir")
    if last and os.path.isdir(last):
        return last
    if os.path.isdir(BATCHES_DIR):
        runs = [os.path.join(BATCHES_DIR, d) for d in os.listdir(BATCHES_DIR)
                if os.path.isdir(os.path.join(BATCHES_DIR, d))]
        if runs:
            return max(runs, key=os.path.getmtime)
    return None


async def _handle_batch_inspect(arg: str):
    """Reload one subject from a batch run and show its own ERP + endpoint (deterministically)."""
    parts = arg.split()
    if not parts:
        await cl.Message(content=(
            "Usage: `/batch-inspect <subject>` (e.g. `/batch-inspect sub-003`), or "
            "`/batch-inspect <run> <subject>` to pick a specific run. Defaults to your last batch."
        )).send()
        return
    # Accept "<sub>" or "<run> <sub>"; the LAST token is the subject.
    sub = parts[-1]
    run_token = " ".join(parts[:-1]) or None
    run_dir = _resolve_batch_run(run_token)
    if run_dir is None:
        await cl.Message(content=("No batch run found. Run `/batch <recipe> <dataset>` first, or "
                                  "pass an existing run name.")).send()
        return

    summary_path = os.path.join(run_dir, "summary.json")
    try:
        with open(summary_path) as fh:
            summary = json.load(fh)
    except Exception as exc:
        await cl.Message(content=f"Could not read `{summary_path}`: {exc}").send()
        return
    rows = summary.get("rows") or []
    row = next((r for r in rows if r.get("sub") == sub), None)
    if row is None:
        subs = ", ".join(r.get("sub", "?") for r in rows) or "(none)"
        await cl.Message(content=f"Subject `{sub}` is not in run "
                         f"`{os.path.basename(run_dir)}`. Subjects: {subs}.").send()
        return

    # 1) Report the subject's stored batch result.
    lines = [f"**Inspecting `{sub}`** (run `{os.path.basename(run_dir)}`)."]
    if not row.get("ok"):
        lines.append(f"This subject FAILED in the batch: {row.get('error', 'unknown error')}.")
    ep = row.get("endpoint_uv")
    if isinstance(ep, (int, float)):
        lat = row.get("latency_ms")
        lines.append(f"- Batch endpoint: **{ep:.3f} µV**" + (f" at {lat} ms" if lat else ""))
    if row.get("qc"):
        lines.append("- QC: " + ", ".join(f"{k}={v}" for k, v in row["qc"].items()))
    if row.get("warnings"):
        lines.append("- Scope warnings: " + "; ".join(map(str, row["warnings"])))
    await cl.Message(content="\n".join(lines)).send()

    # 2) Reload the subject's saved processed recording into the session.
    sub_dir = os.path.join(run_dir, sub)
    fifs = sorted(f for f in os.listdir(sub_dir) if f.endswith(".fif")) if os.path.isdir(sub_dir) else []
    if not fifs:
        await cl.Message(content=(f"No saved recording for `{sub}` (looked in `{sub_dir}`). "
                                  "The batch may have run with derivatives disabled.")).send()
        return
    state_path = os.path.join(sub_dir, fifs[0])
    result, _ = await _run_tool("load_eeg", {"filepath": state_path})
    if result.get("ok") is False:
        await cl.Message(content=f"Could not load `{state_path}`: {result.get('error')}").send()
        return

    # 3) Deterministically re-derive this subject's ERP + endpoint. The saved recording already has
    #    all preprocessing baked in (filter/ICA/re-reference), so re-running only the segmentation +
    #    averaging + measurement steps reproduces the exact batch endpoint with no stochastic re-fit.
    #    If the saved recording is already epoched, skip the (raw-only) binning/epoching steps.
    seg_tools = () if result.get("kind") == "epochs" else ("create_bins", "create_epochs")
    endpoint_tools = seg_tools + ("compute_difference_erp", "compute_erp", "compute_psd",
                                  "measure_component")
    steps = [s for s in (summary.get("spec", {}).get("pipeline") or [])
             if isinstance(s, dict) and s.get("tool") in endpoint_tools]
    if not steps:
        await cl.Message(content=(f"`{sub}` is now loaded (no ERP/measurement step in the plan to "
                                  "replot). Run any command to inspect it.")).send()
        return
    images: list = []
    endpoint_uv = None
    for step in steps:
        r, img = await _run_tool(step["tool"], step.get("args", {}) or {})
        if r.get("ok") is False:
            await cl.Message(content=(f"Re-running `{step['tool']}` on `{sub}` failed: "
                             f"{r.get('error')}. The subject is still loaded for manual "
                             "inspection.")).send()
            return
        if img:  # surface the plot in a visible message, not just the collapsed tool step
            images.append(_png_element(img, f"{sub}-{step['tool']}.png"))
        if step["tool"] == "measure_component" and r.get("amplitude_uv") is not None:
            endpoint_uv = r.get("amplitude_uv")
    ep_txt = f", endpoint {endpoint_uv:.3f} µV" if isinstance(endpoint_uv, (int, float)) else ""
    out = cl.Message(content=(
        f"`{sub}` re-derived from its saved recording (matches the batch{ep_txt}). The subject is "
        "loaded — run any command (e.g. `measure_component`, `/review-ica`, `compute_psd`) to "
        "inspect further."))
    if images:
        out.elements = images
    await out.send()


async def _handle_name_batch(arg: str):
    """Rename the just-run batch (its auto timestamp) to a memorable name."""
    name = arg.strip()
    if not name:
        await cl.Message(content=("Usage: `/name-batch <name>` — renames your last batch run so you "
                                  "can refer to it, e.g. `/name-batch pilot-v1` then "
                                  "`/batch-inspect pilot-v1 sub-003`.")).send()
        return
    if name in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        await cl.Message(content=("Use only letters, digits, dash, underscore or dot in a batch "
                                  "name (no spaces or slashes; not `.`/`..`).")).send()
        return
    src = cl.user_session.get("last_batch_dir")
    if not src or not os.path.isdir(src):
        src = _resolve_batch_run(None)  # fall back to the newest run on disk
    if not src:
        await cl.Message(content="No batch run to name. Run `/batch <recipe> <dataset>` first.").send()
        return
    dst = os.path.join(BATCHES_DIR, name)
    if os.path.dirname(os.path.abspath(dst)) != os.path.abspath(BATCHES_DIR):
        await cl.Message(content="Invalid batch name.").send()
        return
    if os.path.abspath(dst) == os.path.abspath(src):
        await cl.Message(content=f"That run is already named `{name}`.").send()
        return
    if os.path.exists(dst):
        await cl.Message(content=(f"A run named `{name}` already exists — pick another name "
                                  "(or delete the old one).")).send()
        return
    try:
        os.rename(src, dst)
    except Exception as exc:
        await cl.Message(content=f"Could not rename the run: {exc}").send()
        return
    cl.user_session.set("last_batch_dir", dst)
    await cl.Message(content=(f"Renamed the batch run to **`{name}`**. Inspect it with "
                     f"`/batch-inspect {name} <subject>` (or just `/batch-inspect <subject>` while "
                     "it's your most recent run).")).send()


def _run_batch_sync(spec: dict, dataset_dir: str) -> dict:
    """Bridge: run_batch is async, but per-subject steps run through a plain synchronous dispatcher
    (not _run_tool) so a many-subject run doesn't flood the UI with hundreds of step messages.
    Called via cl.make_async, i.e. on a worker thread with no running loop, so asyncio.run is safe."""
    import asyncio

    async def _step(name, args):
        return call_tool(name, args or {}), None

    return asyncio.run(run_batch(spec, dataset_dir, _step))


def _sweep_scope_prefix(request: str) -> tuple[str | None, str]:
    """Extract an explicit leading scope instruction, including quoted paths."""
    match = re.match(r'''^\s*/?scope\s+(?:"([^"]+)"|'([^']+)'|([^,;\n]+))\s*[,;\n]\s*(.+)$''',
                     request, re.I | re.S)
    if not match:
        return None, request
    return next(value.strip() for value in match.groups()[:3] if value is not None), match.group(4).strip()


async def _handle_sweep(request: str):
    """Draft a parameter-sweep spec, validate it, show the grid, and stage it for /run."""
    if not request:
        await cl.Message(content=(
            "Usage: `/sweep <what to vary>` — e.g. `/sweep load /data/sub-002.set, "
            "band-pass 0.1-40 Hz, sweep the high-pass over 0.1, 0.5, 1.0, then measure the "
            "N170 mean over 110-150 ms at PO8`\n\n"
            "Or paste a JSON sweep spec directly (no model): "
            "`/sweep {\"base_pipeline\": [...], \"axes\": [...], \"endpoint\": {...}}`"
        )).send()
        return

    scope_path, request = _sweep_scope_prefix(request)
    if scope_path is not None:
        cl.user_session.set("pending_sweep", None)
        cl.user_session.set("pending_pipeline", None)
        if not await _handle_scope(scope_path):
            return
    spec = _try_parse_sweep(request)
    source = "your spec (verbatim, no model)"
    if spec is None:
        rec = load_recipe(request, kind="sweep")  # a saved recipe name? (slug match, no model)
        if rec is not None:
            spec = rebind_to_current(rec["spec"], SESSION.get("filepath"))
            source = f"saved recipe '{rec['name']}' (rebound to the current recording)"
            if not SESSION.get("filepath"):
                await cl.Message(content="⚠ No recording loaded — `/scope <file>` first so the "
                                 "recipe runs on your data.").send()
        else:
            source = "drafted by the assistant — check the base pipeline, axes, and endpoint"
            ctx = await cl.make_async(retrieve_context)(request)
            async with cl.Step(name="planning sweep", type="llm"):
                spec = await cl.make_async(propose_sweep)(
                    MODEL, request, _with_scope(ctx["text"]), cl.user_session.get("engine"),
                    scope=dict(SCOPE))

    # Separate the planner telemetry (attempts / decline) from the spec before display/run.
    meta = spec.pop("_planner_meta", {}) if isinstance(spec, dict) else {}
    if meta.get("declined"):
        cl.user_session.set("pending_sweep", None)
        why = "; ".join(spec.get("notes") or []) or "no sweepable parameter was identified"
        await cl.Message(content=(
            f"**This doesn't look like a sweep** ({why}). A sweep varies a *parameter of a "
            "tool* (a filter cutoff, an ICA seed, ...). If you meant to run a fixed pipeline, "
            "use `/plan`; otherwise name the parameter and the values to try."
        )).send()
        return

    errors = validate_sweep(spec) + engine_runtime_errors(spec)
    if meta.get("valid") is False:
        errors.extend(e for e in (meta.get("errors") or ["Planner validation failed."]) if e not in errors)
    if errors:
        cl.user_session.set("pending_sweep", None)
        cl.user_session.set("pending_pipeline", None)
        tried = f" (after {meta['attempts']} attempts)" if meta.get("attempts", 1) > 1 else ""
        await cl.Message(content=(
            f"**Could not build a valid sweep{tried}:**\n- " + "\n- ".join(errors)
            + "\n\nRefine the request and `/sweep` again."
        )).send()
        return

    repaired = (f" · self-corrected in {meta['attempts']} attempts"
                if meta.get("attempts", 1) > 1 else "")

    variants, _ = expand_sweep(spec)
    cl.user_session.set("pending_sweep", spec)
    grid = "\n".join(f"{i + 1}. {v['label']}" for i, v in enumerate(variants))
    notes = spec.get("notes") or []
    note_md = ("\n\n**Notes / assumptions:**\n- " + "\n- ".join(notes)) if notes else ""
    final_tool = (spec.get("endpoint") or spec["base_pipeline"][-1])["tool"]
    if final_tool == "filter_eeg":
        note_md += "\n\n**Filter diagnostics:** mean EEG PSD from 0 Hz to Nyquist for each variant, plus an overlaid comparison."
    await cl.Message(content=(
        f"**Proposed sweep** ({source}{repaired}) — **{len(variants)} variants**. Review, then "
        "`/run` to execute *exactly this* (no model in the loop), or `/cancel`:\n"
        f"```json\n{format_config(spec)}\n```"
        f"{engine_review(spec, cl.user_session.get('engine'))}\n**Variants:**\n{grid}{note_md}{_qc_block(spec)}"
    )).send()


def _format_sweep_table(rows: list[dict]) -> str:
    """Markdown comparison table: one row per variant."""
    return format_sweep_table(rows, success="✅", failure="❌")


def _format_stochastic(stochastic: list[dict]) -> str:
    lines = ["**Stochastic (repeat) spread — uncertainty from the random step:**"]
    for g in stochastic:
        grp = g["group"] if isinstance(g["group"], str) else \
            ", ".join(f"{k}={v}" for k, v in g["group"].items())
        lines.append(f"- {grp or '(all seeds)'}: mean {g['mean_uv']:.3f} µV, "
                     f"sd {g['sd_uv']:.3f} µV, spread {g['spread_uv']:.3f} µV (N={g['n']})")
    return "\n".join(lines)


def _plot_spec_curve(rows: list[dict], spec: dict) -> str | None:
    """Specification curve: endpoint vs the swept value (line if a single numeric axis,
    else a bar per variant). Returns a base64 PNG, or None if nothing to plot."""
    import io
    import matplotlib.pyplot as plt

    if any(row.get("output_tool") == "filter_eeg" for row in rows):
        return None  # Filter sweeps compare spectra, not their input cutoffs against themselves.
    ylabel = "endpoint (µV)"
    pts = [(r, r["endpoint_uv"]) for r in rows if r.get("ok") and r.get("endpoint_uv") is not None]
    if not pts:
        # Non-ERP sweep (filter/ICA/PSD): no endpoint amplitude, so plot a numeric metric across
        # the variants -- prefer one that actually varies, else the first numeric metric.
        ok_rows = [r for r in rows if r.get("ok")]
        metric_keys = []
        for r in ok_rows:
            for k, v in (r.get("metrics") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in metric_keys:
                    metric_keys.append(k)
        def _vals(k):
            return [r["metrics"][k] for r in ok_rows
                    if isinstance((r.get("metrics") or {}).get(k), (int, float))
                    and not isinstance(r["metrics"][k], bool)]
        # Only plot a metric that actually VARIES across variants -- a flat line (e.g. n_channels
        # constant) is misleading; the table already shows constant values.
        chosen = next((k for k in metric_keys if len(set(_vals(k))) > 1), None)
        if not chosen:
            return None
        pts = [(r, r["metrics"][chosen]) for r in ok_rows
               if isinstance((r.get("metrics") or {}).get(chosen), (int, float))
               and not isinstance(r["metrics"][chosen], bool)]
        if len(pts) < 2:
            return None  # a single point is not a comparison
        ylabel = chosen
    axes = spec.get("axes") or []
    single_numeric = (len(axes) == 1 and axes[0].get("values")
                      and all(isinstance(v, (int, float)) for v in axes[0]["values"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    if single_numeric:
        param = f"{axes[0]['tool']}.{axes[0]['param']}"
        xy = sorted((list(r["assignments"].values())[0], y) for r, y in pts)
        ax.plot([x for x, _ in xy], [y for _, y in xy], "o-")
        ax.set_xlabel(param)
    else:
        labels = [r["label"] for r, _ in pts]
        ax.bar(range(len(pts)), [y for _, y in pts])
        ax.set_xticks(range(len(pts)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title("Specification curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


async def _run_sweep_and_render(spec: dict):
    """Execute a staged sweep and render the comparison table + specification curve."""
    result = await run_sweep(spec, _run_tool)
    if not result.get("ok"):
        await cl.Message(content=(
            "**Sweep could not run:**\n- " + "\n- ".join(result.get("errors", ["unknown"]))
        )).send()
        return
    rows = result["rows"]
    variant_images = result.pop("images", [])
    summary = result.get("summary") or {}
    comparison_name = result.get("comparison_figure")
    png = (dict(variant_images).get(comparison_name) if comparison_name else
           await cl.make_async(_plot_spec_curve)(rows, spec))
    elements = [_png_element(png, "sweep.png")] if png else []
    cl.user_session.set("last_export", {
        "kind": "sweep", "spec": spec, "results": result,
        "images": ([("specification-curve", png)] if png and not comparison_name else []) + variant_images,
    })
    content = (f"**Sweep complete — {result['n_variants']} variants.**\n\n"
               + _format_sweep_table(rows))
    if summary.get("stochastic"):
        content += "\n\n" + _format_stochastic(summary["stochastic"])
    await cl.Message(content=content, elements=elements).send()
    figures = dict(variant_images)
    for i, row in enumerate(rows, 1):
        details = [cl.File(name=f"variant-{i}-results.json", display="inline", mime="application/json",
                           content=json.dumps(row, indent=2, default=str).encode())]
        if row.get("figure") in figures:
            details.append(_png_element(figures[row["figure"]], row["figure"] + ".png"))
        note = (row.get("output") or {}).get("note")
        psd_diagnostic = row.get("diagnostics", {}).get("psd")
        if psd_diagnostic:
            psd = psd_diagnostic["result"]
            if psd.get("ok"):
                psd_note = (f"PSD: {psd['fmin']:g}-{psd['fmax']:g} Hz, mean across "
                            f"{psd['n_channels']} EEG channels. Spectrum values in the JSON are V^2/Hz.")
                resolution = psd.get("frequency_resolution_hz")
                if resolution is not None:
                    psd_note += f" Frequency spacing: {resolution:g} Hz."
            else:
                psd_note = f"PSD diagnostic failed: {psd.get('error', 'unknown error')}. Filter result is retained."
            note = f"{note}\n\n{psd_note}" if note else psd_note
        labels = (row.get("output") or {}).get("labels")
        label_text = ""
        if isinstance(labels, list) and labels:
            label_text = "\n\n| component (1-based) | label | confidence |\n| --- | --- | --- |\n"
            label_text += "\n".join(
                f"| {item.get('component')} | {item.get('label')} | {item.get('confidence')} |"
                for item in labels if isinstance(item, dict))
        elif isinstance(labels, dict) and labels.get("error"):
            label_text = "\n\n" + str(labels["error"])
        await cl.Message(content=(f"**Variant {i}: {row['label']}**\n\n"
                                  f"Final output: `{row.get('output_tool')}`."
                                  + (f"\n\n{note}" if note else "") + label_text), elements=details).send()


async def _handle_run():
    """Execute the staged pipeline (or sweep) deterministically through the dispatcher."""
    spec = cl.user_session.get("pending_sweep")
    if spec:
        cl.user_session.set("pending_sweep", None)  # consume the approval
        cl.user_session.set("last_export", None)
        await _run_sweep_and_render(spec)
        return
    config = cl.user_session.get("pending_pipeline")
    if not config:
        await cl.Message(content="No pending pipeline. Use `/plan <request>` first.").send()
        return
    cl.user_session.set("pending_pipeline", None)  # consume the approval
    cl.user_session.set("last_export", None)

    # _run_tool is the exact dispatcher the agentic loop uses; here the (name, args)
    # pairs come from the approved config, not from the model.
    results, images = await run_pipeline(config, _run_tool)

    elements = [_png_element(b64, f"{name}.png") for name, b64 in images]
    failed = any(
        isinstance(r["result"], dict) and r["result"].get("ok") is False
        for r in results
    )
    lines = []
    for r in results:
        res = r["result"]
        ok = res.get("ok") if isinstance(res, dict) else None
        mark = "✅" if ok else "❌"
        lines.append(f"{mark} `{r['tool']}` {json.dumps(r['args'], default=str)}")
    header = "Pipeline stopped at a failing step." if failed else "Pipeline completed."
    await cl.Message(content=f"**{header}**\n" + "\n".join(lines),
                     elements=elements).send()

    if not failed:
        cl.user_session.set("last_export", {
            "kind": "pipeline", "spec": config, "results": results, "images": images,
        })

    narration = await cl.make_async(_narrate_results)(results)
    if narration:
        await cl.Message(content=narration).send()


async def _handle_export(arg: str):
    """Write the last completed `/run` as an immutable, user-visible analysis deliverable."""
    name = arg.strip()
    if not name:
        await cl.Message(content=(
            "Usage: `/export <analysis-name>` — after a successful `/run`, writes a reproducible "
            "folder under `data-out/derivatives/eeg-llm/`."
        )).send()
        return
    last = cl.user_session.get("last_export")
    if not isinstance(last, dict):
        await cl.Message(content=(
            "Nothing completed to export. Run an approved `/plan` or `/sweep` first; failed "
            "runs are intentionally not exportable as completed analyses."
        )).send()
        return

    def _save_snapshot(path: str) -> dict:
        return call_tool("save_eeg", {"filepath": path})

    snapshot = _save_snapshot if last["kind"] == "pipeline" else None
    async with cl.Step(name="exporting analysis", type="tool") as step:
        result = await cl.make_async(export_analysis)(
            name=name,
            kind=last["kind"],
            spec=last["spec"],
            results=last["results"],
            scope=dict(SCOPE),
            images=last.get("images"),
            save_snapshot=snapshot,
        )
        step.output = json.dumps(result, indent=2, default=str)
    if not result.get("ok"):
        await cl.Message(content=f"**Export failed:** {result.get('error', 'unknown error')}").send()
        return
    await cl.Message(content=(
        f"**Export complete.** Open `data-out/{result['relative_path']}`.\n"
        f"- configuration, scope, results, and report saved\n"
        f"- processed data: `{result.get('data_file') or 'not applicable for a sweep'}`\n"
        f"- figures: {len(result.get('figures') or [])}"
    )).send()


async def _handle_save(arg: str):
    """Checkpoint the current processed data (.fif snapshot) + the replayable applied-step
    config, so work survives a dashboard restart. Usage: /save <name>."""
    name = arg.split()[0] if arg.split() else ""
    if not name:
        await cl.Message(content="Usage: `/save <name>` — checkpoints current data + applied "
                                 "steps to `sessions/<name>/`.").send()
        return
    if SESSION.get("raw") is None and SESSION.get("epochs") is None:
        await cl.Message(content="Nothing to save — load and process a recording first.").send()
        return
    dest = os.path.join(SESSIONS_DIR, name)
    os.makedirs(dest, exist_ok=True)
    res = await cl.make_async(call_tool)("save_eeg", {"filepath": os.path.join(dest, "state.fif")})
    if not res.get("ok"):
        await cl.Message(content=f"**Save failed:** {res.get('error')}").send()
        return
    steps = cl.user_session.get("applied_steps") or []
    with open(os.path.join(dest, "pipeline.json"), "w") as f:
        json.dump({"pipeline": steps}, f, indent=2, default=str)
    meta = {"name": name, "kind": res.get("applied_to"), "sfreq": res.get("sfreq"),
            "n_steps": len(steps), "state_file": os.path.basename(res["saved"]),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(dest, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    await cl.Message(content=(
        f"**Checkpoint `{name}` saved.**\n"
        f"- state: `{res['saved']}`\n"
        f"- steps recorded: {len(steps)}\n"
        f"- resume any time (even after a restart) with `/resume {name}`"
    )).send()


async def _handle_resume(arg: str):
    """Reload a saved checkpoint: restores the processed data and prints its applied-step
    provenance. Usage: /resume <name> (bare /resume lists checkpoints)."""
    name = arg.split()[0] if arg.split() else ""
    if not name:
        avail = sorted(os.listdir(SESSIONS_DIR)) if os.path.isdir(SESSIONS_DIR) else []
        listing = "\n".join(f"- `{a}`" for a in avail) or "_none saved yet_"
        await cl.Message(content=f"Usage: `/resume <name>`. Saved checkpoints:\n{listing}").send()
        return
    dest = os.path.join(SESSIONS_DIR, name)
    meta_path = os.path.join(dest, "meta.json")
    if not os.path.isfile(meta_path):
        await cl.Message(content=f"No checkpoint named `{name}` in `{SESSIONS_DIR}`.").send()
        return
    with open(meta_path) as f:
        meta = json.load(f)
    res = await cl.make_async(call_tool)(
        "load_eeg", {"filepath": os.path.join(dest, meta["state_file"])})
    if not res.get("ok"):
        await cl.Message(content=f"**Resume failed:** {res.get('error')}").send()
        return
    with open(os.path.join(dest, "pipeline.json")) as f:
        steps = json.load(f).get("pipeline", [])
    cl.user_session.set("applied_steps", steps)
    prov = "\n".join(f"{i + 1}. `{s['tool']}` {json.dumps(s.get('args', {}), default=str)}"
                     for i, s in enumerate(steps)) or "_no recorded steps_"
    await cl.Message(content=(
        f"**Resumed checkpoint `{name}`** (saved {meta.get('saved_at')}).\n"
        f"- loaded `{meta['state_file']}` — kind={res.get('kind')}, sfreq={res.get('sfreq')}\n\n"
        f"**Applied-step provenance:**\n{prov}\n\n"
        "Continue processing from here, or `/compare` to verify."
    )).send()


_RECORDING_EXTS = (".set", ".edf", ".fif", ".bdf", ".vhdr")
# sidecars that must accompany a main recording (dragged together)
_SIDECAR_EXTS = (".fdt", ".eeg", ".vmrk")

import re as _re


def _bids_format_recording(flat_path: str) -> tuple[str, str]:
    """BIDS-format a newly-dropped recording into DATA_DIR.

    Reads with _load_any (tolerant of EEGLAB .set/.fdt pairs), writes via mne-bids into
    DATA_DIR/<sub-XXX>/eeg/<sub-XXX>_task-<task>_eeg.<ext> with EEGLAB format.

    subject: parsed from filename sub-([A-Za-z0-9]+); else next free sub-NNN under DATA_DIR.
    task: parsed from task-([A-Za-z0-9]+); else 'task' (label only, no scientific content).

    Returns (bids_set_path, status_note) on success; raises on failure so the caller can STOP
    and report rather than silently continue (fail-loud contract).
    """
    import mne_bids

    from tools import _load_any

    basename = os.path.basename(flat_path)

    # --- subject: read from filename or assign next free ---
    sub_m = _re.search(r"(?:^|[_\-])sub[-_]([A-Za-z0-9]+)", basename, _re.IGNORECASE)
    if sub_m:
        subject = sub_m.group(1)
    else:
        # next free sub-NNN: scan existing sub-* dirs under DATA_DIR
        existing = set()
        if os.path.isdir(DATA_DIR):
            for d in os.listdir(DATA_DIR):
                m = _re.match(r"sub-(\d+)$", d)
                if m:
                    existing.add(int(m.group(1)))
        n = 1
        while n in existing:
            n += 1
        subject = f"{n:03d}"

    # --- task: read from filename or default 'task' ---
    task_m = _re.search(r"task[-_]([A-Za-z0-9]+)", basename, _re.IGNORECASE)
    task = task_m.group(1) if task_m else "task"
    # 'task' default is intentional: it is a BIDS file-label only, not a scientific annotation.

    kind, raw = _load_any(flat_path)
    if kind != "raw":
        raise RuntimeError(
            f"BIDS auto-format expects a continuous recording; got {kind!r} from {basename}. "
            "Drop epoched files without BIDS-format (use /scope <file> directly)."
        )

    # Strip montage/fiducials to avoid mne-bids 'head frame must contain nasion ...' error.
    # The position information is NOT lost: it comes from channels.tsv / electrodes.tsv written
    # by mne-bids from the file's original chanlocs, and reload re-applies standard_1005.
    raw.set_montage(None, verbose="ERROR")

    bids_path = mne_bids.BIDSPath(
        subject=subject, task=task, datatype="eeg", root=DATA_DIR
    )
    mne_bids.write_raw_bids(
        raw, bids_path, overwrite=True, allow_preload=True, format="EEGLAB", verbose=False
    )

    # Locate the written .set
    eeg_dir = os.path.join(DATA_DIR, f"sub-{subject}", "eeg")
    written = [f for f in os.listdir(eeg_dir) if f.endswith("_eeg.set")]
    if not written:
        raise RuntimeError(f"mne-bids wrote to {eeg_dir} but no _eeg.set found.")
    bids_set = os.path.join(eeg_dir, written[0])

    # Verify round-trip: load the written file back (via _load_any, already imported; app.py does
    # not import mne, and _load_any tolerantly handles the .set/.fdt pair).
    _kind2, raw2 = _load_any(bids_set)
    if len(raw2.ch_names) != len(raw.ch_names):
        raise RuntimeError(
            f"BIDS round-trip channel mismatch: written {len(raw.ch_names)}, "
            f"reloaded {len(raw2.ch_names)}. Aborting BIDS format."
        )

    note = (f"sub-{subject}/eeg/{os.path.basename(bids_set)} "
            f"(task={task!r}, {len(raw2.ch_names)} channels, round-trip OK)")
    return bids_set, note


def _dropped_codebook_name(saved: list) -> str | None:
    cb_file = next((n for n in saved if n.lower().endswith(".codebook.json")), None)
    if cb_file is None:
        for n in saved:
            if not n.lower().endswith(".json"):
                continue
            try:
                with open(os.path.join(DATA_DIR, n)) as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "conditions" in data:
                    cb_file = n
                    break
            except Exception:
                continue
    return cb_file


async def _apply_dropped_codebook(saved: list) -> bool:
    """Bind an explicitly uploaded codebook to the recording the user scopes next."""
    cb_file = _dropped_codebook_name(saved)
    if not cb_file or not SCOPE.get("filepath"):
        return False
    try:
        with open(os.path.join(DATA_DIR, cb_file)) as fh:
            cb = json.load(fh)
        res = set_codebook(conditions=cb.get("conditions"), responses=cb.get("responses"))
    except Exception as e:
        await cl.Message(content=f"⚠ Could not apply codebook `{cb_file}`: {e}").send()
        return False
    if res.get("ok"):
        conds = ", ".join((cb.get("conditions") or {}).keys())
        await cl.Message(content=f"Applied codebook from `{cb_file}` (conditions: {conds}).").send()
        return True
    else:
        await cl.Message(content=f"⚠ Codebook `{cb_file}` not applied: {res.get('error')}").send()
        return False


async def _handle_uploads(msg: cl.Message) -> bool:
    """Save dragged/uploaded files into DATA_DIR, BIDS-format recordings, and auto-scope.

    Returns True if any files were handled. Multi-file formats (a .set needs its .fdt; a .vhdr
    needs .eeg + .vmrk) work by dragging the whole group at once -- all are saved, then each
    recording is BIDS-formatted into DATA_DIR/<sub-XXX>/eeg/. After formatting,
    resolve_data_path can find the file by the original simple name (sub-002.set) via the BIDS
    recursive fallback. A lone sidecar (no main file) is reported, not silently ignored.

    Non-recording files (.json, .tsv, .codebook.json, etc.) are saved flat into DATA_DIR beside
    any already-present recording -- the scope codebook chain (BIDS events.tsv, .codebook.json)
    will find them there or under the eeg/ subfolder.
    """
    files = [(el.name, el.path) for el in (msg.elements or [])
             if getattr(el, "path", None) and getattr(el, "name", None)]
    if not files:
        return False
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    for name, path in files:
        try:
            dest = os.path.join(DATA_DIR, os.path.basename(name))
            shutil.copy(path, dest)
            saved.append(os.path.basename(name))
        except Exception as e:
            await cl.Message(content=f"**Upload failed** for `{name}`: {e}").send()
    if not saved:
        return True
    codebook_name = _dropped_codebook_name(saved)
    if codebook_name:
        cl.user_session.set("pending_codebook_uploads", [codebook_name])

    recordings = [n for n in saved if n.lower().endswith(_RECORDING_EXTS)]
    sidecars_only = [n for n in saved if n.lower().endswith(_SIDECAR_EXTS)]
    listed = ", ".join(f"`{n}`" for n in saved)

    if not recordings:
        if sidecars_only:
            await cl.Message(content=(
                f"Received {listed} into `data-in/`, but that's a **sidecar** with no main "
                f"recording. Drag the main file too (`.set` with its `.fdt`; `.vhdr` with "
                f"`.eeg`+`.vmrk`), then I can scope it.")).send()
        else:
            # .json / .tsv / .codebook.json -- config files: ack and let the user scope
            await cl.Message(content=(
                f"Received {listed} into `data-in/`. "
                f"Use `/scope <recording>` to load the recording these belong to; "
                + ("The uploaded codebook will be applied when you scope it, including a leading "
                   "`/sweep scope <recording>, ...` instruction." if codebook_name else
                   "Matching metadata files are discovered when the recording is scoped.")
            )).send()
        return True

    primary = recordings[0]
    extra = ""
    if len(recordings) > 1:
        extra = f" (BIDS-formatting the first; others: {', '.join(recordings[1:])})"
    await cl.Message(content=f"Received {listed}. BIDS-formatting `{primary}`{extra}…").send()

    # BIDS auto-format: flat file -> sub-XXX/eeg/ layout with events.tsv + channels.tsv.
    flat_path = os.path.join(DATA_DIR, primary)
    try:
        bids_set, note = await cl.make_async(_bids_format_recording)(flat_path)
        await cl.Message(content=f"BIDS-formatted: `{note}`").send()
        # Remove the flat copy now that BIDS layout is in place (avoids confusion on /scope).
        try:
            os.remove(flat_path)
            fdt_flat = flat_path.replace(".set", ".fdt")
            if os.path.exists(fdt_flat):
                os.remove(fdt_flat)
        except Exception:
            pass
        # Scope using the simple original name -- resolve_data_path's BIDS fallback finds it.
        await _handle_scope(primary)
        # _handle_scope applies a pending uploaded codebook after BIDS relocation.
    except Exception as exc:
        # FAIL LOUD: report the BIDS failure, still scope the flat file as a fallback.
        await cl.Message(content=(
            f"**BIDS auto-format failed** for `{primary}`: {exc}\n\n"
            f"The file is still in `data-in/` as a flat file and can be scoped by name. "
            f"BIDS layout will not be available."
        )).send()
        await _handle_scope(primary)

    return True


def _normalize_slash_command(text: str) -> str:
    """Let users type '_' or '-' in a slash command (e.g. /lab_rules == /lab-rules). Only the
    FIRST token is normalized, so filenames/JSON args with underscores are untouched; text that is
    not a slash command is returned unchanged."""
    if not text.startswith("/"):
        return text
    cmd, sep, rest = text.partition(" ")
    return cmd.replace("_", "-") + (sep + rest if sep else "")


@cl.on_message
async def on_message(msg: cl.Message):
    text = _normalize_slash_command(msg.content.strip())

    # --- uploaded/dragged files: save into DATA_DIR + auto-scope the primary recording ---
    if msg.elements:
        handled = await _handle_uploads(msg)
        if handled and not text:
            return

    # --- deterministic pipeline commands (you dictate, code executes) ---
    inspection = _inspect_components_request(text)
    # A request naming specific component numbers is an inspect-those, even phrased as "review the
    # ICA components 1 and 8". A numberless review/overview is the whole-line-up scan view.
    if inspection and re.search(r"\d", inspection):
        await _handle_inspect_components(inspection, text)
        return
    if _review_ica_request(text):
        await _handle_review_ica()
        return
    if inspection is not None:
        await _handle_inspect_components(inspection, text)
        return
    # deterministic in-session "remove ICA components N... [then epoch A vs B and measure ...]"
    if _apply_ica_compound_request(text):
        await _handle_apply_ica_compound(text)
        return
    if text.startswith("/params"):
        await _handle_params(text[len("/params"):].strip())
        return
    if text.startswith("/lab-rules"):
        await _handle_lab_rules(text[len("/lab-rules"):].strip())
        return
    if text.startswith("/scope"):
        await _handle_scope(text[len("/scope"):].strip())
        return
    if text.startswith("/codebook"):
        await _handle_codebook(text[len("/codebook"):].strip())
        return
    if text.startswith("/engine"):
        await _handle_engine(text[len("/engine"):].strip())
        return
    # recipe save-verbs must precede /save (prefix) and /sweep,/plan
    if text.startswith("/save-sweep"):
        await _handle_save_recipe("sweep", text[len("/save-sweep"):].strip())
        return
    if text.startswith("/save-plan"):
        await _handle_save_recipe("plan", text[len("/save-plan"):].strip())
        return
    if text.startswith("/recipes"):
        await _handle_recipes()
        return
    # /batch-inspect must precede /batch (prefix), and the NL form is caught before agentic chat.
    if text.startswith("/name-batch"):
        await _handle_name_batch(text[len("/name-batch"):].strip())
        return
    _bi = _batch_inspect_request(text)
    if _bi is not None:
        await _handle_batch_inspect(_bi)
        return
    if text.startswith("/batch"):
        await _handle_batch(text[len("/batch"):].strip())
        return
    if text.startswith("/sweep"):
        await _handle_sweep(text[len("/sweep"):].strip())
        return
    if text.startswith("/plan"):
        await _handle_plan(text[len("/plan"):].strip())
        return
    if text.startswith("/run"):
        await _handle_run()
        return
    if text.startswith("/export"):
        await _handle_export(text[len("/export"):].strip())
        return
    if text.startswith("/save"):
        await _handle_save(text[len("/save"):].strip())
        return
    if text.startswith("/resume"):
        await _handle_resume(text[len("/resume"):].strip())
        return
    if text.startswith("/cancel"):
        cl.user_session.set("pending_pipeline", None)
        cl.user_session.set("pending_sweep", None)
        await cl.Message(content="Cancelled the pending pipeline / sweep.").send()
        return
    if text.startswith("/compare"):
        await _handle_compare(text[len("/compare"):].strip())
        return
    if text.startswith("/tune"):
        await _handle_tune(text[len("/tune"):].strip())
        return

    # Reliable free-text shortcut: a "what parameters/defaults does <tool> take?" question is
    # answered from the deterministic /params reference, never the model's memory.
    detected = _detect_params_query(text)
    if detected == "ASK":
        await _handle_params("")
        return
    if detected:
        await _handle_params(detected)
        return

    # --- agentic chat (the model decides what to run) ---
    await _handle_agentic(msg)


async def _handle_agentic(msg: cl.Message):
    history: list = cl.user_session.get("history")

    # --- RAG: inject retrieved context for this turn ---
    ctx = await cl.make_async(retrieve_context)(msg.content)
    user_content = msg.content
    if ctx["text"]:
        user_content = (
            "Context (from the local EEG knowledge base):\n"
            f"{ctx['text']}\n\n"
            "----\n"
            f"User question: {msg.content}"
        )
    history.append({"role": "user", "content": user_content})

    pending_images: list[cl.Image] = []

    # --- tool-calling loop (non-streaming so tool_calls parse cleanly) ---
    # final_text holds the model's natural-language answer once it stops calling tools.
    final_text = None
    for _ in range(MAX_TOOL_ROUNDS):
        resp = await cl.make_async(ollama.chat)(
            model=MODEL, messages=history, tools=TOOL_SCHEMAS,
        )
        message = resp["message"]
        tool_calls = message.get("tool_calls") or []

        # Coder models often emit the call as ```json text instead of native
        # tool_calls; recover it so the call actually runs.
        if not tool_calls:
            tool_calls = extract_inline_tool_calls(message.get("content") or "")

        if not tool_calls:
            # The model answered in text — this is the final response.
            final_text = (message.get("content") or "").strip()
            if final_text:
                history.append({"role": "assistant", "content": final_text})
            break

        history.append(message)  # keep the tool-call turn for context
        for tc in tool_calls:
            fn = tc["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result, image_b64 = await _run_tool(fn["name"], args)
            if image_b64:
                pending_images.append(_png_element(image_b64, f"{fn['name']}.png"))
            history.append({
                "role": "tool",
                "name": fn["name"],
                "content": json.dumps(result, default=str)[:4000],
            })
    else:
        history.append({
            "role": "user",
            "content": "Tool budget exhausted. Summarise findings so far for the user.",
        })

    # --- emit the answer ---
    out = cl.Message(content="", elements=pending_images)
    await out.send()

    if final_text:
        full = final_text
        out.content = full
        await out.update()
    else:
        # Budget exhausted, or the model returned an empty final turn:
        # stream a fresh natural-language answer.
        full = ""
        stream = await cl.make_async(ollama.chat)(
            model=MODEL, messages=history, stream=True,
        )
        for part in stream:
            token = part["message"]["content"]
            if token:
                full += token
                await out.stream_token(token)
        history.append({"role": "assistant", "content": full})

    if ctx["sources"]:
        full += "\n\n*Sources: " + ", ".join(ctx["sources"]) + "*"
        out.content = full
        await out.update()
