"""MNE-Python tool wrappers exposed to the LLM.

Each tool returns a JSON-serialisable dict. Figures are returned as base64-encoded
PNGs under the key "image" so the Chainlit layer can render them. A small in-memory
SESSION holds the currently loaded recording and derived objects so tools can chain
(load -> psd -> ica -> erp -> autoreject) within one conversation.

The module exposes:
  TOOL_SCHEMAS : list[dict]            -> OpenAI/Ollama function schemas
  TOOL_FUNCTIONS : dict[str, callable] -> name -> python function for dispatch
  call_tool(name, args) -> dict        -> safe dispatch with error capture
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import traceback

import matplotlib

matplotlib.use("Agg")  # headless backend; we never open windows
import matplotlib.pyplot as plt
import numpy as np

import mne

mne.set_log_level("WARNING")

# Ensure sibling modules (artifact_break_removal, artifact_continuous_detect, ...) always
# import from the tool wrappers' lazy imports below, regardless of the launcher's cwd /
# sys.path (e.g. `chainlit run app.py`, whose sys.path need not include this directory).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

# --------------------------------------------------------------------------- #
# Data-path resolution
# --------------------------------------------------------------------------- #
# When running in a container the user's recordings are mounted at DATA_DIR, so a bare
# filename typed into /scope or /plan (e.g. "sub-002.set") must resolve against it. Outside a
# container DATA_DIR simply won't exist and resolution is a no-op, so absolute-path usage is
# unchanged.
DATA_DIR = os.environ.get("EEG_DATA_DIR", "/data")


def resolve_data_path(filepath: str) -> str:
    """Resolve a user-supplied recording path against DATA_DIR when needed.

    An absolute or existing path passes through unchanged. Otherwise, if the name exists directly
    under DATA_DIR, return that (lets a containerised user type `sub-002.set` for
    /data/sub-002.set). If that flat lookup fails (e.g. after BIDS auto-format, the file lives at
    DATA_DIR/sub-002/eeg/sub-002_task-N170_eeg.set), fall back to a recursive search:
      (a) exact basename match anywhere under DATA_DIR, then
      (b) BIDS-pattern match: the sub-ID embedded in the filename (sub-XXX) combined with the
          extension, i.e. sub-002/eeg/sub-002_*_eeg.set.
    Returns the first match, or the original string so the loader raises a clear FileNotFoundError
    rather than a silent wrong-file load."""
    if not filepath or (os.path.isabs(filepath) and os.path.exists(filepath)):
        return filepath
    if os.path.exists(filepath):
        return os.path.abspath(filepath)
    cand = os.path.join(DATA_DIR, filepath)
    if os.path.exists(cand):
        return cand

    # Recursive fallback for files moved into BIDS layout by _handle_uploads.
    if not os.path.isdir(DATA_DIR):
        return filepath
    basename = os.path.basename(filepath)
    ext = os.path.splitext(basename)[1].lower()

    # (a) exact basename match
    for root, _dirs, files in os.walk(DATA_DIR):
        if basename in files:
            return os.path.join(root, basename)

    # (b) BIDS pattern: sub-XXX stem + same extension -> sub-XXX/eeg/sub-XXX_*_eeg<ext>
    sub_match = re.match(r"^(sub-[A-Za-z0-9]+)", basename, re.IGNORECASE)
    if sub_match:
        sub_id = sub_match.group(1).lower()
        for root, _dirs, files in os.walk(DATA_DIR):
            for f in files:
                if (f.lower().startswith(sub_id) and f.lower().endswith("_eeg" + ext)
                        and os.path.basename(root) == "eeg"):
                    return os.path.join(root, f)

    return filepath


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
SESSION: dict = {
    "raw": None,        # mne.io.Raw
    "ica": None,        # mne.preprocessing.ICA
    "epochs": None,     # mne.Epochs
    "evoked": None,     # mne.Evoked (last ERP / difference wave, for measurement)
    "bins": None,       # dict {events: ndarray, id: {label->code}} from create_bins (ERPLAB-style)
    "artifact_flags": None,  # bool mask over epochs, accumulated by the detect_artifact_* tools
    "filepath": None,
}

# Structural + semantic metadata for the CURRENTLY-scoped recording. Deliberately SEPARATE from
# SESSION so it survives load_eeg's SESSION wipe (the onboarding insight: scope structure once,
# reuse it for every plan). Keyed by filepath; consumers must check it matches the loaded file.
SCOPE: dict = {
    "filepath": None,       # the file this scope describes
    "ch_names": None,       # list[str]
    "sfreq": None,          # float
    "n_channels": None,     # int
    "codes": None,          # {int code: int count} -- event-code inventory actually present
    "codebook": None,       # {"conditions": {label: [[lo,hi]|int,...]}, "responses": {label: code}}
    "codebook_source": None,  # 'bids' | 'file' | 'user' | None
    "warnings": None,       # list[str] -- e.g. codebook/data code mismatches
}

# ERP CORE N170 per-step ground-truth checkpoints (released .set/.fdt files). Keyword ->
# path, so compare_to_checkpoint can be driven in natural language from the chat. Only
# sub-002 is on disk; add subjects by extending _CKPT_BASE / this map.
_CKPT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data/erpcore_n170/sub-002/processed/2_N170")
ERPCORE_CHECKPOINTS = {
    "shifted_ds": f"{_CKPT_BASE}_shifted_ds.set",
    "shifted": f"{_CKPT_BASE}_shifted_ds.set",   # step 1 alias (only ds checkpoint saved)
    "reref_ucbip": f"{_CKPT_BASE}_shifted_ds_reref_ucbip.set",
    "reref": f"{_CKPT_BASE}_shifted_ds_reref_ucbip.set",
    "hpfilt": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt.set",
    "ica_prep1": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_prep1.set",
    "ica_prep2": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_prep2.set",
    "ica_corr": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_corr.set",
    # Downstream stages (post-ICA). cbip/elist/bins are continuous Raw (elist/bins add only
    # ERPLAB eventlist/bin metadata, signal unchanged); epoch/interp/ar are epoched (compared
    # via bin-averaged Evoked). See EPOCHED_CHECKPOINTS for kind dispatch.
    "cbip": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip.set",
    "elist": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist.set",
    "bins": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins.set",
    "epoch": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch.set",
    "interp": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set",
    "ar": f"{_CKPT_BASE}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
    # Script 7 (averaging). ERPLAB `.erp` files, NOT EEGLAB `.set` -- a different container
    # (MATLAB struct with bindata (chan, time, bin)), so they get their own kind below.
    # `_erp_ar` holds the 4 parent bins; `_erp_ar_diff_waves` is the superset (9 bins: the same
    # 4 plus the 5 difference waves), and bin 5 = 'Faces minus Cars' is the published N170.
    "erp": f"{_CKPT_BASE}_erp_ar.erp",
    "erp_diff": f"{_CKPT_BASE}_erp_ar_diff_waves.erp",
}

# Checkpoints whose .set holds epoched (not continuous) data: graded via compare_evoked
# (bin-averaged ERP) against SESSION["epochs"], not the continuous-Raw compare().
EPOCHED_CHECKPOINTS = {"epoch", "interp", "ar"}

# Checkpoints that are ERPLAB `.erp` averaged-ERP files: graded via compare_erp against the
# current SESSION evoked/epochs. Default bin is the published contrast (5 = Faces minus Cars)
# for the difference file, 1 (Faces) for the parent file.
ERP_CHECKPOINTS = {"erp": 1, "erp_diff": 5}

# Canonical EEG frequency bands (Hz)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fig_to_b64(fig) -> str:
    """Render a Matplotlib figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _require_raw():
    raw = SESSION.get("raw")
    if raw is None:
        raise RuntimeError("No EEG loaded. Call load_eeg(filepath=...) first.")
    return raw


def _active():
    """Return (kind, obj) for the active recording: continuous Raw or Epochs.

    Lets the preprocessing/analysis tools work on either continuous data or already-
    epoched data (e.g. EEGLAB epoched .set/.mat such as the ANTS sampleEEGdata).
    """
    if SESSION.get("raw") is not None:
        return "raw", SESSION["raw"]
    if SESSION.get("epochs") is not None:
        return "epochs", SESSION["epochs"]
    raise RuntimeError("No EEG loaded. Call load_eeg(filepath=...) first.")


def _load_eeglab_struct(filepath: str):
    """Read an EEGLAB EEG struct from a .set/.mat file via scipy and build an MNE
    object. Handles single-file EEGLAB data MNE's own reader rejects (old structs).
    Returns ('raw'|'epochs', obj) or None if it is not an EEGLAB struct."""
    import scipy.io as sio
    m = sio.loadmat(filepath, squeeze_me=False, struct_as_record=False)
    if "EEG" not in m:
        return None
    EEG = m["EEG"][0, 0]
    sfreq = float(np.array(EEG.srate).ravel()[0])
    data = np.asarray(EEG.data, dtype=float)          # (nch, pnts[, trials])
    chanlocs = EEG.chanlocs[0]
    ch_names = [str(c.labels[0]) for c in chanlocs]
    info = mne.create_info(ch_names, sfreq, "eeg")
    data = data * 1e-6                                # EEGLAB µV -> MNE volts

    if data.ndim == 3:                               # epoched: (nch, pnts, trials)
        data = np.transpose(data, (2, 0, 1))         # -> (trials, nch, pnts)
        times = np.array(EEG.times).ravel()
        tmin = float(times[0]) / 1000.0
        obj = mne.EpochsArray(data, info, tmin=tmin, verbose="ERROR")
        kind = "epochs"
    else:                                            # continuous
        obj = mne.io.RawArray(data, info, verbose="ERROR")
        kind = "raw"
    try:
        # match_case=False so standard 10-20 names differing only in case (e.g. ERP CORE's FP1/FP2
        # vs the montage's Fp1/Fp2) still receive positions; without it no channel is positioned and
        # ICLabel/topomaps fail ("channel position is missing").
        obj.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose="ERROR")
    except Exception:
        pass
    return kind, obj


def _apply_standard_montage_if_missing(obj) -> bool:
    """Give scalp channels standard 10-20 positions ONLY when the file supplied none.

    ICLabel/topomaps need a position for every scalp channel; many EEGLAB .set files (e.g. ERP
    CORE) import with empty chanlocs. This fills them from standard_1005 by name (case-insensitive,
    so FP1/FP2 match Fp1/Fp2). It never overrides real positions: if ANY channel already has one,
    it is a no-op, so a dataset with genuine electrode coordinates is left untouched. This positions
    channels from their standard names; it does NOT infer channel *types* (still the expert's call).
    """
    import numpy as _np
    locs = _np.array([ch["loc"][:3] for ch in obj.info["chs"]])
    if bool((_np.abs(_np.nan_to_num(locs)).sum(1) > 0).any()):
        return False  # the file already carries positions -- do not override
    try:
        obj.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose="ERROR")
        return True
    except Exception:
        return False


def _load_any(filepath: str):
    """Return ('raw'|'epochs', obj) for any supported file."""
    fp = filepath.lower()
    if fp.endswith((".set", ".mat")):
        try:
            res = _load_eeglab_struct(filepath)
            if res is not None:
                return res
        except Exception:
            pass  # fall back to MNE's reader below
    if fp.endswith(".edf"):
        return "raw", mne.io.read_raw_edf(filepath, preload=True)
    if fp.endswith(".bdf"):
        return "raw", mne.io.read_raw_bdf(filepath, preload=True)
    if fp.endswith(".fif") or fp.endswith(".fif.gz"):
        return "raw", mne.io.read_raw_fif(filepath, preload=True)
    if fp.endswith(".set"):
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
        _apply_standard_montage_if_missing(raw)
        return "raw", raw
    if fp.endswith(".vhdr"):
        return "raw", mne.io.read_raw_brainvision(filepath, preload=True)
    return "raw", mne.io.read_raw(filepath, preload=True)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def _positionless_eeg_channels(obj) -> list:
    """Names of channels typed 'eeg' but lacking a usable scalp position (NaN/zero loc).

    These break topography-based steps (component topographies, ICLabel), which need a
    position for every channel. Usually they are EOG/auxiliary electrodes imported as EEG;
    the expert confirms their type via set_channel_types (never inferred here).
    """
    bad = []
    for ch, typ in zip(obj.info["chs"], obj.get_channel_types()):
        if typ != "eeg":
            continue
        loc = ch["loc"][:3]
        if np.all(loc == 0) or np.any(np.isnan(loc)):
            bad.append(ch["ch_name"])
    return bad


def _channel_warning(obj):
    """Surface (do not fix) channels typed EEG but without a scalp position."""
    flagged = _positionless_eeg_channels(obj)
    if not flagged:
        return None
    example = "{" + f'"{flagged[0]}": "eog"' + "}"
    return {
        "positionless_eeg_channels": flagged,
        "message": (
            f"{len(flagged)} channel(s) are typed EEG but have no scalp position: "
            f"{flagged}. Topography-based steps (component topographies, ICLabel) need a "
            f"position for every channel. If these are EOG/auxiliary channels, set their "
            f"type with set_channel_types (e.g. {example}) so those steps skip them."
        ),
    }


def load_eeg(filepath: str) -> dict:
    """Load an EEG recording (continuous or epoched) from disk into the session."""
    filepath = resolve_data_path(filepath)
    kind, obj = _load_any(filepath)

    if kind == "epochs":
        SESSION.update({"raw": None, "ica": None, "epochs": obj, "bins": None,
                        "filepath": filepath})
        out = {
            "ok": True,
            "filepath": filepath,
            "kind": "epochs",
            "sfreq": float(obj.info["sfreq"]),
            "n_epochs": len(obj),
            "n_channels": len(obj.ch_names),
            "ch_names": obj.ch_names[:64],
            "tmin": float(obj.tmin),
            "tmax": float(obj.tmax),
            "note": "Loaded as epoched data; preprocessing/ERP tools operate on epochs.",
        }
        warn = _channel_warning(obj)
        if warn:
            out["channel_warning"] = warn
        return out

    raw = obj
    SESSION.update({"raw": raw, "ica": None, "epochs": None, "bins": None,
                    "filepath": filepath})
    info = raw.info
    out = {
        "ok": True,
        "filepath": filepath,
        "kind": "raw",
        "sfreq": float(info["sfreq"]),
        "n_channels": len(raw.ch_names),
        "ch_names": raw.ch_names[:64],
        "duration_sec": round(raw.n_times / info["sfreq"], 2),
        "highpass": info.get("highpass"),
        "lowpass": info.get("lowpass"),
        "has_annotations": len(raw.annotations) > 0,
        "n_annotations": len(raw.annotations),
    }
    warn = _channel_warning(raw)
    if warn:
        out["channel_warning"] = warn
    return out


# --------------------------------------------------------------------------- #
# Scope + codebook onboarding
#
# The planner cannot invent dataset-specific facts: which channels exist, and what an event
# code MEANS (code 1 = "face"). The recording holds the numbers; the meaning lives in the
# researcher's protocol. `scope_eeg` reads the structure and resolves the code->condition map
# via a fallback chain -- C (BIDS events sidecar) -> B (a codebook file) -> A (the user sets it
# interactively). Nothing is guessed: a resolved map is always the file's or the user's, and is
# validated against the codes actually present (fail loud on mismatch).
# --------------------------------------------------------------------------- #

def _event_inventory(filepath: str, raw) -> dict:
    """{int code: int count} of event codes present. Prefers the .set native events (keeps
    codes exactly), else MNE annotations."""
    from collections import Counter
    if isinstance(filepath, str) and filepath.lower().endswith(".set"):
        native = _eeglab_native_events(filepath)
        if native is not None:
            codes, _, _native_sfreq = native
            return dict(sorted(Counter(int(c) for c in codes).items()))
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        inv_id = {v: k for k, v in event_id.items()}
        cnt = Counter(int(e[2]) for e in events)
        # map MNE's internal ids back to the original code label where it is an int
        out = {}
        for iid, n in cnt.items():
            lbl = inv_id.get(iid, iid)
            try:
                out[int(lbl)] = out.get(int(lbl), 0) + n
            except (TypeError, ValueError):
                out[str(lbl)] = n
        return dict(sorted(out.items(), key=lambda kv: str(kv[0])))
    except Exception:
        return {}


def _codebook_path(filepath: str) -> str:
    """Durable per-recording codebook path: <dir>/<basename>.codebook.json."""
    base = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(os.path.dirname(os.path.abspath(filepath)), f"{base}.codebook.json")


def _looks_like_response(label: str) -> bool:
    return any(t in str(label).lower() for t in ("response", "resp", "button", "accuracy"))


def _discover_bids_events(filepath: str):
    """Option C: parse a BIDS *_events.tsv beside the recording into a codebook, or None.

    Groups the numeric `value` (trigger code) by `trial_type` (label). Response-like labels go
    to 'responses', the rest to 'conditions'. Faithful: uses only the labels present in the file
    -- if they are coarse (e.g. 'stimulus'/'response'), the map is coarse and the user refines it.
    """
    d = os.path.dirname(os.path.abspath(filepath))
    cands = [f for f in os.listdir(d) if f.endswith("_events.tsv")] if os.path.isdir(d) else []
    if not cands:
        return None
    # prefer the events.tsv whose prefix matches this recording, else the first
    base = os.path.basename(filepath).split("_eeg")[0].split(".")[0]
    match = next((f for f in cands if f.startswith(base)), cands[0])
    path = os.path.join(d, match)
    try:
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            if "value" not in header or "trial_type" not in header:
                return None
            vi, ti = header.index("value"), header.index("trial_type")
            groups: dict = {}
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) <= max(vi, ti):
                    continue
                val, tt = cols[vi], cols[ti]
                if val in ("n/a", ""):
                    continue
                try:
                    code = int(float(val))
                except ValueError:
                    continue
                groups.setdefault(tt, set()).add(code)
    except Exception:
        return None
    if not groups:
        return None
    conditions, responses = {}, {}
    for tt, codes in groups.items():
        codes = sorted(codes)
        if _looks_like_response(tt):
            # a response label typically has few distinct codes -> map label:code(s)
            responses[tt] = codes[0] if len(codes) == 1 else codes
        else:
            conditions[tt] = sorted(codes)
    return {"conditions": conditions, "responses": responses, "_source_file": match}


def _load_codebook_file(filepath: str, requested_path: str | None = None):
    """Prefer the recording's sidecar, then exact matching names at the input root."""
    adjacent = _codebook_path(filepath)
    roots = [DATA_DIR]
    # Older native launchers point EEG_DATA_DIR at data/; uploads now live in data-in/.
    if os.path.basename(os.path.normpath(DATA_DIR)) == "data":
        roots.append(os.path.join(os.path.dirname(DATA_DIR), "data-in"))
    names = [os.path.basename(adjacent)]
    if requested_path:
        names.append(os.path.basename(_codebook_path(requested_path)))
    candidates = [adjacent] + [os.path.join(root, name) for root in roots for name in names]
    for path in dict.fromkeys(candidates):
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                cb = json.load(fh)
            if isinstance(cb, dict) and "conditions" in cb:
                return cb
        except (OSError, ValueError):
            continue
    return None


def resolve_condition_codes(name: str, conditions: dict):
    """Resolve exact labels first, then an unambiguous simple singular/plural alias."""
    wanted = name.strip().casefold()
    exact = [codes for label, codes in conditions.items() if label.strip().casefold() == wanted]
    if exact:
        return exact[0] if len(exact) == 1 else None
    def singular(label):
        label = label.strip().casefold()
        return label[:-1] if label.endswith("s") and not label.endswith("ss") else label
    matches = [codes for label, codes in conditions.items() if singular(label) == singular(wanted)]
    return matches[0] if len(matches) == 1 else None


def _iter_codebook_codes(codebook: dict):
    """Yield every concrete integer code a codebook references (expanding [lo,hi] ranges)."""
    if not isinstance(codebook, dict):
        return
    for section in ("conditions", "responses"):
        for _label, spec in (codebook.get(section) or {}).items():
            items = spec if isinstance(spec, list) else [spec]
            for it in items:
                if isinstance(it, list) and len(it) == 2 and all(isinstance(x, int) for x in it):
                    yield from range(it[0], it[1] + 1)
                elif isinstance(it, int):
                    yield it


def _validate_codebook(codebook: dict, codes_present: dict) -> list:
    """Warnings when a codebook and the recording disagree (never silently)."""
    warns = []
    present = set(int(c) for c in (codes_present or {}) if str(c).lstrip("-").isdigit())
    referenced = set(_iter_codebook_codes(codebook))
    missing = sorted(referenced - present)
    if missing:
        warns.append(f"codebook references {len(missing)} code(s) not in the recording: "
                     f"{missing[:12]}{'...' if len(missing) > 12 else ''}")
    undefined = sorted(present - referenced)
    if undefined and referenced:
        warns.append(f"{len(undefined)} recorded code(s) are not in the codebook: "
                     f"{undefined[:12]}{'...' if len(undefined) > 12 else ''}")
    if not referenced:
        warns.append("codebook defines no codes.")
    return warns


def scope_eeg(filepath: str) -> dict:
    """Scope a recording: read its structure + resolve the event code->condition map.

    Loads the file, records channels / sampling rate / event-code inventory, then resolves a
    codebook via C (BIDS *_events.tsv) -> B (<basename>.codebook.json). Stores everything in the
    module-level SCOPE (which survives load_eeg's SESSION wipe) so later /plan and /sweep drafts
    can build create_bins with real codes. If no codebook is found, SCOPE['codebook'] is None and
    the caller should elicit one (Option A, `set_codebook`). Nothing is invented.
    """
    requested_path = filepath
    filepath = resolve_data_path(filepath)
    kind, obj = _load_any(filepath)
    raw = obj
    SESSION.update({"raw": raw if kind == "raw" else None,
                    "epochs": obj if kind == "epochs" else None,
                    "ica": None, "bins": None, "filepath": filepath})
    codes = _event_inventory(filepath, obj)

    codebook, source = _load_codebook_file(filepath, requested_path), None
    if codebook is not None:
        source = "file"
    else:
        bids = _discover_bids_events(filepath)
        if bids is not None:
            codebook, source = bids, "bids"

    warnings = _validate_codebook(codebook, codes) if codebook else []
    SCOPE.update({"filepath": filepath, "ch_names": list(obj.ch_names),
                  "sfreq": float(obj.info["sfreq"]), "n_channels": len(obj.ch_names),
                  "codes": codes, "codebook": codebook, "codebook_source": source,
                  "warnings": warnings})
    return {
        "ok": True,
        "filepath": filepath,
        "kind": kind,
        "n_channels": len(obj.ch_names),
        "ch_names": list(obj.ch_names),
        "sfreq": float(obj.info["sfreq"]),
        "n_event_codes": len(codes),
        "codes": codes,
        "codebook": codebook,
        "codebook_source": source,
        "codebook_status": ("resolved" if codebook else
                            "MISSING -- provide one with set_codebook (or a "
                            "<basename>.codebook.json)"),
        "warnings": warnings,
    }


def set_codebook(conditions: dict | None = None, responses: dict | None = None,
                 save: bool = True) -> dict:
    """Set the event code->condition map for the scoped recording (Option A / B).

    conditions: {label: codes} where codes is a list mixing bare ints and [lo, hi] ranges,
                e.g. {"face": [[1, 40]], "car": [[41, 80]]}.
    responses:  {label: code} for response-contingent binning, e.g. {"correct": 201}.
    save=True writes a durable <basename>.codebook.json beside the recording so it is reused
    across sessions. Validates against the codes actually present and fails loud on a hard
    mismatch (no overlap at all).
    """
    if not SCOPE.get("filepath"):
        return {"ok": False, "error": "Nothing scoped yet; call scope_eeg(filepath=...) first."}
    if not conditions:
        return {"ok": False, "error": "set_codebook needs `conditions` (label -> codes)."}
    codebook = {"conditions": conditions, "responses": responses or {}}
    warnings = _validate_codebook(codebook, SCOPE.get("codes") or {})
    referenced = set(_iter_codebook_codes(codebook))
    present = set(int(c) for c in (SCOPE.get("codes") or {}) if str(c).lstrip("-").isdigit())
    if referenced and present and not (referenced & present):
        return {"ok": False, "error": (
            "None of the codebook's codes appear in the recording -- wrong codebook or wrong "
            f"file? recording has {sorted(present)[:12]}..."), "warnings": warnings}

    saved_to = None
    if save:
        try:
            saved_to = _codebook_path(SCOPE["filepath"])
            with open(saved_to, "w") as fh:
                json.dump(codebook, fh, indent=2)
        except Exception as exc:
            return {"ok": False, "error": f"could not save codebook: {exc}"}
    SCOPE.update({"codebook": codebook, "codebook_source": "user", "warnings": warnings})
    return {"ok": True, "codebook": codebook, "saved_to": saved_to, "warnings": warnings}


def _fmt_codes(spec) -> str:
    """Human/planner-facing code list. Bare-int runs are collapsed to ranges so a 180-code
    condition renders as '1-80, 101-180', not 180 numbers."""
    items = spec if isinstance(spec, list) else [spec]
    ranges, ints = [], []
    for it in items:
        if isinstance(it, list) and len(it) == 2:
            ranges.append(f"{it[0]}-{it[1]}")
        else:
            ints.append(int(it))
    parts = []
    for lo, hi in _to_ranges(sorted(ints)):
        parts.append(str(lo) if lo == hi else f"{lo}-{hi}")
    return ", ".join(parts + ranges)


def _to_ranges(nums):
    """[1,2,3,5,6] -> [(1,3),(5,6)]."""
    out = []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return [(a, b) for a, b in out]


def scope_context() -> str:
    """A compact planner-facing summary of the scoped recording, or '' if none / stale.

    Injected into the planner context so it can name real channels and build create_bins from
    the real code->condition map. Guarded on filepath: if the loaded file no longer matches what
    was scoped, returns '' rather than stale facts.
    """
    if not SCOPE.get("filepath") or SCOPE["filepath"] != SESSION.get("filepath"):
        return ""
    lines = ["Scoped recording (use these real values; do not invent channels or codes):",
             f"- file: {SCOPE['filepath']} (use this EXACT string as load_eeg's filepath when the "
             f"request says 'load it' / 'the recording'; never a placeholder)",
             f"- sampling rate: {SCOPE['sfreq']} Hz; {SCOPE['n_channels']} channels",
             f"- channels: {', '.join(SCOPE['ch_names'][:40])}"
             + (" ..." if len(SCOPE['ch_names']) > 40 else "")]
    cb = SCOPE.get("codebook")
    if cb and cb.get("conditions"):
        conds = "; ".join(f"{lbl}: codes {_fmt_codes(spec)}"
                          for lbl, spec in cb["conditions"].items())
        lines.append(f"- stimulus conditions -> create_bins `bins` (label + codes): {conds}")
        if cb.get("responses"):
            resp = "; ".join(f"{lbl}={_fmt_codes(c)}" for lbl, c in cb["responses"].items())
            lines.append(f"- response codes (for require_following_code): {resp}")
        lines.append("- Build create_bins from these; for response-contingent binning pass "
                     "require_following_code = the correct-response code.")
    else:
        lines.append("- NO codebook resolved: do not fabricate a code->condition map. If the "
                     "request needs binning/epoching by condition, note that a codebook is "
                     "required.")
    return "\n".join(lines)


def set_channel_types(mapping: dict | None = None) -> dict:
    """Set channel types on the loaded recording (e.g. mark EOG/auxiliary channels).

    `mapping` = {channel_name: type}, with type one of eeg/eog/ecg/emg/misc/stim/etc.
    Typing a channel non-eeg (eog/misc) excludes it from scalp-position steps (component
    topographies, ICLabel) and from the default ICA fit. The expert supplies this mapping;
    channel types are never inferred here.
    """
    if not mapping:
        return {"ok": False,
                "error": "set_channel_types requires `mapping` ({channel: type}); it must "
                         "be read from the request, not defaulted."}
    kind, obj = _active()
    present = set(obj.ch_names)
    unknown = [c for c in mapping if c not in present]
    if unknown:
        return {"ok": False,
                "error": f"Unknown channel(s) {unknown}. Loaded channels include "
                         f"{obj.ch_names[:64]}."}
    try:
        obj.set_channel_types({str(c): str(t) for c, t in mapping.items()})
    except Exception as exc:
        return {"ok": False, "error": f"set_channel_types failed: {exc}"}
    SESSION[kind] = obj
    return {"ok": True, "applied_to": kind, "set": mapping,
            "note": f"Set the type of {len(mapping)} channel(s)."}


def save_eeg(filepath: str) -> dict:
    """Write the current processed recording to disk as FIF so it can be reloaded later.

    Symmetric with load_eeg: saves the active raw or epochs object (with all preprocessing
    applied so far). MNE requires FIF names to end with 'raw.fif'/'raw.fif.gz' (continuous)
    or '-epo.fif'/'_epo.fif' (epoched); the name is normalized to satisfy this.
    """
    kind, obj = _active()
    obj.load_data()
    path = str(filepath)
    if kind == "raw":
        if not path.endswith(("raw.fif", "raw.fif.gz")):
            path = (path[:-4] if path.endswith(".fif") else path) + "_raw.fif"
    else:
        if not path.endswith(("-epo.fif", "_epo.fif", "-epo.fif.gz", "_epo.fif.gz")):
            path = (path[:-4] if path.endswith(".fif") else path) + "_epo.fif"
    try:
        obj.save(path, overwrite=True, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001 -- surface any write failure to the caller
        return {"ok": False, "error": f"save failed: {exc}"}
    return {"ok": True, "saved": path, "applied_to": kind,
            "sfreq": float(obj.info["sfreq"]), "n_channels": len(obj.ch_names)}


def compute_psd(fmin: float = 1.0, fmax: float | None = 45.0) -> dict:
    """Compute power spectral density and per-band power, with a plot.

    Works on continuous or epoched data (epoched: PSD averaged over epochs).
    fmax=None uses Nyquist. Numeric spectrum and band powers remain in SI units;
    plots convert the spectrum to microvolt squared per Hz.
    """
    kind, obj = _active()
    if fmax is None:
        fmax = float(obj.info["sfreq"]) / 2.0
    picks = mne.pick_types(obj.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        picks = mne.pick_types(obj.info, meg=False, eeg=True, misc=True)

    spectrum = obj.compute_psd(fmin=fmin, fmax=fmax, picks=picks)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # epochs -> (n_epochs, n_ch, n_freq): average over epochs, then channels
    if psds.ndim == 3:
        psds = psds.mean(axis=0)
    mean_psd = psds.mean(axis=0)

    # Absolute band power = integral of PSD over the band
    band_power = {}
    total = np.trapz(mean_psd, freqs)
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            abs_p = float(np.trapz(mean_psd[m], freqs[m]))
            band_power[name] = {
                "absolute": abs_p,
                "relative": round(abs_p / total, 4) if total > 0 else None,
            }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(freqs, mean_psd * 1e12, color="#1f77b4")
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)",
           title=f"Mean PSD across {len(picks)} channels")
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.07, color="gray")
    ax.grid(True, which="both", alpha=0.3)

    return {
        "ok": True,
        "fmin": fmin,
        "fmax": fmax,
        "n_channels": int(len(picks)),
        "freqs_hz": freqs.tolist(),
        "mean_psd_v2_hz": mean_psd.tolist(),
        "psd_units": "V^2/Hz",
        "band_power_units": "V^2",
        "frequency_resolution_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else None,
        "band_power": band_power,
        "image": _fig_to_b64(fig),
    }


def run_ica(n_components: int = 15, algorithm: str | None = None, label_components: bool = True,
            engine: str = "mne", picks: list | None = None, random_state: int = 97,
            max_iter: int | str = "auto", iclabel_preprocessing: bool = True,
            l_rate: float | None = None, anneal_step: float | None = None) -> dict:
    """Fit ICA on the loaded recording and (optionally) label components with ICLabel.

    engine : which implementation to fit with. 'mne' (default) = MNE's infomax with MNE's
             defaults; 'eeglab' = the real runica.m run in Octave; 'eeglab-python' = EEGLAB's
             ZCA sphering and l_rate/anneal_step schedule, in pure Python. As with
             filter_eeg/create_epochs, this is the expert's choice and is never auto-detected.
    picks  : channels to fit on (names, or 1-based indices as EEGLAB numbers them). Default =
             every EEG channel. Published pipelines routinely fit a subset.
    iclabel_preprocessing : ICLabel is trained on 1-100 Hz band-passed, average-referenced data,
             so by default the fit runs on a copy preprocessed that way. Set False to decompose
             exactly what is loaded (required to reproduce a pipeline that did not do this).
    l_rate, anneal_step : infomax optimizer schedule. Left as None, each implementation uses its
             own default (MNE's for engine='mne', runica's for the EEGLAB engines). Set them only
             when the request names a schedule -- like `engine`, this is the expert's choice and
             is never inferred. The schedule is not a cosmetic knob: on ERP CORE N170 it moves a
             component's topography agreement by ~0.09 |r|, roughly 10x what the random seed does.
    """
    raw = _require_raw()

    # Resolve algorithm: required, no default allowed (contamination rule).
    _ALGO_MAP = {
        "fastica":          ("fastica",  None),
        "infomax":          ("infomax",  dict(extended=False)),
        "extended infomax": ("infomax",  dict(extended=True)),
        "picard":           ("picard",   None),
    }
    if not algorithm or not algorithm.strip():
        return {
            "ok": False,
            "error": (
                "run_ica requires `algorithm` (one of: fastica, infomax, "
                "extended infomax, picard); it must be read from the request, not defaulted."
            ),
        }
    # Normalize: lowercase, strip, collapse spaces/hyphens/underscores to single space.
    import re as _re
    _algo_norm = _re.sub(r"[\s\-_]+", " ", algorithm.strip().lower())
    # Extra alias: "extendedinfomax" or "ext infomax" -> "extended infomax"
    if _algo_norm in ("extendedinfomax", "ext infomax"):
        _algo_norm = "extended infomax"
    if _algo_norm not in _ALGO_MAP:
        return {
            "ok": False,
            "error": (
                f"Unrecognized algorithm {algorithm!r}. "
                "Allowed values: fastica, infomax, extended infomax, picard."
            ),
        }
    _mne_method, _fit_params = _ALGO_MAP[_algo_norm]
    # Human-readable label for the returned dict.
    _method_label = "infomax (extended)" if _algo_norm == "extended infomax" else _algo_norm

    engine = (engine or "mne").lower()
    if engine not in ("mne", "eeglab", "eeglab-python"):
        return {"ok": False,
                "error": f"Unknown engine {engine!r}; use 'mne', 'eeglab' or 'eeglab-python'."}
    if engine != "mne" and _algo_norm not in ("infomax", "extended infomax"):
        return {"ok": False,
                "error": (f"engine={engine!r} runs EEGLAB's runica, which is infomax only; "
                          f"algorithm={algorithm!r} is not available. Use algorithm='infomax' "
                          "or 'extended infomax', or engine='mne'.")}

    # l_rate/anneal_step describe infomax's annealing schedule; no other algorithm has one.
    # Accepting them elsewhere would silently ignore them, which is worse than refusing.
    _schedule = {k: float(v) for k, v in (("l_rate", l_rate), ("anneal_step", anneal_step))
                 if v is not None}
    if _schedule and _algo_norm not in ("infomax", "extended infomax"):
        return {"ok": False,
                "error": (f"{'/'.join(sorted(_schedule))} set the infomax annealing schedule, "
                          f"which algorithm={algorithm!r} does not have. Use algorithm='infomax' "
                          "or 'extended infomax', or leave them unset.")}
    # runica.m accepts 'lrate'/'annealstep' (runica.m:322, :363), so both EEGLAB engines honour
    # the schedule. Worth knowing when reproducing a paper: runica's *internal* default is
    # 0.00065/log(chans), but pop_runica.m -- what every published pipeline actually calls --
    # overrides it with 0.001 (ica_eeglab.POP_RUNICA_LRATE).

    # ICLabel is trained on 1-100 Hz band-passed, average-referenced data. Labelling a fit of
    # data that was not preprocessed that way degrades the labels silently, so make the caller
    # resolve the contradiction rather than quietly returning worse answers.
    if label_components and not iclabel_preprocessing:
        return {"ok": False,
                "error": ("label_components=True requires iclabel_preprocessing=True (ICLabel is "
                          "trained on 1-100 Hz band-passed, average-referenced data). Set "
                          "label_components=False to fit the data as loaded, or "
                          "iclabel_preprocessing=True to label.")}

    if iclabel_preprocessing:
        raw_proc = raw.copy().filter(1.0, 100.0, picks="eeg", verbose="ERROR")
        raw_proc.set_eeg_reference("average", verbose="ERROR")
    else:
        raw_proc = raw

    # `picks`: names or 1-based indices. _epoch_channel_picks only reads .ch_names, so it works
    # on a Raw as well as on Epochs.
    if picks is not None:
        fit_picks, err = _epoch_channel_picks(raw_proc, picks)
        if err:
            return {"ok": False, "error": err}
    else:
        fit_picks = list(mne.pick_types(raw_proc.info, eeg=True, exclude="bads"))
    if len(fit_picks) < 2:
        return {"ok": False, "error": f"Need >=2 channels to fit ICA, got {len(fit_picks)}."}

    # Components the data can support. An average reference -- applied above when
    # iclabel_preprocessing=True -- removes one degree of freedom; without it the data is full
    # rank and one component per channel is correct (ERP CORE Script #3 fits 31 components on 31
    # channels). This used to be an unconditional n-1 cap that silently returned a 30-component
    # decomposition to a caller asking for 31, which is not "31 minus one": infomax redistributes
    # variance across every component, so the result is a different decomposition under the
    # caller's number. Refuse instead of truncating.
    max_components = len(fit_picks) - (1 if iclabel_preprocessing else 0)
    if n_components > max_components:
        return {"ok": False,
                "error": (
                    f"n_components={n_components} exceeds the {max_components} this fit supports "
                    f"({len(fit_picks)} channels"
                    + (" minus one degree of freedom for the average reference applied by "
                       "iclabel_preprocessing" if iclabel_preprocessing else "")
                    + f"). Lower n_components to {max_components}"
                    + (", or set iclabel_preprocessing=False to fit the data as loaded."
                       if iclabel_preprocessing else "."))}
    n_components = int(n_components)

    if engine == "mne":
        fit_params = dict(_fit_params) if _fit_params else {}
        fit_params.update(_schedule)
        ica = mne.preprocessing.ICA(
            n_components=n_components, method=_mne_method,
            fit_params=fit_params or None, max_iter=max_iter, random_state=random_state,
        )
        ica.fit(raw_proc, picks=fit_picks)
    else:
        from ica_eeglab import fit_eeglab_python, fit_eeglab_octave, ica_from_eeglab_weights
        extended = (_algo_norm == "extended infomax")
        X = raw_proc.get_data(picks=fit_picks)
        n_pca = n_components if n_components < len(fit_picks) else None
        try:
            if engine == "eeglab-python":
                weights, sphere = fit_eeglab_python(X, extended=extended, seed=random_state,
                                                    max_iter=max_iter, n_pca=n_pca,
                                                    **_schedule)
            else:
                weights, sphere = fit_eeglab_octave(X, extended=extended, seed=random_state,
                                                    max_iter=max_iter, n_pca=n_pca,
                                                    **_schedule)
        except Exception as exc:            # fail loud: no fallback to another engine
            return {"ok": False, "error": f"engine={engine!r} failed: {exc}"}
        ica = ica_from_eeglab_weights(weights, sphere,
                                      raw_proc.copy().pick(fit_picks).info)
        n_components = int(ica.n_components_)

    SESSION["ica"] = ica

    labels = None
    if label_components:
        try:
            from mne_icalabel import label_components as iclabel
            res = iclabel(raw_proc, ica, method="iclabel")
            # component numbers are 1-based (EEGLAB/ERP CORE convention) so they match what
            # apply_ica(components=...) and inspect_ica_component(...) expect -- no 0-vs-1 trap.
            labels = [
                {"component": i + 1, "label": lab, "confidence": round(float(p), 3)}
                for i, (lab, p) in enumerate(zip(res["labels"], res["y_pred_proba"]))
            ]
        except Exception as exc:  # icalabel optional / model download issues
            labels = {"error": f"ICLabel unavailable: {exc}"}

    # Component topographies are a convenience preview, not essential output. Some montages
    # (e.g. EOG channels with overlapping/undefined positions) make plot_components raise, so
    # a plotting failure must never abort a successful fit.
    image, plot_note = None, None
    try:
        fig = ica.plot_components(show=False)
        if isinstance(fig, list):
            fig = fig[0]
        image = _fig_to_b64(fig)
    except Exception as exc:
        plot_note = (f"Component topography plot unavailable: {exc} "
                     "If some fitted channels are EOG/auxiliary, type them as EOG "
                     "(set_channel_types) so topographies/ICLabel can run.")

    out = {
        "ok": True,
        "n_components": n_components,
        "method": _method_label,
        "engine": engine,
        "n_fit_channels": len(fit_picks),
        "fit_channels": [raw_proc.ch_names[i] for i in fit_picks],
        "random_state": random_state,
        "iclabel_preprocessing": bool(iclabel_preprocessing),
        "labels": labels,
        "image": image,
    }
    if _schedule:                       # only when the expert set one, so proofs stay traceable
        out["schedule"] = _schedule
    if plot_note:
        out["note"] = plot_note
    return out


def load_ica_eeglab(filepath: str) -> dict:
    """Load a precomputed ICA decomposition from an EEGLAB .set file into the session.

    Use this to apply an *externally computed* ICA (e.g. ERP CORE's published ICA weights)
    for exact replication, instead of recomputing (ICA is non-deterministic, so a re-fit
    cannot reproduce someone else's components). Pairs with apply_ica.
    """
    ica = mne.preprocessing.read_ica_eeglab(filepath)
    SESSION["ica"] = ica
    return {
        "ok": True,
        "source": filepath,
        "n_components": int(ica.n_components_),
        "n_ica_channels": len(ica.ch_names),
        "ch_names": ica.ch_names,
        "note": "ICA decomposition loaded; use apply_ica(exclude=[...]) to remove components.",
    }


def apply_ica(components: list | None = None, exclude: list | None = None,
              label: str | None = None) -> dict:
    """Remove ICA components from the loaded recording (ocular/artifact correction).

    Provide the components to drop in exactly one of these ways:
      - `components` = 1-based ICA component numbers, as EEGLAB reports them and as the
        request states them (converted to 0-based internally); e.g. [3, 4].
      - `exclude` = the 0-based programmatic form.
      - `label` = an artifact CLASS (e.g. 'eye'/'ocular') when the numeric indices are not
        known yet (e.g. right after a fresh run_ica). The class is resolved to concrete
        components at run time from the data itself (ICLabel, falling back to EOG-channel
        correlation for ocular labels), never from a baked answer key.

    Applies the ICA in place to the active raw/epochs (back-projects the remaining
    components). Requires an ICA in the session (run_ica or load_ica_eeglab). The ICA acts
    only on the channels it was fit on (matched by name); other channels untouched.
    """
    ica = SESSION.get("ica")
    if ica is None:
        return {"ok": False,
                "error": "No ICA in session. Run run_ica or load_ica_eeglab first."}
    kind, obj = _active()
    obj.load_data()
    resolve_info = None
    if components is not None:
        ica.exclude = [int(c) - 1 for c in components]   # 1-based -> 0-based
    elif exclude is not None:
        ica.exclude = [int(i) for i in exclude]
    elif label is not None:
        try:
            idx, method, detail = _resolve_ica_by_label(ica, obj, label)
        except Exception as exc:
            return {"ok": False, "error": f"Could not resolve label '{label}': {exc}"}
        if not idx:
            return {"ok": False, "error": f"No components matched label '{label}'."}
        ica.exclude = idx
        resolve_info = {"label": label, "method": method, "detail": detail,
                        "resolved_components_1based": [i + 1 for i in idx]}
    else:
        return {"ok": False,
                "error": "apply_ica needs `components` (1-based numbers) or `label` "
                         "(e.g. 'eye'/'ocular'); neither was supplied."}
    ica.apply(obj, verbose="ERROR")
    SESSION[kind] = obj
    out = {
        "ok": True,
        "applied_to": kind,
        "excluded": list(ica.exclude),
        "n_excluded": len(ica.exclude),
        "note": f"Removed {len(ica.exclude)} ICA component(s) from the {kind} in place.",
    }
    if resolve_info is not None:
        out["resolved_by_label"] = resolve_info
    return out


def _patch_ica_positionless(ica) -> list:
    """Give NaN/zero-position channels in ica.info a unique off-scalp dummy coord.

    Non-scalp channels typed 'eeg' (e.g. HEOG/VEOG imported as EEG) have NaN or all-zero loc,
    which mne's topomap reads as OVERLAPPING positions and refuses to plot -- breaking
    plot_properties / plot_components entirely. Scan ica.info directly (it is an independent copy
    and keeps original channel kinds even after set_channel_types on raw); assign each offender a
    unique x well off the scalp so the renderer skips it cleanly. Returns the patched names.
    """
    patched = []
    for i, ch in enumerate(ica.info["chs"]):
        loc = ch["loc"][:3]
        if np.any(np.isnan(loc)) or np.all(loc == 0):
            ch["loc"][:3] = np.array([3.0 + i * 0.01, 0.0, 0.0])
            patched.append(ch["ch_name"])
    return patched


def inspect_ica_component(components) -> dict:
    """Show detailed properties of ICA component(s) so the EXPERT can decide which to remove.

    Renders each component's topography, epochs image, ERP/time course, and power spectrum
    (mne `plot_properties`) -- the detail needed to judge whether a component is ocular/muscle/
    line-noise vs. brain. This is a REVIEW aid: it removes NOTHING. The expert inspects, then
    calls apply_ica(components=[...]) with the ones they judge to be artifacts.

    components : 1-based ICA component number(s) -- the SAME numbering as run_ica's labels and
                 apply_ica (e.g. 1, or [1, 7]). Requires an ICA in the session (run_ica first).
    """
    ica = SESSION.get("ica")
    if ica is None:
        return {"ok": False, "error": "No ICA in session. Run run_ica first."}
    if components is None:
        return {"ok": False, "error": "inspect_ica_component needs `components` (1-based number(s))."}
    if isinstance(components, (int, float, str)):
        components = [components]
    try:
        picks0 = [int(c) - 1 for c in components]
    except Exception:
        return {"ok": False, "error": f"`components` must be 1-based integers; got {components!r}."}
    n = int(ica.n_components_)
    oor = [p + 1 for p in picks0 if p < 0 or p >= n]
    if oor:
        return {"ok": False, "error": f"Component(s) {oor} out of range 1..{n}."}
    picks0 = picks0[:6]  # cap the rendered set so the image stays readable
    _kind, obj = _active()
    obj.load_data()
    # Non-scalp channels typed 'eeg' (e.g. HEOG/VEOG imported as EEG) have NaN/zero, hence
    # OVERLAPPING, positions -> mne's topomap raises "overlapping positions" and the whole render
    # fails, so a bare run_ica -> inspect returns blank. Give each such channel a unique off-scalp
    # dummy position (same patch review_ica uses) so the renderer skips them cleanly.
    _patch_ica_positionless(ica)
    try:
        figs = ica.plot_properties(obj, picks=picks0, show=False, verbose="ERROR")
    except Exception as exc:
        return {"ok": False, "error": (
            f"plot_properties failed: {exc}. If fitted channels lack scalp positions, type "
            "EOG/auxiliary channels via set_channel_types so topographies can render.")}
    if not isinstance(figs, list):
        figs = [figs]
    # Stack the per-component property figures into one image; fall back to the first on any error.
    try:
        import numpy as _np
        arrs = []
        for f in figs:
            f.canvas.draw()
            arrs.append(_np.asarray(f.canvas.buffer_rgba()))
            plt.close(f)
        w = max(a.shape[1] for a in arrs)
        stacked = _np.vstack([_np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)),
                                      constant_values=255) for a in arrs])
        fig2, ax = plt.subplots(figsize=(stacked.shape[1] / 100, stacked.shape[0] / 100), dpi=100)
        ax.imshow(stacked); ax.axis("off")
        image = _fig_to_b64(fig2); plt.close(fig2)
    except Exception:
        image = _fig_to_b64(figs[0])
        for f in figs:
            plt.close(f)
    return {
        "ok": True,
        "components": [int(c) for c in components][:6],
        "n_components": n,
        "image": image,
        "note": ("Review only -- nothing was removed. Decide which components are artifacts, "
                 "then apply_ica(components=[...]) with those 1-based numbers."),
    }


def _ica_eog_reference_traces(raw):
    """Return (veog, heog, label) 1-D EOG traces for ICA-correlation.

    Prefer bipolar '(uncorr) VEOG' / '(uncorr) HEOG' (single traces); else fall back to
    VEOG_lower and (HEOG_left - HEOG_right). Ported from scripts/tune_ica.py.
    """
    names = raw.ch_names

    def _get(name):
        return raw.get_data(picks=[name])[0]

    if "(uncorr) VEOG" in names and "(uncorr) HEOG" in names:
        return _get("(uncorr) VEOG"), _get("(uncorr) HEOG"), "bipolar (uncorr) VEOG/HEOG"

    veog = _get("VEOG_lower") if "VEOG_lower" in names else None
    heog = None
    if "HEOG_left" in names and "HEOG_right" in names:
        heog = _get("HEOG_left") - _get("HEOG_right")
    return veog, heog, "VEOG_lower / (HEOG_left - HEOG_right)"


def _abs_pearson(a, b):
    """Absolute Pearson r between two 1-D arrays. Returns NaN if either is constant."""
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return abs(float(np.corrcoef(a, b)[0, 1]))


def _ica_eog_correlation_table(raw, ica):
    """|Pearson r| of each ICA component vs the vertical & horizontal EOG.

    Returns (rows, label) where each row is {'component' (1-based), 'r_veog', 'r_heog',
    'r_max'}, sorted by r_max descending. Ported from scripts/tune_ica.py.
    """
    veog, heog, label = _ica_eog_reference_traces(raw)
    sources = ica.get_sources(raw).get_data()  # (n_components, n_times)
    rows = []
    for i in range(sources.shape[0]):
        s = sources[i]
        r_v = _abs_pearson(s, veog)
        r_h = _abs_pearson(s, heog)
        r_max = np.nanmax([r_v if r_v == r_v else -1, r_h if r_h == r_h else -1])
        rows.append({
            "component": i + 1,
            "r_veog": round(float(r_v), 4) if r_v == r_v else None,
            "r_heog": round(float(r_h), 4) if r_h == r_h else None,
            "r_max": round(float(r_max), 4) if r_max >= 0 else None,
        })
    rows.sort(key=lambda d: (d["r_max"] or -1), reverse=True)
    return rows, label


def review_ica() -> dict:
    """Scan view for ICA triage: topography grid + EOG-correlation table + ICLabel labels.

    Shows the full component line-up via ica.plot_components (topography grid) and returns
    an EOG-correlation table (|Pearson r| of each component's time course vs VEOG/HEOG,
    1-based, sorted by r_max descending) plus ICLabel labels where available. This is the
    'frontal topography + high EOG correlation = eye artifact' triage the expert uses before
    calling inspect_ica_component on candidates and then apply_ica(components=[...]).

    Ported from scripts/tune_ica.py (eog_correlation_table / _eog_reference_traces /
    _abs_pearson). Requires an ICA in the session (run_ica first). All component numbers are
    1-based (matching run_ica / apply_ica).
    """
    ica = SESSION.get("ica")
    if ica is None:
        return {"ok": False, "error": "No ICA in session. Run run_ica first."}

    _kind, raw = _active()
    raw.load_data()
    n = int(ica.n_components_)

    # --- Topography grid ---
    # Channels that are typed 'eeg' but lack scalp positions (NaN/zero loc, e.g. HEOG/VEOG
    # imported as EEG) cause 'overlapping positions' errors in plot_components. Patch the ICA's
    # own info with off-scalp dummy positions (unique x coords) so the topomap renderer skips
    # them cleanly without error. The ICA's ch_names are a subset of raw.ch_names (fitted channels
    # only), so only those in ica.info need to be patched.
    _patched_chs = []
    # Scan ica.info directly (not raw) — ica.info is an independent copy and retains
    # original channel kinds even after set_channel_types is called on raw.  Any channel
    # in ica.info with a NaN or all-zero position causes "overlapping positions" in
    # plot_components; give it a unique off-scalp dummy x coord to suppress the error.
    for _idx, _ch in enumerate(ica.info["chs"]):
        _loc = _ch["loc"][:3]
        if np.any(np.isnan(_loc)) or np.all(_loc == 0):
            _ch["loc"][:3] = np.array([3.0 + _idx * 0.01, 0.0, 0.0])
            _patched_chs.append(_ch["ch_name"])

    try:
        figs = ica.plot_components(show=False)
        if not isinstance(figs, list):
            figs = [figs]
        # Tile multiple figures (mne may return >1 for large n_components)
        arrs = []
        for f in figs:
            f.canvas.draw()
            arrs.append(np.asarray(f.canvas.buffer_rgba()))
            plt.close(f)
        if len(arrs) > 1:
            w = max(a.shape[1] for a in arrs)
            tiled = np.vstack([np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)),
                                       constant_values=255) for a in arrs])
            fig2, ax = plt.subplots(figsize=(tiled.shape[1] / 100, tiled.shape[0] / 100), dpi=100)
            ax.imshow(tiled)
            ax.axis("off")
            topo_b64 = _fig_to_b64(fig2)
            plt.close(fig2)
        else:
            fig3, ax3 = plt.subplots(figsize=(arrs[0].shape[1] / 100, arrs[0].shape[0] / 100), dpi=100)
            ax3.imshow(arrs[0])
            ax3.axis("off")
            topo_b64 = _fig_to_b64(fig3)
            plt.close(fig3)
    except Exception as exc:
        return {"ok": False, "error": (
            f"plot_components failed: {exc}. If channels lack scalp positions, type EOG/aux "
            "channels via set_channel_types so topographies can render.")}

    # --- EOG-correlation table ---
    try:
        eog_rows, eog_label = _ica_eog_correlation_table(raw, ica)
        corr_ok = True
    except Exception as exc:
        eog_rows, eog_label = [], str(exc)
        corr_ok = False

    # --- ICLabel labels (best-effort; silently absent when ICLabel unavailable) ---
    iclabel_labels: dict = {}
    iclabel_error = None
    try:
        from mne_icalabel import label_components as _iclabel
        raw_proc = raw.copy().filter(1.0, 100.0, picks="eeg", verbose="ERROR")
        raw_proc.set_eeg_reference("average", verbose="ERROR")
        res = _iclabel(raw_proc, ica, method="iclabel")
        for i, lab in enumerate(res["labels"]):
            iclabel_labels[i + 1] = str(lab)  # 1-based
    except Exception as exc:
        iclabel_error = str(exc)

    # Annotate EOG rows with ICLabel label
    for row in eog_rows:
        row["iclabel"] = iclabel_labels.get(row["component"])

    result = {
        "ok": True,
        "n_components": n,
        "eog_trace_label": eog_label,
        "eog_correlation_table": eog_rows,
        "iclabel_available": bool(iclabel_labels),
        "image": topo_b64,
        "note": (
            "REVIEW ONLY -- nothing removed. Read the topography grid + EOG table: "
            "frontal topography + high r_max = eye artifact candidate. "
            "Inspect candidates with inspect_ica_component, then apply_ica(components=[...])."
        ),
    }
    if not corr_ok:
        result["eog_correlation_error"] = eog_label
    if iclabel_error:
        result["iclabel_error"] = iclabel_error
    return result


_OCULAR_LABELS = {"eye", "eyes", "ocular", "blink", "blinks", "eog",
                  "eye movement", "eye movements"}


def _resolve_ica_by_label(ica, raw, label: str):
    """Resolve an artifact-class label to 0-based ICA component indices.

    Ocular labels ('eye'/'ocular'/...) map to ICLabel's 'eye' class. Tries ICLabel first
    (needs scalp positions, often absent on imported .set files); on failure, falls back to
    correlating each component's time course with the EOG channels (ocular labels only).
    The label names a general class the caller chose; the per-component identification comes
    from the data (classifier / EOG), not from a baked answer key. Returns
    (indices_0based, method, detail); raises ValueError if the label cannot be resolved.
    """
    import re as _re
    norm = _re.sub(r"[\s\-_]+", " ", str(label).strip().lower())
    is_ocular = norm in _OCULAR_LABELS
    target = "eye" if is_ocular else norm

    iclabel_err = None
    try:
        from mne_icalabel import label_components as _iclabel
        raw_proc = raw.copy().filter(1.0, 100.0, picks="eeg", verbose="ERROR")
        raw_proc.set_eeg_reference("average", verbose="ERROR")
        res = _iclabel(raw_proc, ica, method="iclabel")
        labels = list(res["labels"])
        idx = [i for i, lab in enumerate(labels) if str(lab).lower().startswith(target)]
        return idx, "iclabel", f"ICLabel matched {len(idx)} '{target}' component(s)."
    except Exception as exc:
        iclabel_err = str(exc)

    if not is_ocular:
        raise ValueError(
            f"ICLabel unavailable ({iclabel_err}); the EOG-correlation fallback only "
            "supports ocular labels (eye/ocular).")

    import numpy as _np
    sources = ica.get_sources(raw).get_data()   # (n_components, n_times)
    chs = raw.ch_names

    def _trace(preferred, diff_pair):
        for name in preferred:
            if name in chs:
                return raw.get_data(picks=name)[0]
        a, b = diff_pair
        if a in chs and b in chs:
            return raw.get_data(picks=a)[0] - raw.get_data(picks=b)[0]
        return raw.get_data(picks=a)[0] if a in chs else None

    veog = _trace(["(uncorr) VEOG", "VEOG_lower", "VEOG"], ("VEOG_lower", "VEOG_lower"))
    heog = _trace(["(uncorr) HEOG", "HEOG"], ("HEOG_left", "HEOG_right"))
    refs = [t for t in (veog, heog) if t is not None]
    if not refs:
        raise ValueError("no EOG channels found for the correlation fallback.")

    idx = [i for i in range(sources.shape[0])
           if max(abs(_np.corrcoef(sources[i], t)[0, 1]) for t in refs) >= 0.5]
    return idx, "eog_correlation", (
        f"ICLabel unavailable; selected {len(idx)} component(s) with |EOG r| >= 0.5.")


def _resolve_channel(name: str, ch_names: list) -> str | None:
    """Map a requested channel name to an actual one, tolerant of separators/case.

    Exact match wins; otherwise compare on a normalized key (lowercased, non-alphanumerics
    stripped) so 'HEOGleft' resolves to 'HEOG_left', 'veog lower' to 'VEOG_lower', etc.
    Returns the actual channel name, or None if nothing matches.
    """
    if name in ch_names:
        return name
    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    key = norm(name)
    matches = [c for c in ch_names if norm(c) == key]
    return matches[0] if len(matches) == 1 else None


def _default_eog_pairs(ch_names: list) -> tuple[list | None, list | None]:
    """Best-guess [anode, cathode] pairs for HEOG/VEOG from the channel names present.

    Follows the ERP CORE convention (HEOG_left - HEOG_right; VEOG_lower - VEOG_upper, or
    VEOG_lower - FP2 when no upper VEOG lead was recorded). Returns (heog, veog); either may
    be None if no pair is found, and the caller reports which pair was auto-selected.
    """
    present = set(ch_names)
    heog = veog = None
    for a, c in [("HEOG_left", "HEOG_right"), ("HEOGL", "HEOGR"), ("LHEOG", "RHEOG")]:
        if a in present and c in present:
            heog = [a, c]; break
    for a, c in [("VEOG_lower", "VEOG_upper"), ("VEOGL", "VEOGU"), ("VEOG_lower", "FP2")]:
        if a in present and c in present:
            veog = [a, c]; break
    return heog, veog


def derive_bipolar_eog(heog: list | None = None, veog: list | None = None,
                       heog_name: str = "HEOG", veog_name: str = "VEOG") -> dict:
    """Create bipolar EOG channels from monopolar leads.

    `heog`/`veog` are each [anode, cathode] channel names; the bipolar is anode - cathode.
    `heog_name`/`veog_name` name the channels created. Supply them when a downstream step
    refers to the bipolars by a specific label -- ERP CORE calls the post-ICA pair
    '(corr) HEOG' / '(corr) VEOG', and its artifact-detection parameters address them by name.
    When omitted, standard ERP CORE leads are auto-detected from the channels present
    (HEOG_left - HEOG_right; VEOG_lower - VEOG_upper, or VEOG_lower - FP2 if no upper lead);
    the return echoes the selected pairs and whether each was auto-detected or user-supplied.
    Original channels are kept. Bipolar EOG is diagnostic (used for artifact screening); it
    does not enter the average-referenced scalp ERP.
    """
    raw = _require_raw()
    raw.load_data()
    auto_heog, auto_veog = _default_eog_pairs(raw.ch_names)
    heog_source = "user" if heog else "auto"
    veog_source = "user" if veog else "auto"

    def _normalize_pair(user_val, auto_val, label):
        """Ensure a pair is a 2-element [anode, cathode] list.

        A bare string or 1-element list is treated as the anode; the cathode is taken from
        the auto-detect result (same fallback the omitted-arg path uses). A 2-element list
        passes through unchanged. If the single-anode path cannot find a cathode from
        auto-detect, returns None so the caller reports the error.
        """
        if user_val is None:
            return auto_val  # fully auto-detected (may be None if no standard pair found)
        # Normalise: bare string -> 1-element list
        if isinstance(user_val, str):
            user_val = [user_val]
        if len(user_val) == 2:
            return user_val  # already a complete pair -- keep exactly as supplied
        if len(user_val) == 1:
            anode = user_val[0]
            # Reuse the auto-detect cathode for this axis (same logic as omitted-arg path)
            if auto_val and len(auto_val) == 2 and auto_val[0] == anode:
                return auto_val  # auto-detect agrees on the anode -- take its cathode
            # Auto-detect chose a different anode (or found nothing); try to find the
            # cathode from the same auto-detect table by scanning for the anode explicitly
            if label == "veog":
                for a, c in [("VEOG_lower", "VEOG_upper"), ("VEOGL", "VEOGU"),
                              ("VEOG_lower", "FP2")]:
                    if a == anode and c in raw.ch_names:
                        return [a, c]
            elif label == "heog":
                for a, c in [("HEOG_left", "HEOG_right"), ("HEOGL", "HEOGR"),
                              ("LHEOG", "RHEOG")]:
                    if a == anode and c in raw.ch_names:
                        return [a, c]
            return None  # cannot complete the pair
        return None  # empty list -- caller will report

    heog = _normalize_pair(heog, auto_heog, "heog")
    veog = _normalize_pair(veog, auto_veog, "veog")
    if not heog or not veog:
        missing = [n for n, v in (("heog", heog), ("veog", veog)) if not v]
        return {"ok": False,
                "error": f"Could not resolve {', '.join(missing)}: no standard EOG leads found "
                         f"in {raw.ch_names}. Pass {'/'.join(missing)} as [anode, cathode] "
                         "channel names explicitly."}
    created, skipped, remapped = [], [], []
    for name, (anode_req, cathode_req) in [(heog_name, heog), (veog_name, veog)]:
        if name in raw.ch_names:
            skipped.append({"name": name, "reason": "already present"})
            continue
        anode = _resolve_channel(anode_req, raw.ch_names)
        cathode = _resolve_channel(cathode_req, raw.ch_names)
        for req, got in ((anode_req, anode), (cathode_req, cathode)):
            if got is not None and got != req:
                remapped.append({"requested": req, "resolved": got})
        if anode and cathode:
            mne.set_bipolar_reference(raw, anode=anode, cathode=cathode, ch_name=name,
                                      drop_refs=False, copy=False, verbose="ERROR")
            created.append({"name": name, "anode": anode, "cathode": cathode})
        else:
            missing = [req for req, got in ((anode_req, anode), (cathode_req, cathode))
                       if got is None]
            skipped.append({"name": name,
                            "reason": f"could not resolve {missing} in {raw.ch_names}"})
    SESSION["raw"] = raw
    return {
        "ok": True,
        "created": created,
        "skipped": skipped,
        "remapped": remapped,
        "heog": heog,
        "veog": veog,
        "heog_source": heog_source,
        "veog_source": veog_source,
        "n_channels": len(raw.ch_names),
        "note": "Bipolar EOG added (diagnostic; not part of the scalp ERP)."
                + ("" if heog_source == "user" and veog_source == "user"
                   else f" Leads auto-detected (heog={heog_source}, veog={veog_source}).")
                + ("" if not remapped
                   else " Name-matched: "
                        + ", ".join(f"{r['requested']}->{r['resolved']}" for r in remapped) + "."),
    }


def _label_int(label) -> int | None:
    """Parse an event label to an int (e.g. '17' or 'S 17' -> 17), else None."""
    s = str(label).strip()
    try:
        return int(s)
    except ValueError:
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None


def shift_events(ms: float, descriptions: list | None = None,
                 rounding: str = "none", code_ranges: list | None = None) -> dict:
    """Shift event-marker timing in place to correct for a fixed latency (e.g. LCD delay).

    Adds `ms` milliseconds to each annotation onset (positive = later in time). The EEG
    samples are not touched -- only the event times.

    Scope (union; default = all markers):
      descriptions  -> shift only markers whose exact label is in this list.
      code_ranges   -> list of [min, max] inclusive integer ranges; shift markers whose
                       label parses to an int inside any range, e.g. [[1,80],[100,180]].

    rounding: how to quantize the shift.
      'none'    -> exact continuous ms shift (default).
      'nearest' / 'earlier' / 'later' -> quantize the shift to an integer number of
                       samples at the current sampling rate (round / floor-toward-earlier /
                       ceil-toward-later). E.g. 26 ms at 1024 Hz with 'earlier' -> 26 samples.
    """
    raw = _require_raw()
    onsets = raw.annotations.onset
    descr = raw.annotations.description
    sfreq = float(raw.info["sfreq"])

    rounding = (rounding or "none").lower()
    if rounding not in ("none", "nearest", "earlier", "later"):
        return {"ok": False,
                "error": f"rounding must be none|nearest|earlier|later, got '{rounding}'."}
    if rounding == "none":
        applied_samples = None
        shift_s = float(ms) / 1000.0
    else:
        raw_samp = float(ms) / 1000.0 * sfreq
        applied_samples = int({"nearest": np.round, "earlier": np.floor,
                               "later": np.ceil}[rounding](raw_samp))
        shift_s = applied_samples / sfreq

    if descriptions or code_ranges:
        wanted = {str(d) for d in (descriptions or [])}
        ranges = [(int(lo), int(hi)) for lo, hi in (code_ranges or [])]

        def _in_scope(label):
            if str(label) in wanted:
                return True
            code = _label_int(label)
            return code is not None and any(lo <= code <= hi for lo, hi in ranges)

        mask = np.array([_in_scope(d) for d in descr])
    else:
        mask = np.ones(len(onsets), dtype=bool)

    new_onsets = onsets.copy()
    new_onsets[mask] = new_onsets[mask] + shift_s
    raw.set_annotations(
        mne.Annotations(onset=new_onsets, duration=raw.annotations.duration,
                        description=descr, orig_time=raw.annotations.orig_time),
        verbose="ERROR",
    )
    SESSION["raw"] = raw
    return {
        "ok": True,
        "requested_shift_ms": float(ms),
        "applied_shift_ms": round(shift_s * 1000.0, 6),
        "applied_shift_samples": applied_samples,
        "rounding": rounding,
        "sfreq": sfreq,
        "n_shifted": int(mask.sum()),
        "n_events": int(len(onsets)),
        "note": f"Shifted {int(mask.sum())} of {len(onsets)} event markers by "
                f"{round(shift_s * 1000.0, 4)} ms (event times only; EEG samples unchanged).",
    }


def compare_to_checkpoint(checkpoint: str, channels: list | None = None,
                          bipolar: bool = False, gate: str = "tight",
                          plot_channel: str = "PO8") -> dict:
    """EXPERIMENTER-ONLY verification helper. NOT exposed to the LLM (deliberately not in
    TOOL_SCHEMAS/TOOL_FUNCTIONS): the model must never see ERP CORE specifics. Called
    out-of-band by the dashboard's `/compare` command to score the current data.

    Compare the currently loaded recording to an ERP CORE N170 ground-truth checkpoint.

    Scores the in-session data against ERP CORE's released .set checkpoint for a
    preprocessing stage and returns a per-channel agreement table + PASS/REVIEW verdict,
    plus an overlay plot (eeg-llm vs ERP CORE + their difference) for one channel.

    checkpoint : a step keyword (shifted_ds | reref_ucbip | hpfilt) or a path to a .set file.
    bipolar    : True to compare the bipolar HEOG/VEOG channels (renames ERP CORE's
                 '(uncorr) HEOG'/'(uncorr) VEOG' to match).
    gate       : 'tight' (exact-math steps: PASS if min r>=0.999 and max|delta|<1e-4 V) or
                 'mean' (high-pass step: PASS if mean r>=0.98, since MNE's vs EEGLAB's
                 Butterworth differ per channel by design).
    plot_channel : channel to draw in the overlay figure (default PO8).
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from erpcore_compare import compare, compare_evoked, compare_erp, aligned, UV

    path = ERPCORE_CHECKPOINTS.get(checkpoint, checkpoint)
    if not os.path.exists(path):
        return {"ok": False,
                "error": f"Checkpoint not found: '{checkpoint}'. Use a keyword "
                         f"({', '.join(sorted(ERPCORE_CHECKPOINTS))}) or a valid .set path."}

    ref_rename = {"(uncorr) HEOG": "HEOG", "(uncorr) VEOG": "VEOG"} if bipolar else None
    if bipolar and channels is None:
        channels = ["HEOG", "VEOG"]

    # Kind dispatch: epoched checkpoints are graded as bin-averaged ERPs against the current
    # SESSION epochs; continuous checkpoints against the current Raw. See EPOCHED_CHECKPOINTS.
    is_epoched = checkpoint in EPOCHED_CHECKPOINTS
    is_erp = checkpoint in ERP_CHECKPOINTS
    raw = None
    if is_erp:
        # Script 7: grade the averaged waveform itself. Prefer an already-computed difference
        # wave (compute_difference_erp stores it) over re-averaging the epochs, so what gets
        # graded is what the pipeline actually produced.
        got = SESSION.get("evoked")
        if got is None:
            got = SESSION.get("epochs")
        if got is None:
            return {"ok": False,
                    "error": f"Checkpoint '{checkpoint}' is an averaged ERP, but neither an "
                             f"evoked nor epochs are in SESSION. Run compute_erp / "
                             f"compute_difference_erp first."}
        res = compare_erp(got, path, ERP_CHECKPOINTS[checkpoint], channels, ref_rename)
    elif is_epoched:
        epochs = SESSION.get("epochs")
        if epochs is None:
            return {"ok": False,
                    "error": f"Checkpoint '{checkpoint}' holds epoched data, but no epochs are "
                             f"loaded (SESSION['epochs'] is empty). Create/load epochs first."}
        res = compare_evoked(epochs, path, channels, ref_rename)
    else:
        raw = _require_raw()
        res = compare(raw, path, channels, ref_rename)
    min_r, mean_r, worst = res["min_r"], res["mean_r"], res["worst_dmax_V"]

    if gate == "mean":
        verdict = "PASS" if mean_r >= 0.98 else "REVIEW"
        gate_txt = "mean r>=0.98"
    else:
        verdict = "PASS" if (min_r >= 0.999 and worst < 1e-4) else "REVIEW"
        gate_txt = "min r>=0.999 and max|delta|<100 uV"

    # Compact rows (channel, r, max|delta|) so the result stays under the app's 4000-char
    # tool-result cap; magnitude agreement is visible in the overlay plot.
    rows = [
        {"channel": ch, "pearson_r": round(r, 6), "max_abs_diff_uV": round(d * UV, 4)}
        for ch, g, f, d, r in res["rows"]
    ]

    # Overlay figure for one channel: eeg-llm, ERP CORE, and their difference. Raw-only
    # (aligned() reads continuous data); epoched checkpoints rely on the numeric verdict.
    image = None
    try:
        if is_epoched:
            raise RuntimeError("skip overlay for epoched checkpoint")
        common, g, f = aligned(raw, path, ref_rename=ref_rename)
        pick = plot_channel if plot_channel in common else (common[0] if common else None)
        if pick is not None:
            i = common.index(pick)
            sf = raw.info["sfreq"]
            n = min(g.shape[1], int(sf * 10))  # first 10 s is plenty to eyeball
            t = np.arange(n) / sf
            fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
            ax[0].plot(t, g[i, :n] * UV, lw=0.5, color="C0")
            ax[0].set_title(f"eeg-llm — {pick}"); ax[0].set_ylabel("uV")
            ax[1].plot(t, f[i, :n] * UV, lw=0.5, color="C1")
            ax[1].set_title(f"ERP CORE — {pick}"); ax[1].set_ylabel("uV")
            ax[2].plot(t, (g[i, :n] - f[i, :n]) * UV, lw=0.5, color="C3")
            ax[2].set_title("difference (eeg-llm − ERP CORE)")
            ax[2].set_ylabel("uV"); ax[2].set_xlabel("time (s)")
            fig.suptitle(f"{res['checkpoint']}  |  verdict: {verdict}")
            fig.tight_layout()
            image = _fig_to_b64(fig)
    except Exception:
        pass  # numeric result stands on its own if plotting fails

    out = {
        "ok": True,
        "checkpoint": res["checkpoint"],
        "gate": gate_txt,
        "verdict": verdict,
        "n_channels": res["n_channels"],
        "n_samples": res["n_samples"],
        "min_r": round(min_r, 6),
        "mean_r": round(mean_r, 6),
        "worst_max_abs_diff_uV": round(worst * UV, 4),
        "rows": rows,
        "note": f"Compared the loaded recording to ERP CORE's {res['checkpoint']}: "
                f"{verdict} ({gate_txt}).",
    }
    if image is not None:
        out["image"] = image
    return out


def _erp_plot(evoked, plot_style: str, channels: list | None = None,
              tmin: float | None = None, tmax: float | None = None) -> "plt.Figure":
    """Render an ERP figure in one of two styles.

    plot_style='butterfly'  -- all-channel overlay (MNE default, spatial_colors + GFP).
    plot_style='channels'   -- only the named channel(s) as individual traces, with the
                               analysis window (tmin..tmax) shaded via axvspan. Requires
                               at least one channel name; falls back to butterfly if none.
    """
    style = (plot_style or "butterfly").lower()
    if style == "channels" and channels:
        # Filter to channels present in the evoked
        picks = [c for c in channels if c in evoked.ch_names]
        if not picks:
            style = "butterfly"
    if style != "channels":
        return evoked.plot(spatial_colors=True, show=False, gfp=True)

    # Channel-focused view: one line per channel, window shaded.
    times_ms = evoked.times * 1000
    fig, ax = plt.subplots(figsize=(8, 4))
    for ch in picks:
        idx = evoked.ch_names.index(ch)
        ax.plot(times_ms, evoked.data[idx] * 1e6, label=ch)
    # Shade the analysis window if provided
    if tmin is not None and tmax is not None:
        ax.axvspan(tmin * 1000, tmax * 1000, alpha=0.18, color="steelblue",
                   label=f"window {round(tmin*1000)}–{round(tmax*1000)} ms")
    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("ERP — " + ", ".join(picks))
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def compute_erp(tmin: float | None = None, tmax: float | None = None,
                event_id: str | None = None,
                plot_style: str = "butterfly") -> dict:
    """Average epochs to an ERP, with a plot.

    If epoched data is already loaded (e.g. EEGLAB epoched .set/.mat), average it
    directly. Otherwise epoch around annotated events in the continuous recording.

    plot_style='butterfly' (default) -- all-channel overlay (spatial_colors + GFP).
    plot_style='channels'            -- named channel traces only, window shaded; for
                                       compute_erp the GFP-peak channel is used
                                       (no channel arg here; use compute_difference_erp
                                       or measure_component for a specific channel view).
    """
    if SESSION.get("raw") is not None and (tmin is None or tmax is None):
        missing = [n for n, v in (("tmin", tmin), ("tmax", tmax)) if v is None]
        return {"ok": False, "error": (
            f"compute_erp requires {missing} (epoch window in seconds relative to event); "
            "they must be read from the request, not defaulted."
        )}
    use_id = None
    if SESSION.get("raw") is None and SESSION.get("epochs") is not None:
        # Already-epoched data (e.g. ANTS sampleEEGdata): average as-is.
        epochs = SESSION["epochs"]
    else:
        raw = _require_raw()
        events, event_dict = mne.events_from_annotations(raw)
        if len(events) == 0:
            return {"ok": False,
                    "error": "No events/annotations found in the recording; "
                             "cannot compute an ERP. Provide stim or annotated data."}
        use_id = event_dict
        if event_id is not None and event_id in event_dict:
            use_id = {event_id: event_dict[event_id]}
        epochs = mne.Epochs(raw, events, event_id=use_id, tmin=tmin, tmax=tmax,
                            baseline=(None, 0), preload=True, verbose="ERROR")
        SESSION["epochs"] = epochs

    evoked = epochs.average()
    SESSION["evoked"] = evoked  # for measure_component

    # Find the global field power peak as a candidate component latency
    gfp = evoked.data.std(axis=0)
    peak_idx = int(np.argmax(gfp))
    peak_latency_ms = round(float(evoked.times[peak_idx]) * 1000, 1)

    # For 'channels' style in compute_erp, default to the GFP-peak channel
    gfp_ch = evoked.ch_names[int(np.argmax(np.abs(evoked.data[:, peak_idx])))]
    fig = _erp_plot(evoked, plot_style, channels=[gfp_ch], tmin=tmin, tmax=tmax)

    return {
        "ok": True,
        "n_epochs": len(epochs),
        "tmin": tmin,
        "tmax": tmax,
        "event_ids": use_id,
        "peak_gfp_latency_ms": peak_latency_ms,
        "plot_style": (plot_style or "butterfly").lower(),
        "image": _fig_to_b64(fig),
    }


def _epochs_for_event(event_id, tmin: float, tmax: float):
    """Build baseline-corrected epochs for one condition: a single annotation label
    or a list of labels pooled into one condition.

    A list pools all matching events into a single average (e.g. N170 faces = codes
    '1'..'40'). Returns (epochs, n_matched) or raises RuntimeError with a helpful
    message. Shared by compute_difference_erp; mirrors compute_erp's epoching path.
    """
    labels = [event_id] if isinstance(event_id, str) else list(event_id)
    # Bin-aware: if create_bins/create_epochs already produced labelled epochs (e.g. 'B1'),
    # select the condition from those instead of re-epoching raw annotations. The epoch window
    # is the one baked into those epochs.
    se = SESSION.get("epochs")
    if se is not None and all(lab in se.event_id for lab in labels):
        sub = se[labels]
        return sub, len(sub)
    raw = _require_raw()
    events, event_dict = mne.events_from_annotations(raw)
    sel = {lab: event_dict[lab] for lab in labels if lab in event_dict}
    if not sel:
        raise RuntimeError(
            f"None of {labels} found. Available: {sorted(event_dict)[:40]}")
    epochs = mne.Epochs(raw, events, event_id=sel, tmin=tmin, tmax=tmax,
                        baseline=(None, 0), preload=True, verbose="ERROR")
    return epochs, len(epochs)


def compute_difference_erp(event_id_a, event_id_b,
                           tmin: float | None = None, tmax: float | None = None,
                           channels: list | None = None,
                           plot_style: str = "butterfly") -> dict:
    """Average two conditions separately and return the difference wave (A minus B).

    Needed for contrast-based ERP components (e.g. N170 = faces minus cars). Each
    condition is a single annotation label or a list of labels pooled together
    (e.g. faces = ['1',...,'40'], cars = ['41',...,'80']). Each condition is epoched
    and averaged on its own, then subtracted. The difference Evoked is stored for
    measure_component.

    plot_style='butterfly' (default) -- all-channel overlay (spatial_colors + GFP).
    plot_style='channels'            -- only the named channel(s) as traces with
                                       tmin-tmax window shaded. `channels` defaults to the
                                       GFP-peak channel when plot_style='channels' and none
                                       are supplied.
    """
    if tmin is None or tmax is None:
        missing = [n for n, v in (("tmin", tmin), ("tmax", tmax)) if v is None]
        return {"ok": False, "error": (
            f"compute_difference_erp requires {missing} (epoch window in seconds relative to event); "
            "they must be read from the request, not defaulted."
        )}
    try:
        ep_a, n_a = _epochs_for_event(event_id_a, tmin, tmax)
        ep_b, n_b = _epochs_for_event(event_id_b, tmin, tmax)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    if n_a == 0 or n_b == 0:
        return {"ok": False,
                "error": f"No epochs for one condition (a={n_a}, b={n_b})."}

    ev_a = ep_a.average()
    ev_b = ep_b.average()
    diff = mne.combine_evoked([ev_a, ev_b], weights=[1, -1])
    SESSION["evoked"] = diff  # for measure_component

    gfp = diff.data.std(axis=0)
    peak_idx = int(np.argmax(gfp))
    peak_latency_ms = round(float(diff.times[peak_idx]) * 1000, 1)

    # For 'channels' style: default to the GFP-peak channel if none specified
    plot_chs = channels
    if (plot_style or "butterfly").lower() == "channels" and not plot_chs:
        plot_chs = [diff.ch_names[int(np.argmax(np.abs(diff.data[:, peak_idx])))]]
    fig = _erp_plot(diff, plot_style, channels=plot_chs, tmin=tmin, tmax=tmax)

    return {
        "ok": True,
        "condition_a": event_id_a,
        "condition_b": event_id_b,
        "n_epochs_a": n_a,
        "n_epochs_b": n_b,
        "tmin": tmin,
        "tmax": tmax,
        "peak_gfp_latency_ms": peak_latency_ms,
        "plot_style": (plot_style or "butterfly").lower(),
        "image": _fig_to_b64(fig),
    }


def measure_component(tmin: float, tmax: float, channels: list,
                      mode: str = "mean", engine: str = "mne",
                      baseline: list | None = None,
                      plot_style: str = "channels") -> dict:
    """Score the most recent ERP / difference wave in a time window at given channels.

    mode='mean'  -> mean amplitude over the window (µV).
    mode='peak'  -> most extreme (signed) value in the window (µV) + its latency (ms).
    Operates on the Evoked stored by compute_erp / compute_difference_erp.

    `engine` selects the window convention -- the expert's call, never auto-detected:
      'mne'    -> samples with tmin <= t <= tmax.
      'erplab' -> the sample NEAREST each edge (ERPLAB `closest()`), which INCLUDES a sample
                  lying just outside the requested window. At 256 Hz a 110-150 ms request is
                  10 samples under 'mne' (113.28..148.44) but 11 under 'erplab'
                  (109.38..148.44) -- worth 0.22 µV on the ERP CORE N170.

    `baseline` = [start_ms, stop_ms] re-applies a baseline to the ERP before measuring, as
    ERPLAB's pop_geterpvalues does with 'Measure','meanbl'. Its edges use the same `engine`
    convention, so ERP CORE's [-200, 0] INCLUDES t=0 -- note this is the opposite of the
    epoching baseline, which excludes it (see create_epochs). Not a no-op even on already
    baselined epochs: it shifts the ERP CORE N170 by 0.005 µV.

    plot_style='channels' (default for measure_component) -- the named channel(s) as traces
                           with the tmin-tmax window shaded (the legible view when measuring
                           "the N170 at PO8"). Use 'butterfly' for the full all-channel overlay.
    """
    evoked = SESSION.get("evoked")
    if evoked is None:
        return {"ok": False,
                "error": "No ERP available. Call compute_erp or "
                         "compute_difference_erp first."}
    if not channels:
        return {"ok": False, "error": "Provide at least one channel."}

    missing = [c for c in channels if c not in evoked.ch_names]
    if missing:
        return {"ok": False,
                "error": f"Channels not found: {missing}. "
                         f"Available e.g. {evoked.ch_names[:20]}"}

    picks = [evoked.ch_names.index(c) for c in channels]
    engine = (engine or "mne").lower()
    if engine not in ("mne", "erplab"):
        return {"ok": False, "error": f"engine must be 'mne' or 'erplab', got {engine!r}."}

    def _window(lo, hi):
        """Sample indices for [lo, hi] seconds under the selected convention."""
        t = evoked.times
        if engine == "erplab":
            a, b = int(np.argmin(np.abs(t - lo))), int(np.argmin(np.abs(t - hi)))
            return np.arange(a, b + 1)
        return np.flatnonzero((t >= lo) & (t <= hi))

    idx = _window(tmin, tmax)
    if idx.size == 0:
        return {"ok": False,
                "error": f"Window [{tmin}, {tmax}] s outside epoch "
                         f"[{evoked.times[0]:.3f}, {evoked.times[-1]:.3f}] s."}

    data = evoked.data
    if baseline is not None:
        bidx = _window(baseline[0] / 1000.0, baseline[1] / 1000.0)
        if bidx.size == 0:
            return {"ok": False, "error": f"Baseline {baseline} ms outside the epoch."}
        data = data - data[:, bidx].mean(axis=1, keepdims=True)

    # Mean across the selected channels, restricted to the window. Volts -> µV.
    seg = data[np.ix_(picks, idx)].mean(axis=0)  # avg over channels
    times = evoked.times[idx]
    mode = (mode or "mean").lower()

    result = {
        "ok": True,
        "mode": mode,
        "channels": channels,
        "window_ms": [round(tmin * 1000, 1), round(tmax * 1000, 1)],
        "engine": engine,
        # The realised window: a convention mismatch shows up here rather than silently
        # shifting the amplitude.
        "window_samples": int(idx.size),
        "window_actual_ms": [round(float(times[0]) * 1000, 3), round(float(times[-1]) * 1000, 3)],
    }
    if baseline is not None:
        result["baseline_ms"] = list(baseline)
    if mode == "peak":
        # signed peak = most extreme deviation from 0 in the window
        pidx = int(np.argmax(np.abs(seg)))
        result["amplitude_uv"] = round(float(seg[pidx]) * 1e6, 4)
        result["latency_ms"] = round(float(times[pidx]) * 1000, 1)
    else:
        result["amplitude_uv"] = round(float(seg.mean()) * 1e6, 4)

    # Plot: the named channel(s) with the window shaded (default for measure_component).
    try:
        fig = _erp_plot(evoked, plot_style, channels=channels, tmin=tmin, tmax=tmax)
        result["image"] = _fig_to_b64(fig)
        result["plot_style"] = (plot_style or "channels").lower()
    except Exception:
        pass  # plot is a convenience; never block the amplitude result
    return result


def run_autoreject(n_interpolate: int = 4) -> dict:
    """Clean epochs with AutoReject. Builds 1-s fixed-length epochs if none exist."""
    raw = _require_raw()
    epochs = SESSION.get("epochs")
    if epochs is None:
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, baseline=None,
                            preload=True, verbose="ERROR")

    try:
        from autoreject import AutoReject
    except Exception as exc:
        return {"ok": False, "error": f"autoreject not installed: {exc}"}

    ar = AutoReject(n_interpolate=np.array([1, n_interpolate]),
                    random_state=11, n_jobs=1, verbose=False)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    SESSION["epochs"] = epochs_clean

    n_before = len(epochs)
    n_after = len(epochs_clean)
    return {
        "ok": True,
        "n_epochs_before": n_before,
        "n_epochs_after": n_after,
        "n_rejected": n_before - n_after,
        "rejection_rate": round((n_before - n_after) / max(n_before, 1), 3),
        "bad_epoch_fraction": round(float(reject_log.bad_epochs.mean()), 3),
    }


# --------------------------------------------------------------------------- #
# EEGLAB-equivalent tools (asrpy, pyprep, mne-connectivity, fooof, pycrostates)
# --------------------------------------------------------------------------- #
def _epochs_for_analysis(raw, duration: float = 2.0):
    """Return existing epochs or build fixed-length ones from the raw."""
    epochs = SESSION.get("epochs")
    if epochs is not None and len(epochs) > 0:
        return epochs
    events = mne.make_fixed_length_events(raw, duration=duration)
    return mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None,
                      preload=True, verbose="ERROR")


def run_asr(cutoff: float = 20.0) -> dict:
    """Artifact Subspace Reconstruction (EEGLAB clean_rawdata/ASR). Cleans the raw
    in place and reports how much signal variance was corrected."""
    raw = _require_raw()
    try:
        from asrpy import ASR
    except Exception as exc:
        return {"ok": False, "error": f"asrpy not installed: {exc}"}

    raw_f = raw.copy().filter(1.0, None, picks="eeg", verbose="ERROR")
    asr = ASR(sfreq=raw_f.info["sfreq"], cutoff=cutoff)
    asr.fit(raw_f)
    raw_clean = asr.transform(raw_f)

    before = raw_f.get_data(picks="eeg")
    after = raw_clean.get_data(picks="eeg")
    var_before = float(np.var(before))
    var_after = float(np.var(after))
    SESSION["raw"] = raw_clean

    return {
        "ok": True,
        "cutoff_sd": cutoff,
        "variance_before": var_before,
        "variance_after": var_after,
        "variance_removed_frac": round(1 - var_after / var_before, 4) if var_before else None,
        "note": "Loaded recording replaced with the ASR-cleaned version.",
    }


def run_prep(line_freq: float | None = None) -> dict:
    """PREP-style robust bad-channel detection (pyprep). Reports bad channels by
    category without modifying the data."""
    if line_freq is None:
        return {"ok": False, "error": (
            "run_prep requires `line_freq` (Hz, typically 50 or 60); "
            "it must be read from the request, not defaulted."
        )}
    raw = _require_raw()
    if not raw.get_montage():
        try:
            raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
        except Exception:
            pass
    try:
        from pyprep import NoisyChannels
    except Exception as exc:
        return {"ok": False, "error": f"pyprep not installed: {exc}"}

    raw_f = raw.copy().pick("eeg").filter(1.0, None, verbose="ERROR").notch_filter(
        line_freq, verbose="ERROR")
    nc = NoisyChannels(raw_f, random_state=42)
    # RANSAC is montage-sensitive and brittle across numpy versions; the
    # deviation/correlation/HF-noise/flat detectors are the robust core.
    try:
        nc.find_all_bads(ransac=True)
    except Exception:
        nc.find_all_bads(ransac=False)
    bads = nc.get_bads(as_dict=True)
    flat = {k: v for k, v in bads.items() if isinstance(v, list) and v}
    detected = sorted({c for v in flat.values() for c in v})

    # Mark on the loaded recording so interpolate_bads() can repair them.
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(detected))

    return {
        "ok": True,
        "line_freq": line_freq,
        "n_channels": len(raw_f.ch_names),
        "bad_channels": detected,
        "bads_by_category": {k: v for k, v in flat.items()},
        "note": "Bad channels marked on the recording; call interpolate_bads to repair.",
    }


def remove_line_noise(line_freq: float | None = None, n_remove: int = 1) -> dict:
    """Remove power-line noise with Zapline (meegkit dss_line), in place.

    Zapline removes the line-frequency artifact (and its spectral peak) by spatial
    filtering rather than a notch filter, so it avoids the notch's waveform distortion
    around the line frequency. Operates on the continuous EEG channels.
    """
    if line_freq is None:
        return {"ok": False, "error": (
            "remove_line_noise requires `line_freq` (Hz, typically 50 or 60); "
            "it must be read from the request, not defaulted."
        )}
    raw = _require_raw()
    try:
        from meegkit import dss
    except Exception as exc:
        return {"ok": False, "error": f"meegkit not installed: {exc}"}

    raw.load_data()
    eeg = mne.pick_types(raw.info, eeg=True)
    X = raw.get_data(picks=eeg).T              # (n_samples, n_channels)
    sfreq = float(raw.info["sfreq"])

    def _line_power(arr):
        # mean power within +/-1 Hz of the line frequency across channels
        from numpy.fft import rfft, rfftfreq
        f = rfftfreq(arr.shape[0], 1.0 / sfreq)
        band = (f >= line_freq - 1) & (f <= line_freq + 1)
        psd = np.abs(rfft(arr - arr.mean(axis=0), axis=0)) ** 2
        return float(psd[band].mean())

    line_before = _line_power(X)
    out, _artifact = dss.dss_line(X, fline=float(line_freq), sfreq=sfreq,
                                  nremove=int(n_remove))
    line_after = _line_power(out)
    raw._data[eeg] = out.T
    SESSION["raw"] = raw
    return {
        "ok": True,
        "line_freq": line_freq,
        "n_components_removed": int(n_remove),
        "line_power_before": line_before,
        "line_power_after": line_after,
        "line_power_removed_frac": round(1 - line_after / line_before, 4) if line_before else None,
        "note": "Line noise removed with Zapline (spatial filter, no notch distortion). "
                "Metric is power within +/-1 Hz of the line frequency.",
    }


def run_faster(thres: float = 3.0, detect_epochs: bool = True) -> dict:
    """FASTER automatic artifact detection (mne-faster).

    Flags bad channels (and optionally bad epochs) by z-scoring a set of statistical
    metrics across channels/epochs and thresholding at `thres` standard deviations.
    Bad channels are marked on the recording (repair with interpolate_bads); bad epochs
    are reported by index. Builds fixed-length epochs if none are loaded.
    """
    raw = _require_raw()
    try:
        from mne_faster import find_bad_channels, find_bad_epochs
    except Exception as exc:
        return {"ok": False, "error": f"mne-faster not installed: {exc}"}

    if not raw.get_montage():
        try:
            raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
        except Exception:
            pass

    epochs = _epochs_for_analysis(raw).copy().pick("eeg")
    bad_channels = find_bad_channels(epochs, thres=thres)
    # Mark on the loaded recording so interpolate_bads() can repair them.
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bad_channels))

    result = {
        "ok": True,
        "thres_sd": thres,
        "n_epochs": len(epochs),
        "bad_channels": sorted(bad_channels),
        "note": "FASTER bad channels marked; call interpolate_bads to repair.",
    }
    if detect_epochs:
        if bad_channels:
            epochs.info["bads"] = bad_channels
        bad_epochs = find_bad_epochs(epochs, thres=thres)
        result["bad_epochs"] = sorted(int(i) for i in bad_epochs)
        result["n_bad_epochs"] = len(bad_epochs)
    return result


def compute_connectivity(method: str = "wpli", fmin: float | None = None,
                         fmax: float | None = None) -> dict:
    """Spectral connectivity across channels (EEGLAB SIFT-equivalent) in a band.
    Returns a connectivity heatmap and the mean connectivity strength."""
    if fmin is None or fmax is None:
        missing = [n for n, v in (("fmin", fmin), ("fmax", fmax)) if v is None]
        return {"ok": False, "error": (
            f"compute_connectivity requires {missing} (frequency band edges in Hz); "
            "they must be read from the request, not defaulted."
        )}
    raw = _require_raw()
    try:
        from mne_connectivity import spectral_connectivity_epochs
    except Exception as exc:
        return {"ok": False, "error": f"mne-connectivity not installed: {exc}"}

    epochs = _epochs_for_analysis(raw).copy().pick("eeg")
    con = spectral_connectivity_epochs(
        epochs, method=method, mode="multitaper", sfreq=epochs.info["sfreq"],
        fmin=fmin, fmax=fmax, faverage=True, verbose="ERROR")
    mat = con.get_data(output="dense")[:, :, 0]  # (n_ch, n_ch)
    mat = mat + mat.T  # lower-triangular -> symmetric for display
    n = mat.shape[0]
    mean_con = float(mat[np.tril_indices(n, k=-1)].mean())

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0)
    ax.set(title=f"{method.upper()} connectivity {fmin}-{fmax} Hz",
           xlabel="channel", ylabel="channel")
    fig.colorbar(im, ax=ax, shrink=0.8)

    return {
        "ok": True,
        "method": method,
        "fmin": fmin,
        "fmax": fmax,
        "n_channels": n,
        "mean_connectivity": round(mean_con, 4),
        "image": _fig_to_b64(fig),
    }


def run_fooof(fmin: float = 1.0, fmax: float = 40.0) -> dict:
    """Parameterise the power spectrum into aperiodic (1/f) + periodic components
    with FOOOF/specparam. Reports the aperiodic exponent and detected peaks."""
    raw = _require_raw()
    try:
        from fooof import FOOOF
    except Exception as exc:
        return {"ok": False, "error": f"fooof not installed: {exc}"}

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, picks=picks)
    psds, freqs = spectrum.get_data(return_freqs=True)
    mean_psd = psds.mean(axis=0)

    fm = FOOOF(peak_width_limits=(1.0, 8.0), max_n_peaks=6, verbose=False)
    fm.fit(freqs, mean_psd, (fmin, fmax))
    offset, exponent = fm.aperiodic_params_[0], fm.aperiodic_params_[-1]
    peaks = [{"cf_hz": round(float(p[0]), 2),
              "power": round(float(p[1]), 3),
              "bw_hz": round(float(p[2]), 2)} for p in fm.peak_params_]

    fig, ax = plt.subplots(figsize=(7, 4))
    fm.plot(ax=ax, plot_peaks="shade", add_legend=True)
    ax.set_title("FOOOF spectral parameterisation")

    return {
        "ok": True,
        "aperiodic_offset": round(float(offset), 3),
        "aperiodic_exponent": round(float(exponent), 3),
        "r_squared": round(float(fm.r_squared_), 3),
        "n_peaks": len(peaks),
        "peaks": peaks,
        "image": _fig_to_b64(fig),
    }


def compute_microstates(n_states: int = 4) -> dict:
    """EEG microstate analysis (pycrostates modified k-means). Reports global
    explained variance and shows the microstate topographies."""
    raw = _require_raw()
    if not raw.get_montage():
        try:
            raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
        except Exception:
            pass
    try:
        from pycrostates.cluster import ModKMeans
    except Exception as exc:
        return {"ok": False, "error": f"pycrostates not installed: {exc}"}

    raw_eeg = raw.copy().pick("eeg").set_eeg_reference("average", verbose="ERROR")
    n_states = int(max(2, min(n_states, len(raw_eeg.ch_names) - 1)))
    mk = ModKMeans(n_clusters=n_states, random_state=42)
    mk.fit(raw_eeg, n_jobs=1, verbose="ERROR")

    fig = mk.plot(show=False)
    if hasattr(fig, "figure"):
        fig = fig.figure

    return {
        "ok": True,
        "n_states": n_states,
        "global_explained_variance": round(float(mk.GEV_), 4),
        "image": _fig_to_b64(fig),
    }


# --------------------------------------------------------------------------- #
# Core preprocessing: filtering, re-referencing, channel repair, epoching
# --------------------------------------------------------------------------- #
MASTOID_CANDIDATES = ["M1", "M2", "A1", "A2", "TP9", "TP10"]


def _ensure_montage(raw):
    if not raw.get_montage():
        try:
            raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
        except Exception:
            pass


def _remove_dc_segments(obj, kind: str) -> dict:
    """Subtract the per-channel DC offset before filtering, replicating ERPLAB's
    RemoveDC ('on') in pop_basicfilter -> removedc.m.

    For each boundary-delimited segment, the mean of the first round(srate/2) samples
    (0.5 s) is computed per channel and subtracted from the whole segment; if a segment
    is shorter than that window, the whole-segment mean is used instead. Continuous data
    is split at 'BAD boundary'/'boundary' annotations; epoched data treats each epoch as
    its own segment. Applies to EEG channels only (the same picks filter_eeg filters).
    """
    obj.load_data()
    sfreq = float(obj.info["sfreq"])
    win = int(round(sfreq / 2.0))
    picks = mne.pick_types(obj.info, eeg=True)

    def _demean(block):
        n = block.shape[-1]
        w = n if n < win else win
        return block - block[:, :w].mean(axis=1, keepdims=True)

    if kind == "epochs":
        data = obj._data                       # (n_epochs, n_ch, n_times)
        for ep in range(data.shape[0]):
            data[ep][picks, :] = _demean(data[ep][picks, :])
        n_seg = int(data.shape[0])
    else:
        data = obj._data                       # (n_ch, n_times), writable after load_data
        n_times = data.shape[-1]
        edges = [0]
        for onset, desc in zip(obj.annotations.onset, obj.annotations.description):
            if str(desc).strip().lower() in ("bad boundary", "boundary"):
                s = int(round(float(onset) * sfreq))
                if 0 < s < n_times:
                    edges.append(s)
        edges = sorted(set(edges + [n_times]))
        for s, e in zip(edges[:-1], edges[1:]):
            data[picks, s:e] = _demean(data[picks, s:e])
        n_seg = len(edges) - 1
    return {"dc_window_samples": win, "dc_n_segments": int(n_seg)}


def _iir_segment_edges(obj, sfreq: float, n_times: int) -> list:
    """Boundary-delimited segment edges (sample indices), identical convention to
    _remove_dc_segments so DC removal and the filter split the record the same way."""
    edges = [0]
    for onset, desc in zip(obj.annotations.onset, obj.annotations.description):
        if str(desc).strip().lower() in ("bad boundary", "boundary"):
            s = int(round(float(onset) * sfreq))
            if 0 < s < n_times:
                edges.append(s)
    return sorted(set(edges + [n_times]))


def _erplab_filtfilt(obj, kind: str, l_freq, h_freq, order: int) -> dict:
    """Apply a Butterworth filter the ERPLAB/MATLAB way, in place, EEG picks only.

    This is the zero-phase *application* pop_basicfilter -> MATLAB filtfilt uses, as
    opposed to MNE's SOS + sosfiltfilt: same order-N Butterworth design, but applied on
    the b,a (transfer-function) form via scipy.signal.filtfilt with MATLAB's odd-reflection
    padding and MATLAB's pad length nfact = 3*(nfilt-1), and re-initialized per
    inter-boundary segment. Matching the pad length + b,a form is what closes the gap that
    SOS + sosfiltfilt leaves (differing padlen formula and per-section initial conditions).
    """
    from scipy.signal import butter, filtfilt
    sfreq = float(obj.info["sfreq"])
    nyq = sfreq / 2.0
    # ERPLAB's 'Order' is the EFFECTIVE two-pass slope (e.g. 'Order',2 = 12 dB/oct). The
    # forward-backward filtfilt doubles a single-pass Butterworth's slope, so ERPLAB designs
    # at half: a requested effective order 2 is a design order 1 (6 dB/oct, doubled to 12).
    # This is the fix that makes the high-pass bit-exact vs ERP CORE's hpfilt checkpoint;
    # designing at the full order gives an effective 4th-order (24 dB/oct) filter (~140 uV off).
    design_order = max(1, int(round(order / 2)))
    if l_freq and h_freq:
        b, a = butter(design_order, [l_freq / nyq, h_freq / nyq], btype="bandpass")
        band = f"{l_freq}-{h_freq} Hz bandpass"
    elif l_freq:
        b, a = butter(design_order, l_freq / nyq, btype="highpass")
        band = f"{l_freq} Hz highpass"
    elif h_freq:
        b, a = butter(design_order, h_freq / nyq, btype="lowpass")
        band = f"{h_freq} Hz lowpass"
    else:
        return {"engine": "erplab", "filt_applied": False}

    nfact = 3 * (max(len(b), len(a)) - 1)          # MATLAB filtfilt's nfact (6 for order 2)
    picks = mne.pick_types(obj.info, eeg=True)

    def _apply(block):
        # MATLAB filtfilt requires len > nfact and errors otherwise; degrade gracefully
        # on a pathologically short segment by shrinking the pad to the largest valid.
        pad = nfact if block.shape[-1] > nfact else max(block.shape[-1] - 1, 0)
        return filtfilt(b, a, block, axis=-1, padtype="odd", padlen=pad)

    if kind == "epochs":
        data = obj._data
        for ep in range(data.shape[0]):
            data[ep][picks, :] = _apply(data[ep][picks, :])
        n_seg = int(data.shape[0])
    else:
        data = obj._data
        n_times = data.shape[-1]
        edges = _iir_segment_edges(obj, sfreq, n_times)
        for s, e in zip(edges[:-1], edges[1:]):
            data[picks, s:e] = _apply(data[picks, s:e])
        n_seg = len(edges) - 1

    # Best-effort filter-metadata update (MNE Info is locked in recent versions).
    try:
        with obj.info._unlock():
            if l_freq:
                obj.info["highpass"] = float(l_freq)
            if h_freq:
                obj.info["lowpass"] = float(h_freq)
    except Exception:
        pass

    return {"engine": "erplab", "iir_order": int(order), "design_order": int(design_order),
            "filt_band": band, "filt_n_segments": int(n_seg),
            "filt_pad_samples": int(nfact)}


def _apply_windowed_sinc_fir(obj, l_freq, h_freq, window, beta, filter_length,
                             trans_bandwidth, max_ripple: float = 0.002) -> dict:
    """Design and apply a zero-phase windowed-sinc FIR over the EEG channels.

    One implementation for every window ('hamming'|'hann'|'blackman'|'kaiser'|...), so a
    sweep over fir_window compares the window, not the backend. High-pass, low-pass and
    band-pass are all handled via scipy.firwin; applied non-causally with filtfilt.
    Number of taps: an explicit int filter_length, else estimated from the transition
    width by the same Kaiser length rule as _design_aa_fir.
    """
    from scipy.signal import firwin, filtfilt

    fs = float(obj.info["sfreq"])
    if isinstance(filter_length, str) and filter_length.strip().isdigit():
        filter_length = int(filter_length)
    if isinstance(filter_length, int):
        numtaps = int(filter_length)
    else:  # 'auto' -> size from the transition band
        tbw = float(trans_bandwidth) if isinstance(trans_bandwidth, (int, float)) else 2.0
        atten_db = -20.0 * np.log10(max_ripple)
        dw = 2.0 * np.pi * tbw / fs
        numtaps = int(np.ceil((atten_db - 7.95) / (2.285 * dw))) + 1
    if numtaps % 2 == 0:                              # force Type-I (odd) for zero phase
        numtaps += 1

    win = (window, beta) if str(window).lower() == "kaiser" else window
    if l_freq and h_freq:
        taps = firwin(numtaps, [l_freq, h_freq], window=win, fs=fs, pass_zero=False)
    elif l_freq and not h_freq:                       # high-pass
        taps = firwin(numtaps, l_freq, window=win, fs=fs, pass_zero=False)
    else:                                             # low-pass
        taps = firwin(numtaps, h_freq, window=win, fs=fs, pass_zero=True)

    picks = mne.pick_types(obj.info, eeg=True)
    data = obj.get_data()
    if data.ndim == 2:                               # Raw: (n_ch, n_times)
        data[picks] = filtfilt(taps, [1.0], data[picks], axis=-1)
    else:                                            # Epochs: (n_epochs, n_ch, n_times)
        data[:, picks] = filtfilt(taps, [1.0], data[:, picks], axis=-1)
    obj._data = data
    return {
        "fir_window": window,
        "fir_beta": beta if str(window).lower() == "kaiser" else None,
        "fir_num_taps": int(numtaps),
        "fir_backend": "scipy.firwin+filtfilt",
    }


def filter_eeg(l_freq: float | None = None, h_freq: float | None = None,
               notch: float | None = None, method: str = "fir",
               iir_order: int | None = None, remove_dc: bool = False,
               engine: str = "mne",
               fir_window: str | None = None, fir_beta: float = 5.0,
               fir_design: str = "firwin",
               filter_length: str | int = "auto",
               l_trans_bandwidth: float | str = "auto",
               h_trans_bandwidth: float | str = "auto") -> dict:
    """Band-pass (and optional notch) filter the loaded recording in place.

    Works on continuous or epoched data.

    method='fir' (default) -> MNE zero-phase windowed-sinc FIR.
    method='iir'           -> zero-phase Butterworth IIR (two-pass filtfilt),
                              matching toolboxes that default to Butterworth
                              (e.g. FieldTrip 'but', ERP CORE's 0.1-30 Hz filter).
    remove_dc=True         -> subtract the per-channel DC offset before filtering,
                              replicating ERPLAB RemoveDC (first round(srate/2)-sample
                              per-segment mean). See _remove_dc_segments.
    engine='mne' (default) -> apply via MNE's zero-phase machinery (IIR: SOS + sosfiltfilt).
    engine='erplab'        -> apply a Butterworth the ERPLAB/MATLAB way (b,a-form filtfilt,
                              MATLAB pad length + odd reflection, per inter-boundary
                              segment); reproduces pop_basicfilter to machine precision.
                              Implies a Butterworth design regardless of `method`.
                              See _erplab_filtfilt.

    FIR window / length (method='fir', engine='mne'):
      fir_window=None (default) -> MNE's default FIR (hamming window, auto length);
                                   preserves prior behaviour bit-for-bit.
      fir_window set            -> route through a scipy windowed-sinc (firwin) applied
                                   zero-phase (filtfilt) so ALL window types run through
                                   ONE implementation and are directly comparable in a
                                   sweep. Supports 'hamming' | 'hann' | 'blackman' |
                                   'kaiser' (Kaiser uses fir_beta). MNE cannot do Kaiser.
      filter_length             -> 'auto' (estimate from the transition band) or an int
                                   number of taps (forced odd for zero phase).
      fir_design / l_trans_bandwidth / h_trans_bandwidth -> passed to MNE when
                                   fir_window is None; the scipy path uses filter_length
                                   (or l_trans_bandwidth as the transition width).
    """
    if l_freq is None and h_freq is None:
        return {"ok": False, "error": (
            "filter_eeg needs at least a high-pass (l_freq) or low-pass (h_freq) cutoff; "
            "neither was given."
        )}
    kind, obj = _active()
    obj.load_data()
    dc_info = _remove_dc_segments(obj, kind) if remove_dc else None
    method = (method or "fir").lower()
    engine = (engine or "mne").lower()
    if (engine == "erplab" or method == "iir") and iir_order is None:
        return {"error": "iir_order is required for a Butterworth/IIR filter "
                         "(method='iir' or engine='erplab'); there is no default order."}
    erplab_info = None
    fir_info = None
    if engine == "erplab":
        erplab_info = _erplab_filtfilt(obj, kind, l_freq, h_freq, iir_order)
    elif method == "iir":
        iir_params = dict(order=int(iir_order), ftype="butter")
        obj.filter(l_freq, h_freq, picks="eeg", method="iir",
                   iir_params=iir_params, verbose="ERROR")
    elif fir_window is not None:
        fir_info = _apply_windowed_sinc_fir(
            obj, l_freq, h_freq, fir_window, fir_beta, filter_length,
            l_trans_bandwidth)
    else:
        obj.filter(l_freq, h_freq, picks="eeg", method="fir", fir_design=fir_design,
                   filter_length=filter_length, l_trans_bandwidth=l_trans_bandwidth,
                   h_trans_bandwidth=h_trans_bandwidth, verbose="ERROR")
    applied_notch = []
    if notch:
        nyq = obj.info["sfreq"] / 2.0
        freqs = [notch * k for k in (1, 2, 3) if notch * k < nyq]
        obj.notch_filter(freqs, picks="eeg", verbose="ERROR")
        applied_notch = freqs
    SESSION[kind] = obj
    result = {
        "ok": True,
        "applied_to": kind,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "notch_freqs": applied_notch,
        "remove_dc": bool(remove_dc),
        "engine": engine,
        "highpass": obj.info.get("highpass"),
        "lowpass": obj.info.get("lowpass"),
        "note": f"Filter applied in place to the loaded {kind}.",
    }
    if dc_info:
        result.update(dc_info)
    if erplab_info:
        result.update(erplab_info)
    if fir_info:
        result.update(fir_info)
    return result


def _design_aa_fir(work_fs: float, cutoff_hz: float, window: str, beta: float,
                   transition_bandwidth: float | None, order: int | None,
                   max_ripple: float):
    """Design a zero-phase windowed-sinc low-pass anti-aliasing FIR.

    Returns (taps, info_dict). The order is either given explicitly or estimated from
    the Kaiser design formula off (transition_bandwidth, max_ripple) -- the same length
    rule EEGLAB's pop_firwsord uses. `window`/`beta` set the window independently, so any
    published windowed-sinc anti-aliaser (Kaiser beta, cutoff, transition) can be matched.
    """
    from scipy.signal import firwin

    if order is not None:
        numtaps = int(order) + 1
        est = None
    else:
        tbw = transition_bandwidth if transition_bandwidth else 0.25 * (work_fs / 2.0)
        atten_db = -20.0 * np.log10(max_ripple)
        dw = 2.0 * np.pi * tbw / work_fs           # transition width in rad/sample
        est = int(np.ceil((atten_db - 7.95) / (2.285 * dw)))
        numtaps = est + 1
    if numtaps % 2 == 0:                            # force Type-I (odd length) for zero phase
        numtaps += 1

    win = (window, beta) if window.lower() == "kaiser" else window
    taps = firwin(numtaps, cutoff_hz, window=win, fs=work_fs)   # unit-gain low-pass
    return taps, {
        "window": window,
        "beta": beta if window.lower() == "kaiser" else None,
        "cutoff_hz": round(cutoff_hz, 4),
        "transition_bandwidth_hz": (round(transition_bandwidth, 4)
                                    if transition_bandwidth else None),
        "num_taps": int(numtaps),
        "order": int(numtaps - 1),
        "estimated_order": est,
        "max_ripple": max_ripple,
    }


def resample(sfreq: float | None = None,
             antialias: str = "auto",
             window: str = "hamming",
             beta: float = 5.0,
             cutoff: float | None = None,
             transition_bandwidth: float | None = None,
             order: int | None = None,
             max_ripple: float = 0.002,
             phase: str = "zero") -> dict:
    """Resample the loaded recording to a new sampling rate (in place).

    Resample BEFORE epoching to avoid event-timing jitter; any epochs already in the
    session are dropped so they are rebuilt against the new rate.

    antialias:
      'auto' (default) -> the library's built-in anti-aliasing low-pass.
      'fir'            -> an explicit, fully specified windowed-sinc low-pass applied
                          before decimation. Use this to reproduce a specific published
                          anti-aliasing filter. Configure it with:
        window                -> 'hamming' (default) | 'hann' | 'blackman' | 'kaiser' | ...
        beta                  -> Kaiser window shape parameter (when window='kaiser').
        cutoff                -> low-pass cutoff in Hz (defaults to the new Nyquist).
        transition_bandwidth  -> transition-band width in Hz; sets the filter length.
        order                 -> explicit FIR order (overrides transition_bandwidth).
        max_ripple            -> passband/stopband deviation used to size the order
                                 from the transition width (default 0.002 ~= 54 dB).
        phase                 -> 'zero' (default, non-causal symmetric) or 'minimum'
                                 (causal minimum-phase). Only applies when antialias='fir'.
    """
    if sfreq is None:
        return {"ok": False, "error": (
            "resample requires `sfreq` (target sampling rate in Hz); "
            "it must be read from the request, not defaulted."
        )}
    kind, obj = _active()
    obj.load_data()
    old = float(obj.info["sfreq"])
    target = float(sfreq)
    if target <= 0:
        return {"ok": False, "error": "sfreq must be positive."}
    nyq = target / 2.0
    lowpass = obj.info.get("lowpass")
    aliasing_risk = bool(lowpass and lowpass > nyq)

    fir_info = None
    if (antialias or "auto").lower() == "fir":
        if target >= old:
            return {"ok": False,
                    "error": "FIR anti-aliasing is for downsampling only (target < current)."}
        ratio = old / target
        q = int(round(ratio))
        if abs(ratio - q) > 1e-6:
            return {"ok": False,
                    "error": f"FIR anti-aliasing needs an integer decimation ratio; "
                             f"{old} -> {target} Hz is {ratio:.4f}x. Pick a divisor rate."}
        phase_mode = (phase or "zero").lower()
        if phase_mode not in ("zero", "minimum"):
            return {"ok": False,
                    "error": f"phase must be 'zero' or 'minimum', got '{phase}'."}
        from scipy.signal import fftconvolve, lfilter, minimum_phase

        cutoff_hz = float(cutoff) if cutoff is not None else nyq
        if cutoff_hz > nyq:
            return {"ok": False,
                    "error": f"cutoff {cutoff_hz} Hz is above the new Nyquist {nyq} Hz; "
                             "it would not prevent aliasing."}
        taps, fir_info = _design_aa_fir(old, cutoff_hz, window, beta,
                                        transition_bandwidth, order, max_ripple)
        fir_info["phase"] = phase_mode
        # Apply the configured filter exactly once, then decimate -- no second filter
        # (mirrors EEGLAB: firws band-limit, then polyphase decimate by the same design).
        data = obj.get_data()
        if phase_mode == "minimum":
            # Causal minimum-phase: derive the min-phase kernel and apply forward-only
            # (no delay compensation), so group delay is real -- the opposite of zero-phase.
            mp = minimum_phase(taps)
            kernel = mp.reshape((1,) * (data.ndim - 1) + (-1,))
            filtered = lfilter(kernel.ravel(), 1.0, data, axis=-1)
        else:
            # Zero-phase, edge-padded FIR (mirrors firfiltdcpadded): pad with the boundary
            # value and convolve 'valid' so the filtered signal keeps the original length.
            pad = (len(taps) - 1) // 2
            kernel = taps.reshape((1,) * (data.ndim - 1) + (-1,))
            padded = np.pad(data, [(0, 0)] * (data.ndim - 1) + [(pad, pad)], mode="edge")
            filtered = fftconvolve(padded, kernel, mode="valid", axes=-1)
        decimated = filtered[..., ::q]

        info2 = obj.info.copy()
        with info2._unlock():
            info2["sfreq"] = target
            info2["lowpass"] = min(info2.get("lowpass") or nyq, cutoff_hz)
        if kind == "raw":
            obj2 = mne.io.RawArray(decimated, info2, first_samp=0, verbose="ERROR")
            obj2.set_annotations(obj.annotations)
        else:
            obj2 = mne.EpochsArray(decimated, info2, events=obj.events,
                                   tmin=obj.tmin, event_id=obj.event_id, verbose="ERROR")
        obj = obj2
    else:
        obj.resample(target, verbose="ERROR")
    SESSION[kind] = obj
    if kind == "raw":
        SESSION["epochs"] = None  # raw-derived epochs would be stale at the old rate

    return {
        "ok": True,
        "applied_to": kind,
        "old_sfreq": old,
        "new_sfreq": float(obj.info["sfreq"]),
        "new_nyquist": nyq,
        "antialias": (antialias or "auto").lower(),
        "aa_filter": fir_info,
        "aliasing_note": (
            "Existing low-pass is above the new Nyquist; anti-alias filter still "
            "applied, but consider low-passing below %.1f Hz first." % nyq
        ) if aliasing_risk else "OK: anti-aliasing applied.",
        "note": f"Resampled the {kind} in place.",
    }


def set_reference(ref_type: str | None = None,
                  channels: list | None = None,
                  add_implicit: list | None = None) -> dict:
    """Re-reference the loaded recording.

    ref_type: 'average' | 'mastoids' (linked-mastoid) | 'channels' (use `channels`).
    add_implicit: channel names to add back as zeros BEFORE referencing (the online
    reference electrode that was not recorded, e.g. FieldTrip's implicitref). Needed for
    true linked-mastoid when one mastoid was the online reference.
    """
    if ref_type is None:
        return {"ok": False, "error": (
            "set_reference requires `ref_type` ('average', 'mastoids', or 'channels'); "
            "it must be read from the request, not defaulted."
        )}
    kind, obj = _active()
    obj.load_data()
    ref_type = ref_type.lower()

    added = []
    if add_implicit:
        to_add = [c for c in add_implicit if c not in obj.ch_names]
        if to_add:
            obj = mne.add_reference_channels(obj, to_add)
            SESSION[kind] = obj
            added = to_add

    if ref_type == "average":
        obj.set_eeg_reference("average", projection=False, verbose="ERROR")
        used = "average"
    elif ref_type in ("mastoids", "mastoid", "linked-mastoid"):
        ref = channels or [c for c in MASTOID_CANDIDATES if c in obj.ch_names]
        ref = [c for c in ref if c in obj.ch_names]
        if not ref:
            return {"ok": False,
                    "error": "No mastoid channels found. Expected one of "
                             f"{MASTOID_CANDIDATES}; pass them explicitly via `channels`."}
        obj.set_eeg_reference(ref, verbose="ERROR")
        used = ref
    elif ref_type in ("channels", "channel", "custom"):
        ref = [c for c in (channels or []) if c in obj.ch_names]
        if not ref:
            return {"ok": False,
                    "error": "ref_type='channels' requires valid `channels` present "
                             "in the recording."}
        obj.set_eeg_reference(ref, verbose="ERROR")
        used = ref
    else:
        return {"ok": False,
                "error": f"Unknown ref_type '{ref_type}'. Use 'average', "
                         "'mastoids', or 'channels'."}

    SESSION[kind] = obj
    return {"ok": True, "reference": used, "added_implicit": added, "applied_to": kind,
            "note": f"Re-reference applied in place to the loaded {kind}."}


def interpolate_bads(channels: list | None = None) -> dict:
    """Interpolate bad channels (spherical splines). Uses channels marked by run_prep
    unless an explicit list is given."""
    raw = _require_raw()
    raw.load_data()
    _ensure_montage(raw)
    if channels:
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(channels))
    bads = list(raw.info["bads"])
    if not bads:
        return {"ok": True, "interpolated": [],
                "note": "No bad channels marked; nothing to interpolate. "
                        "Run run_prep first or pass `channels`."}
    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    SESSION["raw"] = raw
    return {"ok": True, "interpolated": bads, "n_interpolated": len(bads)}


def _epoch_channel_picks(epochs, channels):
    """Resolve `channels` to 0-based picks. Accepts channel NAMES (preferred) or 1-based
    integer indices (EEGLAB/ERPLAB convention, as ERP CORE's AR parameter sheets state them).
    Returns (picks, error_or_None)."""
    if not channels:
        return None, "Provide `channels` (names, or 1-based indices as EEGLAB numbers them)."
    picks = []
    for c in channels:
        if isinstance(c, str):
            if c not in epochs.ch_names:
                return None, f"Channel {c!r} not found. Available: {epochs.ch_names[:20]}"
            picks.append(epochs.ch_names.index(c))
        else:
            i = int(c) - 1  # 1-based -> 0-based
            if not (0 <= i < len(epochs.ch_names)):
                return None, (f"Channel index {c} out of range 1..{len(epochs.ch_names)}.")
            picks.append(i)
    return picks, None


def _flag_epochs(mask, detector, params):
    """OR a detector's mask into the session's accumulated artifact flags (ERPLAB semantics:
    each detection pass adds to the epoch's global reject flag)."""
    prev = SESSION.get("artifact_flags")
    if prev is None or len(prev) != len(mask):
        prev = np.zeros(len(mask), dtype=bool)
    SESSION["artifact_flags"] = prev | mask
    return {"ok": True, "detector": detector, "params": params,
            "n_flagged_this_pass": int(mask.sum()),
            "n_flagged_total": int(SESSION["artifact_flags"].sum()),
            "n_epochs": int(len(mask)),
            "note": "Flags accumulated in session; call reject_flagged_epochs to drop them."}


def _epoch_data_uv():
    """(epochs, data_in_uV, times_ms) or (None, error, None)."""
    epochs = SESSION.get("epochs")
    if epochs is None:
        return None, "No epochs in session; create or load epochs first.", None
    return epochs, epochs.get_data() * 1e6, epochs.times * 1000.0


def detect_artifacts_extreme_value(channels=None, threshold_min: float | None = None,
                                   threshold_max: float | None = None,
                                   tmin_ms: float | None = None,
                                   tmax_ms: float | None = None) -> dict:
    """Flag epochs whose signal leaves [threshold_min, threshold_max] uV inside the time
    window, on any listed channel (ERPLAB pop_artextval, "simple voltage threshold").
    Deterministic: recomputed from the data, never inherited. All parameters are required --
    take them from the request / the study's AR parameter sheet."""
    from artifact_epoch_reject import extreme_value
    epochs, uv, tms = _epoch_data_uv()
    if epochs is None:
        return {"ok": False, "error": uv}
    missing = [n for n, v in (("threshold_min", threshold_min), ("threshold_max", threshold_max),
                              ("tmin_ms", tmin_ms), ("tmax_ms", tmax_ms)) if v is None]
    if missing:
        return {"ok": False, "error": f"detect_artifacts_extreme_value requires {missing} "
                "(no defaults; take the values from the request)."}
    picks, err = _epoch_channel_picks(epochs, channels)
    if err:
        return {"ok": False, "error": err}
    mask = extreme_value(uv, tms, picks, threshold_min, threshold_max, tmin_ms, tmax_ms)
    return _flag_epochs(mask, "extreme_value",
                        {"channels": channels, "threshold_min": threshold_min,
                         "threshold_max": threshold_max, "tmin_ms": tmin_ms, "tmax_ms": tmax_ms})


def detect_artifacts_moving_window(channels=None, threshold: float | None = None,
                                   tmin_ms: float | None = None, tmax_ms: float | None = None,
                                   window_size_ms: float | None = None,
                                   window_step_ms: float | None = None) -> dict:
    """Flag epochs whose peak-to-peak amplitude within a sliding window exceeds `threshold` uV
    on any listed channel (ERPLAB pop_artmwppth). Used for both general C.R.A.P. artifacts and
    blinks (on a VEOG channel). Deterministic: recomputed, never inherited. All parameters are
    required -- take them from the request / the study's AR parameter sheet."""
    from artifact_epoch_reject import moving_window_peak_to_peak
    epochs, uv, tms = _epoch_data_uv()
    if epochs is None:
        return {"ok": False, "error": uv}
    missing = [n for n, v in (("threshold", threshold), ("tmin_ms", tmin_ms),
                              ("tmax_ms", tmax_ms), ("window_size_ms", window_size_ms),
                              ("window_step_ms", window_step_ms)) if v is None]
    if missing:
        return {"ok": False, "error": f"detect_artifacts_moving_window requires {missing} "
                "(no defaults; take the values from the request)."}
    picks, err = _epoch_channel_picks(epochs, channels)
    if err:
        return {"ok": False, "error": err}
    mask = moving_window_peak_to_peak(uv, tms, epochs.info["sfreq"], picks, threshold,
                                      tmin_ms, tmax_ms, window_size_ms, window_step_ms)
    return _flag_epochs(mask, "moving_window_peak_to_peak",
                        {"channels": channels, "threshold": threshold, "tmin_ms": tmin_ms,
                         "tmax_ms": tmax_ms, "window_size_ms": window_size_ms,
                         "window_step_ms": window_step_ms})


def detect_artifacts_step(channels=None, threshold: float | None = None,
                          tmin_ms: float | None = None, tmax_ms: float | None = None,
                          window_size_ms: float | None = None,
                          window_step_ms: float | None = None) -> dict:
    """Flag epochs containing a step-like shift (saccade / horizontal eye movement): within a
    sliding window, |mean(first half) - mean(second half)| exceeds `threshold` uV on any listed
    channel (ERPLAB pop_artstep). Deterministic: recomputed, never inherited. All parameters are
    required -- take them from the request / the study's AR parameter sheet."""
    from artifact_epoch_reject import step_like
    epochs, uv, tms = _epoch_data_uv()
    if epochs is None:
        return {"ok": False, "error": uv}
    missing = [n for n, v in (("threshold", threshold), ("tmin_ms", tmin_ms),
                              ("tmax_ms", tmax_ms), ("window_size_ms", window_size_ms),
                              ("window_step_ms", window_step_ms)) if v is None]
    if missing:
        return {"ok": False, "error": f"detect_artifacts_step requires {missing} "
                "(no defaults; take the values from the request)."}
    picks, err = _epoch_channel_picks(epochs, channels)
    if err:
        return {"ok": False, "error": err}
    mask = step_like(uv, tms, epochs.info["sfreq"], picks, threshold,
                     tmin_ms, tmax_ms, window_size_ms, window_step_ms)
    return _flag_epochs(mask, "step_like",
                        {"channels": channels, "threshold": threshold, "tmin_ms": tmin_ms,
                         "tmax_ms": tmax_ms, "window_size_ms": window_size_ms,
                         "window_step_ms": window_step_ms})


def reject_flagged_epochs() -> dict:
    """Drop the epochs flagged by the detect_artifacts_* passes (ERPLAB's 'Criterion','good'
    at averaging time). Call after all detection passes; ERP averages then exclude them."""
    epochs = SESSION.get("epochs")
    if epochs is None:
        return {"ok": False, "error": "No epochs in session."}
    flags = SESSION.get("artifact_flags")
    if flags is None:
        return {"ok": False, "error": "No artifact flags in session; run a detect_artifacts_* "
                "pass first (flags are computed from the data, not inherited)."}
    if len(flags) != len(epochs):
        return {"ok": False, "error": f"Flag count {len(flags)} != epoch count {len(epochs)}."}
    n_before = len(epochs)
    SESSION["epochs"] = epochs[~flags]
    SESSION["artifact_flags"] = None
    return {"ok": True, "n_before": n_before, "n_flagged": int(flags.sum()),
            "n_kept": int((~flags).sum()),
            "note": "Flagged epochs dropped; ERP averages now exclude them."}


def _event_code_int(label):
    """An MNE/ERPLAB annotation label -> its integer event code, or None if not numeric."""
    try:
        return int(re.sub(r"[^0-9-]", "", str(label)))
    except (ValueError, TypeError):
        return None


def _eeglab_native_events(filepath):
    """Read (codes, sample0, native_sfreq) straight from an EEGLAB .set's EEG.event struct,
    or None.

    Why this exists: after a downsample, EEGLAB/ERPLAB event latencies are *fractional*
    (e.g. 9581.75 at 256 Hz). MNE rounds them to integer samples on import; ERPLAB keeps the
    fraction and **truncates** at epoch time, i.e. sample0 = floor(latency - 1) (latency is
    1-based). Those two rules disagree for any event with a .5/.75 fraction, shifting a subset
    of epochs by one sample. Using the native latencies with ERPLAB's floor reproduces the
    released epochs bit-exactly (r = 1.0); MNE's rounding does not.

    Also returns EEG.srate (the .set's original sampling rate) so the caller can detect and
    correct for a resample applied after load_eeg -- native latencies are in samples at the
    .set's original rate, not at the current raw.info['sfreq'].
    """
    try:
        import scipy.io as sio
        EEG = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)["EEG"]
        ev = np.atleast_1d(EEG.event)
        lat = np.array([float(e.latency) for e in ev])
        codes = [_event_code_int(e.type) for e in ev]
        native_sfreq = float(EEG.srate)
        return codes, np.floor(lat - 1.0).astype(int), native_sfreq
    except Exception:
        return None


def create_bins(bins=None, require_following_code: int | None = None,
                engine: str = "mne") -> dict:
    """Assign time-locking events to bins (ERPLAB BINLISTER), from the loaded continuous
    recording's event codes. Stores the bin-tagged event set in SESSION['bins'] for
    create_epochs to consume. This is the `bins` pipeline stage.

    bins : list of {"label": <str>, "codes": [[lo, hi], ...] and/or [c1, c2, ...]} -- maps a
           set of stimulus event codes (inclusive [lo, hi] ranges and/or bare integers) to a
           bin label. The code->bin mapping comes from the request / event-code scheme; nothing
           is defaulted (no task-specific values baked in).
    require_following_code : if set, a time-locking event is binned only when the immediately
           following event equals this code (ERPLAB response contingency, e.g. 201 = correct
           response). Omit for unconditional binning.
    engine : which event-latency convention to use -- these disagree after a downsample, when
           latencies are fractional (e.g. 9581.75 at 256 Hz).
           'mne' (default) -> MNE's annotation samples (fraction ROUNDED on import).
           'erplab'        -> the source .set's native latencies, TRUNCATED as ERPLAB does
                              (sample0 = floor(latency-1)). Required to reproduce ERPLAB/EEGLAB
                              epochs bit-exactly; rounding shifts a subset of epochs by one
                              sample. Pair with create_epochs(engine='erplab').
    """
    if not bins:
        return {"ok": False, "error": "create_bins requires `bins` (label -> codes); take the "
                "code->bin mapping from the request/event scheme (no default)."}
    raw = _require_raw()
    engine = (engine or "mne").lower()
    if engine not in ("mne", "erplab"):
        return {"ok": False, "error": f"Unknown engine {engine!r}; use 'mne' or 'erplab'."}
    if engine == "erplab":
        src = SESSION.get("filepath")
        native = (_eeglab_native_events(src)
                  if isinstance(src, str) and src.lower().endswith(".set") else None)
        if native is None:
            return {"ok": False, "error": (
                "engine='erplab' needs the source EEGLAB .set to read native (fractional) event "
                f"latencies, but they could not be read from {src!r}. Load the .set directly, or "
                "use engine='mne'.")}
        codes, samples, native_sfreq = native
        # Rescale native latencies when the pipeline resampled after load_eeg. Native
        # latencies are in samples at the .set's original rate; if current sfreq differs
        # (e.g. 1024 -> 256 Hz, ratio 0.25) the samples must be multiplied by that ratio
        # before they are used as indices into the resampled data. When no resample occurred
        # (ratio == 1.0 exactly) this is a strict no-op so bit-exact results are preserved.
        current_sfreq = float(raw.info["sfreq"])
        ratio = current_sfreq / native_sfreq
        if ratio != 1.0:
            # Keep ERPLAB's floor convention: floor(latency - 1) was already applied to the
            # fractional native samples; now scale those integer samples and re-floor to land
            # on the nearest sample in the resampled grid.
            samples = np.floor(samples * ratio).astype(int)
        latency_source = "eeglab_native_floor"
    else:
        events, event_dict = mne.events_from_annotations(raw)
        inv = {v: k for k, v in event_dict.items()}
        codes = [_event_code_int(inv[c]) for c in events[:, 2]]
        samples = events[:, 0]
        latency_source = "mne_annotations_rounded"

    def _match_bin(code):
        for b in bins:
            for spec in b.get("codes", []):
                if isinstance(spec, (list, tuple)) and len(spec) == 2:
                    if spec[0] <= code <= spec[1]:
                        return b["label"]
                elif code == spec:
                    return b["label"]
        return None

    labels = [b["label"] for b in bins]
    bin_id = {lab: i + 1 for i, lab in enumerate(labels)}
    tagged, per_bin = [], {lab: 0 for lab in labels}
    for i in range(len(codes)):
        code = codes[i]
        if code is None:
            continue
        lab = _match_bin(code)
        if lab is None:
            continue
        if require_following_code is not None:
            nxt = codes[i + 1] if i + 1 < len(codes) else None
            if nxt != require_following_code:
                continue
        tagged.append([int(samples[i]), 0, bin_id[lab]])
        per_bin[lab] += 1

    if not tagged:
        return {"ok": False, "error": "No events matched the bin definitions (check codes / "
                "require_following_code against the recording's event stream)."}
    SESSION["bins"] = {"events": np.array(tagged, dtype=int), "id": dict(bin_id)}
    return {"ok": True, "n_binned": len(tagged), "per_bin": per_bin,
            "require_following_code": require_following_code,
            "engine": engine, "latency_source": latency_source,
            "note": "Bins stored in session; call create_epochs to epoch them."}


def create_epochs(tmin: float | None = None, tmax: float | None = None,
                  event_id: str | None = None,
                  fixed_length_sec: float | None = None,
                  baseline: bool | list | None = True, engine: str = "mne") -> dict:
    """Epoch the recording. If bins were defined by create_bins, epochs are cut per bin;
    otherwise event-based from annotations by default; pass `fixed_length_sec` for
    resting-state fixed-length epochs.

    engine : baseline-window convention (they differ by one sample at t=0).
             'mne' (default) -> baseline (None, 0), which INCLUDES the t=0 sample.
             'erplab'        -> baseline excludes t=0 (ERPLAB's pre-stimulus window), needed
                                to reproduce ERPLAB epochs bit-exactly. Pair with
                                create_bins(engine='erplab').
    """
    if fixed_length_sec is None and (tmin is None or tmax is None):
        missing = [n for n, v in (("tmin", tmin), ("tmax", tmax)) if v is None]
        return {"ok": False, "error": (
            f"create_epochs requires {missing} for event-based epoching; "
            "they must be read from the request, not defaulted. "
            "(Pass fixed_length_sec instead for resting-state fixed-length epochs.)"
        )}
    raw = _require_raw()

    if fixed_length_sec:
        events = mne.make_fixed_length_events(raw, duration=fixed_length_sec)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=fixed_length_sec,
                            baseline=None, preload=True, verbose="ERROR")
        SESSION["epochs"] = epochs
        return {"ok": True, "mode": "fixed_length", "n_epochs": len(epochs),
                "duration_sec": fixed_length_sec}

    engine = (engine or "mne").lower()
    if engine not in ("mne", "erplab"):
        return {"ok": False, "error": f"Unknown engine {engine!r}; use 'mne' or 'erplab'."}
    # ERPLAB's pre-stimulus baseline stops just before t=0; MNE's (None, 0) includes it.
    base_end = (-1.0 / raw.info["sfreq"]) if engine == "erplab" else 0

    # `baseline` accepts a bool (True -> pre-stimulus window (None, base_end); False -> off) OR an
    # explicit [lo, hi] window (MNE convention). A window in ms (|value| > 30) is converted to s;
    # its upper edge honours the engine's t=0 convention. This tolerance exists because the planner
    # naturally emits a window here (as measure_component uses), and a window == baseline-on.
    def _resolve_baseline(b):
        if isinstance(b, (list, tuple)) and len(b) == 2:
            lo, hi = b
            scale = 1000.0 if (abs(lo) > 30 or abs(hi) > 30) else 1.0  # ms -> s
            lo = None if lo in (None, 0) else lo / scale
            hi_s = hi / scale if hi not in (None,) else 0
            # keep the engine's t=0 convention when the window ends at 0
            hi_out = base_end if abs(hi_s) < 1e-9 else hi_s
            return (lo, hi_out)
        return (None, base_end) if b else None
    base = _resolve_baseline(baseline)

    # Bin-aware path: if create_bins tagged events, epoch those (labelled by bin).
    binset = SESSION.get("bins")
    if binset is not None:
        events, use_id = binset["events"], dict(binset["id"])
        epochs = mne.Epochs(raw, events, event_id=use_id, tmin=tmin, tmax=tmax,
                            baseline=base, preload=True, verbose="ERROR")
        SESSION["epochs"] = epochs
        return {"ok": True, "mode": "binned", "n_epochs": len(epochs),
                "per_bin": {lab: int((events[:, 2] == i).sum()) for lab, i in use_id.items()},
                "tmin": tmin, "tmax": tmax, "engine": engine,
                "baseline": (str(base) if base else "off")}

    events, event_dict = mne.events_from_annotations(raw)
    if len(events) == 0:
        return {"ok": False,
                "error": "No events/annotations found. Provide annotated/stim data, "
                         "or pass fixed_length_sec for resting-state epochs."}
    use_id = event_dict
    if event_id is not None and event_id in event_dict:
        use_id = {event_id: event_dict[event_id]}

    epochs = mne.Epochs(raw, events, event_id=use_id, tmin=tmin, tmax=tmax,
                        baseline=base, preload=True, verbose="ERROR")
    SESSION["epochs"] = epochs
    return {"ok": True, "mode": "event_based", "n_epochs": len(epochs),
            "event_ids": use_id, "tmin": tmin, "tmax": tmax, "engine": engine,
            "baseline": (str(base) if base else "off")}


def delete_break_segments(time_threshold_ms: float | None = None,
                          start_buffer_ms: float | None = None,
                          end_buffer_ms: float | None = None,
                          use_code_ranges=None) -> dict:
    """Delete break-period segments from the loaded continuous recording, in place
    (ERPLAB pop_erplabDeleteTimeSegments). All timing/scope parameters are required and
    have no defaults -- supply them from the request. See artifact_break_removal."""
    from artifact_break_removal import delete_break_segments as _impl
    raw = _require_raw()
    missing = [n for n, v in (("time_threshold_ms", time_threshold_ms),
                              ("start_buffer_ms", start_buffer_ms),
                              ("end_buffer_ms", end_buffer_ms),
                              ("use_code_ranges", use_code_ranges)) if v is None]
    if missing:
        return {"ok": False, "error": "delete_break_segments requires "
                + ", ".join(missing) + " (no defaults; take the values from the request)."}
    ranges = tuple(tuple(r) for r in use_code_ranges)
    res = _impl(raw, time_threshold_ms=time_threshold_ms,
                start_buffer_ms=start_buffer_ms, end_buffer_ms=end_buffer_ms,
                use_code_ranges=ranges)
    SESSION["raw"] = res.pop("raw")
    res["ok"] = True
    res["applied_to"] = "raw"
    return res


def continuous_artifact_detect(ampth: float, winms: float, stepms: float,
                               channel_ranges=None) -> dict:
    """Detect and delete continuous-EEG segments whose peak-to-peak amplitude exceeds
    `ampth` uV within a moving window (ERPLAB pop_continuousartdet + eeg_eegrej),
    in place. Semi-automatic pre-ICA cleaning: ampth/winms/stepms and the channel
    ranges to scan are per-recording and must all be supplied. See
    artifact_continuous_detect."""
    from artifact_continuous_detect import (
        continuous_artifact_detect as _detect, apply_deletion as _apply)
    raw = _require_raw()
    if channel_ranges is None:
        return {"ok": False, "error": "continuous_artifact_detect requires channel_ranges "
                "(1-based inclusive [min,max] channel ranges to scan); no default."}
    # 1-based inclusive ranges -> 0-based channel indices, clamped to the montage
    nchan = len(raw.ch_names)
    chans = sorted({c - 1 for lo, hi in channel_ranges
                    for c in range(int(lo), int(hi) + 1) if 0 < c <= nchan})
    info = _detect(raw, ampth=ampth, winms=winms, stepms=stepms, chan_array=chans)
    if info["deleted_spans"]:
        SESSION["raw"] = _apply(raw, info["deleted_spans"])
    info["ok"] = True
    info["applied_to"] = "raw"
    return info


# --------------------------------------------------------------------------- #
# Schemas + dispatch
# --------------------------------------------------------------------------- #
TOOL_FUNCTIONS = {
    "load_eeg": load_eeg,
    "set_channel_types": set_channel_types,
    "save_eeg": save_eeg,
    "filter_eeg": filter_eeg,
    "resample": resample,
    "set_reference": set_reference,
    "run_prep": run_prep,
    "interpolate_bads": interpolate_bads,
    "run_asr": run_asr,
    "remove_line_noise": remove_line_noise,
    "run_faster": run_faster,
    "run_ica": run_ica,
    "load_ica_eeglab": load_ica_eeglab,
    "apply_ica": apply_ica,
    "inspect_ica_component": inspect_ica_component,
    "review_ica": review_ica,
    "derive_bipolar_eog": derive_bipolar_eog,
    "shift_events": shift_events,
    "delete_break_segments": delete_break_segments,
    "continuous_artifact_detect": continuous_artifact_detect,
    "run_autoreject": run_autoreject,
    "detect_artifacts_extreme_value": detect_artifacts_extreme_value,
    "detect_artifacts_moving_window": detect_artifacts_moving_window,
    "detect_artifacts_step": detect_artifacts_step,
    "reject_flagged_epochs": reject_flagged_epochs,
    "create_bins": create_bins,
    "create_epochs": create_epochs,
    "compute_psd": compute_psd,
    "compute_erp": compute_erp,
    "compute_difference_erp": compute_difference_erp,
    "measure_component": measure_component,
    "run_fooof": run_fooof,
    "compute_connectivity": compute_connectivity,
    "compute_microstates": compute_microstates,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "load_eeg",
            "description": "Load an EEG recording from disk into the session. "
                           "Must be called before any analysis tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to an EEG file (.edf, .fif, .bdf, .set, .vhdr).",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_eeg",
            "description": "Save the current processed recording to disk as a FIF file so it "
                           "can be reloaded later. Use when the request asks to save, export, "
                           "or checkpoint the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Output path; '.fif' is appended/normalized to the MNE "
                                       "raw/epochs naming convention.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_eeg",
            "description": "Band-pass and optional notch filter the loaded recording "
                           "in place. At least one of l_freq or h_freq must be set. "
                           "For a HIGH-PASS-ONLY filter set l_freq and leave h_freq=null. "
                           "method='fir' (default, zero-phase windowed-sinc) or 'iir' "
                           "(zero-phase / non-causal Butterworth); set iir_order to the "
                           "Butterworth filter order when method='iir' (see the roll-off "
                           "table in the planning rules to convert dB/octave to order).",
            "parameters": {
                "type": "object",
                "properties": {
                    "l_freq": {"type": "number",
                               "description": "High-pass edge in Hz (null to skip)."},
                    "h_freq": {"type": "number",
                               "description": "Low-pass edge in Hz (null to skip). For a HIGH-PASS-ONLY "
                                              "filter (the request gives only a high-pass / low "
                                              "cutoff and no upper/low-pass edge), set this to null."},
                    "notch": {"type": "number",
                              "description": "Line-noise freq to notch (50 or 60); harmonics auto-added."},
                    "method": {"type": "string",
                               "description": "'fir' (default, zero-phase windowed-sinc) or "
                                              "'iir' for a zero-phase (non-causal) Butterworth "
                                              "filter. IMPORTANT: if the request names a "
                                              "'Butterworth' filter (or gives a Butterworth/IIR "
                                              "order), set 'iir' and the matching iir_order; do "
                                              "NOT leave the 'fir' default when a Butterworth is "
                                              "requested."},
                    "iir_order": {"type": "number",
                                  "description": "Butterworth filter order, required when "
                                                 "method='iir' or engine='erplab' (there is no "
                                                 "default order, always set it explicitly). If "
                                                 "the request gives the roll-off in dB/octave, "
                                                 "look up the order: 6 dB/oct=1, 12 dB/oct=2, "
                                                 "18 dB/oct=3, 24 dB/oct=4, 30 dB/oct=5, "
                                                 "36 dB/oct=6, 48 dB/oct=8. Otherwise use the "
                                                 "order named in the request."},
                    "remove_dc": {"type": "boolean",
                                  "description": "If the request says to remove the DC offset or "
                                                 "DC bias (e.g. 'remove the DC offsets and "
                                                 "high-pass filter'), set true. Replicates ERPLAB "
                                                 "RemoveDC: subtracts the first 0.5 s per-segment "
                                                 "mean per channel before the filter. Default "
                                                 "false; leave false when no DC removal is asked."},
                    "engine": {"type": "string",
                               "description": "Application backend: 'mne' (default) applies MNE's "
                                              "zero-phase machinery (IIR: SOS + sosfiltfilt); "
                                              "'erplab' reproduces ERPLAB/MATLAB pop_basicfilter "
                                              "bit-exactly (b,a-form filtfilt, MATLAB padding, "
                                              "per-segment). Under 'erplab', iir_order is ERPLAB's "
                                              "EFFECTIVE two-pass slope (a '2nd-order Butterworth / "
                                              "12 dB per octave' is iir_order=2, applied as a "
                                              "design-order-1 filtfilt); do NOT halve it yourself. "
                                              "IMPORTANT: if the request says to reproduce ERP CORE "
                                              "/ ERPLAB / the MATLAB pipeline exactly, or to match a "
                                              "MATLAB/ERPLAB filter bit-for-bit, set 'erplab' (it "
                                              "implies a Butterworth design); otherwise leave 'mne'."},
                    "fir_window": {"type": "string",
                                   "description": "FIR window for method='fir' (engine='mne'). "
                                                  "Leave unset for MNE's default. Set to compare "
                                                  "window types in a sweep: 'hamming', 'hann', "
                                                  "'blackman', or 'kaiser' (Kaiser needs fir_beta). "
                                                  "When set, ALL windows run through one scipy "
                                                  "windowed-sinc backend so they are comparable."},
                    "fir_beta": {"type": "number",
                                 "description": "Kaiser window shape parameter (used only when "
                                                "fir_window='kaiser'). Higher = more stopband "
                                                "attenuation, wider transition. Default 5.0."},
                    "fir_design": {"type": "string",
                                   "description": "MNE FIR design when fir_window is unset: "
                                                  "'firwin' (default) or 'firwin2'."},
                    "filter_length": {"type": ["integer", "string"],
                                      "description": "FIR length: the string 'auto' (default, sized "
                                                     "from the transition band) or an integer number "
                                                     "of taps (e.g. 847). Sweep this to study the "
                                                     "effect of filter length."},
                    "l_trans_bandwidth": {"type": "number",
                                          "description": "High-pass transition-band width in Hz "
                                                         "('auto' if unset). In the scipy window path "
                                                         "it also sizes 'auto' filter_length."},
                    "h_trans_bandwidth": {"type": "number",
                                          "description": "Low-pass transition-band width in Hz "
                                                         "('auto' if unset), MNE FIR path only."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resample",
            "description": "Resample the loaded recording to a new sampling rate in Hz. "
                           "Do this before epoching. Anti-aliasing is applied before "
                           "decimation: 'auto' uses the built-in low-pass, or set "
                           "antialias='fir' to specify an exact windowed-sinc filter "
                           "(window, cutoff, transition band, order) when a particular "
                           "anti-aliasing filter is called for.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sfreq": {"type": "number",
                              "description": "Target sampling rate in Hz."},
                    "antialias": {"type": "string",
                                  "description": "'auto' (default, built-in low-pass) or "
                                                 "'fir' for an explicit windowed-sinc "
                                                 "anti-aliasing filter defined below. Set 'fir' "
                                                 "whenever the request names a specific window, "
                                                 "cutoff, order, or a 'windowed-sinc'/'FIR' "
                                                 "anti-aliasing filter — do not leave it 'auto'."},
                    "window": {"type": "string",
                               "description": "FIR window when antialias='fir'. Default "
                                              "'hamming'; also 'hann', 'blackman', 'kaiser'. "
                                              "IMPORTANT: if the request says 'Kaiser' or "
                                              "'Kaiser-windowed', set 'kaiser' (and beta) — do "
                                              "NOT leave the hamming default. Match the window "
                                              "named in the request exactly; only use the "
                                              "default when no window is named."},
                    "beta": {"type": "number",
                             "description": "Kaiser window shape parameter; REQUIRED whenever "
                                            "window='kaiser' (e.g. the request says 'beta=5' "
                                            "or 'β=5'). Ignored for other windows."},
                    "cutoff": {"type": "number",
                               "description": "Anti-aliasing low-pass cutoff in Hz "
                                              "(defaults to the new Nyquist)."},
                    "transition_bandwidth": {"type": "number",
                                             "description": "Transition-band width in Hz; "
                                                            "sets the FIR order."},
                    "order": {"type": "number",
                              "description": "Explicit FIR order (overrides transition_bandwidth)."},
                    "max_ripple": {"type": "number",
                                   "description": "Passband/stopband deviation used to size "
                                                  "the order from the transition width "
                                                  "(default 0.002)."},
                    "phase": {"type": "string",
                              "description": "FIR phase when antialias='fir': 'zero' "
                                             "(default, non-causal symmetric) or 'minimum' "
                                             "(causal minimum-phase)."},
                },
                "required": ["sfreq"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shift_events",
            "description": "Shift event-marker timing to correct for a fixed acquisition or "
                           "display latency (positive ms = later in time). Shifts the "
                           "event times only, not the EEG samples. By default all markers "
                           "are shifted; scope with descriptions and/or code_ranges. Set "
                           "rounding to quantize the shift to whole samples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {"type": "number",
                           "description": "Milliseconds to add to each event onset (e.g. 26)."},
                    "descriptions": {"type": "array", "items": {"type": "string"},
                                     "description": "Optional: only shift markers with these exact labels."},
                    "code_ranges": {"type": "array",
                                    "items": {"type": "array", "items": {"type": "number"}},
                                    "description": "List of [min,max] inclusive integer code "
                                                   "ranges to shift (e.g. [[10,19],[30,39]]). "
                                                   "When empty or omitted, all markers are "
                                                   "shifted."},
                    "rounding": {"type": "string",
                                 "description": "Quantize the shift to whole samples at the "
                                                "current rate: 'none' (default, exact ms), "
                                                "'nearest', 'earlier' (floor), or 'later' (ceil). "
                                                "IMPORTANT: if the request says to quantize/round "
                                                "the shift to whole samples ('round to samples', "
                                                "'floor'/'earlier', 'round earlier'), set the "
                                                "matching mode ('earlier' = floor); do NOT leave "
                                                "'none' when rounding is specified. Keep 'none' "
                                                "only when no rounding is requested."},
                },
                "required": ["ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reference",
            "description": "Re-reference the loaded recording. 'average', 'mastoids' "
                           "(linked-mastoid, auto-detects M1/M2/A1/A2/TP9/TP10), or "
                           "'channels' with an explicit list. An AVERAGE reference over all "
                           "sites (e.g. 'average of the 33 sites') is ref_type='average' with "
                           "NO channels arg — it already averages every EEG channel; do not "
                           "pass a channel count. Use add_implicit to add back a non-recorded "
                           "online reference (e.g. M1) before linked-mastoid referencing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_type": {"type": "string",
                                 "description": "One of: average, mastoids, channels."},
                    "channels": {"type": "array", "items": {"type": "string"},
                                 "description": "List of channel NAMES (strings), only for "
                                 "ref_type='mastoids' or 'channels'. Never a count: 'average "
                                 "of the 33 sites' is ref_type='average' (all EEG channels), "
                                 "NOT channels=33. Omit entirely for an average reference."},
                    "add_implicit": {"type": "array", "items": {"type": "string"},
                                     "description": "Channels to add back as zeros before referencing (implicit online reference)."},
                },
                "required": ["ref_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_line_noise",
            "description": "Remove power-line noise (50 or 60 Hz) with Zapline, a spatial "
                           "filter that removes the line artifact without the waveform "
                           "distortion a notch filter introduces. Use instead of a notch "
                           "when preserving the signal near the line frequency matters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_freq": {"type": "number",
                                  "description": "Line frequency in Hz (50 or 60)."},
                    "n_remove": {"type": "number",
                                 "description": "Number of line components to remove (default 1)."},
                },
                "required": ["line_freq"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_faster",
            "description": "FASTER automatic artifact detection: flags bad channels (and "
                           "optionally bad epochs) by z-scoring statistical metrics and "
                           "thresholding at thres standard deviations. Marks bad channels "
                           "for interpolate_bads to repair; reports bad-epoch indices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thres": {"type": "number",
                              "description": "Z-score threshold in SD (default 3)."},
                    "detect_epochs": {"type": "boolean",
                                      "description": "Also detect bad epochs (default true)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "interpolate_bads",
            "description": "Spherical-spline interpolate bad channels. Uses channels "
                           "marked by run_prep unless an explicit list is given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {"type": "array", "items": {"type": "string"},
                                 "description": "Optional explicit bad-channel names to add and repair."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_artifacts_extreme_value",
            "description": "Flag epochs whose signal leaves a voltage range (uV) within a time "
                           "window, on any listed channel (ERPLAB simple voltage threshold / "
                           "pop_artextval). Flags accumulate across detection passes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {"type": "array",
                                 "description": "Channel names, or 1-based indices as EEGLAB "
                                                "numbers them.",
                                 "items": {"type": ["string", "integer"]}},
                    "threshold_min": {"type": "number", "description": "Lower voltage bound (uV)."},
                    "threshold_max": {"type": "number", "description": "Upper voltage bound (uV)."},
                    "tmin_ms": {"type": "number", "description": "Window start (ms)."},
                    "tmax_ms": {"type": "number", "description": "Window end (ms)."},
                },
                "required": ["channels", "threshold_min", "threshold_max", "tmin_ms", "tmax_ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_artifacts_moving_window",
            "description": "Flag epochs whose peak-to-peak amplitude in a sliding window exceeds "
                           "a threshold (uV) on any listed channel (ERPLAB pop_artmwppth). Used "
                           "for general artifacts and for blinks on a VEOG channel. Flags "
                           "accumulate across detection passes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {"type": "array",
                                 "description": "Channel names, or 1-based indices as EEGLAB "
                                                "numbers them.",
                                 "items": {"type": ["string", "integer"]}},
                    "threshold": {"type": "number", "description": "Peak-to-peak threshold (uV)."},
                    "tmin_ms": {"type": "number", "description": "Window start (ms)."},
                    "tmax_ms": {"type": "number", "description": "Window end (ms)."},
                    "window_size_ms": {"type": "number", "description": "Sliding window size (ms)."},
                    "window_step_ms": {"type": "number", "description": "Sliding window step (ms)."},
                },
                "required": ["channels", "threshold", "tmin_ms", "tmax_ms",
                             "window_size_ms", "window_step_ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_artifacts_step",
            "description": "Flag epochs containing a step-like shift (saccade / horizontal eye "
                           "movement): within a sliding window, |mean(first half) - mean(second "
                           "half)| exceeds a threshold (uV) on any listed channel (ERPLAB "
                           "pop_artstep). Flags accumulate across detection passes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {"type": "array",
                                 "description": "Channel names, or 1-based indices as EEGLAB "
                                                "numbers them.",
                                 "items": {"type": ["string", "integer"]}},
                    "threshold": {"type": "number", "description": "Step threshold (uV)."},
                    "tmin_ms": {"type": "number", "description": "Window start (ms)."},
                    "tmax_ms": {"type": "number", "description": "Window end (ms)."},
                    "window_size_ms": {"type": "number", "description": "Sliding window size (ms)."},
                    "window_step_ms": {"type": "number", "description": "Sliding window step (ms)."},
                },
                "required": ["channels", "threshold", "tmin_ms", "tmax_ms",
                             "window_size_ms", "window_step_ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_flagged_epochs",
            "description": "Drop the epochs flagged by the detect_artifacts_* passes, so ERP "
                           "averages exclude them (ERPLAB's 'good' averaging criterion). Call "
                           "after all detection passes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_bins",
            "description": "Assign time-locking events to bins (ERPLAB BINLISTER) by event "
                           "code, from the loaded continuous recording. Stores the bin-tagged "
                           "events for create_epochs. Use before create_epochs for "
                           "condition-averaged ERPs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bins": {
                        "type": "array",
                        "description": "List of bins, each {\"label\": <str>, \"codes\": "
                                       "[[lo, hi], ...] and/or [c1, c2, ...]}. Codes are "
                                       "inclusive ranges and/or bare integers, taken from the "
                                       "request/event scheme.",
                        "items": {"type": "object"},
                    },
                    "require_following_code": {
                        "type": "integer",
                        "description": "If set, bin a time-locking event only when the "
                                       "immediately following event equals this code (response "
                                       "contingency, e.g. 201 = correct response).",
                    },
                    "engine": {
                        "type": "string",
                        "enum": ["mne", "erplab"],
                        "description": "Event-latency convention, which differs after a "
                                       "downsample (fractional latencies). 'mne' = MNE's "
                                       "rounded annotation samples. 'erplab' = the .set's "
                                       "native latencies truncated as ERPLAB does; use when "
                                       "reproducing an ERPLAB/EEGLAB pipeline bit-exactly.",
                    },
                },
                "required": ["bins"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_epochs",
            "description": "Epoch the recording. Uses create_bins output if present; else "
                           "event-based from annotations by default; "
                           "pass fixed_length_sec for resting-state fixed-length epochs. "
                           "Stored for downstream ERP/AutoReject/connectivity tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tmin": {"type": "number", "description": "Epoch start (s) relative to event."},
                    "tmax": {"type": "number", "description": "Epoch end (s) relative to event."},
                    "event_id": {"type": "string",
                                 "description": "Optional single annotation label to epoch."},
                    "fixed_length_sec": {"type": "number",
                                         "description": "If set, make fixed-length epochs of this duration (resting state)."},
                    "baseline": {"type": ["boolean", "array"], "items": {"type": "number"},
                                 "description": "Pre-stimulus baseline correction: true (apply the "
                                 "pre-stimulus window) / false (off), OR an explicit [lo, hi] window "
                                 "(MNE convention; ms or seconds)."},
                    "engine": {
                        "type": "string",
                        "enum": ["mne", "erplab"],
                        "description": "Baseline-window convention. 'mne' = (None, 0), which "
                                       "includes the t=0 sample. 'erplab' = baseline stops just "
                                       "before t=0; use when reproducing an ERPLAB/EEGLAB "
                                       "pipeline bit-exactly.",
                    },
                },
                "required": ["tmin", "tmax"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_psd",
            "description": "Compute power spectral density and per-band (delta/theta/"
                           "alpha/beta/gamma) power for the loaded recording.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fmin": {"type": "number", "description": "Low frequency bound (Hz)."},
                    "fmax": {"type": ["number", "null"], "description": "High frequency bound (Hz); null uses Nyquist."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ica",
            "description": "Fit ICA on the loaded recording and optionally label "
                           "components (brain/eye/muscle/heart/line/other) with ICLabel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_components": {"type": "integer",
                                     "description": "Number of ICA components to fit."},
                    "algorithm": {
                        "type": "string",
                        "enum": ["fastica", "infomax", "extended-infomax", "picard"],
                        "description": (
                            "ICA algorithm to use. Choose the one named in the request. "
                            "There is no default — this parameter is required. "
                            "Options: 'fastica', 'infomax', 'extended-infomax', 'picard'."
                        ),
                    },
                    "label_components": {"type": "boolean",
                                         "description": "Run ICLabel auto-labelling."},
                    "engine": {
                        "type": "string",
                        "enum": ["mne", "eeglab", "eeglab-python"],
                        "description": (
                            "Which implementation to fit with, when the request names one. "
                            "'mne' (default) = MNE's infomax with MNE's defaults; 'eeglab' = "
                            "the real runica.m run in Octave; 'eeglab-python' = EEGLAB's "
                            "sphering and learning-rate schedule in pure Python. Do not guess "
                            "this — set it only if the request asks for a specific toolbox's ICA."
                        ),
                    },
                    "picks": {
                        "type": "array",
                        "items": {"type": ["string", "number"]},
                        "description": (
                            "Channels to fit ICA on, as names or 1-based indices. Default: "
                            "every EEG channel. Set it when the request names a channel subset."
                        ),
                    },
                    "random_state": {"type": "integer",
                                     "description": "Random seed for the decomposition."},
                    "max_iter": {
                        "type": ["integer", "string"],
                        "description": "Maximum training iterations ('auto' or an integer)."},
                    "iclabel_preprocessing": {
                        "type": "boolean",
                        "description": (
                            "Default true: fit a copy band-passed 1-100 Hz and average-"
                            "referenced, which is what ICLabel expects. Set false to decompose "
                            "exactly the data as loaded, which is what a published pipeline "
                            "specifying its own ICA input requires. Labelling requires true."
                        ),
                    },
                    "l_rate": {
                        "type": "number",
                        "description": (
                            "Infomax learning rate. Do not guess this — set it only if the "
                            "request names a learning rate or asks for a specific toolbox's "
                            "schedule. Omitted, each engine uses its own default. Infomax only. "
                            "Note EEGLAB has two different 'defaults': runica.m's internal "
                            "0.00065/log(channels), and the 0.001 that pop_runica hardcodes for "
                            "every published pipeline."
                        ),
                    },
                    "anneal_step": {
                        "type": "number",
                        "description": (
                            "Infomax annealing factor, e.g. 0.98. Do not guess this — set it "
                            "only if the request names an annealing schedule. Same restrictions "
                            "as l_rate."
                        ),
                    },
                },
                "required": ["algorithm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_ica_eeglab",
            "description": "Load a precomputed ICA decomposition from an EEGLAB .set file "
                           "into the session, to apply externally computed ICA weights "
                           "instead of re-fitting. Pairs with apply_ica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string",
                                 "description": "Path to an EEGLAB .set containing ICA weights."},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_ica",
            "description": "Remove ICA components (ocular/artifact correction) from the "
                           "loaded recording in place, back-projecting the remaining "
                           "components. Requires an ICA in the session (run_ica or "
                           "load_ica_eeglab). Supply EITHER `components` (numeric indices, "
                           "when known) OR `label` (an artifact class, when the indices are "
                           "not known yet, e.g. right after a fresh run_ica).",
            "parameters": {
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "number"},
                                   "description": "1-based ICA component numbers to remove "
                                                  "(e.g. [3,4]). Use this when you know the "
                                                  "specific indices."},
                    "label": {"type": "string",
                              "description": "Artifact CLASS to remove (e.g. 'eye' or "
                                             "'ocular') when the numeric indices are not "
                                             "known. Resolved to concrete components at run "
                                             "time from the data (ICLabel, falling back to "
                                             "EOG correlation for ocular labels). Use this "
                                             "instead of guessing indices after a fresh "
                                             "run_ica."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_ica_component",
            "description": "Show detailed properties (topography, epochs image, ERP/time course, "
                           "power spectrum) of ICA component(s) so the EXPERT can decide which are "
                           "artifacts. A REVIEW aid that removes nothing; after inspecting, the "
                           "expert calls apply_ica(components=[...]). Requires an ICA in the "
                           "session (run_ica first).",
            "parameters": {
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "number"},
                                   "description": "1-based ICA component number(s) to inspect "
                                                  "(same numbering as run_ica's labels and "
                                                  "apply_ica), e.g. [1] or [1,7]."},
                },
                "required": ["components"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_ica",
            "description": "Scan view for ICA triage: renders the full component topography grid "
                           "(ica.plot_components) AND returns an EOG-correlation table (|Pearson r| "
                           "of each component's source vs VEOG/HEOG, 1-based, sorted by r_max "
                           "descending) plus ICLabel labels where available. Use this BEFORE "
                           "inspect_ica_component to see the whole line-up and identify candidate "
                           "ocular components (frontal topography + high r_max = eye artifact). "
                           "REVIEW ONLY -- removes nothing; the expert then calls "
                           "apply_ica(components=[...]). Requires an ICA in the session "
                           "(run_ica first).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_channel_types",
            "description": "Set channel types on the loaded recording, e.g. mark EOG/"
                           "auxiliary channels that were imported as EEG. Type a channel "
                           "'eog'/'misc' to exclude it from scalp-position steps (component "
                           "topographies, ICLabel) and from the default ICA fit. The expert "
                           "specifies this mapping; types are never inferred.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mapping": {
                        "type": "object",
                        "description": "Map of channel name -> type (eeg, eog, ecg, emg, "
                                       "misc, stim). E.g. {\"HEOG_left\": \"eog\", "
                                       "\"HEOG_right\": \"eog\", \"VEOG_lower\": \"eog\"}.",
                    },
                },
                "required": ["mapping"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "derive_bipolar_eog",
            "description": "Create bipolar EOG channels from monopolar leads. Specify heog "
                           "and veog each as [anode, cathode]. Diagnostic channels; the "
                           "original channels are kept.",
            "parameters": {
                "type": "object",
                "properties": {
                    "heog": {"type": "array", "items": {"type": "string"},
                             "description": "[anode, cathode] channel names for the HEOG bipolar."},
                    "veog": {"type": "array", "items": {"type": "string"},
                             "description": "[anode, cathode] channel names for the VEOG bipolar."},
                    "heog_name": {"type": "string",
                                  "description": "Name for the created HEOG channel "
                                                 "(default 'HEOG'). Set it when the request "
                                                 "names the bipolar, e.g. '(corr) HEOG'."},
                    "veog_name": {"type": "string",
                                  "description": "Name for the created VEOG channel "
                                                 "(default 'VEOG')."},
                },
                "required": ["heog", "veog"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_erp",
            "description": "Epoch around annotated events and average to an ERP. "
                           "Requires events/annotations in the recording.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tmin": {"type": "number", "description": "Epoch start (s, relative to event)."},
                    "tmax": {"type": "number", "description": "Epoch end (s, relative to event)."},
                    "event_id": {"type": "string",
                                 "description": "Optional single annotation label to average."},
                    "plot_style": {"type": "string",
                                   "description": "'butterfly' (default, all-channel overlay with "
                                                  "GFP) or 'channels' (GFP-peak channel only, "
                                                  "window shaded). Use 'channels' when the user "
                                                  "asks for a channel-focused or single-channel view."},
                },
                "required": ["tmin", "tmax"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_difference_erp",
            "description": "Average two conditions separately and return the "
                           "difference wave (A minus B). Each condition is one annotation "
                           "label or a list of labels pooled together. Requires annotated events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id_a": {"type": ["string", "array"], "items": {"type": "string"},
                                   "description": "Label or list of labels for condition A (minuend)."},
                    "event_id_b": {"type": ["string", "array"], "items": {"type": "string"},
                                   "description": "Label or list of labels for condition B (subtrahend)."},
                    "tmin": {"type": "number", "description": "Epoch start (s, relative to event)."},
                    "tmax": {"type": "number", "description": "Epoch end (s, relative to event)."},
                    "channels": {"type": "array", "items": {"type": "string"},
                                 "description": "Channel(s) to show in 'channels' plot_style (e.g. "
                                                "['PO8']). Defaults to the GFP-peak channel if "
                                                "plot_style='channels' and this is omitted."},
                    "plot_style": {"type": "string",
                                   "description": "'butterfly' (default, all-channel overlay with "
                                                  "GFP) or 'channels' (named channel(s) only, "
                                                  "tmin-tmax window shaded). Use 'channels' for a "
                                                  "legible single-channel view (e.g. 'show the N170 "
                                                  "at PO8')."},
                },
                "required": ["event_id_a", "event_id_b", "tmin", "tmax"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_component",
            "description": "Score the most recent ERP / difference wave: mean or peak "
                           "amplitude (and latency) in a time window at given channels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tmin": {"type": "number",
                             "description": "Window start in SECONDS. If the request gives the "
                                            "window in milliseconds, CONVERT: 110 ms -> 0.11, "
                                            "130 ms -> 0.13. Never pass the raw ms number."},
                    "tmax": {"type": "number",
                             "description": "Window end in SECONDS (convert from ms: 150 ms -> "
                                            "0.15, 200 ms -> 0.20)."},
                    "channels": {"type": "array", "items": {"type": "string"},
                                 "description": "Channel names to average over."},
                    "mode": {"type": "string",
                             "description": "'mean' (mean amplitude) or 'peak' (signed peak + latency)."},
                    "engine": {"type": "string",
                               "description": "Window convention: 'mne' keeps samples inside "
                                              "[tmin, tmax]; 'erplab' snaps each edge to the "
                                              "NEAREST sample (ERPLAB closest()), which can "
                                              "include a sample just outside the window. State "
                                              "it only if the request names a toolbox."},
                    "baseline": {"type": "array", "items": {"type": "number"},
                                 "description": "Optional [start_ms, stop_ms] baseline re-applied "
                                                "to the ERP before measuring, as ERPLAB's "
                                                "'meanbl' measure does."},
                    "plot_style": {"type": "string",
                                   "description": "'channels' (default for measure_component -- the "
                                                  "named channel(s) as traces with the tmin-tmax "
                                                  "window shaded, the legible view when measuring "
                                                  "'the N170 at PO8') or 'butterfly' (full all-channel "
                                                  "overlay)."},
                },
                "required": ["tmin", "tmax", "channels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_autoreject",
            "description": "Clean epochs with the AutoReject algorithm and report "
                           "rejection statistics. Builds 1-s epochs if none exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_interpolate": {"type": "integer",
                                      "description": "Max channels to interpolate per epoch."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_asr",
            "description": "Artifact Subspace Reconstruction (EEGLAB clean_rawdata/ASR): "
                           "removes high-amplitude transient artifacts and replaces the "
                           "loaded recording with the cleaned version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cutoff": {"type": "number",
                               "description": "Rejection threshold in SDs (lower = more aggressive)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_prep",
            "description": "PREP-style robust bad-channel detection (pyprep). Lists noisy/"
                           "flat/deviant channels by category without modifying the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_freq": {"type": "number",
                                  "description": "Power-line frequency in Hz (50 or 60)."},
                },
                "required": ["line_freq"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_connectivity",
            "description": "Spectral connectivity across channels in a frequency band "
                           "(EEGLAB SIFT-equivalent). Returns a heatmap and mean strength.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string",
                               "description": "Connectivity metric: wpli, pli, coh, plv, or imcoh."},
                    "fmin": {"type": "number", "description": "Band low edge (Hz)."},
                    "fmax": {"type": "number", "description": "Band high edge (Hz)."},
                },
                "required": ["fmin", "fmax"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_fooof",
            "description": "Parameterise the power spectrum into aperiodic (1/f) and "
                           "periodic (peaks) components with FOOOF/specparam. Reports the "
                           "aperiodic exponent, offset, fit R^2, and detected peaks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fmin": {"type": "number", "description": "Fit range low edge (Hz)."},
                    "fmax": {"type": "number", "description": "Fit range high edge (Hz)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_microstates",
            "description": "EEG microstate analysis via modified k-means (pycrostates). "
                           "Returns the microstate topographies and global explained variance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_states": {"type": "integer",
                                 "description": "Number of microstate maps to fit."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_break_segments",
            "description": "Delete break-period segments from the loaded CONTINUOUS "
                           "recording, in place (ERPLAB pop_erplabDeleteTimeSegments). "
                           "Removes stretches between successive stimulus event codes "
                           "whose gap is >= time_threshold_ms, preserving a buffer of "
                           "data around the flanking codes; each interior deletion "
                           "inserts a boundary. A cleaning step on continuous data. All "
                           "parameters are required -- take the threshold, buffers, and "
                           "code ranges from the request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_threshold_ms": {"type": "number",
                                          "description": "Minimum gap (ms) between successive "
                                                         "use-codes to count as a break and be "
                                                         "deleted."},
                    "start_buffer_ms": {"type": "number",
                                        "description": "Data (ms) kept after the code preceding a "
                                                       "break."},
                    "end_buffer_ms": {"type": "number",
                                      "description": "Data (ms) kept before the code following a "
                                                     "break."},
                    "use_code_ranges": {"type": "array",
                                        "items": {"type": "array", "items": {"type": "number"}},
                                        "description": "List of [min,max] inclusive event-code "
                                                       "ranges that delimit trials (e.g. "
                                                       "[[10,19],[30,39]]). Gaps are measured "
                                                       "between these codes."},
                },
                "required": ["time_threshold_ms", "start_buffer_ms",
                             "end_buffer_ms", "use_code_ranges"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "continuous_artifact_detect",
            "description": "Detect and delete segments of the loaded CONTINUOUS recording "
                           "whose peak-to-peak amplitude exceeds a threshold within a "
                           "moving window (ERPLAB pop_continuousartdet + eeg_eegrej). "
                           "Semi-automatic pre-ICA cleaning for large muscle artifacts / "
                           "extreme voltage offsets. ampth/winms/stepms and the channel "
                           "ranges to scan are per-recording (hand-tuned by visual "
                           "inspection) and have NO defaults, so all must be supplied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ampth": {"type": "number",
                              "description": "Peak-to-peak amplitude threshold in microvolts; a "
                                             "window is flagged if any scanned channel exceeds it."},
                    "winms": {"type": "number",
                              "description": "Moving-window width in ms."},
                    "stepms": {"type": "number",
                               "description": "Moving-window step in ms."},
                    "channel_ranges": {"type": "array",
                                       "items": {"type": "array", "items": {"type": "number"}},
                                       "description": "List of [min,max] inclusive, 1-based "
                                                      "channel-number ranges to scan for "
                                                      "artifacts (e.g. [[1,64]] for channels 1 to "
                                                      "64). A window is flagged if any scanned "
                                                      "channel exceeds ampth."},
                },
                "required": ["ampth", "winms", "stepms", "channel_ranges"],
            },
        },
    },
]


def call_tool(name: str, args: dict) -> dict:
    """Safely dispatch a tool call, capturing any exception as an error dict."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return fn(**(args or {}))
    except Exception as exc:
        return {"ok": False, "error": str(exc),
                "traceback": traceback.format_exc(limit=3)}
