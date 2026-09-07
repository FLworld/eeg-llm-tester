"""ERPLAB `pop_continuousartdet` (peak-to-peak) reproduction for eeg-llm.

Faithful port of ERPLAB v8's continuous artifact detection as used in ERP CORE
N170 Script #2 (`basicrap.m` -> `eeg_eegrej`), the "semi-automatic" muscle /
extreme-voltage segment removal that prepares continuous data for ICA.

Traced from lucklab/erplab `functions/basicrap.m` and
`pop_functions/pop_continuousartdet.m` (ERPLAB master). Behaviour Script #2 relies
on (it passes only ampth/winms/stepms/chanArray, review off):

  winpnts = floor(winms * fs / 1000);  stepnts = floor(stepms * fs / 1000)
  For each inter-boundary segment [bp1, bp2] (1-based, from `latebound`):
    j = bp1; while j <= bp2 - (winpnts-1):
        t1 = j+1;  t2 = j + winpnts - 1          # NOTE ERPLAB off-by-one:
                                                 # first sample of each step skipped,
                                                 # effective width = winpnts-1 samples
        per channel ch in chanArray: p2p = abs(max - min) over data[ch, t1:t2]
        if any p2p > ampth  -> flag window [t1, t2]   (numChanThreshold=1, strict >)
        j = j + stepnts
  All flagged windows (WinRej) go to eeg_eegrej, which deletes their UNION
  (overlapping windows merge -> contiguous deleted spans) and inserts boundaries.

Defaults confirmed from source and NOT overridden by Script #2:
  threshType='peak-to-peak', numChanThreshold=1, firstdet='on' (speed only),
  shortisi/shortseg/winoffset = []  -> NO join / discard / offset step.

Thresholds (ampth/winms/stepms) in ERP CORE are PER SUBJECT, hand-tuned by visual
inspection and stored in `ICA_Prep_Values_N170.xls` (not on disk). This module is
parameterised on them; it does not embed any subject's values.
"""

from __future__ import annotations

import math

import mne
import numpy as np


# --------------------------------------------------------------------------- #
# Boundary handling (matches basicrap.m `latebound`)
# --------------------------------------------------------------------------- #
def _segment_edges(n_times: int, boundary_samples) -> list[int]:
    """Inter-boundary segment edges as 0-based half-open cut points [0 .. n_times].

    ERPLAB builds latebound = round([0, (boundary_latency - 0.5), pnts]); segment q
    is 1-based samples (latebound(q)+1 .. latebound(q+1)). In 0-based half-open terms
    that is exactly [edge_q, edge_{q+1}), so the edge list below is the direct port
    (boundary_samples are 1-based EEGLAB latencies, typically x.5)."""
    edges = {0, int(n_times)}
    for lat in (boundary_samples or []):
        e = int(round(float(lat) - 0.5))
        if 0 < e < n_times:
            edges.add(e)
    return sorted(edges)


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of half-open [start, end) intervals (merges overlapping/abutting),
    as eeg_eegrej does before deleting."""
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #
def continuous_artifact_detect(raw, ampth, winms, stepms, chan_array=range(31),
                               boundary_samples=None) -> dict:
    """Flag (and describe) continuous-EEG segments to delete, ERPLAB-faithfully.

    raw              : mne.io.Raw (continuous). Data assumed in Volts (MNE standard);
                       compared to ampth after converting to microvolts.
    ampth            : peak-to-peak threshold in microvolts (single value; ERPLAB's
                       1-value form == symmetric p2p).
    winms, stepms    : moving-window width / step in ms.
    chan_array       : 0-based channel indices to scan (ERP CORE Script #2 uses
                       MATLAB 1:31 == Python range(31): scalp + monopolar EOG, NOT
                       the 2 bipolar EOG channels).
    boundary_samples : 1-based EEGLAB boundary latencies to segment on. If None, they
                       are read from raw.annotations ('boundary' / 'BAD boundary').

    Returns a dict with the flagged windows, the merged deleted spans (0-based,
    half-open sample indices), and a summary. Deletes nothing; call apply_deletion().
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("continuous_artifact_detect requires a continuous mne.io.Raw")

    fs = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    chan_array = list(chan_array)

    winpnts = math.floor(winms * fs / 1000.0)   # basicrap: floor(windowms*fs/1000)
    stepnts = math.floor(stepms * fs / 1000.0)
    if winpnts < 2 or stepnts < 1:
        raise ValueError(f"winms/stepms too small for fs={fs}: winpnts={winpnts}, "
                         f"stepnts={stepnts}")

    if boundary_samples is None:
        boundary_samples = [
            float(onset) * fs + 1.0  # annotation onset (s) -> ~1-based sample latency
            for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
            if str(desc).strip().lower() in ("boundary", "bad boundary")
        ]
    edges = _segment_edges(n_times, boundary_samples)

    data_uv = raw.get_data() * 1e6              # Volts -> microvolts (ERPLAB units)
    sub = data_uv[chan_array, :]

    winrej: list[tuple[int, int]] = []          # 0-based half-open [start, end)
    win_p2p: list = []                          # per-channel p2p for each flagged window
    # iterate inter-boundary segments; window loop resets in each (matches basicrap)
    for seg0, seg1 in zip(edges[:-1], edges[1:]):
        bp1 = seg0 + 1                          # 1-based first sample of segment
        bp2 = seg1                              # 1-based last sample of segment
        j = bp1
        while j <= bp2 - (winpnts - 1):
            t1 = j + 1                          # ERPLAB off-by-one
            t2 = j + winpnts - 1
            w = sub[:, t1 - 1:t2]               # 1-based t1..t2 -> py slice [t1-1:t2]
            p2p = w.max(axis=1) - w.min(axis=1)  # abs() unnecessary: max >= min
            if np.any(p2p > ampth):             # strict > , numChanThreshold=1
                winrej.append((t1 - 1, t2))     # py half-open [t1-1, t2)
                win_p2p.append(p2p)             # display-only: which channels drove it
            j += stepnts

    spans = _merge(winrej)
    n_del = sum(e - s for s, e in spans)
    # Per-span triggering channels: for each merged span, the max windowed p2p per
    # channel across the windows composing it; channels over ampth are the triggers
    # (display/diagnostic only -- does not affect detection). ch index k -> raw channel.
    ch_names = raw.ch_names
    triggers_per_span: list = []
    for (S, E) in spans:
        maxp = None
        for (ws, we), p in zip(winrej, win_p2p):
            if ws >= S and we <= E:
                maxp = p if maxp is None else np.maximum(maxp, p)
        if maxp is None:
            triggers_per_span.append([])
            continue
        order = [int(k) for k in np.argsort(maxp)[::-1] if maxp[k] > ampth]
        triggers_per_span.append(
            [(ch_names[chan_array[k]], round(float(maxp[k]), 1)) for k in order])
    return {
        "triggers_per_span": triggers_per_span,  # per span: [(ch_name, p2p_uv), ...]
        "engine": "erplab:pop_continuousartdet",
        "ampth": float(ampth), "winms": float(winms), "stepms": float(stepms),
        "sfreq": fs, "winpnts": winpnts, "stepnts": stepnts,
        "chan_array": chan_array,
        "n_windows_flagged": len(winrej),
        "deleted_spans": spans,                 # 0-based half-open sample indices
        "n_segments": len(spans),
        "n_deleted_samples": int(n_del),
        "n_times": n_times,
        "total_ms_removed": round(n_del / fs * 1000.0, 1),
        "pct_removed": round(100.0 * n_del / n_times, 3) if n_times else 0.0,
    }


def windowed_p2p_envelope(raw, winms, stepms, chan_array=None, boundary_samples=None):
    """Display helper: max-across-channels peak-to-peak per moving window, at window
    centres (seconds). Mirrors continuous_artifact_detect's windowing so the envelope
    lines up with what the detector thresholds -- for plotting the p2p curve against
    ampth. Returns (centres_s, env_uv). Does not affect detection."""
    fs = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    if chan_array is None:
        chan_array = range(len(raw.ch_names))
    chan_array = list(chan_array)
    winpnts = math.floor(winms * fs / 1000.0)
    stepnts = math.floor(stepms * fs / 1000.0)
    if winpnts < 2 or stepnts < 1:
        return [], []
    if boundary_samples is None:
        boundary_samples = [
            float(onset) * fs + 1.0
            for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
            if str(desc).strip().lower() in ("boundary", "bad boundary")
        ]
    edges = _segment_edges(n_times, boundary_samples)
    sub = raw.get_data() * 1e6
    sub = sub[chan_array, :]
    centres, env = [], []
    for seg0, seg1 in zip(edges[:-1], edges[1:]):
        bp1, bp2, j = seg0 + 1, seg1, seg0 + 1
        while j <= bp2 - (winpnts - 1):
            t1, t2 = j + 1, j + winpnts - 1
            w = sub[:, t1 - 1:t2]
            centres.append(((t1 - 1) + t2) / 2.0 / fs)
            env.append(float((w.max(axis=1) - w.min(axis=1)).max()))
            j += stepnts
    return centres, env


def apply_deletion(raw, spans):
    """Return a new Raw with `spans` (0-based half-open) removed and boundary
    annotations at the joins (mne.concatenate_raws inserts 'BAD boundary'), the
    eeg_eegrej equivalent."""
    spans = _merge(list(spans))
    if not spans:
        return raw.copy()
    fs = float(raw.info["sfreq"])
    n = int(raw.n_times)
    keep, prev = [], 0
    for s, e in spans:
        if s > prev:
            keep.append((prev, s))
        prev = max(prev, e)
    if prev < n:
        keep.append((prev, n))
    segs = [raw.copy().crop(tmin=s / fs, tmax=(e - 1) / fs) for s, e in keep]
    return mne.concatenate_raws(segs) if len(segs) > 1 else segs[0]


# --------------------------------------------------------------------------- #
# .set helpers for validation (read boundaries / reconstruct ground-truth spans)
# --------------------------------------------------------------------------- #
def read_set_boundaries(path: str):
    """Return (boundary_latencies_1based, pnts, srate) from an EEGLAB .set via scipy.
    _load_eeglab_struct (tools.py) drops EEG.event, so we read it directly here."""
    import scipy.io as sio
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    EEG = m["EEG"]
    lat = []
    ev = np.atleast_1d(EEG.event) if hasattr(EEG, "event") else []
    for e in ev:
        t = e.type
        ts = t if isinstance(t, str) else str(t)
        if ts.strip().lower() == "boundary":
            lat.append(float(e.latency))
    return sorted(lat), int(EEG.pnts), float(EEG.srate)


def _boundary_events(path: str):
    """Return sorted [(latency_1based, duration_samples)] for boundary events in a
    .set file (via scipy)."""
    import scipy.io as sio
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    EEG = m["EEG"]
    evs = []
    for e in np.atleast_1d(EEG.event):
        t = e.type
        ts = t if isinstance(t, str) else str(t)
        if ts.strip().lower() == "boundary":
            evs.append((float(e.latency), float(getattr(e, "duration", 0) or 0)))
    return sorted(evs)


def reconstruct_deleted_spans(prep2_path: str, prep1_path: str | None = None):
    """Reconstruct deletions in the INPUT timeline from boundary events.

    EEGLAB boundary events carry latency (position in the shortened record, 1-based,
    x.5) and duration (# samples removed there); walking them with a running offset
    maps each cut back to input coordinates.

    IMPORTANT: prep2 carries BOTH Part-1 (break removal) and Part-2 (continuousartdet)
    boundaries. To isolate Part-2's *net* deletions pass prep1_path -- boundaries
    already present in prep1 are subtracted. For a clean subject (e.g. ERP CORE
    sub-002) Part-2 deletes nothing, so this returns ([], 0).

    Returns (spans_0based_halfopen, total_removed)."""
    b2 = _boundary_events(prep2_path)
    if prep1_path is not None:
        b1 = {(round(l, 1), round(d, 1)) for l, d in _boundary_events(prep1_path)}
        b2 = [(l, d) for l, d in b2 if (round(l, 1), round(d, 1)) not in b1]
    spans, cum = [], 0
    for lat, dur in b2:
        d = int(round(dur))
        if d <= 0:
            continue
        start = int(round(lat - 0.5)) + cum     # -> input 0-based sample
        spans.append((start, start + d))
        cum += d
    return spans, cum


def span_overlap(spans_a, spans_b, n_times: int) -> dict:
    """Sample-level comparison of two half-open span lists over a record of n_times.
    Returns intersection / union / a-only / b-only sample counts and overlap ratios."""
    a = np.zeros(n_times, bool)
    b = np.zeros(n_times, bool)
    for s, e in spans_a:
        a[max(0, s):min(n_times, e)] = True
    for s, e in spans_b:
        b[max(0, s):min(n_times, e)] = True
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return {
        "a_samples": int(a.sum()), "b_samples": int(b.sum()),
        "intersection": inter, "union": union,
        "a_only": int(np.sum(a & ~b)), "b_only": int(np.sum(b & ~a)),
        "iou": round(inter / union, 4) if union else 1.0,
        "recall_of_b": round(inter / int(b.sum()), 4) if b.sum() else 1.0,
    }
