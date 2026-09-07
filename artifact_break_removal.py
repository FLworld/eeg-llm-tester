"""ERP CORE N170 Script #2, Part 1: break-period removal.

Faithful reimplementation of ERPLAB's `pop_erplabDeleteTimeSegments` as called in
`Script2_ICA_Prep.m`:

    pop_erplabDeleteTimeSegments(EEG, 'timeThresholdMS',2000,
        'startEventcodeBufferMS',2000, 'endEventcodeBufferMS',2000,
        'ignoreUseEventcodes',[1:80 100:180], 'ignoreUseType','Use')

Removes stretches of continuous EEG in the break periods between trial blocks,
defined as gaps >= timeThresholdMS between successive *use* event codes (the
stimulus codes 1-80 and 100-180). A start/end buffer of data is preserved around
the flanking codes; each interior deletion inserts a `boundary` annotation.

Algorithm traced from lucklab/erplab `erplab_deleteTimeSegments.m`:
  * event latencies are ROUNDED to integer samples (line 161)
  * timeThresholdSample = round(timeThresholdMS * srate/1000); buffers likewise
  * sample 1 and EEG.pnts are appended to the analysed-sample list if absent
  * per gap [t1,t2] with (t2-t1) >= threshold, the rejection window is
      t1 == 1     -> [1,            t2 - endBuffer]      (no buffer at record start)
      t2 == pnts  -> [t1 + startBuffer, pnts]            (no buffer at record end)
      otherwise   -> [t1 + startBuffer, t2 - endBuffer]  (kept iff min < max)
  * eeg_eegrej removes each window [a,b] inclusive; a deletion at the very start
    or very end produces no boundary event (it is not a discontinuity).

Framework-agnostic: takes and returns an MNE Raw (loaded e.g. via
mne.io.read_raw_eeglab, so EEGLAB events live in raw.annotations as numeric-string
descriptions). Does not touch tools.SESSION, so it is trivial to register later.
"""

from __future__ import annotations

import math

import mne
import numpy as np


def _mround(x: float) -> int:
    """MATLAB/ERPLAB round: half away from zero. MNE re-derives the EEGLAB latency
    from a float onset ((L-1)/srate), so a true half-sample latency (L = k+0.5) can
    land ~1e-4 BELOW x.5 and Python's round() would drop it down. EEG latencies here
    are quarter-samples, so anything within a small tol of .5 is a true half and must
    round UP (as MATLAB does). Otherwise defer to normal round."""
    frac = x - math.floor(x)
    if abs(frac - 0.5) < 1e-3:            # a true half-sample latency -> round up
        return int(math.floor(x) + 1)
    return int(round(x))


def _use_code_samples(raw, use_code_ranges) -> list[int]:
    """1-based latencies of events whose numeric code falls in any (lo, hi) inclusive
    range -- ERPLAB's `analyzedSamples` (round(latency), MATLAB half-away-from-zero)."""
    sfreq = float(raw.info["sfreq"])
    out = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        d = str(desc).strip()
        if not (d.isdigit() or (d.startswith("-") and d[1:].isdigit())):
            continue
        code = int(d)
        if any(lo <= code <= hi for lo, hi in use_code_ranges):
            # EEGLAB latency is 1-based: L = onset*srate + 1; ERPLAB rounds it.
            out.append(_mround(onset * sfreq + 1.0))
    return sorted(out)


def _rejection_windows(use_samples, pnts, thr, start_buf, end_buf):
    """ERPLAB rejection windows as 1-based inclusive [a, b] sample pairs."""
    analysed = list(use_samples)
    if not analysed or analysed[0] != 1:
        analysed = [1] + analysed
    if analysed[-1] != pnts:
        analysed = analysed + [pnts]

    windows = []
    last = analysed[0]
    for s in analysed[1:]:
        if abs(s - last) >= thr:
            t1, t2 = last, s
            if t1 == 1:
                win = (1, t2 - end_buf)
            elif t2 == pnts:
                win = (t1 + start_buf, pnts)
            else:
                lo, hi = t1 + start_buf, t2 - end_buf
                win = (lo, hi) if lo < hi else None
            if win is not None:
                windows.append(win)
        last = s
    return windows


def delete_break_segments(raw, time_threshold_ms: float = 2000.0,
                          start_buffer_ms: float = 2000.0,
                          end_buffer_ms: float = 2000.0,
                          use_code_ranges=((1, 80), (100, 180))) -> dict:
    """Delete break-period segments from a continuous MNE Raw, ERPLAB-style.

    Returns a dict with the new Raw and provenance:
      raw, n_rejection_windows, rejection_windows (1-based inclusive, original
      samples), n_boundaries, samples_removed, duration_removed_s,
      n_times_before/after.
    """
    raw.load_data()
    sfreq = float(raw.info["sfreq"])
    pnts = int(raw.n_times)
    thr = int(round(time_threshold_ms * sfreq / 1000.0))
    start_buf = int(round(start_buffer_ms * sfreq / 1000.0))
    end_buf = int(round(end_buffer_ms * sfreq / 1000.0))

    use_samples = _use_code_samples(raw, use_code_ranges)
    windows = _rejection_windows(use_samples, pnts, thr, start_buf, end_buf)

    # Complement of the (1-based inclusive) rejection windows = kept spans, as
    # 0-based half-open [start, stop) sample indices for MNE cropping.
    removed = np.zeros(pnts, bool)
    for a, b in windows:
        lo = max(0, int(a) - 1)          # 1-based inclusive -> 0-based
        hi = min(pnts, int(b))           # inclusive b -> half-open stop
        removed[lo:hi] = True

    keep = ~removed
    if not keep.any():
        raise ValueError("break removal would delete the entire recording")

    # Contiguous kept spans.
    edges = np.diff(keep.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    stops = list(np.where(edges == -1)[0] + 1)
    if keep[0]:
        starts = [0] + starts
    if keep[-1]:
        stops = stops + [pnts]
    spans = list(zip(starts, stops))

    # Crop each kept span and concatenate; boundary annotations at interior joins.
    pieces = []
    for s0, s1 in spans:
        piece = raw.copy().crop(tmin=s0 / sfreq,
                                tmax=(s1 - 1) / sfreq,
                                include_tmax=True)
        pieces.append(piece)
    new_raw = pieces[0]
    if len(pieces) > 1:
        new_raw = mne.concatenate_raws(pieces)

    # A deletion touching sample 0 or the final sample is an edge deletion -> no
    # boundary. Interior deletions each become one 'boundary' annotation at the
    # collapsed join position.
    kept_cumsum = np.cumsum(keep)
    boundary_onsets = []
    for a, b in windows:
        if int(a) <= 1 or int(b) >= pnts:
            continue
        collapsed = (kept_cumsum[int(a) - 2] - 0.5) / sfreq  # EEGLAB -0.5 convention
        boundary_onsets.append(max(0.0, collapsed))

    if boundary_onsets:
        ann = new_raw.annotations
        new_raw.set_annotations(ann + mne.Annotations(
            onset=boundary_onsets,
            duration=[0.0] * len(boundary_onsets),
            description=["boundary"] * len(boundary_onsets),
            orig_time=ann.orig_time))

    samples_removed = int(removed.sum())
    return {
        "raw": new_raw,
        "n_rejection_windows": len(windows),
        "rejection_windows": windows,
        "n_boundaries": len(boundary_onsets),
        "boundary_onsets_s": boundary_onsets,
        "samples_removed": samples_removed,
        "duration_removed_s": samples_removed / sfreq,
        "n_times_before": pnts,
        "n_times_after": int(new_raw.n_times),
        "time_threshold_samples": thr,
        "buffer_samples": (start_buf, end_buf),
    }
