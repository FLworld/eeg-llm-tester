"""ERPLAB-equivalent epoch artifact *detection* (ERP CORE Script #6).

Three deterministic detectors, matching ERPLAB's pop_artextval / pop_artmwppth / pop_artstep.
Because they are deterministic (unlike ICA), they are reproduced here rather than inherited from
the released flags: same data + same parameters must yield the same flagged epochs.

ERPLAB semantics reproduced exactly (each verified against ERP CORE sub-002, 21/21 flags):
  * Data are compared in MICROVOLTS (MNE stores volts -- callers convert).
  * Time windows are inclusive in ms.
  * **ms -> samples uses FLOOR**, for both window size and window step. This matters: rounding
    the step instead coarsens the slide and misses borderline epochs (2 of 21 for sub-002).
  * Detectors ACCUMULATE: ERPLAB ORs each pass into the epoch's global reject flag.

All parameters are required by the callers; nothing task-specific is defaulted here.
"""
import numpy as np


def _win_indices(times_ms, tmin_ms, tmax_ms):
    """Inclusive sample indices for a [tmin, tmax] ms window."""
    return np.where((times_ms >= tmin_ms - 1e-9) & (times_ms <= tmax_ms + 1e-9))[0]


def _sliding(n, sfreq, winsize_ms, winstep_ms):
    """(start, stop) sample pairs for ERPLAB's sliding window. FLOOR on the ms->sample
    conversion; a full-length window is used if the requested one exceeds the segment."""
    w = max(1, min(int(np.floor(winsize_ms / 1000.0 * sfreq)), n))
    s = max(1, int(np.floor(winstep_ms / 1000.0 * sfreq)))
    out = [(i, i + w) for i in range(0, n - w + 1, s)]
    return out or [(0, n)]


def extreme_value(data_uv, times_ms, channels, threshold_min, threshold_max,
                  tmin_ms, tmax_ms):
    """pop_artextval: flag an epoch if ANY sample on ANY listed channel leaves
    [threshold_min, threshold_max] (uV) inside the time window."""
    idx = _win_indices(times_ms, tmin_ms, tmax_ms)
    seg = data_uv[:, channels, :][:, :, idx]
    return ((seg < threshold_min) | (seg > threshold_max)).any(axis=(1, 2))


def moving_window_peak_to_peak(data_uv, times_ms, sfreq, channels, threshold,
                               tmin_ms, tmax_ms, winsize_ms, winstep_ms):
    """pop_artmwppth: slide a window; flag if peak-to-peak (max-min) within any window on any
    listed channel exceeds `threshold` (uV)."""
    idx = _win_indices(times_ms, tmin_ms, tmax_ms)
    seg = data_uv[:, channels, :][:, :, idx]
    flag = np.zeros(seg.shape[0], dtype=bool)
    for a, b in _sliding(len(idx), sfreq, winsize_ms, winstep_ms):
        w = seg[:, :, a:b]
        flag |= ((w.max(axis=2) - w.min(axis=2)) > threshold).any(axis=1)
    return flag


def step_like(data_uv, times_ms, sfreq, channels, threshold,
              tmin_ms, tmax_ms, winsize_ms, winstep_ms):
    """pop_artstep: slide a window; flag if |mean(first half) - mean(second half)| within any
    window on any listed channel exceeds `threshold` (uV). Detects step-like shifts
    (saccades / horizontal eye movements)."""
    idx = _win_indices(times_ms, tmin_ms, tmax_ms)
    seg = data_uv[:, channels, :][:, :, idx]
    flag = np.zeros(seg.shape[0], dtype=bool)
    for a, b in _sliding(len(idx), sfreq, winsize_ms, winstep_ms):
        half = (b - a) // 2
        if half < 1:
            continue
        d = np.abs(seg[:, :, a:a + half].mean(axis=2) - seg[:, :, a + half:b].mean(axis=2))
        flag |= (d > threshold).any(axis=1)
    return flag
