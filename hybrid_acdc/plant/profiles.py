from __future__ import annotations

from typing import List

import numpy as np


def expand_24h_to_time(profile_24: List[float], t: np.ndarray, compress_24h_to_seconds: float) -> np.ndarray:
    """Expand a 24-point hourly profile to a simulation timeline (piecewise constant)."""
    if len(profile_24) != 24:
        raise ValueError('profile_24 must have length 24')
    if compress_24h_to_seconds <= 0:
        raise ValueError('compress_24h_to_seconds must be positive')

    t = np.asarray(t, dtype=float)
    # Map time -> hour index
    hour_len = compress_24h_to_seconds / 24.0
    idx = np.floor(t / hour_len).astype(int)
    idx = np.clip(idx, 0, 23)
    arr = np.asarray(profile_24, dtype=float)
    return arr[idx]


def add_step_events(base: np.ndarray, t: np.ndarray, events: List[dict]) -> np.ndarray:
    """Add step events to a base profile.

    Each event is a dict with keys:
      - 't_s': start time in seconds
      - 'delta': value to add
      - optional 't_e': end time in seconds (if omitted, step persists)
    """
    y = np.array(base, dtype=float, copy=True)
    for ev in events:
        ts = float(ev['t_s'])
        delta = float(ev['delta'])
        te = float(ev.get('t_e', t[-1] + 1.0))
        mask = (t >= ts) & (t < te)
        y[mask] += delta
    return y
