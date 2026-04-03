from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CostWeights:
    # Normalize squared voltage error by the +/-8 V band so costs stay in a
    # reasonable range and the optimizer doesn't get dominated by raw scaling.
    w_vdc: float = 1.0 / (8.0 ** 2)
    w_vac: float = 5.0
    w_alpha: float = 0.05
    w_pgrid: float = 0.01
    w_id: float = 0.002


def compute_cost(
    signals,
    vdc_bounds: tuple[float, float],
    weights: CostWeights,
    vac_bounds: tuple[float, float] | None = None,
) -> float:
    """Compute a scalar cost from simulation signals.

    The cost is designed to be sensitive to controller gains and power sharing.
    """
    t = signals.t
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    e_v = signals.vdc - signals.vdc_ref

    # Soft constraint penalty for violating DC voltage band
    vmin, vmax = vdc_bounds
    vio_low = np.maximum(0.0, vmin - signals.vdc)
    vio_high = np.maximum(0.0, signals.vdc - vmax)
    pen_band = vio_low**2 + vio_high**2

    # Soft constraint penalty for violating AC RMS band (absolute around 230 Vrms)
    pen_vac = 0.0
    if vac_bounds is not None:
        vmin_ac, vmax_ac = vac_bounds
        vioA = np.maximum(0.0, vmin_ac - signals.vA_rms) + np.maximum(0.0, signals.vA_rms - vmax_ac)
        vioB = np.maximum(0.0, vmin_ac - signals.vB_rms) + np.maximum(0.0, signals.vB_rms - vmax_ac)
        pen_vac = np.sum(vioA**2 + vioB**2) * dt

    # Sharing smoothness: discourage aggressive alpha movements
    dalpha = np.diff(signals.alpha, prepend=signals.alpha[0])

    # Grid power smoothness (avoid oscillatory exchange)
    dpgrid = np.diff(signals.p_grid, prepend=signals.p_grid[0])

    # Current magnitude proxy
    i_mag = np.sqrt(signals.id1**2 + signals.iq1**2) + np.sqrt(signals.id2**2 + signals.iq2**2)

    J = (
        weights.w_vdc * np.sum(e_v**2) * dt
        # Strongly enforce the DC safety band (850 +/- 8 V).
        + 2.0e5 * np.sum(pen_band) * dt
        + weights.w_vac * pen_vac
        + weights.w_alpha * np.sum(dalpha**2) * dt
        + weights.w_pgrid * np.sum(dpgrid**2) * dt
        + weights.w_id * np.sum(i_mag**2) * dt
    )
    return float(J)
