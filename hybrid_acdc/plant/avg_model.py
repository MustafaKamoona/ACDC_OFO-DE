from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .profiles import expand_24h_to_time, add_step_events


@dataclass
class SimSignals:
    t: np.ndarray
    vdc: np.ndarray
    vdc_ref: np.ndarray
    p_pv: np.ndarray
    p_wind: np.ndarray
    p_dc: np.ndarray
    pA: np.ndarray
    pB: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    alpha: np.ndarray
    pref_total: np.ndarray
    p_grid: np.ndarray
    id1: np.ndarray
    id2: np.ndarray
    iq1: np.ndarray
    iq2: np.ndarray
    vA_rms: np.ndarray
    vB_rms: np.ndarray


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def simulate_averaged(
    cfg: Dict[str, Any],
    controller: Any,
    seed: int = 0,
) -> SimSignals:
    """Run the averaged (fast RMS) simulation.

    Model: DC-bus energy balance + first-order VSC P/Q tracking.

    Notes:
    - This is intentionally lightweight and reproducible for conference work.
    - The control coupling is explicit so PID/OFO/OFO-NN changes affect dynamics.
    """
    rng = np.random.default_rng(seed)

    sim = cfg['sim']
    dt = float(sim['dt'])
    t_end = float(sim['t_end'])
    t = np.arange(0.0, t_end + 1e-12, dt)

    compress = float(sim.get('compress_24h_to_seconds', t_end))

    # Grid / line model for RMS voltage constraint (very lightweight)
    grid = cfg.get('grid', {})
    v_phase_rms_nom = float(grid.get('v_phase_rms_nom', 230.0))
    f_grid = float(sim.get('f_grid', 50.0))
    w_grid = 2.0 * np.pi * f_grid
    lines = cfg.get('lines', {})
    zA = complex(float(lines.get('PCC_A', {}).get('R', 0.08)), w_grid*float(lines.get('PCC_A', {}).get('L', 1.8e-3)))
    zB = complex(float(lines.get('PCC_B', {}).get('R', 0.10)), w_grid*float(lines.get('PCC_B', {}).get('L', 2.2e-3)))
    ZA = abs(zA)
    ZB = abs(zB)

    # DC bus parameters
    dc = cfg['dc_bus']
    vdc_nom = float(dc['vdc_nom'])
    Cdc = float(dc['Cdc'])
    vdc_min = float(dc.get('vdc_min', vdc_nom - 8.0))
    vdc_max = float(dc.get('vdc_max', vdc_nom + 8.0))

    # VSC dynamics (simple first order)
    vsc = cfg['vsc']
    tau_p = float(vsc.get('tau_p', 0.15))  # s
    tau_q = float(vsc.get('tau_q', 0.20))

    # Secondary power exchange limits
    sec = cfg['secondary']
    P_import_max = float(sec.get('P_import_max_kw', 500.0)) * 1e3
    P_export_max = float(sec.get('P_export_max_kw', 500.0)) * 1e3
    enable_export = bool(sec.get('enable_export', True))

    # Profiles (kW -> W)
    prof = cfg['profiles']
    pv_kw = prof['pv_24h_kw']
    wind_kw = prof['wind_24h_kw']
    dc_kw = prof['dc_load_24h_kw']
    A_kw = prof['ac_loadA_24h_kw']
    B_kw = prof['ac_loadB_24h_kw']

    p_pv = expand_24h_to_time(pv_kw, t, compress) * 1e3
    p_wind = expand_24h_to_time(wind_kw, t, compress) * 1e3
    p_dc = expand_24h_to_time(dc_kw, t, compress) * 1e3
    pA = expand_24h_to_time(A_kw, t, compress) * 1e3
    pB = expand_24h_to_time(B_kw, t, compress) * 1e3

    # Optional step events
    # Accept both short keys (pv_steps, ...) and the explicit YAML keys (pv_steps, dc_load_steps, ...)
    step_key_map = [
        (('pv_steps', 'pv_24h_steps'), 'p_pv'),
        (('wind_steps', 'wind_24h_steps'), 'p_wind'),
        (('dc_steps', 'dc_load_steps'), 'p_dc'),
        (('acA_steps', 'ac_loadA_steps'), 'pA'),
        (('acB_steps', 'ac_loadB_steps'), 'pB'),
    ]

    for keys, target in step_key_map:
        events = None
        for kname in keys:
            if kname in prof:
                events = prof.get(kname, [])
                break
        if not events:
            continue
        if target == 'p_pv':
            p_pv = add_step_events(p_pv, t, events)
        elif target == 'p_wind':
            p_wind = add_step_events(p_wind, t, events)
        elif target == 'p_dc':
            p_dc = add_step_events(p_dc, t, events)
        elif target == 'pA':
            pA = add_step_events(pA, t, events)
        elif target == 'pB':
            pB = add_step_events(pB, t, events)

    # Reactive power proportional to P (simple)
    q_over_p = float(prof.get('Q_over_P', 0.20))
    qA = q_over_p * pA
    qB = q_over_p * pB

    # State arrays
    n = len(t)
    vdc = np.zeros(n)
    vdc_ref = np.full(n, vdc_nom)
    p1 = np.zeros(n)
    p2 = np.zeros(n)
    q1 = np.zeros(n)
    q2 = np.zeros(n)
    alpha_arr = np.zeros(n)
    pref_total_arr = np.zeros(n)
    p_grid = np.zeros(n)
    id1 = np.zeros(n)
    id2 = np.zeros(n)
    iq1 = np.zeros(n)
    iq2 = np.zeros(n)
    vA_rms = np.full(n, v_phase_rms_nom)
    vB_rms = np.full(n, v_phase_rms_nom)

    # init
    vdc[0] = vdc_nom
    controller.reset(vdc0=vdc_nom)

    # Simple mapping constants: P ~ k_id * id, Q ~ k_iq * iq
    k_id = float(vsc.get('k_id', 1.0e3))   # W per A (abstract scaling)
    k_iq = float(vsc.get('k_iq', 1.0e3))

    # main loop
    for k in range(n - 1):
        meas = {
            't': float(t[k]),
            'vdc': float(vdc[k]),
            'vdc_ref': float(vdc_ref[k]),
            'p_pv': float(p_pv[k]),
            'p_wind': float(p_wind[k]),
            'p_dc': float(p_dc[k]),
            'pA': float(pA[k]),
            'pB': float(pB[k]),
            'p1': float(p1[k]),
            'p2': float(p2[k]),
            'q1': float(q1[k]),
            'q2': float(q2[k]),
        }

        u = controller.step(meas, dt=dt)
        alpha = float(u['alpha'])
        pref_total = float(u['pref_total'])
        q1_ref = float(u.get('q1_ref', qA[k]))
        q2_ref = float(u.get('q2_ref', qB[k]))

        # Bound power exchange
        if not enable_export:
            pref_total = max(pref_total, 0.0)
        pref_total = _clip(pref_total, -P_export_max, P_import_max)
        alpha = _clip(alpha, 0.0, 1.0)

        # Split between VSCs
        pref1 = alpha * pref_total + pA[k]
        pref2 = (1.0 - alpha) * pref_total + pB[k]

        # VSC first-order tracking
        p1[k+1] = p1[k] + (dt / tau_p) * (pref1 - p1[k])
        p2[k+1] = p2[k] + (dt / tau_p) * (pref2 - p2[k])
        q1[k+1] = q1[k] + (dt / tau_q) * (q1_ref - q1[k])
        q2[k+1] = q2[k] + (dt / tau_q) * (q2_ref - q2[k])

        # Currents (abstract)
        id1[k+1] = p1[k+1] / k_id
        id2[k+1] = p2[k+1] / k_id
        iq1[k+1] = q1[k+1] / k_iq
        iq2[k+1] = q2[k+1] / k_iq

        # Approximate bus RMS voltages via per-phase drop |Z|*I_rms (simple but effective for constraints)
        # Compute per-phase current magnitudes from three-phase apparent power.
        s1 = np.sqrt(p1[k+1]**2 + q1[k+1]**2)
        s2 = np.sqrt(p2[k+1]**2 + q2[k+1]**2)
        i1_rms = s1 / (3.0 * max(vA_rms[k], 1.0))
        i2_rms = s2 / (3.0 * max(vB_rms[k], 1.0))
        vA_rms[k+1] = max(0.0, v_phase_rms_nom - ZA * i1_rms)
        vB_rms[k+1] = max(0.0, v_phase_rms_nom - ZB * i2_rms)

        # Grid power exchange positive=import to DC via VSCs
        # We define p_grid as pref_total (import/export command), after limits
        p_grid[k] = pref_total

        # DC bus power balance
        p_ac_total = p1[k+1] + p2[k+1] - (pA[k] + pB[k])  # net sent to grid (approx)
        # DC side power needed by AC loads + exchange
        p_needed_from_dc = (pA[k] + pB[k]) + max(0.0, -pref_total)  # if exporting, draw extra
        p_supplied_to_dc = (p_pv[k] + p_wind[k]) + max(0.0, pref_total)  # if importing, adds

        p_net = p_supplied_to_dc - p_dc[k] - p_needed_from_dc
        # Vdc dynamics: C * V * dV/dt = p_net
        vdot = p_net / (Cdc * max(vdc[k], 1.0))
        vdc[k+1] = vdc[k] + dt * vdot
        # Numerical safety clamp (not a controller clamp): prevents runaway values during poor initial gains.
        vdc[k+1] = _clip(vdc[k+1], vdc_nom - 40.0, vdc_nom + 40.0)

        alpha_arr[k] = alpha
        pref_total_arr[k] = pref_total

    # final sample of p_grid
    p_grid[-1] = pref_total_arr[-2] if n > 1 else 0.0

    return SimSignals(
        t=t,
        vdc=vdc,
        vdc_ref=vdc_ref,
        p_pv=p_pv,
        p_wind=p_wind,
        p_dc=p_dc,
        pA=pA,
        pB=pB,
        p1=p1,
        p2=p2,
        q1=q1,
        q2=q2,
        alpha=alpha_arr,
        pref_total=pref_total_arr,
        p_grid=p_grid,
        id1=id1,
        id2=id2,
        iq1=iq1,
        iq2=iq2,
        vA_rms=vA_rms,
        vB_rms=vB_rms,
    )
