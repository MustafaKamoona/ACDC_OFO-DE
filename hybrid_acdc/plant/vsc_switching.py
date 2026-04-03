from __future__ import annotations

import numpy as np


def tri_carrier(t: np.ndarray, f_sw: float) -> np.ndarray:
    """Triangular carrier in [-1,1]."""
    T = 1.0 / f_sw
    x = (t / T) % 1.0
    return np.where(x < 0.5, 4*x - 1.0, -4*(x-0.5) + 1.0)


def pwm_phase_voltage(v_ref: np.ndarray, vdc: float, carrier: np.ndarray) -> np.ndarray:
    """SPWM (bipolar) phase output voltage using normalized carrier."""
    m = np.clip(v_ref / (vdc/2.0), -0.999, 0.999)
    gating = (m >= carrier).astype(float)
    return (2*gating - 1.0) * (vdc/2.0)


def simulate_l_filter_switching(
    vdc: float,
    v_phase_rms: float,
    f_grid: float,
    f_sw: float,
    Lf: float,
    Rf: float,
    i_ref_abc: np.ndarray,
    dt: float,
    t_end: float,
) -> dict:
    """Short-window EMT-like switching sim for THD.

    This uses per-phase independent L filter to a stiff grid voltage.
    A simple proportional current controller generates v_ref_abc.
    """
    t = np.arange(0.0, t_end, dt)
    w = 2*np.pi*f_grid
    Vp = np.sqrt(2.0) * v_phase_rms

    # Stiff grid voltages
    va = Vp*np.sin(w*t)
    vb = Vp*np.sin(w*t - 2*np.pi/3)
    vc = Vp*np.sin(w*t + 2*np.pi/3)
    v_grid = np.vstack([va, vb, vc]).T

    carrier = tri_carrier(t, f_sw)

    i = np.zeros((t.size, 3))

    # Simple proportional current control to generate v_ref
    # (In the paper you will mention: switching model used only for THD validation)
    Kp_sw = 8.0  # V/A

    for k in range(1, t.size):
        e = i_ref_abc[k-1] - i[k-1]
        v_ref = v_grid[k-1] + Kp_sw*e
        v_inv = np.array([
            pwm_phase_voltage(v_ref[0], vdc, carrier[k-1]),
            pwm_phase_voltage(v_ref[1], vdc, carrier[k-1]),
            pwm_phase_voltage(v_ref[2], vdc, carrier[k-1]),
        ])
        di = (v_inv - v_grid[k-1] - Rf*i[k-1]) / Lf
        i[k] = i[k-1] + di*dt

    return {"t": t, "i_abc": i, "v_grid_abc": v_grid}


def simulate_switching_window(
    vdc: float,
    v_phase_rms: float,
    f_grid: float,
    f_sw: float,
    L: float,
    R: float,
    i_ref_abc: np.ndarray,
    dt: float,
    t_end: float,
) -> dict:
    """Compatibility wrapper."""
    return simulate_l_filter_switching(
        vdc=vdc,
        v_phase_rms=v_phase_rms,
        f_grid=f_grid,
        f_sw=f_sw,
        Lf=L,
        Rf=R,
        i_ref_abc=i_ref_abc,
        dt=dt,
        t_end=t_end,
    )


def compute_thd(i_abc: np.ndarray, f_grid: float, dt: float, n_harmonics: int = 40) -> float:
    """Compute THD (%) from phase-a current using an FFT.

    Uses the last 10 grid cycles by default (caller should window data appropriately).
    """
    ia = i_abc[:, 0]
    N = len(ia)
    # Remove DC
    ia = ia - np.mean(ia)
    freqs = np.fft.rfftfreq(N, d=dt)
    I = np.abs(np.fft.rfft(ia))
    # Find fundamental bin closest to f_grid
    k1 = int(np.argmin(np.abs(freqs - f_grid)))
    I1 = I[k1] + 1e-12
    # Sum squared harmonics up to n_harmonics
    harm_power = 0.0
    for h in range(2, n_harmonics + 1):
        kh = int(np.argmin(np.abs(freqs - h * f_grid)))
        harm_power += I[kh] ** 2
    thd = np.sqrt(harm_power) / I1 * 100.0
    return float(thd)
