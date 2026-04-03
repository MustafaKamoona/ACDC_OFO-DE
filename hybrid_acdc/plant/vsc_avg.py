from __future__ import annotations

import numpy as np


class VSC_Averaged:
    """Averaged grid-following VSC with inner dq current PI control.

    Grid-aligned frame: v_q = 0.

    Plant (L filter):
      di_d/dt = (v_inv_d - v_d + wL i_q - R i_d)/L
      di_q/dt = (v_inv_q - v_q - wL i_d - R i_q)/L

    Power (three-phase):
      P = 1.5*(v_d*i_d + v_q*i_q)
      Q = 1.5*(v_q*i_d - v_d*i_q)

    In this simplified model, v_d is assumed stiff (grid voltage magnitude).
    """

    def __init__(self, Lf: float, Rf: float, v_mod_max: float, vdc_nom: float, f_grid: float, v_phase_rms: float):
        self.L = float(Lf)
        self.R = float(Rf)
        self.v_mod_max = float(v_mod_max)
        self.f_grid = float(f_grid)
        self.w = 2.0 * np.pi * self.f_grid
        self.v_phase_rms = float(v_phase_rms)
        self.vd = np.sqrt(2.0) * self.v_phase_rms  # phase-to-neutral peak
        self.vq = 0.0

        self.id = 0.0
        self.iq = 0.0

        self.i_int_d = 0.0
        self.i_int_q = 0.0

        # Inner current PI defaults (can be overridden)
        self.kp_i = 2.0
        self.ki_i = 400.0

        self.vdc_nom = float(vdc_nom)

    def set_current_pi(self, kp: float, ki: float) -> None:
        self.kp_i = float(kp)
        self.ki_i = float(ki)

    def step(self, dt: float, vdc: float, id_ref: float, iq_ref: float) -> dict:
        # Current errors
        e_d = id_ref - self.id
        e_q = iq_ref - self.iq

        # Integrators
        self.i_int_d += e_d * dt
        self.i_int_q += e_q * dt

        # PI outputs (voltage commands)
        v_cmd_d = self.kp_i * e_d + self.ki_i * self.i_int_d
        v_cmd_q = self.kp_i * e_q + self.ki_i * self.i_int_q

        # Decoupling terms
        v_inv_d = v_cmd_d + self.vd - self.w * self.L * self.iq
        v_inv_q = v_cmd_q + self.vq + self.w * self.L * self.id

        # Modulation limit (approx): |v_inv| <= m_max * vdc/2
        v_lim = self.v_mod_max * max(vdc, 1e-3) / 2.0
        mag = np.sqrt(v_inv_d**2 + v_inv_q**2)
        if mag > v_lim:
            scale = v_lim / mag
            v_inv_d *= scale
            v_inv_q *= scale

        # Plant integration (Euler)
        did = (v_inv_d - self.vd + self.w * self.L * self.iq - self.R * self.id) / self.L
        diq = (v_inv_q - self.vq - self.w * self.L * self.id - self.R * self.iq) / self.L
        self.id += did * dt
        self.iq += diq * dt

        # Power
        P = 1.5 * (self.vd * self.id + self.vq * self.iq)
        Q = 1.5 * (self.vq * self.id - self.vd * self.iq)

        return {
            "id": self.id,
            "iq": self.iq,
            "vd": self.vd,
            "vq": self.vq,
            "v_inv_d": v_inv_d,
            "v_inv_q": v_inv_q,
            "P": P,
            "Q": Q,
        }
