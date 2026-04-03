from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pid_reference import PID, PIDParams
from .nn_disturbance import OnlineLinearEstimator, DisturbanceNNConfig


# ======================================================================
# Outer-loop configuration
# ======================================================================


@dataclass
class OuterLoopConfig:
    """Outer-loop limits and nominal settings."""

    vdc_nom: float
    P_import_max: float  # [W] positive (grid -> DC)
    P_export_max: float  # [W] positive magnitude (DC -> grid)


# ======================================================================
# PID outer controller
# ======================================================================


class PIDOuterController:
    """Reference PID outer loop.

    Controlled variable:
      - DC-link voltage vdc (track vdc_ref)

    Manipulated variables (outputs):
      - pref_total : total grid power exchange command [W]
                    positive => import AC->DC
                    negative => export DC->AC
      - alpha      : power-sharing factor (VSC-1 share), 0..1

    Notes:
      - Implements *anti-windup* (conditional integration) because the base PID
        class only saturates the output.
      - Includes a ramp-rate limiter (dp_max) for realistic and stable power
        commands.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        alpha: float,
        outer: OuterLoopConfig,
        dp_max_kw_per_s: float = 5000.0,
    ):
        self.outer = outer
        self.alpha = float(alpha)

        self.pid = PID(
            PIDParams(
                kp=float(kp),
                ki=float(ki),
                u_min=-float(outer.P_export_max),
                u_max=float(outer.P_import_max),
            )
        )

        # Ramp limiter (W/s) to avoid unrealistically sharp grid commands.
        self.pref_prev = 0.0
        self.dp_max = float(dp_max_kw_per_s) * 1e3

    def reset(self, vdc0: float) -> None:  # vdc0 reserved for future use
        self.pid.reset()
        self.pref_prev = 0.0

    def _pid_aw(self, e: float, dt: float) -> float:
        """PID with conditional integration anti-windup."""

        kp = float(self.pid.p.kp)
        ki = float(self.pid.p.ki)

        # candidate (unsaturated)
        u_unsat = kp * e + ki * float(self.pid.int_e)

        u_sat = float(np.clip(u_unsat, float(self.pid.p.u_min), float(self.pid.p.u_max)))

        # conditional integration (classic anti-windup)
        if u_sat == u_unsat:
            self.pid.int_e += float(e) * float(dt)

        return u_sat

    def step(self, meas: dict, dt: float) -> dict:
        e = float(meas["vdc_ref"] - meas["vdc"])

        # PID + anti-windup
        pref_cmd = self._pid_aw(e=e, dt=float(dt))

        # Ramp-rate limiting
        dp = float(np.clip(pref_cmd - self.pref_prev, -self.dp_max * dt, +self.dp_max * dt))
        pref = float(self.pref_prev + dp)
        self.pref_prev = pref

        # safety (should already be enforced by PID saturation)
        pref = float(np.clip(pref, -self.outer.P_export_max, self.outer.P_import_max))

        return {"pref_total": pref, "alpha": float(self.alpha)}


# ======================================================================
# OFO-NN (PID + online disturbance estimator)
# ======================================================================


class OFONNOuterController(PIDOuterController):
    """PID outer loop augmented with a lightweight online disturbance estimator.

    The estimator learns a *lumped* power imbalance term d_hat [kW] and adds it
    to the PID command before saturation/ramp limiting.

    This improves transient tracking under renewable/load steps while keeping
    the structure close to the reference PID.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        alpha: float,
        outer: OuterLoopConfig,
        nn_cfg: DisturbanceNNConfig,
        dp_max_kw_per_s: float = 5000.0,
        d_hat_limit_kw: float = 200.0,
        learn_on_error_v: float = 0.25,
    ):
        super().__init__(
            kp=kp,
            ki=ki,
            alpha=alpha,
            outer=outer,
            dp_max_kw_per_s=dp_max_kw_per_s,
        )
        self.est = OnlineLinearEstimator(n_features=5, cfg=nn_cfg)
        self.d_hat_limit_kw = float(d_hat_limit_kw)
        self.learn_on_error_v = float(learn_on_error_v)

    def _phi(self, meas: dict) -> np.ndarray:
        """Feature vector (kW-normalized)."""

        return np.array(
            [
                1.0,
                float(meas["p_pv"]) / 1e3,
                float(meas["p_wind"]) / 1e3,
                float(meas["p_dc"]) / 1e3,
                (float(meas["pA"]) + float(meas["pB"])) / 1e3,
            ],
            dtype=float,
        )

    def step(self, meas: dict, dt: float) -> dict:
        e = float(meas["vdc_ref"] - meas["vdc"])

        # PID baseline (anti-windup)
        u_pid = float(self._pid_aw(e=e, dt=float(dt)))

        # Disturbance estimate (kW)
        phi = self._phi(meas)
        d_hat_kw = float(np.clip(self.est.predict(phi), -self.d_hat_limit_kw, +self.d_hat_limit_kw))

        # Combine and saturate to power bounds
        pref_cmd = float(u_pid + d_hat_kw * 1e3)
        pref_cmd = float(np.clip(pref_cmd, -self.outer.P_export_max, self.outer.P_import_max))

        # Ramp-rate limiting
        dp = float(np.clip(pref_cmd - self.pref_prev, -self.dp_max * dt, +self.dp_max * dt))
        pref = float(self.pref_prev + dp)
        self.pref_prev = pref

        # Online learning target: net deficit estimate (kW)
        if abs(e) > self.learn_on_error_v:
            target_kw = (
                (float(meas["p_dc"]) + float(meas["pA"]) + float(meas["pB"]))
                - (float(meas["p_pv"]) + float(meas["p_wind"]))
            ) / 1e3
            target_kw = float(np.clip(target_kw, -500.0, 500.0))
            self.est.update(phi, target=float(target_kw))

        return {"pref_total": float(pref), "alpha": float(self.alpha)}
