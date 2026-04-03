from __future__ import annotations


class DCBus:
    """Simple DC-link capacitor model using power balance."""

    def __init__(self, vdc0: float, Cdc: float, vdc_min: float, vdc_max: float):
        self.vdc = float(vdc0)
        self.Cdc = float(Cdc)
        self.vdc_min = float(vdc_min)
        self.vdc_max = float(vdc_max)
        self._eps = 1e-6

    def step(self, dt: float, P_in_w: float, P_out_w: float) -> float:
        v = self.vdc
        denom = self.Cdc * (v if v > self._eps else self._eps)
        dv = (P_in_w - P_out_w) * dt / denom
        v_new = v + dv
        # Soft clamp
        if v_new < self.vdc_min:
            v_new = self.vdc_min
        elif v_new > self.vdc_max:
            v_new = self.vdc_max
        self.vdc = v_new
        return v_new
