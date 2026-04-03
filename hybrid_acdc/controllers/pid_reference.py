from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDParams:
    kp: float
    ki: float
    u_min: float
    u_max: float


class PID:
    def __init__(self, params: PIDParams):
        self.p = params
        self.int_e = 0.0

    def reset(self) -> None:
        self.int_e = 0.0

    def step(self, e: float, dt: float) -> float:
        self.int_e += e * dt
        u = self.p.kp * e + self.p.ki * self.int_e
        if u > self.p.u_max:
            u = self.p.u_max
        elif u < self.p.u_min:
            u = self.p.u_min
        return u
