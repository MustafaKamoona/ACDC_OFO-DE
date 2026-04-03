from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OFOConfig:
    n_iter: int = 60
    step_size: float = 0.08
    perturb: float = 0.2
    seed: int = 7

    # bounds
    kp_bounds: tuple[float, float] = (0.2, 30.0)
    ki_bounds: tuple[float, float] = (1.0, 5000.0)
    alpha_bounds: tuple[float, float] = (0.05, 0.95)

    # per-parameter nominal step sizes (scaled by `step_size`)
    step_vec: tuple[float, float, float] = (0.4, 80.0, 0.02)


def _clip_theta(theta: np.ndarray, cfg: OFOConfig) -> np.ndarray:
    kp = float(np.clip(theta[0], *cfg.kp_bounds))
    ki = float(np.clip(theta[1], *cfg.ki_bounds))
    a = float(np.clip(theta[2], *cfg.alpha_bounds))
    return np.array([kp, ki, a], dtype=float)


def spsa_update(theta: np.ndarray, J_plus: float, J_minus: float, delta: np.ndarray, cfg: OFOConfig) -> np.ndarray:
    # gradient estimate
    g_hat = (J_plus - J_minus) / (2.0 * cfg.perturb) * delta
    # Use a sign-SPSA update to avoid numerical explosion from large costs.
    # This also makes convergence behavior more interpretable for papers.
    step = cfg.step_size * np.array(cfg.step_vec, dtype=float)
    theta_new = theta - step * np.sign(g_hat)
    return _clip_theta(theta_new, cfg)


def make_delta(rng: np.random.Generator) -> np.ndarray:
    # Rademacher +/-1
    d = rng.integers(0, 2, size=3)
    d = 2 * d - 1
    return d.astype(float)
