from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DisturbanceNNConfig:
    # NOTE: This estimator runs *online* at every simulation step.
    # Keep the learning rate conservative to avoid numerical blow-ups.
    lr: float = 1e-3
    seed: int = 7
    err_clip: float = 50.0
    grad_clip: float = 5.0


class OnlineLinearEstimator:
    """A light online disturbance estimator.

    We use a linear model d_hat = w^T phi, updated by SGD.
    This is intentionally simple, stable, and fast (paper-friendly).
    """

    def __init__(self, n_features: int, cfg: DisturbanceNNConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.w = 0.01 * self.rng.standard_normal(n_features)

    def predict(self, phi: np.ndarray) -> float:
        return float(np.dot(self.w, phi))

    def update(self, phi: np.ndarray, target: float) -> None:
        # Defensive: avoid NaN/Inf contamination.
        phi = np.asarray(phi, dtype=float)
        if not np.all(np.isfinite(phi)) or not np.isfinite(target):
            return

        y = self.predict(phi)
        if not np.isfinite(y):
            return

        err = float(y - float(target))
        # Clip the instantaneous error to avoid exploding updates
        err = float(np.clip(err, -self.cfg.err_clip, self.cfg.err_clip))

        # Normalized SGD (stabilizes when feature magnitudes vary)
        denom = 1.0 + float(np.dot(phi, phi))
        grad = (err / denom) * phi
        # Clip gradient magnitude
        gnorm = float(np.linalg.norm(grad) + 1e-12)
        if gnorm > self.cfg.grad_clip:
            grad = grad * (self.cfg.grad_clip / gnorm)

        self.w -= float(self.cfg.lr) * grad


# Backward-compatible alias used by some snippets
OnlineDisturbanceEstimator = OnlineLinearEstimator
