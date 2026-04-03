from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ofo import OFOOptimizer
from .nn_disturbance import OnlineDisturbanceEstimator


@dataclass
class OFONNBundle:
    ofo: OFOOptimizer
    nn: OnlineDisturbanceEstimator


class OFONNController:
    """Combines OFO + online disturbance estimator.

    - NN estimates a lumped disturbance power d_hat (W)
    - OFO updates (Kp,Ki,alpha) using a cost that is reduced by NN compensation
    """

    def __init__(self, ofo: OFOOptimizer, nn: OnlineDisturbanceEstimator):
        self.ofo = ofo
        self.nn = nn

    def disturbance_comp(self, feat: np.ndarray) -> float:
        return float(self.nn.predict(feat))

    def update_nn(self, feat: np.ndarray, target: float) -> None:
        self.nn.update(feat, target)
