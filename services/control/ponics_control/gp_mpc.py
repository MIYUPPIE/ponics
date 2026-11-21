"""GP-MPC: the nominal MPC with the learned pH residual in its prediction model.

    EC_{k+1} = EC_k + g_ec u_ab + d_ec                                  (nominal)
    pH_{k+1} = pH_k + g_ph (base - acid) + a(z_k) + b(z_k) (acid - base) + d_ph

a, b come from the structured residual GP (contract: ponics_contracts.residual). The
model is exactly linear in u for fixed z_k, so each step k gets

    B_k = B + [[0, 0, 0], [0, b_k, -b_k]],      offset_k = (d_ec, a_k + d_ph)

z_k (pH, EC, clock, age) is taken along the previous solution's predicted path, shifted
by one step, which is the usual GP-MPC linearisation; all N GP evaluations go in one batch.

Support: where z_k is outside the box the GP was trained on (1st-99th percentile of pH,
EC, age) the residual is dropped and the nominal model is used for that step.

Safety clamp: the corrected pH gain per mL is kept between GAIN_MIN and GAIN_MAX times the
nominal gain, with the nominal sign. A GP far from its data cannot flip the sign of acid.
GP variance is not used yet (constraint tightening is a follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ponics_contracts import CONTROL_DT_S, DAY_S, Observation
from ponics_contracts.residual import RESIDUAL_VERSION, ResidualModel, features

from ponics_control.mpc import Mpc

LIGHTS_ON_H, LIGHTS_OFF_H = 6.0, 22.0
GAIN_MIN, GAIN_MAX = 0.25, 8.0


@dataclass
class GpMpc(Mpc):
    residual: ResidualModel | None = None
    name: str = "gp_mpc"

    def __post_init__(self):
        if self.residual is None:
            raise ValueError("GpMpc needs a residual model")
        if getattr(self.residual, "version", None) != RESIDUAL_VERSION:
            raise ValueError("residual model was trained for a different feature contract")
        super().__post_init__()

    def reset(self) -> None:
        super().reset()
        self.path = None  # predicted x_1..x_N of the last solve
        self.last_b = np.zeros(0)

    def horizon_model(self, obs: Observation, x0: np.ndarray, B: np.ndarray):
        N = self.params.horizon
        X = np.tile(x0, (N, 1)) if self.path is None else np.vstack([x0, self.path[1:]])
        rows = []
        for k in range(N):
            t = obs.t + k * CONTROL_DT_S
            hour = (t % DAY_S) / 3600.0
            rows.append(features(X[k, 1], X[k, 0], hour, LIGHTS_ON_H <= hour < LIGHTS_OFF_H, t / DAY_S))
        Zr = np.array(rows)
        a, b = self.residual.parts(Zr)
        inside = self.residual.support(Zr)
        a, b = np.where(inside, a, 0.0), np.where(inside, b, 0.0)  # nominal outside the data
        self.last_support = inside
        g = B[1, 2]  # nominal pH gain per mL (positive, base direction)
        gain = np.clip(g - b, GAIN_MIN * g, GAIN_MAX * g)  # true gain per mL of net base
        Bs, offs = [], np.zeros((N, 2))
        for k in range(N):
            Bk = B.copy()
            Bk[1, 1], Bk[1, 2] = -gain[k], gain[k]
            Bs.append(Bk)
            offs[k] = self.est.d + np.array([0.0, a[k]])
        self.last_b = b
        return Bs, offs, offs[0] - self.est.d

    def solve(self, x0, r, Bs, offsets, umax) -> np.ndarray:
        U = super().solve(x0, r, Bs, offsets, umax)
        path, x = np.zeros((len(U), 2)), x0.copy()
        for k in range(len(U)):
            x = x + Bs[k] @ U[k] + offsets[k]
            path[k] = x
        self.path = path
        return U
