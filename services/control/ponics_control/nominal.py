"""The nominal model a control engineer writes from the recipe sheet and one jar titration.
No speciation, no plant model.

    EC_{k+1} = EC_k + g_ec * u_ab + d_ec
    pH_{k+1} = pH_k + g_ph * (u_base - u_acid) + d_ph

g_ec   = EC_full / (FULL_STRENGTH_ML_PER_L * V)       dS/m per mL of A (+ same of B)
g_ph   = C_adj / (beta * V)                           pH per mL of 0.5 M acid or base
d_*    = per-step drift (crop uptake, CO2, top-up water), estimated online

EC_FULL and BETA are what a jar test of fresh full-strength solution made with the
train-suite tap water gives (see docs/DESIGN.md, "nominal model"). The true buffer
capacity changes with pH, dissolved CO2, phosphate and tap-water alkalinity, and the
true drift changes with time of day and crop size. That mismatch is the residual the
GP has to learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ponics_contracts import Dose, Observation

EC_FULL = 1.80  # dS/m per full-strength dose (jar test slope, soft tap water)
BETA = 0.30  # mmol/L per pH unit at pH 5.8 (jar test after a few hours of aeration)
C_ADJ = 0.5  # mmol/mL of the acid and base stocks
FULL_ML_PER_L = 10.0


@dataclass(frozen=True)
class NominalModel:
    ec_full: float = EC_FULL
    beta: float = BETA

    def B(self, volume_l: float) -> np.ndarray:
        """2 x 3 input matrix, state (ec, ph), input (ab, acid, base)."""
        g_ec = self.ec_full / (FULL_ML_PER_L * volume_l)
        g_ph = C_ADJ / (self.beta * volume_l)
        return np.array([[g_ec, 0.0, 0.0], [0.0, -g_ph, g_ph]])

    def predict(self, x: np.ndarray, u: np.ndarray, volume_l: float, d: np.ndarray | None = None) -> np.ndarray:
        out = x + self.B(volume_l) @ u
        return out if d is None else out + d


class DriftEstimator:
    """Offset-free MPC disturbance estimate: d <- d + a * (x_meas - x_pred).
    Reset after a maintenance dump, which is a step change, not drift."""

    def __init__(self, model: NominalModel, alpha: float = 0.1, clip: tuple[float, float] = (0.01, 0.03)):
        self.model, self.alpha, self.clip = model, alpha, np.array(clip)
        self.reset()

    def reset(self) -> None:
        self.d = np.zeros(2)
        self.prev: tuple[np.ndarray, np.ndarray, float] | None = None
        self.prev_extra = np.zeros(2)

    def update(self, obs: Observation) -> np.ndarray:
        """Fold the new reading into the drift estimate and return it as the state x."""
        x = np.array([obs.ec, obs.ph])
        if obs.maintenance:
            self.reset()
        elif self.prev is not None:
            x0, u0, v0 = self.prev
            pred = self.model.predict(x0, u0, v0, self.d) + self.prev_extra
            self.d = np.clip(self.d + self.alpha * (x - pred), -self.clip, self.clip)
        return x

    def commit(self, x: np.ndarray, dose: Dose, volume_l: float, extra_pred: np.ndarray | None = None) -> None:
        """Remember this step so the next reading can be compared with its prediction.
        extra_pred: anything the caller's model added to the nominal step-0 prediction (GP mean)."""
        self.prev = (x, np.array([dose.ab_ml, dose.acid_ml, dose.base_ml]), volume_l)
        self.prev_extra = np.zeros(2) if extra_pred is None else np.asarray(extra_pred, dtype=float)
