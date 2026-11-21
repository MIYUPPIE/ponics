"""Contract between ponics_learning (fits the residual) and ponics_control (uses it).

The nominal model is good for EC (mixing is linear; the EC residual is sensor noise,
R^2 = 0.12 on a held-out seed with an unstructured GP) and poor for pH (the buffer is
nonlinear and drifts as phosphate and CO2 change). So the residual is on pH only, and
is structured to be linear in the net acid dose n = acid_ml - base_ml:

    dpH_res(z, n) = a(z) + b(z) * n
    dpH_res       = pH_{k+1} - (pH_k + g_ph (base - acid))       (nominal step removed)

a(z) is the drift the nominal model misses (uptake, CO2 exchange, top-up alkalinity);
b(z) is the correction to the pH gain per mL (b < 0 means acid acts harder than nominal).
One exact GP with kernel k_a(z,z') + n n' k_b(z,z') gives both, and the prediction is
exactly linear in the dose, so the MPC stays a QP with no linearisation error in u.

v3 adds support(): the GP is only trusted inside the 1st to 99th percentile box of the
training pH, EC and age.

A first version (v1) used one unstructured GP on (z, ab, acid, base). Its slopes in the
dose were up to 5x wrong, because in closed-loop data acid is dosed exactly when pH is
rising (confounding). See docs/DESIGN.md, "Residual model".

Features z (FEATURES, in this order):
    ph, ec            measured state (buffer depends on pH; EC tracks accumulated salts)
    sin_h, cos_h      time of day (uptake and CO2 follow the light cycle)
    lights            1 if lights on
    age               days since transplant / 35 (crop size, phosphate depletion)

Bump RESIDUAL_VERSION when any of this changes: saved models carry it and are refused
on mismatch.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

RESIDUAL_VERSION = 3
FEATURES = ("ph", "ec", "sin_h", "cos_h", "lights", "age")
CYCLE_DAYS = 35.0


def features(ph: float, ec: float, hour: float, lights_on: bool, day: float) -> np.ndarray:
    a = 2.0 * math.pi * hour / 24.0
    return np.array([ph, ec, math.sin(a), math.cos(a), 1.0 if lights_on else 0.0, day / CYCLE_DAYS])


def net_acid(u) -> float:
    """n = acid_ml - base_ml, from u = (ab, acid, base)."""
    return float(u[1] - u[2])


class ResidualModel(Protocol):
    version: int

    def parts(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Z: (m, len(FEATURES)). Returns a (m,) drift and b (m,) gain correction per mL."""
        ...

    def support(self, Z: np.ndarray) -> np.ndarray:
        """(m,) bool: True where z lies inside the region the model was trained on.
        Outside it the controller must fall back to the nominal model."""
        ...


class ZeroResidual:
    """Predicts nothing. GP-MPC with this model must equal the nominal MPC exactly."""

    version = RESIDUAL_VERSION

    def parts(self, Z: np.ndarray):
        m = len(Z)
        return np.zeros(m), np.zeros(m)

    def support(self, Z: np.ndarray):
        return np.ones(len(Z), dtype=bool)
