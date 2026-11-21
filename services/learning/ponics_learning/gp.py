"""Structured residual GP for pH: dpH_res = a(z) + b(z) * n   (contract: ponics_contracts.residual).

One exact GP over inputs (z, n) with kernel

    K((z,n),(z',n')) = s_a^2 exp(-|z-z'|^2_La / 2) + n n' s_b^2 exp(-|z-z'|^2_Lb / 2) + s_w^2 [same point]

ARD length scales La, Lb over the standardised z. Hyperparameters maximise the log
marginal likelihood (L-BFGS-B, fixed start, so a fit is reproducible). Posterior means:

    a(z*) = k_a(z*, Z) alpha,     b(z*) = k_b(z*, Z) (n o alpha),     alpha = K^-1 y

Plain numpy/scipy: scikit-learn kernels cannot restrict a kernel to some input columns,
and the model is small (about 1000 points, 15 hyperparameters).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from ponics_contracts.residual import FEATURES, RESIDUAL_VERSION

D = len(FEATURES)
N_SCALE = 10.0  # n is divided by this inside the kernel, so s_b is per 10 mL


def _sqexp(A: np.ndarray, B: np.ndarray, ls: np.ndarray) -> np.ndarray:
    Al, Bl = A / ls, B / ls
    d2 = (Al * Al).sum(1)[:, None] + (Bl * Bl).sum(1)[None, :] - 2.0 * Al @ Bl.T
    return np.exp(-0.5 * np.maximum(d2, 0.0))


class ResidualGp:
    version = RESIDUAL_VERSION

    def __init__(self):
        self.theta = None  # log [s_a, La(D), s_b, Lb(D), s_w]
        self.meta: dict = {}

    # ---- hyperparameters ---------------------------------------------------
    @staticmethod
    def _unpack(theta):
        e = np.exp(theta)
        return e[0], e[1:1 + D], e[1 + D], e[2 + D:2 + 2 * D], e[2 + 2 * D]

    def _K(self, theta, Z, n):
        sa, la, sb, lb, sw = self._unpack(theta)
        nn = np.outer(n, n)
        return sa**2 * _sqexp(Z, Z, la) + nn * sb**2 * _sqexp(Z, Z, lb) + (sw**2 + 1e-8) * np.eye(len(Z))

    def _nll(self, theta, Z, n, y):
        """Negative log marginal likelihood and its analytic gradient in log-hyperparameters:
        dNLL/dt = 0.5 tr((K^-1 - alpha alpha') dK/dt)."""
        sa, la, sb, lb, sw = self._unpack(theta)
        diff2 = (Z[:, None, :] - Z[None, :, :]) ** 2  # (m, m, D)
        Ka = sa**2 * np.exp(-0.5 * (diff2 / la**2).sum(-1))
        Kb = np.outer(n, n) * sb**2 * np.exp(-0.5 * (diff2 / lb**2).sum(-1))
        K = Ka + Kb + (sw**2 + 1e-8) * np.eye(len(Z))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        alpha = cho_solve(c, y)
        nll = 0.5 * y @ alpha + np.log(np.diag(c[0])).sum()
        W = cho_solve(c, np.eye(len(Z))) - np.outer(alpha, alpha)
        g = np.empty_like(theta)
        g[0] = (W * Ka).sum()
        g[1:1 + D] = 0.5 * np.einsum("ij,ijd->d", W * Ka, diff2) / la**2
        g[1 + D] = (W * Kb).sum()
        g[2 + D:2 + 2 * D] = 0.5 * np.einsum("ij,ijd->d", W * Kb, diff2) / lb**2
        g[2 + 2 * D] = np.trace(W) * sw**2
        return nll, g

    def fit(self, Z: np.ndarray, u_net: np.ndarray, y: np.ndarray) -> "ResidualGp":
        self.zm, self.zs = Z.mean(0), Z.std(0) + 1e-9
        self.lo, self.hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        Zs, n, ys = (Z - self.zm) / self.zs, u_net / N_SCALE, (y - self.ym) / self.ys
        theta0 = np.log(np.r_[0.5, np.ones(D) * 2.0, 0.5, np.ones(D) * 2.0, 0.3])
        bounds = [(np.log(1e-3), np.log(1e2))] + [(np.log(0.05), np.log(1e3))] * D
        bounds = bounds * 2 + [(np.log(1e-3), np.log(1.0))]
        res = minimize(self._nll, theta0, args=(Zs, n, ys), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300})
        self.theta = res.x
        self.Z, self.n = Zs, n
        self.alpha = cho_solve(cho_factor(self._K(self.theta, Zs, n), lower=True), ys)
        self.meta["nll"] = float(res.fun)
        self.meta["hyper"] = {"s_a": float(np.exp(res.x[0])), "La": np.exp(res.x[1:1 + D]).round(3).tolist(),
                              "s_b": float(np.exp(res.x[1 + D])), "Lb": np.exp(res.x[2 + D:2 + 2 * D]).round(3).tolist(),
                              "s_w": float(np.exp(res.x[-1]))}
        return self

    # ---- prediction ----------------------------------------------------------
    def parts(self, Z: np.ndarray):
        sa, la, sb, lb, _ = self._unpack(self.theta)
        Zs = (np.atleast_2d(Z) - self.zm) / self.zs
        a = sa**2 * _sqexp(Zs, self.Z, la) @ self.alpha
        b = sb**2 * _sqexp(Zs, self.Z, lb) @ (self.n * self.alpha)
        # back to pH units: y = ym + ys * (a + b * n / N_SCALE)
        return self.ym + self.ys * a, self.ys * b / N_SCALE

    SUPPORT_DIMS = (0, 1, 5)  # ph, ec, age; the clock features always lie in range

    def support(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)
        d = list(self.SUPPORT_DIMS)
        return np.all((Z[:, d] >= self.lo[d]) & (Z[:, d] <= self.hi[d]), axis=1)

    def predict(self, Z: np.ndarray, u_net: np.ndarray) -> np.ndarray:
        a, b = self.parts(Z)
        return a + b * np.asarray(u_net)

    # ---- persistence ----------------------------------------------------------
    _FIELDS = ("theta", "zm", "zs", "lo", "hi", "ym", "ys", "Z", "n", "alpha", "meta")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"version": self.version} | {k: getattr(self, k) for k in self._FIELDS}, f)

    @classmethod
    def load(cls, path: Path) -> "ResidualGp":
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["version"] != RESIDUAL_VERSION:
            raise ValueError(f"{path}: residual contract v{d['version']}, code expects v{RESIDUAL_VERSION}; retrain")
        m = cls()
        for k in cls._FIELDS:
            setattr(m, k, d[k])
        return m
