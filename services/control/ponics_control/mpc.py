"""Offset-free linear MPC on the nominal model, solved as a box-constrained QP (OSQP).

Horizon N steps of 15 min. Decision variable U = (u_0 .. u_{N-1}), u = (ab, acid, base) >= 0.
Prediction, with per-step offsets c_k (drift estimate, plus the GP mean in GP-MPC):

    x_{k+1} = x_k + B_k u_k + c_k

Cost  sum_k (x_k - r)' Q (x_k - r) + cost' u_k + rho |u_k|^2 + |T (u_k - u_{k-1})|^2_Rdu
      T u = (ab, acid - base): moves are penalised on A+B and on the net pH action.

The move penalty R_du is what keeps the loop stable when the true buffer capacity is up
to 3x off the nominal one (it is, at pH 5.5: see docs/DESIGN.md). It was tuned on a linear
plant with exactly that gain error (tests/test_controllers.py::test_mpc_robust_to_gain_error).

Q weighs EC error at 1/0.1^2 and pH error at 1/0.15^2. `cost` is the chemical price per
mL relative to tracking, so the controller does not dose acid and base together and does
not chase sensor noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import osqp
from scipy import sparse

from ponics_contracts import Controller, Dose, Observation

from ponics_control.nominal import DriftEstimator, NominalModel


@dataclass(frozen=True)
class MpcParams:
    horizon: int = 16
    q_ec: float = 1.0 / 0.1**2
    q_ph: float = 1.0 / 0.15**2
    cost: tuple[float, float, float] = (0.01, 0.02, 0.02)  # per mL of ab, acid, base
    rho: float = 1e-4
    r_du: tuple[float, float] = (5e-4, 0.05)  # move suppression on A+B and on net acid (acid - base)
    alpha: float = 0.1  # drift estimator gain


@dataclass
class Mpc(Controller):
    params: MpcParams = field(default_factory=MpcParams)
    model: NominalModel = field(default_factory=NominalModel)
    name: str = "mpc"

    def __post_init__(self):
        self.est = DriftEstimator(self.model, self.params.alpha)
        self.reset()

    def reset(self) -> None:
        self.est.reset()
        self.plan = None  # last solution, N x 3
        self.u_prev = np.zeros(3)
        self.last_solve_ms = 0.0

    # Hook for GP-MPC: per-step input matrices and offsets along the horizon.
    def horizon_model(self, obs: Observation, x0: np.ndarray, B: np.ndarray):
        N = self.params.horizon
        return [B] * N, np.tile(self.est.d, (N, 1)), np.zeros(2)

    def act(self, obs: Observation) -> Dose:
        p, lim = self.params, obs.limits
        x0 = self.est.update(obs)
        B = self.model.B(obs.volume_l)
        Bs, offsets, first_extra = self.horizon_model(obs, x0, B)
        U = self.solve(x0, np.array([obs.setpoints.ec, obs.setpoints.ph]), Bs, offsets,
                       np.array([lim.ab_ml, lim.acid_ml, lim.base_ml]))
        self.plan = U
        u = U[0]
        self.u_prev = u.copy()
        dose = Dose(float(u[0]), float(u[1]), float(u[2]))
        # The estimator compares the next reading with the nominal prediction plus
        # whatever extra the model added for step 0 (GP mean).
        extra = Bs[0] @ u - B @ u + first_extra
        self.est.commit(x0, dose, obs.volume_l, extra)
        return dose

    def solve(self, x0, r, Bs, offsets, umax) -> np.ndarray:
        t0 = time.perf_counter()
        p = self.params
        N, m, n = p.horizon, 3, 2
        # X = X0 + S U, X stacks x_1..x_N.
        S = np.zeros((n * N, m * N))
        X0 = np.zeros(n * N)
        acc = x0.copy()
        for k in range(N):
            acc = acc + offsets[k]
            X0[n * k:n * k + n] = acc
            for j in range(k + 1):
                S[n * k:n * k + n, m * j:m * j + m] = Bs[j]
        Q = np.diag(np.tile([p.q_ec, p.q_ph], N))
        R = np.tile(r, N)
        # D U - e0 stacks u_k - u_{k-1}, with u_{-1} the last applied input.
        D = np.eye(m * N) - np.eye(m * N, k=-m)
        e0 = np.zeros(m * N)
        e0[:m] = self.u_prev
        # Penalise moves of (ab, acid - base), not of acid and base separately: with
        # separate penalties, cancelling acid with base is cheaper than cutting acid fast.
        T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])
        Rdu = np.kron(np.eye(N), T.T @ np.diag(p.r_du) @ T)
        P = 2.0 * (S.T @ Q @ S + p.rho * np.eye(m * N) + D.T @ Rdu @ D)
        q = 2.0 * S.T @ Q @ (X0 - R) + np.tile(p.cost, N) - 2.0 * D.T @ Rdu @ e0
        A = sparse.eye(m * N, format="csc")
        solver = osqp.OSQP()
        solver.setup(sparse.triu(sparse.csc_matrix(P), format="csc"), q, A, np.zeros(m * N), np.tile(umax, N),
                     verbose=False, eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=20000)
        res = solver.solve()
        U = np.clip(np.asarray(res.x, dtype=float).reshape(N, m), 0.0, umax)
        if not np.all(np.isfinite(U)):
            U = np.zeros((N, m))
        self.last_solve_ms = (time.perf_counter() - t0) * 1e3
        return U
