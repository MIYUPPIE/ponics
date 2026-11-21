"""Residual data from closed-loop runs on the train suite only.

The logging controller is the nominal MPC plus random dither (extra 2 to 10 mL acid or
base, or up to 30 mL A+B, 8% of steps) and random holds (no dose, 10% of steps), so the
data covers off-setpoint pH, the doses the GP-MPC will choose, and pure drift at every pH. Logging with the controller whose model is being corrected is the usual GP-MPC
practice. PID was tried first and rejected: kicked into the weak-buffer region near
pH 5 it falls into an acid/base limit cycle (true gain ~4x nominal there), which is a
finding about PID, not useful data. The A+B dither is one-sided (only at or below the
EC setpoint), because nutrients cannot be taken back out. Readings are sensor readings,
as they would be on real hardware. Steps that span a maintenance dump
are dropped: a dump is a scheduled event, not dynamics.
"""

from __future__ import annotations

import numpy as np

from ponics_bench.runner import observe
from ponics_bench.scenarios import EVAL_SEED_MAX, Scenario
from ponics_contracts import DAY_S, CONTROL_DT_S, Dose, Setpoints
from ponics_contracts.residual import features, net_acid
from ponics_control.mpc import Mpc
from ponics_control.nominal import NominalModel
from ponics_sensors import Probes
from ponics_sim.reservoir import NftSystem

TRAIN_SEED_MIN = 1000


def check_seed(seed: int, suite: str) -> None:
    if seed <= EVAL_SEED_MAX:
        raise ValueError(f"seed {seed} is an evaluation seed; learning must use seeds >= {TRAIN_SEED_MIN}")
    if suite != "train":
        raise ValueError("learning may only use the train suite")


class Dithered(Mpc):
    def __init__(self, seed: int, p: float = 0.08, p_hold: float = 0.10):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.p, self.p_hold = p, p_hold
        self.name = "mpc_dither"

    def act(self, obs):
        d = super().act(obs)
        ab, acid, base = d.ab_ml, d.acid_ml, d.base_ml
        r = self.rng.random()
        if r < self.p_hold:
            # Hold: no dose this step. Without holds, low pH is almost always followed by
            # base, and the GP cannot tell drift a(z) from gain b(z) there (it learned a
            # +1.3 pH/step "drift" at pH 4.6 and drove the tank further down).
            return Dose(0.0, 0.0, 0.0)
        if r < self.p_hold + self.p:
            kind = self.rng.integers(3)
            if kind == 0 and obs.ec < obs.setpoints.ec + 0.05:
                ab += self.rng.uniform(5.0, 30.0)
            elif kind == 1 and obs.ph > 5.3:
                acid += self.rng.uniform(2.0, 10.0)
            elif kind == 2 and obs.ph < 6.5:
                base += self.rng.uniform(2.0, 10.0)
        return Dose(ab, acid, base)


def collect(seed: int, days: int = 35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns z features (m, 6), net acid dose n (m,) in mL, and the pH residual y (m,)."""
    check_seed(seed, "train")
    sc = Scenario("train", seed, "sensor", days)
    sys = NftSystem(sc.config())
    probes = Probes(sc.sensor_seed)
    ctrl = Dithered(seed)
    ctrl.reset()
    nom = NominalModel()
    sp = Setpoints()
    Z, N, Y = [], [], []
    prev = None
    for _ in range(int(days * DAY_S / CONTROL_DT_S)):
        obs = observe(sys, probes, sp)
        if prev is not None and not obs.maintenance:
            o0, u0 = prev
            pred = nom.predict(np.array([o0.ec, o0.ph]), u0, o0.volume_l)
            Z.append(features(o0.ph, o0.ec, o0.hour, o0.lights_on, o0.day))
            N.append(net_acid(u0))
            Y.append(obs.ph - pred[1])
        dose = sys.step(ctrl.act(obs))
        prev = (obs, np.array([dose.ab_ml, dose.acid_ml, dose.base_ml]))
    return np.array(Z), np.array(N), np.array(Y)
