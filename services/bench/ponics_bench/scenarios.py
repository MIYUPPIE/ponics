"""Seeded scenarios. A Scenario fixes the tap water, climate, cultivar jitter and all
sensor noise, so (controller, scenario) -> episode is deterministic.

Suites
  train   soft tap water (alkalinity ~2 mM, Na 1 mM), standard 22/18 C room.
  unseen  hard tap water (alkalinity ~3.8 mM, Ca 2 mM, Na 1.5 mM) and a 28/22 C room with
          low humidity, so transpiration and the buffer are both well outside train.
          Hard water builds salts faster, so the grower dumps half the tank weekly
          (days 7, 14, 21, 28) instead of on days 14 and 28.
          No learned controller may train on it.

Seeds 0..99 are evaluation seeds. Learning code must use seeds >= 1000 (enforced in
ponics_learning). Feedback 'sensor' gives the controller probe readings; 'truth' gives
the true pH and EC25, to separate control error from measurement error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ponics_chem.recipes import TAP_HARD, TAP_SOFT
from ponics_sim.climate import HEAT, STANDARD
from ponics_sim.crop import CropParams
from ponics_sim.reservoir import SystemConfig

SUITES = ("train", "unseen")
FEEDBACK = ("sensor", "truth")
EVAL_SEED_MAX = 999


@dataclass(frozen=True)
class Scenario:
    suite: str
    seed: int
    feedback: str = "sensor"
    days: int = 35

    def config(self) -> SystemConfig:
        rng = np.random.default_rng([self.seed, SUITES.index(self.suite)])
        tap, clim = (TAP_SOFT, STANDARD) if self.suite == "train" else (TAP_HARD, HEAT)
        clim = clim.jittered(float(rng.normal(0, 1.0)), float(rng.uniform(-0.05, 0.05)))
        crop = CropParams().jittered(*(float(x) for x in rng.uniform(-0.1, 0.1, 3)))
        dumps = (14, 28) if self.suite == "train" else (7, 14, 21, 28)
        return SystemConfig(tap=tap, climate=clim, crop=crop, dump_days=dumps)

    @property
    def sensor_seed(self) -> int:
        return self.seed * 7919 + SUITES.index(self.suite) + 17


def grid(seeds=range(10), suites=SUITES, feedback=("sensor",), days: int = 35) -> list[Scenario]:
    return [Scenario(s, k, f, days) for s in suites for k in seeds for f in feedback]
