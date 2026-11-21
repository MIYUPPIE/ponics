"""pH and EC probes with the errors that matter in a grow room.

pH probe (glass electrode, e.g. Atlas EZO-pH class):
  reading = 7 + slope * (pH - 7) + offset + noise,  quantised to 0.01
  offset starts at N(0, 0.02), random-walks at 0.03 pH/sqrt(day), and is reset to
  N(0, 0.02) at each weekly recalibration (days 7, 14, 21, 28 at 09:00).
  slope = 1 + N(0, 0.015) per seed; noise sd 0.015.

EC probe (two-electrode conductivity cell with temperature compensation):
  reading = gain * EC25 * (1 + a_true (T-25)) / (1 + a_probe (T-25)) + noise
  The probe compensates at 2.0 %/C; nutrient solutions are nearer 1.9 %/C.
  gain = 1 + N(0, 0.02) per seed; noise sd 1% of reading + 0.005; quantised to 0.01.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ponics_contracts import DAY_S


@dataclass(frozen=True)
class SensorParams:
    ph_cal_sd: float = 0.02
    ph_drift_per_sqrt_day: float = 0.03
    ph_slope_sd: float = 0.015
    ph_noise: float = 0.015
    recal_days: tuple[int, ...] = (7, 14, 21, 28)
    recal_hour: float = 9.0
    ec_gain_sd: float = 0.02
    ec_alpha_true: float = 0.019
    ec_alpha_probe: float = 0.020
    ec_noise_rel: float = 0.01
    ec_noise_abs: float = 0.005


class Probes:
    def __init__(self, seed: int, params: SensorParams | None = None):
        self.p = params or SensorParams()
        self.rng = np.random.default_rng(seed)
        self.ph_slope = 1.0 + self.rng.normal(0, self.p.ph_slope_sd)
        self.ec_gain = 1.0 + self.rng.normal(0, self.p.ec_gain_sd)
        self.ph_offset = self.rng.normal(0, self.p.ph_cal_sd)
        self._last_t = 0.0
        self._recal_done: set[int] = set()

    def _advance(self, t: float) -> None:
        dt_day = max(t - self._last_t, 0.0) / DAY_S
        self._last_t = t
        self.ph_offset += self.rng.normal(0, self.p.ph_drift_per_sqrt_day * math.sqrt(dt_day))
        day, hour = int(t // DAY_S), (t % DAY_S) / 3600.0
        if day in self.p.recal_days and day not in self._recal_done and hour >= self.p.recal_hour:
            self._recal_done.add(day)
            self.ph_offset = self.rng.normal(0, self.p.ph_cal_sd)

    def read(self, t: float, ph: float, ec25: float, t_water: float) -> tuple[float, float]:
        self._advance(t)
        p = self.p
        ph_r = 7.0 + self.ph_slope * (ph - 7.0) + self.ph_offset + self.rng.normal(0, p.ph_noise)
        comp = (1 + p.ec_alpha_true * (t_water - 25.0)) / (1 + p.ec_alpha_probe * (t_water - 25.0))
        ec_clean = self.ec_gain * ec25 * comp
        ec_r = ec_clean + self.rng.normal(0, p.ec_noise_rel * ec_clean + p.ec_noise_abs)
        return round(ph_r, 2), round(max(ec_r, 0.0), 2)


__all__ = ["SensorParams", "Probes"]
