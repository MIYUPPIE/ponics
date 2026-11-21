"""Indoor growth-room climate. Deterministic functions of time; per-seed offsets come
from the scenario. Lights are on 06:00 to 22:00 (16 h photoperiod)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ponics_contracts import DAY_S


@dataclass(frozen=True)
class ClimateProfile:
    name: str
    t_day: float  # air C, lights on
    t_night: float
    rh_day: float  # 0..1
    rh_night: float
    ppfd: float  # umol/m2/s at canopy, lights on
    co2_ppm: float
    water_mean: float  # C
    water_amp: float  # C, diurnal half-swing, peaks mid-photoperiod
    lights_on_h: float = 6.0
    photoperiod_h: float = 16.0

    def jittered(self, dt_air: float, dppfd: float) -> "ClimateProfile":
        return ClimateProfile(
            self.name, self.t_day + dt_air, self.t_night + dt_air, self.rh_day, self.rh_night,
            self.ppfd * (1.0 + dppfd), self.co2_ppm, self.water_mean + 0.5 * dt_air, self.water_amp,
            self.lights_on_h, self.photoperiod_h,
        )


STANDARD = ClimateProfile("standard", 22.0, 18.0, 0.65, 0.75, 250.0, 800.0, 20.0, 1.0)
HEAT = ClimateProfile("heat", 28.0, 22.0, 0.50, 0.60, 250.0, 600.0, 24.0, 1.5)


@dataclass(frozen=True)
class ClimateNow:
    lights_on: bool
    t_air: float
    rh: float
    vpd: float  # kPa
    ppfd: float
    co2_ppm: float
    t_water: float


def saturation_vp(t_c: float) -> float:
    """Tetens, kPa."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def climate_at(p: ClimateProfile, t: float) -> ClimateNow:
    hour = (t % DAY_S) / 3600.0
    on = p.lights_on_h <= hour < p.lights_on_h + p.photoperiod_h
    t_air = p.t_day if on else p.t_night
    rh = p.rh_day if on else p.rh_night
    phase = 2.0 * math.pi * (hour - (p.lights_on_h + p.photoperiod_h / 2.0 - 6.0)) / 24.0
    t_w = p.water_mean + p.water_amp * math.sin(phase)
    return ClimateNow(on, t_air, rh, saturation_vp(t_air) * (1.0 - rh), p.ppfd if on else 0.0, p.co2_ppm, t_w)
