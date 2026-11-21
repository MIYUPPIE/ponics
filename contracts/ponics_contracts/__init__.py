"""Shared types. Every service imports these and nothing from another service's internals.

Units, used everywhere:
  concentrations   mmol/L (mM)
  amounts          mmol
  volumes          L (reservoir, water) and mL (pump doses)
  EC               dS/m at 25 C (numerically equal to mS/cm)
  time             s since transplant (t = 0 at 00:00 on day 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONTRACT_VERSION = 1

# One control decision every 15 min, physics integrated every 5 min.
CONTROL_DT_S = 900.0
PHYSICS_DT_S = 300.0
SUBSTEPS = int(CONTROL_DT_S // PHYSICS_DT_S)
DAY_S = 86400.0

# Strong ions (fully dissociated) and their charge. Phosphate and carbonate are weak
# acids and are carried as totals P_T and C_T; their charge depends on pH.
STRONG_IONS: dict[str, int] = {"K": 1, "Ca": 2, "Mg": 2, "Na": 1, "NO3": -1, "SO4": -2, "Cl": -1}
WEAK_TOTALS: tuple[str, ...] = ("P", "C")
ALL_COMPONENTS: tuple[str, ...] = tuple(STRONG_IONS) + WEAK_TOTALS


@dataclass(frozen=True)
class Setpoints:
    ec: float = 1.8  # dS/m
    ph: float = 5.8
    ec_band: float = 0.2  # +/- around ec counts as "in band"
    ph_lo: float = 5.5
    ph_hi: float = 6.5


@dataclass(frozen=True)
class PumpLimits:
    """Largest dose one control step may command. Doses are clipped to these by the sim."""

    ab_ml: float = 100.0  # mL of stock A and the same mL of stock B
    acid_ml: float = 40.0  # 0.5 M HNO3
    base_ml: float = 40.0  # 0.5 M KOH


@dataclass(frozen=True)
class Dose:
    """One control step's command. ab_ml is dosed from A and from B (equal volumes)."""

    ab_ml: float = 0.0
    acid_ml: float = 0.0
    base_ml: float = 0.0

    def clipped(self, lim: PumpLimits) -> "Dose":
        c = lambda v, hi: min(max(float(v), 0.0), hi)  # noqa: E731
        return Dose(c(self.ab_ml, lim.ab_ml), c(self.acid_ml, lim.acid_ml), c(self.base_ml, lim.base_ml))


@dataclass(frozen=True)
class Observation:
    """What a controller may see. ec and ph come from the sensors (or the truth, in
    `truth` feedback mode). Everything else is known to a grower without a sensor."""

    t: float  # s since transplant
    ec: float
    ph: float
    water_temp: float  # C
    lights_on: bool
    volume_l: float
    setpoints: Setpoints = field(default_factory=Setpoints)
    limits: PumpLimits = field(default_factory=PumpLimits)
    maintenance: bool = False  # a partial dump and refill happened since the last step

    @property
    def day(self) -> float:
        return self.t / DAY_S

    @property
    def hour(self) -> float:
        return (self.t % DAY_S) / 3600.0


class Controller:
    """Protocol for every controller. `act` is called once per CONTROL_DT_S."""

    name: str = "controller"

    def reset(self) -> None:  # called before each episode
        pass

    def act(self, obs: Observation) -> Dose:  # pragma: no cover - interface
        raise NotImplementedError


__all__ = [
    "CONTRACT_VERSION",
    "CONTROL_DT_S",
    "PHYSICS_DT_S",
    "SUBSTEPS",
    "DAY_S",
    "STRONG_IONS",
    "WEAK_TOTALS",
    "ALL_COMPONENTS",
    "Setpoints",
    "PumpLimits",
    "Dose",
    "Observation",
    "Controller",
]
