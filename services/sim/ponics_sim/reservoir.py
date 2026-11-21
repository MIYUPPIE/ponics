"""The "truth" plant: one 100 L reservoir feeding 72 lettuce plants in NFT channels.

Each 5 min physics step:
  1. speciate the solution (pH, EC, dissolved CO2)
  2. crop rates: growth, transpiration, ion uptake, root CO2
  3. remove uptake and transpired water; the float valve tops up with tap water
  4. CO2 exchange with room air (first order, k_La), plus root respiration CO2
Each 15 min control step: the dose is added first, then three physics steps run.

Maintenance: on each day in `dump_days` at 08:00, half the solution is drained and
refilled with tap water. Every controller sees the same events.

Every mmol that enters or leaves is booked in `ledger`, so mass balance is testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ponics_chem import co2_saturation_mM, speciate
from ponics_chem.recipes import ACID, BASE, STOCK_A, STOCK_B, TapWater
from ponics_contracts import ALL_COMPONENTS, CONTROL_DT_S, DAY_S, PHYSICS_DT_S, SUBSTEPS, Dose, PumpLimits

from ponics_sim.climate import ClimateProfile, climate_at
from ponics_sim.crop import CropParams, CropState, advance, fresh_weight, rates


@dataclass(frozen=True)
class SystemConfig:
    tap: TapWater
    climate: ClimateProfile
    crop: CropParams = field(default_factory=CropParams)
    volume_l: float = 100.0
    n_plants: int = 72
    kla_per_h: float = 2.0  # CO2 exchange with air, aerated reservoir
    initial_ph: float = 5.8
    initial_ec: float = 1.8
    dump_days: tuple[int, ...] = (14, 28)
    dump_fraction: float = 0.5
    limits: PumpLimits = field(default_factory=PumpLimits)


@dataclass
class Ledger:
    """Cumulative mmol per component, and litres/mL of every input."""

    added: dict = field(default_factory=lambda: {k: 0.0 for k in ALL_COMPONENTS})
    removed: dict = field(default_factory=lambda: {k: 0.0 for k in ALL_COMPONENTS})
    stock_ml: float = 0.0  # A + B together
    acid_ml: float = 0.0
    base_ml: float = 0.0
    prep_acid_ml: float = 0.0  # used to set up the initial solution, not by the controller
    water_l: float = 0.0  # tap water added after the initial fill
    drained_l: float = 0.0
    transpired_l: float = 0.0


class NftSystem:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.reset()

    # ---- setup -------------------------------------------------------------
    def reset(self) -> None:
        c = self.cfg
        self.t = 0.0
        self.volume = 0.0
        self.amounts = {k: 0.0 for k in ALL_COMPONENTS}
        self.ledger = Ledger()
        self.tap = c.tap.composition()
        # Fill with tap water so that tap + stocks + acid = volume_l (three passes).
        tap_l = c.volume_l
        for _ in range(3):
            self.amounts = {k: 0.0 for k in ALL_COMPONENTS}
            self.volume = 0.0
            self._add(self.tap, tap_l, water=False)
            ab_ml, acid_ml = self._mix_to(c.initial_ec, c.initial_ph)
            tap_l = c.volume_l - (2 * ab_ml + acid_ml) / 1000.0
        self._add(STOCK_A, ab_ml / 1000.0)
        self._add(STOCK_B, ab_ml / 1000.0)
        self._add(ACID, acid_ml / 1000.0)
        self.ledger = Ledger()  # the prepared solution is the starting point, not an input
        for k, v in self.amounts.items():
            self.ledger.added[k] = v
        self.ledger.prep_acid_ml = acid_ml
        self.ledger_volume0 = self.volume
        self.crop = CropState(dw=c.crop.dw0)
        self.maintenance_flag = False
        self._dumped: set[int] = set()
        self._spec = speciate(self.conc)
        r = rates(c.crop, self.crop, self.conc, self._spec.ec, self._spec.ph, climate_at(c.climate, 12 * 3600.0))
        self.crop.g24 = max(r.growth, 0.0) * c.climate.photoperiod_h / 24.0

    def _trial(self, ab_ml: float, acid_ml: float):
        t = dict(self.amounts)
        for stock in (STOCK_A, STOCK_B):
            for k, v in stock.items():
                t[k] += v * ab_ml / 1000.0
        t["NO3"] += ACID["NO3"] * acid_ml / 1000.0
        vol = self.volume + (2 * ab_ml + acid_ml) / 1000.0
        return speciate({k: v / vol for k, v in t.items()})

    def _acid_to(self, ab_ml: float, ph: float) -> float:
        lo, hi = 0.0, 4000.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self._trial(ab_ml, mid).ph > ph:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _mix_to(self, ec: float, ph: float) -> tuple[float, float]:
        """How a grower prepares the tank: A+B to the EC target, acid to the pH target."""
        lo, hi = 0.0, 3000.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._trial(mid, self._acid_to(mid, ph)).ec < ec:
                lo = mid
            else:
                hi = mid
        ab = 0.5 * (lo + hi)
        return ab, self._acid_to(ab, ph)

    def _add(self, conc: dict, litres: float, water: bool = False) -> None:
        for k, v in conc.items():
            self.amounts[k] += v * litres
            self.ledger.added[k] += v * litres
        self.volume += litres
        if water:
            self.ledger.water_l += litres

    def _remove(self, comp: str, mmol: float) -> None:
        take = min(max(mmol, 0.0), self.amounts[comp])
        self.amounts[comp] -= take
        self.ledger.removed[comp] += take

    # ---- state -------------------------------------------------------------
    @property
    def conc(self) -> dict:
        return {k: v / self.volume for k, v in self.amounts.items()}

    @property
    def ph(self) -> float:
        return self._spec.ph

    @property
    def ec(self) -> float:
        return self._spec.ec

    @property
    def day(self) -> float:
        return self.t / DAY_S

    def climate(self):
        return climate_at(self.cfg.climate, self.t)

    def fresh_weight_g(self) -> float:
        return fresh_weight(self.cfg.crop, self.crop)

    # ---- dynamics ----------------------------------------------------------
    def apply_dose(self, dose: Dose) -> Dose:
        d = dose.clipped(self.cfg.limits)
        if d.ab_ml > 0:
            self._add(STOCK_A, d.ab_ml / 1000.0)
            self._add(STOCK_B, d.ab_ml / 1000.0)
            self.ledger.stock_ml += 2 * d.ab_ml
        if d.acid_ml > 0:
            self._add(ACID, d.acid_ml / 1000.0)
            self.ledger.acid_ml += d.acid_ml
        if d.base_ml > 0:
            self._add(BASE, d.base_ml / 1000.0)
            self.ledger.base_ml += d.base_ml
        return d

    def step(self, dose: Dose) -> Dose:
        """One control interval. Returns the dose actually applied (after clipping)."""
        self.maintenance_flag = False
        applied = self.apply_dose(dose)
        for _ in range(SUBSTEPS):
            self._physics(PHYSICS_DT_S)
        self._maybe_dump()
        self._spec = speciate(self.conc)
        return applied

    def _physics(self, dt: float) -> None:
        c, dt_h = self.cfg, dt / 3600.0
        spec = speciate(self.conc)
        clim = self.climate()
        conc = self.conc
        r = rates(c.crop, self.crop, conc, spec.ec, spec.ph, clim)
        n = c.n_plants
        for ion, u in r.uptake.items():
            self._remove(ion, u * n * dt_h)
        water = r.transpiration * n * dt_h
        self.volume -= water
        self.ledger.transpired_l += water
        # Float valve: top up to the set volume with tap water.
        self._add(self.tap, max(c.volume_l - self.volume, 0.0), water=True)
        # CO2: root respiration in, exchange with room air.
        resp = r.root_co2 * n * dt_h
        self.amounts["C"] += resp
        self.ledger.added["C"] += resp
        sat = co2_saturation_mM(clim.co2_ppm)
        co2 = spec.co2_aq  # start-of-substep speciation; uptake in 5 min barely moves it
        new = sat + (co2 - sat) * math.exp(-c.kla_per_h * dt_h)
        delta = (new - co2) * self.volume
        if delta >= 0:
            self.amounts["C"] += delta
            self.ledger.added["C"] += delta
        else:
            self._remove("C", -delta)
        advance(c.crop, self.crop, r, spec.ph, spec.ec, dt_h)
        self.last_rates = r
        self.t += dt

    def _maybe_dump(self) -> None:
        c = self.cfg
        day = int(self.t // DAY_S)
        hour = (self.t % DAY_S) / 3600.0
        if day in c.dump_days and day not in self._dumped and hour >= 8.0:
            self._dumped.add(day)
            f = c.dump_fraction
            for k in list(self.amounts):
                self._remove(k, self.amounts[k] * f)
            drained = self.volume * f
            self.volume -= drained
            self.ledger.drained_l += drained
            self._add(self.tap, drained, water=True)
            self.maintenance_flag = True


__all__ = ["SystemConfig", "NftSystem", "Ledger", "CONTROL_DT_S"]
