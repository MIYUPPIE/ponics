"""Closed-loop episode: sim + probes + controller, logged every control step."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ponics_contracts import CONTROL_DT_S, DAY_S, Controller, Observation, Setpoints
from ponics_sensors import Probes
from ponics_sim.reservoir import Ledger, NftSystem

from ponics_bench.scenarios import Scenario


@dataclass
class EpisodeLog:
    scenario: Scenario
    controller: str
    setpoints: Setpoints
    t: list = field(default_factory=list)
    ph: list = field(default_factory=list)  # truth
    ec: list = field(default_factory=list)  # truth, EC25
    ph_meas: list = field(default_factory=list)
    ec_meas: list = field(default_factory=list)
    ab_ml: list = field(default_factory=list)
    acid_ml: list = field(default_factory=list)
    base_ml: list = field(default_factory=list)
    stress: list = field(default_factory=list)
    fw: list = field(default_factory=list)  # mean fresh weight per plant, g
    solve_ms: list = field(default_factory=list)
    fw_g: float = 0.0
    root_health: float = 1.0
    ledger: Ledger | None = None
    finished: bool = False
    error: str = ""

    def arrays(self) -> dict:
        return {k: np.asarray(getattr(self, k)) for k in
                ("t", "ph", "ec", "ph_meas", "ec_meas", "ab_ml", "acid_ml", "base_ml", "stress", "fw", "solve_ms")}


def observe(sys: NftSystem, probes: Probes | None, sp: Setpoints) -> Observation:
    clim = sys.climate()
    if probes is None:
        ph, ec = sys.ph, sys.ec
    else:
        ph, ec = probes.read(sys.t, sys.ph, sys.ec, clim.t_water)
    return Observation(sys.t, ec, ph, clim.t_water, clim.lights_on, sys.volume, sp, sys.cfg.limits, sys.maintenance_flag)


def run_episode(ctrl: Controller, sc: Scenario, sp: Setpoints | None = None) -> EpisodeLog:
    sp = sp or Setpoints()
    sys = NftSystem(sc.config())
    probes = Probes(sc.sensor_seed) if sc.feedback == "sensor" else None
    ctrl.reset()
    log = EpisodeLog(sc, getattr(ctrl, "name", type(ctrl).__name__), sp)
    steps = int(sc.days * DAY_S / CONTROL_DT_S)
    for _ in range(steps):
        obs = observe(sys, probes, sp)
        t0 = time.perf_counter()
        dose = ctrl.act(obs)
        dt_ms = (time.perf_counter() - t0) * 1e3
        log.t.append(sys.t)
        log.ph.append(sys.ph)
        log.ec.append(sys.ec)
        log.ph_meas.append(obs.ph)
        log.ec_meas.append(obs.ec)
        applied = sys.step(dose)
        log.ab_ml.append(applied.ab_ml)
        log.acid_ml.append(applied.acid_ml)
        log.base_ml.append(applied.base_ml)
        log.stress.append(sys.last_rates.f_stress)
        log.fw.append(sys.fresh_weight_g())
        log.solve_ms.append(dt_ms)
    log.fw_g = sys.fresh_weight_g()
    log.root_health = sys.crop.root_health
    log.ledger = sys.ledger
    log.finished = True
    return log
