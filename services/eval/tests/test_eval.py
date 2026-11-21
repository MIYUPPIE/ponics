import numpy as np

from ponics_bench.runner import EpisodeLog
from ponics_bench.scenarios import Scenario
from ponics_contracts import Setpoints
from ponics_eval.cli import check
from ponics_eval.metrics import aggregate, episode_metrics
from ponics_sim.reservoir import Ledger


def fake_log(ph, ec):
    n = len(ph)
    log = EpisodeLog(Scenario("train", 0), "x", Setpoints())
    log.t = list(np.arange(n) * 900.0)
    log.ph, log.ec = list(ph), list(ec)
    log.ph_meas, log.ec_meas = list(ph), list(ec)
    log.ab_ml = log.acid_ml = log.base_ml = [0.0] * n
    log.stress, log.solve_ms, log.fw = [1.0] * n, [1.0] * n, [150.0] * n
    log.fw_g, log.ledger, log.finished = 150.0, Ledger(), True
    return log


def test_metrics_on_known_trace():
    ph = np.r_[np.full(80, 5.8), np.full(16, 7.2)]
    ec = np.r_[np.full(48, 1.8), np.full(48, 1.5)]
    m = episode_metrics(fake_log(ph, ec))
    assert abs(m["ph_in_band"] - 80 / 96) < 1e-9
    assert abs(m["ph_severe_h"] - 4.0) < 1e-9
    assert abs(m["ec_low_frac"] - 0.5) < 1e-9
    agg = aggregate([m, m])
    assert agg["n"] == 2 and agg["ph_in_band_worst"] == m["ph_in_band"]


def test_check_flags_regressions_and_requirements():
    base = {"mpc|train|sensor": {"ph_rms": 0.10, "ph_severe_h": 0.0, "ec_low_frac": 0.0, "ec_rms": 0.05,
                                  "acid_l": 3.0, "base_l": 0.2, "cost": 1.0, "ph_in_band": 1.0,
                                  "ec_in_band": 1.0, "fw_g": 150.0, "root_health": 1.0, "solve_p99_ms": 5.0}}
    same = {k: dict(v) for k, v in base.items()}
    assert check(same, base, {}) == []
    worse = {k: dict(v) for k, v in base.items()}
    worse["mpc|train|sensor"]["ph_rms"] = 0.2
    worse["mpc|train|sensor"]["fw_g"] = 100.0
    fails = check(worse, base, {"mpc|train|sensor": {"ph_in_band": 0.95}})
    assert len(fails) == 2
    fails = check(same, base, {"mpc|train|sensor": {"ph_in_band": 1.01}})
    assert len(fails) == 1 and fails[0].startswith("REQUIREMENT")
