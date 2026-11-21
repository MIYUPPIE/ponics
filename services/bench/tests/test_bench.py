from ponics_bench.runner import run_episode
from ponics_bench.scenarios import Scenario, grid
from ponics_control.classic import RuleController


def test_scenarios_are_deterministic_and_suites_differ():
    a, b = Scenario("train", 3).config(), Scenario("train", 3).config()
    assert a == b
    u = Scenario("unseen", 3).config()
    assert u.tap.name == "hard" and a.tap.name == "soft" and u.dump_days != a.dump_days


def test_grid_size():
    assert len(grid(seeds=range(4), feedback=("sensor", "truth"))) == 16


def test_short_episode_is_reproducible():
    l1 = run_episode(RuleController(), Scenario("train", 0, "sensor", days=1))
    l2 = run_episode(RuleController(), Scenario("train", 0, "sensor", days=1))
    assert l1.finished and len(l1.ph) == 96
    assert l1.ph == l2.ph and l1.ph_meas == l2.ph_meas and l1.fw_g == l2.fw_g
