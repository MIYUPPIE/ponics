import pytest

from ponics_bench.scenarios import Scenario
from ponics_contracts import ALL_COMPONENTS, Dose
from ponics_sim.climate import STANDARD, climate_at
from ponics_sim.crop import f_ph, f_salinity
from ponics_sim.reservoir import NftSystem


@pytest.fixture(scope="module")
def fresh():
    return NftSystem(Scenario("train", 0).config())


def test_initial_tank_is_mixed_to_targets(fresh):
    assert fresh.ph == pytest.approx(5.8, abs=0.01)
    assert fresh.ec == pytest.approx(1.8, abs=0.01)
    assert fresh.volume == pytest.approx(100.0, abs=0.02)


def test_mass_balance_and_volume_over_a_day():
    s = NftSystem(Scenario("train", 1).config())
    for k in range(96):
        s.step(Dose(5.0 if k % 4 == 0 else 0.0, 2.0, 0.5))
    for c in ALL_COMPONENTS:
        assert s.amounts[c] == pytest.approx(s.ledger.added[c] - s.ledger.removed[c], abs=1e-6), c
    # The float valve only adds water; doses can overfill by a few tens of mL at night.
    assert 100.0 - 1e-6 <= s.volume <= 100.2
    added = s.ledger.water_l + (s.ledger.stock_ml + s.ledger.acid_ml + s.ledger.base_ml) / 1000
    assert s.volume == pytest.approx(s.ledger_volume0 + added - s.ledger.transpired_l, abs=1e-6)


def test_open_loop_ph_rises_and_crop_grows():
    s = NftSystem(Scenario("train", 2).config())
    fw0 = s.fresh_weight_g()
    for _ in range(96):
        s.step(Dose())
    assert s.ph > 6.0  # CO2 degassing + nitrate uptake push pH up
    assert s.fresh_weight_g() > fw0


def test_dump_happens_once_at_eight():
    cfg = Scenario("train", 3).config()
    from dataclasses import replace

    s = NftSystem(replace(cfg, dump_days=(0,)))
    na0 = s.amounts["Na"]
    flags = []
    for _ in range(40):
        s.step(Dose())
        flags.append(s.maintenance_flag)
    assert sum(flags) == 1 and flags.index(True) == 31  # t = 8:00 after 32 steps
    assert s.ledger.drained_l == pytest.approx(50.0, abs=0.5)
    assert s.amounts["Na"] == pytest.approx(na0, rel=0.2)  # refill with tap brings Na back


def test_stress_curves():
    assert f_ph(5.8) == 1.0 and f_ph(6.5) == 1.0 and f_ph(7.5) == pytest.approx(0.7)
    assert f_ph(4.0) < 0.5
    assert f_salinity(2.0) == 1.0 and f_salinity(4.5) < 1.0


def test_lights_cycle():
    assert not climate_at(STANDARD, 5 * 3600).lights_on
    assert climate_at(STANDARD, 6 * 3600).lights_on
    assert not climate_at(STANDARD, 22 * 3600).lights_on


def test_pumps_are_clipped(fresh):
    s = NftSystem(Scenario("train", 4).config())
    d = s.step(Dose(1e6, 1e6, 0))
    assert d.ab_ml == 100.0 and d.acid_ml == 40.0
