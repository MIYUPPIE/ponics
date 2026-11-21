import pytest

from ponics_chem import carbon_for_ph, speciate, strong_charge
from ponics_chem.recipes import ACID, STOCK_A, STOCK_B, TAP_HARD, TAP_SOFT


def test_pure_water_is_neutral():
    assert speciate({}).ph == pytest.approx(7.0, abs=0.01)


def test_strong_acid_with_activity():
    # 1 mM HNO3: [H+] = 1e-3, gamma ~0.965 (Davies) -> pH 3.015
    assert speciate({"NO3": 1.0}).ph == pytest.approx(3.015, abs=0.005)


def test_kcl_conductivity_standard():
    # 0.01 M KCl reference solution: 1.413 mS/cm at 25 C
    assert speciate({"K": 10.0, "Cl": 10.0}).ec == pytest.approx(1.413, rel=0.005)


def test_phosphate_buffer_at_pka2():
    # Equimolar H2PO4-/HPO4--: pH ~ pKa2 (7.20) minus the activity shift
    s = speciate({"K": 3.0, "P": 2.0})  # K 3 mM balances 1 mM H2PO4- + 1 mM HPO4--
    assert 6.9 < s.ph < 7.2


def test_charge_balance_closes():
    s = speciate({"K": 7, "Ca": 3.5, "Mg": 1.5, "NO3": 12, "SO4": 1.75, "P": 1.5, "C": 0.5})
    sp = s.species
    pos = sp["K"] + 2 * sp["Ca"] + 2 * sp["Mg"] + sp["H"]
    neg = sp["NO3"] + 2 * sp["SO4"] + sp["H2PO4"] + 2 * sp["HPO4"] + 3 * sp["PO4"] + sp["HCO3"] + 2 * sp["CO3"] + sp["OH"]
    assert pos == pytest.approx(neg, rel=1e-6)


def test_acid_lowers_ph_monotonically():
    base = {"K": 7, "Ca": 3.5, "Mg": 1.5, "NO3": 12, "SO4": 1.75, "P": 1.5, "C": 0.05}
    phs = [speciate(base | {"NO3": 12 + a}).ph for a in (0, 0.1, 0.2, 0.5, 1.0)]
    assert all(x > y for x, y in zip(phs, phs[1:]))


def test_stocks_are_charge_balanced():
    # Stock A has only strong ions; B carries P as H2PO4-, so its strong charge = +P.
    assert strong_charge(STOCK_A) == pytest.approx(0.0, abs=1e-9)
    assert strong_charge(STOCK_B) * 1e3 == pytest.approx(STOCK_B["P"], rel=1e-9)
    assert strong_charge(ACID) < 0


@pytest.mark.parametrize("tap", [TAP_SOFT, TAP_HARD])
def test_tap_water_carbon_reproduces_ph(tap):
    assert speciate(tap.composition()).ph == pytest.approx(tap.ph, abs=0.01)
    assert carbon_for_ph(tap.ions, tap.ph) > 1.0  # alkaline waters carry > 1 mM C_T
