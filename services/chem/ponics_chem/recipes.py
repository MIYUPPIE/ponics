"""Stock solutions, pH adjusters and tap waters, as component concentrations in mM.

Full-strength lettuce solution (1 L of A + 1 L of B into 100 L), mM:
    NO3 12.0  H2PO4 1.5  SO4 1.75  K 7.0  Ca 3.5  Mg 1.5
which is a standard leafy-greens recipe (Sonneveld & Voogt, *Plant Nutrition of
Greenhouse Crops*, 2009, lettuce NFT range). Salts: A = Ca(NO3)2 + KNO3,
B = KNO3 + KH2PO4 + MgSO4 + K2SO4, kept apart so Ca never meets phosphate or sulfate
at stock strength.

Acid and base add only their strong ion: HNO3 adds NO3-, KOH adds K+. The H+ or OH-
appears through the charge balance in ponics_chem.speciate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ponics_chem import carbon_for_ph

# 100x concentrates, mM of components.
STOCK_A = {"Ca": 350.0, "K": 250.0, "NO3": 950.0}
STOCK_B = {"K": 450.0, "NO3": 250.0, "P": 150.0, "Mg": 150.0, "SO4": 175.0}
ACID = {"NO3": 500.0}  # 0.5 M HNO3
BASE = {"K": 500.0}  # 0.5 M KOH

FULL_STRENGTH_ML_PER_L = 10.0  # mL of A (and of B) per L of solution


@dataclass(frozen=True)
class TapWater:
    name: str
    ions: dict  # mM strong ions
    ph: float

    def composition(self) -> dict:
        """Strong ions plus the C_T that gives the stated pH."""
        return self.ions | {"C": carbon_for_ph(self.ions, self.ph)}


# Soft municipal water, alkalinity ~2 mM. Train suite.
TAP_SOFT = TapWater("soft", {"Ca": 1.0, "Mg": 0.3, "Na": 1.0, "K": 0.05, "Cl": 1.0, "SO4": 0.3, "NO3": 0.1}, 7.6)
# Hard borehole water, alkalinity ~3.8 mM, Na 1.5 mM. Unseen suite.
TAP_HARD = TapWater("hard", {"Ca": 2.0, "Mg": 0.8, "Na": 1.5, "K": 0.1, "Cl": 1.5, "SO4": 0.8, "NO3": 0.3}, 7.8)


def mix(parts: list[tuple[dict, float]]) -> dict:
    """Amounts (mmol) from (concentration mM, volume L) pairs."""
    out: dict = {}
    for conc, vol in parts:
        for k, v in conc.items():
            out[k] = out.get(k, 0.0) + v * vol
    return out


__all__ = ["STOCK_A", "STOCK_B", "ACID", "BASE", "FULL_STRENGTH_ML_PER_L", "TapWater", "TAP_SOFT", "TAP_HARD", "mix"]
