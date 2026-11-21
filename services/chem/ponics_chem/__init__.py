"""Equilibrium chemistry of a nutrient solution: pH from charge balance, EC from ionic
conductances. This is the "truth" chemistry of the simulator. Controllers never import it.

Model
-----
Strong ions (K, Ca, Mg, Na, NO3, SO4, Cl) are fully dissociated. Phosphate (P_T) and
inorganic carbon (C_T) are weak acids. H+ is found from electroneutrality:

    S + h - Kw'/h - P_T * nP(h) - C_T * nC(h) = 0,     S = sum z_i c_i over strong ions

nP, nC are the mean negative charge per P and per C at [H+] = h. Primed constants are
conditional (activity corrected) constants from the Davies equation at the solution's
ionic strength, which is iterated to a fixed point. pH = -log10(gamma_1 * h).

EC uses Debye-Hueckel-Onsager-corrected equivalent ionic conductances. It reproduces
the 0.01 M KCl standard (1.413 mS/cm at 25 C) to 0.1%. Ion pairing (MgSO4, CaSO4) is
ignored, so EC of sulfate-rich solutions is overestimated by a few percent.

All at 25 C. Temperature enters only through the EC probe (see ponics_sensors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from ponics_contracts import STRONG_IONS

# Thermodynamic constants at 25 C.
KW = 10**-13.995
KP = (10**-2.148, 10**-7.198, 10**-12.35)  # H3PO4 / H2PO4- / HPO4-- dissociation
KC = (10**-6.352, 10**-10.329)  # CO2(aq)+H2O / HCO3- dissociation
K_HENRY_CO2 = 10**-1.468  # mol/(L atm)
DAVIES_A = 0.509

# Limiting equivalent conductances, S cm^2 / eq, 25 C (CRC Handbook).
LAMBDA0 = {
    "H": 349.65, "OH": 198.0, "K": 73.48, "Na": 50.08, "Ca": 59.47, "Mg": 53.0,
    "Cl": 76.31, "NO3": 71.42, "SO4": 80.0, "HCO3": 44.5, "CO3": 69.3,
    "H2PO4": 36.0, "HPO4": 57.0, "PO4": 69.0,
}
CHARGE = {**STRONG_IONS, "H": 1, "OH": -1, "HCO3": -1, "CO3": -2, "H2PO4": -1, "HPO4": -2, "PO4": -3}
ONSAGER_B1, ONSAGER_B2_ION = 0.2289, 60.32 / 2.0


def gamma(z: int, ionic_strength: float) -> float:
    s = math.sqrt(max(ionic_strength, 0.0))
    return 10 ** (-DAVIES_A * z * z * (s / (1.0 + s) - 0.3 * ionic_strength))


@dataclass(frozen=True)
class Speciation:
    ph: float
    h: float  # mol/L
    ionic_strength: float  # mol/L
    species: dict  # mM of every charged species incl. H, OH, phosphate and carbonate forms
    co2_aq: float  # mM dissolved CO2 (H2CO3*)
    ec: float  # dS/m at 25 C


def _fractions3(h, k1, k2, k3):
    d = h**3 + k1 * h * h + k1 * k2 * h + k1 * k2 * k3
    return h**3 / d, k1 * h * h / d, k1 * k2 * h / d, k1 * k2 * k3 / d


def _fractions2(h, k1, k2):
    d = h * h + k1 * h + k1 * k2
    return h * h / d, k1 * h / d, k1 * k2 / d


def strong_charge(conc_mM: dict) -> float:
    """S = sum z c over strong ions, mol/L."""
    return sum(z * conc_mM.get(ion, 0.0) for ion, z in STRONG_IONS.items()) * 1e-3


def _solve_h(S: float, PT: float, CT: float, I: float) -> float:
    g1, g2, g3 = gamma(1, I), gamma(2, I), gamma(3, I)
    kw = KW / (g1 * g1)
    k1, k2, k3 = KP[0] / (g1 * g1), KP[1] / g2, KP[2] * g2 / (g1 * g3)
    c1, c2 = KC[0] / (g1 * g1), KC[1] / g2

    def f(x):
        h = 10.0**x
        _, a1, a2, a3 = _fractions3(h, k1, k2, k3)
        _, b1, b2 = _fractions2(h, c1, c2)
        return S + h - kw / h - PT * (a1 + 2 * a2 + 3 * a3) - CT * (b1 + 2 * b2)

    return 10.0 ** brentq(f, -14.5, 0.5, xtol=1e-10)


def speciate(conc_mM: dict) -> Speciation:
    """Solve pH and EC for a solution given component concentrations in mM.
    Keys: strong ions of STRONG_IONS, plus 'P' (total phosphate) and 'C' (total inorganic carbon)."""
    S = strong_charge(conc_mM)
    PT = conc_mM.get("P", 0.0) * 1e-3
    CT = conc_mM.get("C", 0.0) * 1e-3
    I_strong = 0.5 * sum(z * z * conc_mM.get(ion, 0.0) for ion, z in STRONG_IONS.items()) * 1e-3
    I = I_strong + 0.5 * PT + 0.5 * CT  # first guess: H2PO4- and HCO3-
    for _ in range(6):
        h = _solve_h(S, PT, CT, I)
        sp = _species(h, PT, CT, I)
        I_new = I_strong + 0.5 * sum(CHARGE[k] ** 2 * v for k, v in sp.items()) * 1e-3
        if abs(I_new - I) < 1e-6 * max(I, 1e-6):  # pH moves < 1e-6 per this change in I
            break
        I = I_new
    g1 = gamma(1, I)
    co2 = CT * 1e3 - sp["HCO3"] - sp["CO3"]
    all_sp = {ion: conc_mM.get(ion, 0.0) for ion in STRONG_IONS} | sp
    return Speciation(-math.log10(g1 * h), h, I, all_sp, max(co2, 0.0), conductivity(all_sp, I))


def _species(h, PT, CT, I) -> dict:
    g1, g2, g3 = gamma(1, I), gamma(2, I), gamma(3, I)
    k1, k2, k3 = KP[0] / (g1 * g1), KP[1] / g2, KP[2] * g2 / (g1 * g3)
    c1, c2 = KC[0] / (g1 * g1), KC[1] / g2
    _, a1, a2, a3 = _fractions3(h, k1, k2, k3)
    _, b1, b2 = _fractions2(h, c1, c2)
    return {
        "H": h * 1e3, "OH": KW / (g1 * g1) / h * 1e3,
        "H2PO4": PT * a1 * 1e3, "HPO4": PT * a2 * 1e3, "PO4": PT * a3 * 1e3,
        "HCO3": CT * b1 * 1e3, "CO3": CT * b2 * 1e3,
    }


def conductivity(species_mM: dict, ionic_strength: float) -> float:
    """EC in dS/m (= mS/cm) at 25 C."""
    s = math.sqrt(max(ionic_strength, 0.0))
    shield = s / (1.0 + s)
    ec = 0.0
    for ion, c in species_mM.items():
        z = abs(CHARGE[ion])
        lam = LAMBDA0[ion] - z * (ONSAGER_B1 * LAMBDA0[ion] + ONSAGER_B2_ION) * shield
        ec += c * 1e-3 * z * max(lam, 0.0)
    return ec


def carbon_for_ph(conc_mM: dict, ph: float) -> float:
    """Total inorganic carbon (mM) that makes a solution sit at `ph`. Used to describe
    tap water by its pH and its strong ions, which is how a water report gives it."""
    lo, hi = 0.0, 50.0
    base = dict(conc_mM)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if speciate(base | {"C": mid}).ph > ph:
            lo = mid  # more CO2 lowers pH
        else:
            hi = mid
    return 0.5 * (lo + hi)


def co2_saturation_mM(ppm: float) -> float:
    """Dissolved CO2 in equilibrium with air at `ppm` CO2 and 1 atm."""
    return K_HENRY_CO2 * ppm * 1e-6 * 1e3


__all__ = [
    "Speciation", "speciate", "conductivity", "strong_charge", "gamma",
    "carbon_for_ph", "co2_saturation_mM", "KW", "KP", "KC",
]
