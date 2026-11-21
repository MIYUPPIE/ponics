"""Butterhead lettuce in NFT: light-use-efficiency growth, transpiration, nutrient uptake,
and how root-zone EC, pH and nutrient supply slow it down.

Per plant, per hour:

  f_int  = 1 - exp(-k LAI),  LAI = DW * SLA / area                 light interception
  gross  = LUE * PAR * area * f_int * f_T * f_CO2 * f_stress         g DW / h
  maint  = m * DW * Q10^((T-20)/10) / 24
  dDW/dt = gross - maint
  E      = f_int * area * (a_rad PPFD + a_vpd VPD) * f_salE * root   L / h  (night: 15% of VPD term)
  U_i    = g24 * content_i * C_i / (Km_i + C_i) * root              mmol / h  (demand-driven ions)
  U_Na,Cl = E * C * (1 - exclusion)                                   (passive, mass flow)

g24 is growth smoothed over ~12 h, so uptake continues at night as it does in real crops.
f_stress = min over N, P, K, Ca, Mg of the supply factor, times f_salinity, f_pH and root health.
Root health falls irreversibly while pH < 4.5, pH > 7.8 or EC > 4.

Calibrated to land near 150 to 200 g fresh weight at 35 days under 250 umol/m2/s for 16 h,
which is the usual indoor butterhead range. Parameters and sources are in docs/DESIGN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# mmol per g dry weight, butterhead lettuce (N 4.5%, P 0.6%, K 7%, Ca 1.2%, Mg 0.35%, S 0.3%).
CONTENT = {"NO3": 3.2, "P": 0.19, "K": 1.8, "Ca": 0.30, "Mg": 0.14, "SO4": 0.09}
KM_UPTAKE = {"NO3": 0.10, "P": 0.005, "K": 0.05, "Ca": 0.05, "Mg": 0.05, "SO4": 0.05}
KG_GROWTH = {"NO3": 2.0, "P": 0.10, "K": 1.0, "Ca": 0.5, "Mg": 0.2}
# Supply factor is 1 down to 40% of full strength, then falls.
C_REF = {"NO3": 0.4 * 12.0, "P": 0.4 * 1.5, "K": 0.4 * 7.0, "Ca": 0.4 * 3.5, "Mg": 0.4 * 1.5}
EXCLUSION = {"Na": 0.85, "Cl": 0.70}


@dataclass(frozen=True)
class CropParams:
    area_m2: float = 0.05  # per plant, 20 plants/m2
    dw0: float = 0.15  # g at transplant
    sla: float = 0.035  # m2/g
    k_ext: float = 0.8
    lue: float = 0.55  # g DW / mol PAR intercepted at 400 ppm
    maint: float = 0.012  # 1/day at 20 C
    q10: float = 2.0
    a_rad: float = 0.00072  # L / (m2 h) per umol/m2/s
    a_vpd: float = 0.097  # L / (m2 h kPa)
    content_scale: float = 1.0
    dry_matter: float = 0.05  # DW / FW
    root_resp: float = 0.0035  # mmol CO2 / h / g DW into the solution

    def jittered(self, lue: float, sla: float, content: float) -> "CropParams":
        return replace(self, lue=self.lue * (1 + lue), sla=self.sla * (1 + sla), content_scale=1 + content)


@dataclass
class CropState:
    dw: float  # g per plant
    root_health: float = 1.0
    g24: float = 0.0  # smoothed growth rate, g/h per plant


@dataclass(frozen=True)
class CropRates:
    growth: float  # g/h per plant (net)
    transpiration: float  # L/h per plant
    uptake: dict  # mmol/h per plant
    root_co2: float  # mmol/h per plant
    f_stress: float
    f_ph: float
    f_nut: float
    f_sal: float


def f_ph(ph: float) -> float:
    d = max(5.5 - ph, ph - 6.5, 0.0)
    if d <= 1.0:
        return 1.0 - 0.3 * d
    return max(0.1, 0.7 - 0.5 * (d - 1.0))


def f_salinity(ec: float) -> float:
    return min(1.0, max(0.2, 1.0 - 0.15 * max(0.0, ec - 2.5)))


def f_supply(conc: dict) -> float:
    worst = 1.0
    for ion, kg in KG_GROWTH.items():
        c = max(conc.get(ion, 0.0), 0.0)
        ref = C_REF[ion] / (kg + C_REF[ion])
        worst = min(worst, (c / (kg + c)) / ref)
    return min(worst, 1.0)


def rates(p: CropParams, s: CropState, conc: dict, ec: float, ph: float, clim) -> CropRates:
    lai = s.dw * p.sla / p.area_m2
    f_int = 1.0 - math.exp(-p.k_ext * lai)
    fT = math.exp(-(((clim.t_air - 22.0) / 9.0) ** 2))
    fco2 = 1.0 + 0.25 * math.log2(max(clim.co2_ppm, 200.0) / 400.0)
    fp, fn, fs = f_ph(ph), f_supply(conc), f_salinity(ec)
    stress = fp * fn * fs * s.root_health
    par_mol_h = clim.ppfd * 3600e-6
    gross = p.lue * par_mol_h * p.area_m2 * f_int * fT * fco2 * stress
    maint = p.maint * s.dw * p.q10 ** ((clim.t_air - 20.0) / 10.0) / 24.0
    salE = min(1.0, max(0.3, 1.0 - 0.08 * max(0.0, ec - 2.0)))
    vpd_term = p.a_vpd * clim.vpd * (1.0 if clim.lights_on else 0.15)
    E = f_int * p.area_m2 * (p.a_rad * clim.ppfd + vpd_term) * salE * s.root_health
    demand = max(s.g24, 0.0) * p.content_scale * s.root_health
    up = {}
    for ion, content in CONTENT.items():
        c = max(conc.get(ion, 0.0), 0.0)
        up[ion] = demand * content * c / (KM_UPTAKE[ion] + c)
    for ion, ex in EXCLUSION.items():
        up[ion] = E * max(conc.get(ion, 0.0), 0.0) * (1.0 - ex)
    return CropRates(gross - maint, E, up, p.root_resp * s.dw, stress, fp, fn, fs)


def advance(p: CropParams, s: CropState, r: CropRates, ph: float, ec: float, dt_h: float) -> None:
    s.dw = max(s.dw + r.growth * dt_h, 0.01)
    a = 1.0 - math.exp(-dt_h / 12.0)
    s.g24 += a * (max(r.growth, 0.0) - s.g24)
    if ph < 4.5 or ph > 7.8 or ec > 4.0:
        s.root_health = max(0.0, s.root_health - 0.02 * dt_h)


def fresh_weight(p: CropParams, s: CropState) -> float:
    return s.dw / p.dry_matter
