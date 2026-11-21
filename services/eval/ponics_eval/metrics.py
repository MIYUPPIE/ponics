"""Per-episode metrics on the TRUE pH and EC (sensor error is part of what is scored).

ph_in_band     share of time 5.5 <= pH <= 6.5
ph_rms         RMS of pH - setpoint
ph_severe_h    hours with pH < 5.0 or > 7.0 (root damage starts beyond 4.5 / 7.8)
ec_low_frac    share of time EC < setpoint - 0.2. This is the controllable EC failure:
               EC above band comes from salts the controller cannot remove.
ec_in_band     share of time |EC - setpoint| <= 0.2
ec_rms
fw_g           mean fresh weight per plant at harvest (day 35)
stock_l        A + B concentrate used by the controller (L, both stocks)
acid_l, base_l 0.5 M HNO3 / KOH used by the controller (L)
cost           mean stage cost of the MPC objective, on the truth (same for all controllers)
root_health    1 = undamaged
solve_p99_ms   controller compute time per decision
"""

from __future__ import annotations

import numpy as np

from ponics_bench.runner import EpisodeLog
from ponics_contracts import CONTROL_DT_S
from ponics_control.mpc import MpcParams

LOWER_IS_BETTER = ("ph_rms", "ph_severe_h", "ec_low_frac", "ec_rms", "acid_l", "base_l", "cost")
HIGHER_IS_BETTER = ("ph_in_band", "ec_in_band", "fw_g", "root_health")


def episode_metrics(log: EpisodeLog, p: MpcParams = MpcParams()) -> dict:
    a = log.arrays()
    sp = log.setpoints
    ph, ec = a["ph"], a["ec"]
    hours = CONTROL_DT_S / 3600.0
    track = p.q_ec * (ec - sp.ec) ** 2 + p.q_ph * (ph - sp.ph) ** 2
    chem = p.cost[0] * a["ab_ml"] + p.cost[1] * a["acid_ml"] + p.cost[2] * a["base_ml"]
    L = log.ledger
    return {
        "finished": float(log.finished),
        "ph_in_band": float(np.mean((ph >= sp.ph_lo) & (ph <= sp.ph_hi))),
        "ph_rms": float(np.sqrt(np.mean((ph - sp.ph) ** 2))),
        "ph_severe_h": float(np.sum((ph < 5.0) | (ph > 7.0)) * hours),
        "ec_low_frac": float(np.mean(ec < sp.ec - sp.ec_band)),
        "ec_in_band": float(np.mean(np.abs(ec - sp.ec) <= sp.ec_band)),
        "ec_rms": float(np.sqrt(np.mean((ec - sp.ec) ** 2))),
        "fw_g": float(log.fw_g),
        "stock_l": L.stock_ml / 1000.0,
        "acid_l": L.acid_ml / 1000.0,
        "base_l": L.base_ml / 1000.0,
        "cost": float(np.mean(track + chem)),
        "root_health": float(log.root_health),
        "solve_p99_ms": float(np.percentile(a["solve_ms"], 99)),
    }


def aggregate(ms: list[dict]) -> dict:
    keys = ms[0].keys()
    out = {k: float(np.mean([m[k] for m in ms])) for k in keys}
    out |= {f"{k}_sd": float(np.std([m[k] for m in ms])) for k in ("ph_rms", "fw_g", "cost")}
    out["ph_in_band_worst"] = float(min(m["ph_in_band"] for m in ms))
    out["solve_p99_ms"] = float(max(m["solve_p99_ms"] for m in ms))
    out["n"] = len(ms)
    return out
