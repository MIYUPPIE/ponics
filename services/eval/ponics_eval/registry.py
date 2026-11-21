"""Controller names -> constructors. Learned controllers load from models/ and fail loudly
if the model is missing (run `ponics-train`)."""

from __future__ import annotations

from pathlib import Path

from ponics_contracts import Controller

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"
NAMES = ("rule", "pid", "mpc", "gp_mpc", "mpc_tight", "gp_mpc_tight", "ppo")
LEARNED = {"gp_mpc": MODELS / "gp.pkl", "gp_mpc_tight": MODELS / "gp.pkl", "ppo": MODELS / "ppo.zip"}
# "tight" = move penalty on net acid 0.003 instead of 0.05: the aggressive tuning that only
# a model with the right buffer gain survives (docs/DESIGN.md, finding 5).
TIGHT_R_DU = (5e-4, 0.003)


def make(name: str) -> Controller:
    if name in LEARNED and not LEARNED[name].exists():
        raise FileNotFoundError(f"{LEARNED[name]} missing: run `ponics-train {'ppo' if name == 'ppo' else 'gp'}`")
    if name == "rule":
        from ponics_control.classic import RuleController

        return RuleController()
    if name == "pid":
        from ponics_control.classic import PidController

        return PidController()
    if name == "mpc":
        from ponics_control.mpc import Mpc

        return Mpc()
    if name == "mpc_tight":
        from ponics_control.mpc import Mpc, MpcParams

        return Mpc(params=MpcParams(r_du=TIGHT_R_DU), name="mpc_tight")
    if name in ("gp_mpc", "gp_mpc_tight"):
        from ponics_control.gp_mpc import GpMpc
        from ponics_control.mpc import MpcParams
        from ponics_learning.gp import ResidualGp

        params = MpcParams(r_du=TIGHT_R_DU) if name == "gp_mpc_tight" else MpcParams()
        return GpMpc(params=params, residual=ResidualGp.load(LEARNED[name]), name=name)
    if name == "ppo":
        from ponics_learning.rl import PpoController

        return PpoController(LEARNED["ppo"])
    raise KeyError(name)
