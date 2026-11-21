"""The two controllers a grower actually runs today.

RuleController   commercial doser logic: fixed shots when a reading leaves a dead band.
PidController    PI on EC (dosing A+B) and split-range PI on pH (acid above, base below),
                 conditional-integration anti-windup, gains from the nominal model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ponics_contracts import Controller, Dose, Observation

from ponics_control.nominal import NominalModel


@dataclass
class RuleController(Controller):
    ec_low: float = 0.1  # dose when EC < setpoint - ec_low
    ab_shot_ml: float = 30.0
    ph_high: float = 6.1
    ph_low: float = 5.5
    acid_shot_ml: float = 10.0
    base_shot_ml: float = 5.0
    name: str = "rule"

    def act(self, obs: Observation) -> Dose:
        sp = obs.setpoints
        ab = self.ab_shot_ml if obs.ec < sp.ec - self.ec_low else 0.0
        acid = self.acid_shot_ml if obs.ph > self.ph_high else 0.0
        base = self.base_shot_ml if obs.ph < self.ph_low else 0.0
        return Dose(ab, acid, base)


@dataclass
class PidController(Controller):
    """Gains are set so one step closes about half of an error under the nominal model."""

    model: NominalModel = None  # type: ignore[assignment]
    ec_close: float = 0.5  # fraction of EC error removed by P action per step
    ec_ti_steps: float = 16.0
    ph_close: float = 0.4
    ph_ti_steps: float = 24.0
    ph_deadband: float = 0.05
    name: str = "pid"

    def __post_init__(self):
        self.model = self.model or NominalModel()
        self.reset()

    def reset(self) -> None:
        self.i_ec = 0.0
        self.i_ph = 0.0

    def act(self, obs: Observation) -> Dose:
        sp, lim = obs.setpoints, obs.limits
        B = self.model.B(obs.volume_l)
        if obs.maintenance:
            self.reset()
        # EC: positive action only (nutrients cannot be removed).
        e_ec = sp.ec - obs.ec
        kp_ec = self.ec_close / B[0, 0]
        u_ab = kp_ec * e_ec + self.i_ec
        if 0.0 < u_ab < lim.ab_ml or (u_ab <= 0 and e_ec > 0) or (u_ab >= lim.ab_ml and e_ec < 0):
            self.i_ec += kp_ec * e_ec / self.ec_ti_steps
        self.i_ec = min(max(self.i_ec, 0.0), lim.ab_ml)
        # pH: u > 0 means acid, u < 0 means base.
        e_ph = obs.ph - sp.ph
        if abs(e_ph) < self.ph_deadband:
            e_ph = 0.0
        kp_ph = self.ph_close / B[1, 2]
        u_ph = kp_ph * e_ph + self.i_ph
        sat_hi, sat_lo = lim.acid_ml, -lim.base_ml
        if sat_lo < u_ph < sat_hi or (u_ph >= sat_hi and e_ph < 0) or (u_ph <= sat_lo and e_ph > 0):
            self.i_ph += kp_ph * e_ph / self.ph_ti_steps
        self.i_ph = min(max(self.i_ph, sat_lo), sat_hi)
        return Dose(max(u_ab, 0.0), max(u_ph, 0.0), max(-u_ph, 0.0))
