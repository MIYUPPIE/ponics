import numpy as np
import pytest

from ponics_contracts import Observation
from ponics_contracts.residual import RESIDUAL_VERSION, ZeroResidual
from ponics_control.classic import PidController, RuleController
from ponics_control.gp_mpc import GAIN_MAX, GpMpc
from ponics_control.mpc import Mpc, MpcParams
from ponics_control.nominal import NominalModel


def obs(ec=1.8, ph=5.8, t=0.0, maint=False):
    return Observation(t, ec, ph, 20.0, True, 100.0, maintenance=maint)


def run_linear(ctrl, gain=1.0, x0=(1.6, 6.3), drift=(-0.002, 0.01), steps=60):
    """Linear plant: nominal B with the pH gain scaled by `gain`, constant drift."""
    B = NominalModel().B(100.0) * np.array([[1.0], [gain]])
    x, xs, us = np.array(x0, float), [], []
    for k in range(steps):
        d = ctrl.act(obs(x[0], x[1], k * 900.0))
        u = np.array([d.ab_ml, d.acid_ml, d.base_ml])
        x = x + B @ u + np.array(drift)
        xs.append(x.copy())
        us.append(u)
    return np.array(xs), np.array(us)


def test_rule_shots():
    d = RuleController().act(obs(ec=1.5, ph=6.4))
    assert d.ab_ml > 0 and d.acid_ml > 0 and d.base_ml == 0
    d = RuleController().act(obs(ec=1.8, ph=5.3))
    assert d.ab_ml == 0 and d.base_ml > 0 and d.acid_ml == 0


def test_pid_signs_and_split_range():
    p = PidController()
    d = p.act(obs(ec=1.6, ph=6.2))
    assert d.ab_ml > 0 and d.acid_ml > 0 and d.base_ml == 0
    p.reset()
    d = p.act(obs(ec=2.0, ph=5.4))
    assert d.ab_ml == 0 and d.base_ml > 0 and d.acid_ml == 0


@pytest.mark.parametrize("C,tol", [(PidController, 0.06), (Mpc, 0.03)])  # PID has a 0.05 pH dead band
def test_converges_on_nominal_plant(C, tol):
    xs, us = run_linear(C())
    assert abs(xs[-10:, 1] - 5.8).max() < tol
    assert abs(xs[-10:, 0] - 1.8).max() < 0.03


def test_mpc_respects_pump_limits_and_never_doses_both():
    xs, us = run_linear(Mpc(), x0=(1.0, 7.5))
    assert (us >= 0).all() and (us[:, 0] <= 100 + 1e-6).all() and (us[:, 1:] <= 40 + 1e-6).all()
    assert (np.minimum(us[:, 1], us[:, 2]) < 0.5).all()


@pytest.mark.parametrize("gain", [0.5, 2.0, 3.0, 4.0])
def test_mpc_robust_to_gain_error(gain):
    """The true buffer is up to ~3x weaker than nominal near pH 5.5. The move penalty
    must keep the loop stable there (this is what tuned r_du)."""
    xs, _ = run_linear(Mpc(), gain=gain, steps=80)
    assert abs(xs[-20:, 1] - 5.8).max() < 0.05


def test_unpenalised_mpc_is_not_robust():
    xs, _ = run_linear(Mpc(params=MpcParams(r_du=(5e-4, 0.0))), gain=3.0, steps=80)
    assert abs(xs[-20:, 1] - 5.8).max() > 0.3


def test_gp_mpc_with_zero_residual_equals_mpc():
    a, b = Mpc(), GpMpc(residual=ZeroResidual())
    xa, ua = run_linear(a, gain=2.0)
    xb, ub = run_linear(b, gain=2.0)
    assert np.allclose(ua, ub, atol=1e-6)


class GainResidual(ZeroResidual):
    """True gain = `mult` x nominal everywhere; b is the correction per mL of net acid."""

    def __init__(self, mult, a=0.0, inside=True):
        self.mult, self.a, self.inside = mult, a, inside
        self.g = NominalModel().B(100.0)[1, 2]

    def parts(self, Z):
        m = len(Z)
        return np.full(m, self.a), np.full(m, -(self.mult - 1.0) * self.g)

    def support(self, Z):
        return np.full(len(Z), self.inside)


def test_gp_mpc_uses_learned_gain():
    """With the right gain correction, GP-MPC on a 3x plant settles faster and doses less
    acid in total than the nominal MPC."""
    xg, ug = run_linear(GpMpc(residual=GainResidual(3.0)), gain=3.0, steps=40)
    xn, un = run_linear(Mpc(), gain=3.0, steps=40)
    err = lambda xs: np.abs(xs[:12, 1] - 5.8).sum()  # noqa: E731
    assert err(xg) < err(xn)


def test_gp_mpc_ignores_residual_outside_support():
    wild = GainResidual(1.0, a=5.0, inside=False)  # absurd drift, but flagged out of support
    xg, ug = run_linear(GpMpc(residual=wild))
    xn, un = run_linear(Mpc())
    assert np.allclose(ug, un, atol=1e-6)


def test_gp_mpc_gain_is_clamped():
    c = GpMpc(residual=GainResidual(100.0))
    c.act(obs(ph=6.2))
    B = NominalModel().B(100.0)
    Bs, _, _ = c.horizon_model(obs(ph=6.2, t=900.0), np.array([1.8, 6.2]), B)
    assert Bs[0][1, 2] == pytest.approx(GAIN_MAX * B[1, 2])


def test_gp_mpc_rejects_wrong_contract():
    class Old(ZeroResidual):
        version = RESIDUAL_VERSION - 1

    with pytest.raises(ValueError):
        GpMpc(residual=Old())
