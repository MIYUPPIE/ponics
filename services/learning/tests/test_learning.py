import numpy as np
import pytest

from ponics_contracts.residual import FEATURES
from ponics_learning.cli import subsample
from ponics_learning.data import check_seed
from ponics_learning.gp import ResidualGp
from ponics_learning.rl import N_OBS, ObsBuilder, action_to_dose
from ponics_contracts import Observation


def test_eval_seeds_and_unseen_suite_are_off_limits():
    with pytest.raises(ValueError):
        check_seed(5, "train")
    with pytest.raises(ValueError):
        check_seed(5000, "unseen")
    check_seed(1000, "train")


def test_structured_gp_recovers_drift_and_gain():
    """y = a(pH) + b(pH) * n with a known a, b. The GP must separate them."""
    rng = np.random.default_rng(0)
    m = 160
    Z = np.zeros((m, len(FEATURES)))
    Z[:, 0] = rng.uniform(5.2, 6.6, m)
    Z[:, 1] = rng.uniform(1.6, 2.0, m)
    h = rng.uniform(0, 24, m)
    Z[:, 2], Z[:, 3] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    Z[:, 4] = (h > 6) & (h < 22)
    Z[:, 5] = rng.uniform(0, 1, m)
    n = rng.uniform(-15, 15, m)
    a_true = 0.03 * (6.0 - Z[:, 0])
    b_true = -0.02 * (6.6 - Z[:, 0])
    y = a_true + b_true * n + rng.normal(0, 0.005, m)
    gp = ResidualGp().fit(Z, n, y)
    a, b = gp.parts(Z)
    assert np.sqrt(np.mean((b - b_true) ** 2)) < 0.004
    assert np.sqrt(np.mean((a - a_true) ** 2)) < 0.01
    assert gp.support(Z).mean() > 0.9
    far = Z[:1].copy()
    far[0, 0] = 3.0
    assert not gp.support(far)[0]


def test_subsample_is_deterministic_and_keeps_doses():
    n = np.zeros(1000)
    n[::10] = 5.0
    a, b = subsample(n, 200, 0), subsample(n, 200, 0)
    assert (a == b).all() and len(a) == 200 and (np.abs(n[a]) > 2).sum() == 100


def test_action_mapping_never_doses_acid_and_base():
    from ponics_learning.rl import AB_MAX, ADJ_MAX

    d = action_to_dose(np.array([0.5, 0.2]))
    assert d.ab_ml == 0.5 * AB_MAX and abs(d.acid_ml - 0.2 * ADJ_MAX) < 1e-9 and d.base_ml == 0
    d = action_to_dose(np.array([-1.0, -2.0]))
    assert d.ab_ml == 0 and d.acid_ml == 0 and d.base_ml == ADJ_MAX


def test_observation_vector():
    b = ObsBuilder()
    v = b(Observation(3600 * 12, 1.8, 5.8, 20, True, 100))
    assert v.shape == (N_OBS,) and v.dtype == np.float32 and abs(v[0]) < 1e-6


def test_marginal_likelihood_gradient_is_exact():
    from scipy.optimize import approx_fprime

    from ponics_learning.gp import D

    rng = np.random.default_rng(1)
    m = 40
    Z, n, y = rng.normal(size=(m, D)), rng.normal(size=m), rng.normal(size=m)
    g = ResidualGp()
    th = rng.normal(scale=0.3, size=2 * D + 3)
    num = approx_fprime(th, lambda t: g._nll(t, Z, n, y)[0], 1e-6)
    assert np.abs(g._nll(th, Z, n, y)[1] - num).max() < 1e-3 * max(1.0, np.abs(num).max())


def test_env_never_uses_eval_scenarios_even_if_seeded_low():
    from ponics_learning.rl import PonicsEnv

    env = PonicsEnv(seed0=3000, days=1)
    env.reset(seed=0)  # SB3 does this
    assert env.sys.cfg == __import__("ponics_bench.scenarios", fromlist=["Scenario"]).Scenario("train", 3000, days=1).config()
    obs, r, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (10,) and np.isfinite(r) and not term
