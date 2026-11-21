from ponics_contracts import CONTROL_DT_S, PHYSICS_DT_S, SUBSTEPS, Dose, Observation, PumpLimits
from ponics_contracts.residual import FEATURES, ZeroResidual, features, net_acid


def test_dose_clipping():
    d = Dose(-5, 500, 3).clipped(PumpLimits())
    assert (d.ab_ml, d.acid_ml, d.base_ml) == (0.0, 40.0, 3.0)


def test_steps_divide():
    assert SUBSTEPS * PHYSICS_DT_S == CONTROL_DT_S


def test_observation_clock():
    o = Observation(t=86400 * 2 + 3600 * 7.5, ec=1.8, ph=5.8, water_temp=20, lights_on=True, volume_l=100)
    assert abs(o.day - 2.3125) < 1e-9 and abs(o.hour - 7.5) < 1e-9


def test_residual_features_and_zero_model():
    z = features(5.8, 1.8, 6.0, True, 17.5)
    assert z.shape == (len(FEATURES),) and abs(z[2] - 1.0) < 1e-12 and abs(z[5] - 0.5) < 1e-12
    assert net_acid((10, 4, 1)) == 3.0
    a, b = ZeroResidual().parts(z[None])
    assert a[0] == 0 and b[0] == 0 and ZeroResidual().support(z[None])[0]
