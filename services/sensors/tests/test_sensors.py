import numpy as np

from ponics_sensors import Probes


def test_same_seed_same_readings():
    a, b = Probes(5), Probes(5)
    ra = [a.read(t * 900.0, 5.8, 1.8, 20.0) for t in range(50)]
    rb = [b.read(t * 900.0, 5.8, 1.8, 20.0) for t in range(50)]
    assert ra == rb


def test_readings_are_near_truth():
    p = Probes(1)
    r = np.array([p.read(t * 900.0, 5.8, 1.8, 25.0) for t in range(96)])
    assert abs(r[:, 0].mean() - 5.8) < 0.15
    assert abs(r[:, 1].mean() - 1.8) < 0.1


def test_recalibration_resets_offset():
    p = Probes(2)
    p.ph_offset = 0.5
    p.read(7 * 86400 + 9.5 * 3600, 5.8, 1.8, 20.0)
    assert abs(p.ph_offset) < 0.1


def test_temperature_compensation_error_is_small():
    p = Probes(3)
    p.ec_gain = 1.0
    p.p = type(p.p)(ec_noise_rel=0.0, ec_noise_abs=0.0)
    _, ec = p.read(0.0, 5.8, 1.8, 15.0)
    assert abs(ec - 1.8) < 0.04
