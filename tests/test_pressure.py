"""Independent classical pressure checks; prepared but not executed."""
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.linalg import expm
from shared.problem import Problem, load_observations
from classical.reference import ClassicalForward

ROOT = Path(__file__).resolve().parents[1]


def problem():
    return Problem.load(ROOT/'configs/small_1d.json')


def test_pressure_matches_independent_second_order_system():
    p = problem()
    theta = np.array([1.0, 1.1])
    speeds = p.speeds(theta)
    laplacian = np.array([[-2., 1.], [1., -2.]])/p.config['spacing_m']**2
    generator = np.block([[np.zeros((2, 2)), np.eye(2)],
                          [np.diag(speeds**2)@laplacian, np.zeros((2, 2))]])
    expected = np.zeros(p.trace_shape)
    for si, source in enumerate(p.sources):
        initial = np.zeros(4)
        initial[source] = p.config['source']['amplitude_pa']
        for ti, time in enumerate(p.times):
            expected[si, ti] = (expm(time*generator)@initial)[p.receivers]
    np.testing.assert_allclose(ClassicalForward(p).predict(theta).mean, expected, atol=1e-11)


def test_initial_pressure_is_independent_of_source_speed():
    p = problem()
    p.times = np.array([0.0])
    actual = ClassicalForward(p).predict([0.9, 1.2]).mean[:, 0, :]
    np.testing.assert_allclose(actual, np.eye(2), atol=1e-12)


def test_midpoint_source_pressure_increment():
    p = problem()
    p.config['source'] = {'kind': 'pressure_rate', 'bin_edges_s': [0., 2e-6, 4e-6],
                          'values_pa_s': [1e6, -1e6]}
    np.testing.assert_allclose(p.impulses(3e-6), [(2e-6, 2.), (0.5e-6, -1.)])


def test_mismatched_acquisition_is_rejected(tmp_path):
    p = problem()
    path = tmp_path/'observations.npz'
    np.savez(path, pressure_pa=np.ones(p.trace_shape), acquisition_signature='different')
    with pytest.raises(ValueError, match='different acquisition'):
        load_observations(path, p)


@pytest.mark.parametrize('field,value', [('sources', [0.5, 1]), ('region_ids', [0.1, 1])])
def test_fractional_identifiers_cannot_silently_round(tmp_path, field, value):
    cfg = json.loads((ROOT/'configs/small_1d.json').read_text())
    cfg[field] = value
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='integers'):
        Problem.load(path)
