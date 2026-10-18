"""Static-review regression cases; not run without execution authorization."""
from types import SimpleNamespace
import numpy as np
from shared.evidence import Prediction
from shared.inversion import invert


def fixture(initial=1.0):
    return SimpleNamespace(
        initial=np.array([initial]), lower=np.array([0.8]), upper=np.array([1.3]),
        config={'optimizer': {'iterations': 1, 'difference_step': 0.03,
                              'learning_rate': 0.1, 'max_parameter_step': 0.03}})


def test_corrected_loss_has_exact_binomial_expectation():
    # Enumerate all outcomes of two independent +/-1 shots; no Monte Carlo.
    probability_zero = 0.7
    mean = 2*probability_zero-1
    probabilities = [(1-probability_zero)**2,
                     2*probability_zero*(1-probability_zero), probability_zero**2]
    expected_loss = 0.0
    for overlap, probability in zip((-1., 0., 1.), probabilities):
        def predict(parameters):
            return Prediction(np.array([overlap]), np.array([0.5]),
                              np.array([1-overlap**2]))
        _, history = invert(fixture(), np.array([1.]), predict, lambda rows: None)
        expected_loss += probability*history[0]['relative_pressure_loss']
    np.testing.assert_allclose(expected_loss, 0.5*(mean-1)**2, atol=1e-14)


def test_near_bound_uses_probe_anchored_at_current_model():
    probes = []
    def predict(parameters):
        probes.append(parameters.copy())
        return Prediction(parameters.copy(), np.zeros(1), np.zeros(1))
    _, history = invert(fixture(0.81), np.array([1.]), predict, lambda rows: None)
    np.testing.assert_allclose(probes[1], [0.84])
    np.testing.assert_allclose(probes[2], [0.81])
    np.testing.assert_allclose(history[0]['gradient'], [-0.175])


def test_recorded_step_is_actual_projected_displacement():
    def predict(parameters):
        return Prediction(parameters.copy(), np.zeros(1), np.zeros(1))
    result, history = invert(fixture(1.29), np.array([10.]), predict, lambda rows: None)
    np.testing.assert_allclose(history[0]['step'], result-np.array([1.29]))
