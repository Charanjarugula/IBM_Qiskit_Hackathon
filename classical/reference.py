"""Exact sparse classical propagation of the SAME reduced pressure problem as the gate path.

This is the matrix-exponential reference approach already used in Anik's qwave/classical.py,
with physical pressure initialization/readout and source quadrature made explicit.
It is separate from the reusable Stride full-grid adjoint baseline.
"""
import numpy as np
from scipy.sparse.linalg import expm_multiply
from ._core.evidence import Prediction


class ClassicalForward:
    def __init__(self, problem):
        self.problem = problem
        self.calls = 0
        self.evolutions = 0

    def predict(self, parameters):
        self.calls += 1
        p = self.problem
        c = p.speeds(parameters)
        H = p.hamiltonian(parameters)
        output = np.zeros(p.trace_shape)
        for si, source in enumerate(p.sources):
            initial = np.zeros(H.shape[0], dtype=complex)
            initial[source] = 1
            for ti, time in enumerate(p.times):
                for duration, pressure_increment in p.impulses(time):
                    state = expm_multiply(-1j*duration*H, initial)
                    if np.max(np.abs(state.imag)) > 1e-9:
                        raise ArithmeticError('Real acoustic state acquired an imaginary component')
                    output[si, ti] += pressure_increment * c[p.receivers]/c[source] * state[p.receivers].real
                    self.evolutions += 1
        return Prediction(output, np.zeros_like(output), np.zeros_like(output))

    def statistics(self):
        return {'forward_calls': self.calls, 'sparse_evolutions': self.evolutions,
                'quantum_shots': 0, 'reference': 'exact discrete evolution, midpoint source quadrature'}
