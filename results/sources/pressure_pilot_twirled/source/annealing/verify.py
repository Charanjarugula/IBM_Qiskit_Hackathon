"""Small algebra/API checks, not a wave experiment. python -m annealing.verify"""
import itertools
import numpy as np
import dimod
from .qubo import encoding, least_squares_bqm, normalized_ising
from .forward import Settings, AnnealingForward


def main():
    A = np.array([[1., -.2], [.2, 1.]])
    b, center = np.array([-.23, .31]), np.array([.1, -.1])
    W = encoding(2, 3, .5)
    bqm = least_squares_bqm(A, b, center, W)
    spin, scale = normalized_ising(bqm)
    for bits in itertools.product((0, 1), repeat=6):
        z = np.asarray(bits)
        residual = A@(center+W@z)-b
        expected = float(residual@residual)
        assert np.isclose(bqm.energy(dict(enumerate(bits))), expected, atol=1e-12)
        assert np.isclose(spin.energy(dict(enumerate(2*z-1)))*scale, expected, atol=1e-12)
    assert np.all(W@np.zeros(6) == 0)
    assert W[0, 0] < 0 and W[0, 1] > 0
    # The deterministic CN control must conserve norm for a skew generator.
    G = np.array([[0., -2.], [2., 0.]])
    C = np.eye(2)-.1*G/2
    update = np.eye(2)+np.linalg.solve(C, .1*G)
    assert np.allclose(update.T@update, np.eye(2), atol=1e-12)
    # Verify both installed sampler APIs on a one-variable fixture, not acoustics.
    from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler
    fixture = dimod.BinaryQuadraticModel({0: 1.}, {}, 0., dimod.SPIN)
    for algorithm, sampler in [('sqa', PathIntegralAnnealingSampler()), ('sa', SimulatedAnnealingSampler())]:
        options = dict(num_reads=2, num_sweeps=2, seed=7, beta_schedule_type='custom')
        if algorithm == 'sqa':
            options.update(Hp_field=[.1, 4.], Hd_field=[2., 0.])
        else:
            options.update(beta_schedule=[.1, 4.], randomize_order=True)
        result = sampler.sample(fixture, **options)
        assert result.record.num_occurrences.sum() == 2 and set(result.variables) == {0}
    # Exact enumeration substitutes for annealing to test decoding/refinement.
    # No probabilistic convergence assumption belongs in a correctness test.
    logs = []
    solver = AnnealingForward(Settings(bits=3, refinements=3, reads=64, sweeps=2), 'sa', 7, logs.append)
    class Enumerate:
        def sample(self, model, **kwargs):
            return dimod.ExactSolver().sample(model)
    solver.sampler = Enumerate()
    answer = solver.solve(A, b, {})
    assert np.linalg.norm(A@answer-b) < np.linalg.norm(b)*.03
    assert all(logs[i]['best_residual_l2'] >= logs[i+1]['best_residual_l2'] for i in range(len(logs)-1))
    assert solver.calls == 3 and np.all(np.isfinite(answer))
    print('Passed: all 64 QUBO energies, Ising scaling, signed encoding, CN norm, sampler APIs, refinement.')


if __name__ == '__main__':
    main()
