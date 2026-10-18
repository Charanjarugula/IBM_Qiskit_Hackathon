"""Anneal the same acoustic equations as the gate circuit; estimate signed pressure.

G=-iH is real skew-symmetric. Crank-Nicolson uses
(I-hG/2)(y_next-y)=hGy. Each linear solve becomes a QUBO.
SQA uses D-Wave's path-integral Monte Carlo on a transverse-field Ising model:
https://github.com/dwavesystems/dwave-samplers . It is a classical simulation
of annealing, not an IBM job or a real-time quantum-hardware emulator.
No classical exact solution, observations, or fitted truth seed the annealer.
"""
from dataclasses import dataclass
from math import ceil
from time import perf_counter
import numpy as np
import dimod
from .qubo import encoding, least_squares_bqm, normalized_ising


@dataclass(frozen=True)
class Settings:
    bits: int = 6
    refinements: int = 3
    reads: int = 16
    sweeps: int = 200
    beta_end: float = 128.0
    transverse_start: float = 2.0
    max_variables: int = 512
    max_bqm_calls: int = 2000
    max_seconds: float = 600.0

    def validate(self):
        for key in ('bits', 'refinements', 'reads', 'sweeps', 'max_variables', 'max_bqm_calls'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not 2 <= self.bits <= 10 or self.sweeps < 2 or self.refinements > 8:
            raise ValueError('Use 2–10 bits, 1–8 refinements and at least two sweeps')
        if (not np.isfinite([self.beta_end, self.transverse_start, self.max_seconds]).all()
                or self.beta_end <= 0.1 or self.transverse_start <= 0 or self.max_seconds <= 0):
            raise ValueError('Invalid annealing schedule or time budget')


def resource_plan(problem, settings):
    settings.validate()
    dimension = problem.grid.n_vertices+problem.grid.n_edges
    steps = len(problem.sources)*sum(
        max(1, ceil(duration/problem.config['trotter_step_s']))
        for time in problem.times for duration, _ in problem.impulses(time))
    calls = steps*settings.refinements
    if dimension*settings.bits > settings.max_variables or calls > settings.max_bqm_calls:
        raise ValueError('Annealing variable/solve budget exceeded before any sampling')
    return {'state_components': dimension, 'binary_variables_per_bqm': dimension*settings.bits,
            'time_steps_per_repeat': steps, 'bqm_calls_per_repeat_at_most': calls,
            'annealing_reads_per_repeat_at_most': calls*settings.reads,
            'sweeps_per_read': settings.sweeps,
            'scope': 'One fixed sound-speed map; forward prediction only, no inversion',
            'time_step_s': problem.config['trotter_step_s']}


def generator(problem, parameters):
    matrix = -1j*problem.hamiltonian(parameters).toarray()
    if np.max(np.abs(matrix.imag)) > 1e-12 or not np.allclose(matrix.real, -matrix.real.T):
        raise ValueError('Expected a real skew-symmetric acoustic generator')
    return matrix.real


def pressure_forward(problem, parameters, solve_step):
    """Shared acquisition and midpoint source quadrature; no amplitude squaring."""
    G, c = generator(problem, parameters), problem.speeds(parameters)
    pressure = np.zeros(problem.trace_shape)
    for si, source in enumerate(problem.sources):
        for ti, time in enumerate(problem.times):
            for duration, increment in problem.impulses(time):
                steps = max(1, ceil(duration/problem.config['trotter_step_s']))
                h = duration/steps
                state = np.zeros(len(G))
                state[source] = 1
                A = np.eye(len(G))-h*G/2
                for step in range(steps):
                    rhs = h*G@state
                    state += solve_step(A, rhs, {'source': int(source), 'time_s': float(time),
                                                'duration_s': duration, 'step': step})
                    if not np.isfinite(state).all():
                        raise ValueError('Nonfinite annealed acoustic state')
                pressure[si, ti] += increment*c[problem.receivers]/c[source]*state[problem.receivers]
    return pressure


def direct_cn(problem, parameters):
    # Separate numerical control; never passed into AnnealingForward.
    return pressure_forward(problem, parameters, lambda A, rhs, _: np.linalg.solve(A, rhs))


class AnnealingForward:
    def __init__(self, settings, algorithm, seed, record):
        from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler
        settings.validate()
        if algorithm not in ('sqa', 'sa') or not 0 <= seed < 2**31:
            raise ValueError('Expected sqa/sa and a seed in [0, 2**31)')
        self.settings, self.algorithm, self.record = settings, algorithm, record
        self.rng = np.random.default_rng(seed)
        self.sampler = PathIntegralAnnealingSampler() if algorithm == 'sqa' else SimulatedAnnealingSampler()
        progress = np.linspace(0, 1, settings.sweeps)
        self.Hp = np.geomspace(0.1, settings.beta_end, settings.sweeps)
        self.Hd = settings.transverse_start*(1-progress)**2  # Nonzero driver, ending at zero.
        self.started, self.calls = perf_counter(), 0

    def expired(self):
        return perf_counter()-self.started >= self.settings.max_seconds

    def solve(self, A, rhs, context):
        s = self.settings
        radius = float(np.linalg.norm(rhs))
        if radius == 0:
            return np.zeros_like(rhs)
        # sigma_min(I-hG/2)>=1, so ||delta||<=||rhs||: a physical, truth-free box.
        center = np.zeros_like(rhs)
        best_residual = float(rhs@rhs)
        for refinement in range(s.refinements):
            if self.expired() or self.calls >= s.max_bqm_calls:
                raise TimeoutError('Annealing budget exhausted; partial evidence retained')
            W = encoding(len(rhs), s.bits, radius)
            bqm = least_squares_bqm(A, rhs, center, W)
            spin, scale = normalized_ising(bqm)
            seed = int(self.rng.integers(0, 2**31))
            options = dict(num_reads=s.reads, seed=seed, num_sweeps=s.sweeps,
                           interrupt_function=self.expired)
            if self.algorithm == 'sqa':
                options.update(beta_schedule_type='custom', Hp_field=self.Hp, Hd_field=self.Hd)
            else:
                options.update(beta_schedule_type='custom', beta_schedule=self.Hp,
                               randomize_order=True)
            start = perf_counter()
            samples = self.sampler.sample(spin, **options)
            elapsed = perf_counter()-start
            self.calls += 1
            if self.expired() or int(samples.record.num_occurrences.sum()) != s.reads:
                raise TimeoutError('Sampler interrupted; no completed result is reported')
            binary = samples.change_vartype(dimod.BINARY, inplace=False)
            candidate_bits = np.array([binary.first.sample[i] for i in range(W.shape[1])])
            candidate = center+W@candidate_bits
            residual = A@candidate-rhs
            candidate_residual = float(residual@residual)
            accepted = candidate_residual < best_residual
            if accepted:
                center, best_residual = candidate, candidate_residual
            self.record({**context, 'call': self.calls, 'refinement': refinement, 'seed': seed,
                         'radius': radius, 'ising_bias_scale': scale,
                         'accepted': accepted, 'best_residual_l2': best_residual**0.5,
                         'anneal_seconds': elapsed, 'best_bits': candidate_bits.tolist(),
                         'sample_energies': binary.record.energy.tolist(),
                         'sample_occurrences': binary.record.num_occurrences.tolist()})
            radius /= 4
        return center
