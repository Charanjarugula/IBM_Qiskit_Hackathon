"""Explicit resource ceilings and raw Sampler options. No IBM SDK imports."""
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class Limits:
    shots: int
    max_data_qubits: int
    max_circuits: int
    max_total_shots: int
    max_jobs: int
    circuits_per_job: int
    max_execution_seconds_per_job: int
    max_total_execution_seconds: int
    max_logical_gates: int
    max_compiled_gates: int
    max_depth: int
    max_two_qubit_gates: int

    @classmethod
    def load(cls, path):
        limits = cls(**json.loads(Path(path).read_text()))
        if any(type(value) is not int or value < 1 for value in vars(limits).values()):
            raise ValueError('Hardware limits must be positive integers')
        if limits.shots < 2 or limits.max_data_qubits > 8:
            raise ValueError('Need at least two shots and at most eight data qubits')
        return limits

    def check_totals(self, circuits, jobs):
        if circuits > self.max_circuits or circuits*self.shots > self.max_total_shots:
            raise ValueError('Circuit/shot budget exceeded')
        if jobs > self.max_jobs or jobs*self.max_execution_seconds_per_job > self.max_total_execution_seconds:
            raise ValueError('Job count or reserved execution-time ceiling exceeded')


def plan(problem, limits, mode):
    per_call = len(problem.sources)*len(problem.receivers)*sum(len(problem.impulses(t)) for t in problem.times)
    calls = (problem.config['optimizer']['iterations']*(1+2*len(problem.initial))+1
             if mode == 'invert' else 1)
    if mode in ('pilot', 'experiment'):
        # Minimum four readout calibrations; the final count depends on placement.
        per_call = per_call*(2 if mode == 'experiment' else 1)+4
        calls = 3 if mode == 'experiment' else 1
    jobs = calls*ceil(per_call/limits.circuits_per_job)
    circuits = calls*per_call
    limits.check_totals(circuits, jobs)
    return {'mode': mode, 'forward_calls': calls, 'circuits': circuits,
            'shots': circuits*limits.shots, 'jobs': jobs,
            'counts_are_lower_bounds': mode in ('pilot', 'experiment'),
            'execution_seconds_ceiling': jobs*limits.max_execution_seconds_per_job,
            'timing_note': 'Ceiling, not a runtime estimate; excludes queue and compilation time'}


def raw_sampler_options(limits):
    # Sampler has no Estimator resilience_level. Do not pass one.
    return {'default_shots': limits.shots,
            'max_execution_time': limits.max_execution_seconds_per_job,
            'dynamical_decoupling': {'enable': False},
            'twirling': {'enable_gates': False, 'enable_measure': False},
            'execution': {'init_qubits': True, 'meas_type': 'classified'}}
