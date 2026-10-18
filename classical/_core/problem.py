"""Matched constant-density pressure problem, with explicit physical units."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from .grid import Grid


@dataclass
class Problem:
    config: dict
    grid: Grid
    base: np.ndarray
    regions: np.ndarray
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sources: np.ndarray
    receivers: np.ndarray
    times: np.ndarray

    @classmethod
    def load(cls, path):
        cfg = json.loads(Path(path).read_text())
        if cfg.get('schema') != 1:
            raise ValueError('Expected configuration schema 1')
        shape = cfg['shape']
        if not 1 <= len(shape) <= 3 or any(type(x) is not int or x < 2 for x in shape):
            raise ValueError('Use a 1D/2D/3D grid with at least two cells per axis')
        grid = Grid(shape)
        base = np.asarray(cfg['base_speed_ratio'], dtype=float).reshape(-1)
        def integer_array(values, name):
            raw = np.asarray(values)
            if raw.dtype.kind not in 'iu':
                raise ValueError(f'{name} must contain integers, not rounded coordinates')
            return raw.astype(int)
        regions = integer_array(cfg['region_ids'], 'region_ids').reshape(-1)
        initial = np.asarray(cfg['initial_parameters'], dtype=float)
        lower = np.asarray(cfg['parameter_lower'], dtype=float)
        upper = np.asarray(cfg['parameter_upper'], dtype=float)
        if base.size != grid.n_vertices or regions.size != base.size:
            raise ValueError('One base speed and region ID required per grid cell')
        if initial.ndim != 1 or not 1 <= len(initial) <= 16:
            raise ValueError('Matched finite-difference inversion supports 1–16 parameters')
        if lower.shape != initial.shape or upper.shape != initial.shape:
            raise ValueError('Parameter bounds must match initial parameters')
        if not all(np.isfinite(a).all() for a in (base, initial, lower, upper)):
            raise ValueError('Nonfinite model values')
        if np.any(base <= 0) or np.any(lower <= 0) or np.any(lower >= upper):
            raise ValueError('Sound speeds and parameter intervals must be positive')
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError('Initial parameters outside bounds')
        if set(regions[regions >= 0]) != set(range(len(initial))) or np.any(regions < -1):
            raise ValueError('Region IDs must be -1 (fixed) or cover each parameter')
        sources = integer_array(cfg['sources'], 'sources')
        receivers = integer_array(cfg['receivers'], 'receivers')
        for indices in (sources, receivers):
            if indices.ndim != 1 or not len(indices) or len(set(indices)) != len(indices):
                raise ValueError('Acquisition lists must be nonempty and unique')
            if np.any(indices < 0) or np.any(indices >= grid.n_vertices):
                raise ValueError('Source/receiver outside grid')
        times = np.asarray(cfg['times_s'], dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(times <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError('Receiver times must be positive and strictly increasing')
        for key in ('spacing_m', 'reference_speed_m_s', 'trotter_step_s'):
            if not np.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f'{key} must be positive')
        source = cfg['source']
        if source['kind'] == 'initial_pressure':
            if not np.isfinite(source['amplitude_pa']) or source['amplitude_pa'] == 0:
                raise ValueError('Initial pressure amplitude must be finite and nonzero')
        elif source['kind'] == 'pressure_rate':
            edges = np.asarray(source['bin_edges_s'], dtype=float)
            rates = np.asarray(source['values_pa_s'], dtype=float)
            if edges.ndim != 1 or rates.ndim != 1 or not rates.size:
                raise ValueError('Pressure-rate source needs nonempty one-dimensional arrays')
            if len(edges) != len(rates) + 1 or edges[0] != 0 or np.any(np.diff(edges) <= 0):
                raise ValueError('Pressure-rate source needs increasing bins starting at zero')
            if not np.isfinite(edges).all() or not np.isfinite(rates).all() or not np.any(rates):
                raise ValueError('Invalid pressure-rate pulse')
        else:
            raise ValueError('Source kind must be initial_pressure or pressure_rate')
        optimizer = cfg['optimizer']
        if type(optimizer['iterations']) is not int or optimizer['iterations'] < 1:
            raise ValueError('Optimizer iterations must be a positive integer')
        for key in ('difference_step', 'learning_rate', 'max_parameter_step'):
            if not np.isfinite(optimizer[key]) or optimizer[key] <= 0:
                raise ValueError(f'Optimizer {key} must be finite and positive')
        return cls(cfg, grid, base, regions, initial, lower, upper, sources, receivers, times)

    def speeds(self, parameters):
        theta = np.asarray(parameters, dtype=float)
        if theta.shape != self.initial.shape or not np.isfinite(theta).all():
            raise ValueError('Invalid parameter vector')
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            raise ValueError('Parameters outside physical bounds')
        ratio = self.base.copy()
        for k, value in enumerate(theta):
            ratio[self.regions == k] = value
        return ratio * self.config['reference_speed_m_s']

    def hamiltonian(self, parameters):
        # pi = p/c, z = auxiliary flux, d(pi,z)/dt = (K z, -K.T pi).
        c = self.speeds(parameters)
        K = sp.diags(c) @ self.grid.incidence() / self.config['spacing_m']
        return sp.bmat([[None, 1j*K], [-1j*K.T, None]], format='csr')

    def impulses(self, time):
        """(propagation duration, physical pressure increment) for each source contribution.

        A finite waveform drives p_t, not p_tt. Midpoint quadrature approximates its
        Duhamel integral; its error is separate from the circuit's Trotter error.
        """
        source = self.config['source']
        if source['kind'] == 'initial_pressure':
            return [(float(time), float(source['amplitude_pa']))]
        edges = source['bin_edges_s']
        out = []
        for a, b, rate in zip(edges[:-1], edges[1:], source['values_pa_s']):
            end = min(b, time)
            if end > a and rate != 0:
                out.append((time - (a + end)/2, rate * (end-a)))
        return out

    @property
    def acquisition_signature(self):
        # Exclude optimizer, truth and circuit approximation settings deliberately.
        keys = ('schema', 'shape', 'spacing_m', 'reference_speed_m_s',
                'sources', 'receivers', 'times_s', 'source')
        payload = {k: self.config[k] for k in keys}
        payload.update(boundary='zero_pressure_ghost', density='constant', units='Pa',
                       model='pressure_flux_v1', source_quadrature='midpoint_impulses')
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def comparison_signature(self):
        # Same inverse problem and update settings, without truth or solver accuracy.
        keys = ('base_speed_ratio', 'region_ids', 'initial_parameters',
                'parameter_lower', 'parameter_upper', 'optimizer')
        payload = {k: self.config[k] for k in keys}
        payload.update(acquisition=self.acquisition_signature,
                       objective='variance_corrected_pressure_l2_v1')
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def trace_shape(self):
        return (len(self.sources), len(self.times), len(self.receivers))


def load_observations(path, problem):
    with np.load(path, allow_pickle=False) as data:
        if str(data['acquisition_signature'].item()) != problem.acquisition_signature:
            raise ValueError('Observations use a different acquisition/physics convention')
        observed = np.asarray(data['pressure_pa'], dtype=float)
    if observed.shape != problem.trace_shape or not np.isfinite(observed).all():
        raise ValueError('Invalid observed pressure array')
    if np.linalg.norm(observed) == 0:
        raise ValueError('Zero observations do not define this normalized objective')
    return observed
