"""Offline CSV/JSON comparison of matched fixed-map gate and annealing runs.

Reads saved outputs only. Never simulates, connects to IBM or submits jobs.
Completion, repeat spread and an explicit accuracy target are reported separately.
Use --require-accuracy to fail the command when any method misses the target;
noisy stress cases are allowed to miss it and must still be shown.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from classical._core.problem import Problem
from .reference import load_reference


def read(path):
    return json.loads(Path(path).read_text())


def pressure(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Invalid pressure array')
    return array


def metrics(values, expected):
    difference = values-expected
    norm = float(np.linalg.norm(expected))
    if norm <= 1e-12:
        raise ValueError('Near-zero reference does not support relative error')
    return {'rmse_pa': float(np.sqrt(np.mean(difference**2))),
            'relative_l2': float(np.linalg.norm(difference)/norm)}


def validate_run(directory, reference, problem):
    manifest = read(directory/'manifest.json')
    hashes = manifest['source_sha256']
    # Validate provenance without requiring unrelated reporting code to be identical.
    for name, digest in hashes.items():
        path = (directory/'source'/name).resolve()
        if not path.is_relative_to((directory/'source').resolve()):
            raise ValueError('Invalid source path in manifest')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Changed source snapshot: {name}')
    for name in ('classical/_core/problem.py', 'classical/_core/grid.py', 'classical/reference.py'):
        if hashes.get(name) != reference['source_sha256'][name]:
            raise ValueError('Different shared wave model; do not combine these runs')
    other = Problem.load(directory/'config.json')
    comparison = read(directory/'comparison.json')
    expected = {'acquisition_signature': problem.acquisition_signature,
                'parameters': reference['parameters'], 'base_speed_ratio': problem.base.tolist(),
                'region_ids': problem.regions.tolist()}
    if (any(comparison.get(key) != value for key, value in expected.items())
            or other.acquisition_signature != problem.acquisition_signature
            or other.base.tolist() != problem.base.tolist()
            or other.regions.tolist() != problem.regions.tolist()
            or other.config['trotter_step_s'] != reference['trotter_step_s']):
        raise ValueError('Runs differ in map, acquisition or time step; do not combine them')
    completed = directory/'completed.json'
    complete = (completed.exists() and read(completed).get('status') == 'completed'
                and not (directory/'failure.json').exists())
    return manifest['engine'], complete


def validate_gate_counts(directory, rows, problem, settings, complete):
    """Rebuild every completed prediction from saved counts, not summary claims."""
    from .readout import ancilla_statistics
    groups = {}
    for path in sorted(directory.glob('counts_*.json')):
        record = read(path)
        groups.setdefault(record['forward_call'], []).append(record)
    expected = [(int(source), float(time), int(receiver), float(duration))
                for source in problem.sources for time in problem.times
                for duration, _ in problem.impulses(time) for receiver in problem.receivers]
    if complete and set(groups) != {row['repeat'] for row in rows}:
        raise ValueError('Missing or extra gate forward calls')
    speeds = problem.speeds(settings['parameters'])
    for row in rows:
        records = groups.get(row['repeat'], [])
        actual = [(r['source'], r['time_s'], r['receiver'], r['evolution_time_s']) for r in records]
        if sorted(actual) != sorted(expected):
            raise ValueError('Incomplete or duplicated gate observations')
        mean, variance = np.zeros(problem.trace_shape), np.zeros(problem.trace_shape)
        for record in records:
            if record['parameters'] != settings['parameters'] or record['shots'] != settings['shots']:
                raise ValueError('Gate counts describe different parameters or shots')
            si = list(problem.sources).index(record['source'])
            ti = list(problem.times).index(record['time_s'])
            ri = list(problem.receivers).index(record['receiver'])
            increment = dict(problem.impulses(record['time_s']))[record['evolution_time_s']]
            scale = increment*speeds[record['receiver']]/speeds[record['source']]
            if not np.isclose(scale, record['physical_scale_pa'], rtol=1e-12, atol=1e-12):
                raise ValueError('Incorrect physical pressure scaling')
            overlap, se, _ = ancilla_statistics(record['counts'], record['shots'])
            mean[si, ti, ri] += scale*overlap
            variance[si, ti, ri] += (scale*se)**2
        if (not np.allclose(mean, pressure(row['pressure_pa'], problem.trace_shape), rtol=1e-12, atol=1e-12)
                or not np.allclose(np.sqrt(variance), pressure(row['standard_error_pa'], problem.trace_shape),
                                   rtol=1e-12, atol=1e-12)):
            raise ValueError('Saved pressure/uncertainty does not match raw gate counts')


def summarize(label, directory, rows, requested, complete, expected, ideal, target, shape):
    good = [row for row in rows if row['status'] == 'completed']
    values = [pressure(row['pressure_pa'], shape) for row in good]
    for row, value in zip(good, values):
        row['error_vs_exact'] = metrics(value, expected)
        row['error_vs_ideal_circuit'] = metrics(value, ideal)
        if not np.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0:
            raise ValueError('Invalid CPU timing')
    complete = bool(complete and len(good) == requested and len(rows) == requested)
    passes = complete and all(row['error_vs_exact']['relative_l2'] <= target for row in good)
    mean = np.mean(values, axis=0) if values else None
    return {'method': label, 'run_directory': str(directory), 'complete': complete,
            'completed_repeats': len(good), 'requested_repeats': requested,
            'all_repeats_within_target': bool(passes), 'relative_l2_target': target,
            'mean_pressure_pa': mean.tolist() if values else None,
            'error_of_mean_vs_exact': metrics(mean, expected) if values else None,
            'repeat_standard_deviation_pa': np.std(values, axis=0, ddof=1).tolist() if len(values)>1 else None,
            'mean_forward_wall_seconds': float(np.mean([row['wall_seconds'] for row in good])) if good else None,
            'all_repeats': rows}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--gate-runs', nargs='+', required=True)
    cli.add_argument('--annealing-runs', nargs='*', default=[])
    cli.add_argument('--output', required=True)
    cli.add_argument('--max-relative-error', type=float, default=0.01)
    cli.add_argument('--require-accuracy', action='store_true')
    args = cli.parse_args(argv)
    if not np.isfinite(args.max_relative_error) or args.max_relative_error <= 0:
        cli.error('Accuracy target must be finite and positive')
    output = Path(args.output)
    if output.exists():
        cli.error('Use a new output directory')
    directories = [Path(path).resolve() for path in args.gate_runs+args.annealing_runs]
    if len(set(directories)) != len(directories):
        cli.error('A run cannot appear twice')
    first = directories[0]
    problem = Problem.load(first/'config.json')
    reference = load_reference(first/'reference.json', problem, read(first/'settings.json')['parameters'])
    exact = pressure(reference['classical_pressure_pa'], problem.trace_shape)
    ideal = pressure(reference['ideal_pressure_pa'], problem.trace_shape)
    summary = []
    for index, directory in enumerate(directories):
        engine, complete = validate_run(directory, reference, problem)
        settings = read(directory/'settings.json')
        if settings['parameters'] != reference['parameters']:
            raise ValueError('Settings describe different fixed parameters')
        requested = settings['repetitions']
        if type(requested) is not int or not 1 <= requested <= 10:
            raise ValueError('Invalid repetition count')
        rows = read(directory/'repeats.json') if (directory/'repeats.json').exists() else []
        if index < len(args.gate_runs):
            if engine != 'gate_forward_benchmark':
                raise ValueError('Expected fixed-map gate benchmark, not an inversion run')
            load_reference(directory/'reference.json', problem, reference['parameters'])
            if len({row['repeat'] for row in rows}) != len(rows):
                raise ValueError('Duplicate gate repetition')
            if any(type(row['repeat']) is not int or not 0 <= row['repeat'] < requested for row in rows):
                raise ValueError('Invalid gate repetition index')
            validate_gate_counts(directory, rows, problem, settings, complete)
            label = 'gate_ideal' if settings['noise_probability'] == 0 else 'gate_depolarizing'
            item = summarize(label, directory, rows, requested, complete, exact, ideal,
                             args.max_relative_error, problem.trace_shape)
            item.update(noise_probability=settings['noise_probability'], shots=settings['shots'],
                        execution=read(directory/'completed.json') if (directory/'completed.json').exists() else None)
            summary.append(item)
            classical_path = directory/'classical.json'
            classical_rows = read(classical_path) if classical_path.exists() else []
            if (len({row['repeat'] for row in classical_rows}) != len(classical_rows)
                    or any(type(row['repeat']) is not int or not 0 <= row['repeat'] < requested
                           for row in classical_rows)):
                raise ValueError('Invalid classical repetition indices')
            summary.append(summarize('classical_exact', directory,
                [dict(row, status='completed') for row in classical_rows], requested, complete,
                exact, ideal, args.max_relative_error, problem.trace_shape))
        else:
            if engine != 'annealing_forward_comparison':
                raise ValueError('Expected an annealing forward run')
            if (not settings['algorithms'] or len(set(settings['algorithms'])) != len(settings['algorithms'])
                    or any(name not in ('sa', 'sqa') for name in settings['algorithms'])):
                raise ValueError('Invalid annealing algorithm selection')
            if len({(row['algorithm'], row['seed']) for row in rows}) != len(rows):
                raise ValueError('Duplicate annealing repetition')
            if any(not settings['seed'] <= row['seed'] < settings['seed']+requested for row in rows):
                raise ValueError('Unexpected annealing repeat seed')
            if any(row['algorithm'] not in settings['algorithms'] for row in rows):
                raise ValueError('Unexpected annealing algorithm')
            for algorithm in settings['algorithms']:
                selected = [row for row in rows if row['algorithm'] == algorithm]
                summary.append(summarize(algorithm, directory, selected, requested, complete,
                    exact, ideal, args.max_relative_error, problem.trace_shape))
    result = {'scope': 'Matched fixed-map forward waves, not full reconstruction or the paper benchmark',
              'parameters': reference['parameters'], 'acquisition_signature': problem.acquisition_signature,
              'ideal_circuit_error_vs_exact': metrics(ideal, exact), 'methods': summary,
              'all_runs_complete': all(item['complete'] for item in summary),
              'all_methods_within_target': all(item['all_repeats_within_target'] for item in summary),
              'timing_note': 'CPU forward wall times. Gate times include compilation and count recording; '
                             'annealing includes QUBO construction and recording. Classical times include '
                             'matrix construction/evolution. Excludes CLI startup; no QPU speedup claim.',
              'uncertainty_note': 'Repeat SD describes repeat variability, not systematic error or a '
                                  'confidence interval. Saved gate SE describes shot uncertainty only.'}
    output.mkdir(parents=True)
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'run_directory', 'complete', 'completed_repeats', 'requested_repeats',
                         'all_repeats_within_target', 'relative_l2_target', 'rmse_of_mean_pa',
                         'relative_l2_of_mean', 'mean_forward_wall_seconds'])
        for item in summary:
            error = item['error_of_mean_vs_exact'] or {}
            writer.writerow([item[key] for key in ('method', 'run_directory', 'complete',
                'completed_repeats', 'requested_repeats', 'all_repeats_within_target', 'relative_l2_target')]
                +[error.get('rmse_pa'), error.get('relative_l2'), item['mean_forward_wall_seconds']])
    print(f"Complete: {result['all_runs_complete']}; all repeats within target: {result['all_methods_within_target']}")
    if not result['all_runs_complete'] or (args.require_accuracy and not result['all_methods_within_target']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
