"""Offline experiment report from saved IBM counts. Never connects or submits.

python -m ibm_hardware.report --run-directory runs/experiment --execute
"""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import numpy as np
from .readout import analyze_pilot

LABELS = {'original': 'Original raw', 'optimized': 'Optimized raw', 'corrected': 'Readout corrected'}
COLORS = {'original': '#c45b26', 'optimized': '#3156b8', 'corrected': '#008477', 'ideal': '#20252b'}


def pressure_draws(records, counts, shots, shape, rng, samples):
    """Smoothed parametric bootstrap; paired raw/corrected counts, shared calibrations.

    Half-count smoothing avoids zero uncertainty for all-zero/all-one samples.
    This approximate IID model does not include hardware drift or model bias.
    """
    size = int(np.prod(shape))
    output = {key: np.zeros((samples, size)) for key in LABELS}
    calibrations = {}
    for record, raw in zip(records, counts):
        if record['kind'] == 'calibration':
            key = record['readout_qubit'], record['prepared_state']
            pooled = calibrations.setdefault(key, [0, 0])
            pooled[0] += raw.get('1', 0)
            pooled[1] += shots
    models = {}
    for qubit in {q for q, _ in calibrations}:
        errors = []
        for state in (0, 1):
            ones, total = calibrations[(qubit, state)]
            p1 = rng.binomial(total, (ones+0.5)/(total+1), size=samples)/total
            errors.append(p1 if state == 0 else 1-p1)
        e0, e1 = errors
        contrast = 1-e0-e1
        if np.any(contrast <= 0.1):
            raise ValueError('Calibration uncertainty is too large for stable bootstrap inversion')
        models[qubit] = e0, e1, contrast
    for record, raw in zip(records, counts):
        if record['kind'] != 'wave':
            continue
        index = np.ravel_multi_index((record['source_index'], record['time_index'], record['receiver_index']), shape)
        overlap = 1-2*rng.binomial(shots, (raw.get('1', 0)+0.5)/(shots+1), size=samples)/shots
        scale, variant = record['physical_scale_pa'], record['variant']
        output[variant][:, index] += scale*overlap
        if variant == 'optimized':
            e0, e1, contrast = models[record['readout_qubit']]
            output['corrected'][:, index] += scale*(overlap+e0-e1)/contrast
    return output


def rmse(values, reference):
    return float(np.sqrt(np.mean((np.asarray(values)-reference)**2)))


def generate_report(directory):
    from classical._core.problem import Problem
    from simulator.reference import load_reference
    directory = Path(directory)
    plan = json.loads((directory/'experiment_plan.json').read_text())
    raw_variants = [key for key in ('original', 'optimized')
                    if any(r.get('variant') == key for r in plan['wave_records'])]
    if 'optimized' not in raw_variants:
        raise ValueError('Experiment has no optimized wave arm')
    problem = Problem.load(directory/'config.json')
    reference = load_reference(directory/'reference.json', problem, plan['parameters'])
    ideal = np.asarray(reference['ideal_pressure_pa']).reshape(-1)
    classical = np.asarray(reference['classical_pressure_pa']).reshape(-1)
    rounds, payloads, usage = [], [], []
    for repetition in range(plan['repetitions']):
        prefix = f'job_{repetition:04d}'
        count_path = directory/(prefix+'_counts.json')
        if not count_path.exists():
            count_path = directory/(prefix+'_recovered')/(prefix+'_counts.json')
        if not count_path.exists():
            continue
        intent = json.loads((directory/(prefix+'_intent.json')).read_text())
        submitted = json.loads((directory/(prefix+'_submitted.json')).read_text())
        records, shots = intent['records'], intent['shots']
        if (intent['backend'] != plan['backend'] or shots != plan['shots_per_circuit']
                or records != plan['records_by_repetition'][repetition]):
            raise ValueError('Saved job does not match the predeclared experiment')
        counts = json.loads(count_path.read_text())
        if len(records) != len(counts):
            raise ValueError('Result count differs from predeclared circuits')
        old = [(r, c) for r, c in zip(records, counts) if r.get('variant') == 'original']
        new = [(r, c) for r, c in zip(records, counts) if r.get('variant') != 'original']
        b = analyze_pilot([r for r, _ in new], [c for _, c in new], shots, problem.trace_shape)
        values = {'optimized': b['raw_pressure_pa']}
        errors = {'optimized': b['raw_standard_error_pa']}
        if old:
            a = analyze_pilot([r for r, _ in old], [c for _, c in old], shots, problem.trace_shape, mitigate=False)
            values['original'], errors['original'] = a['raw_pressure_pa'], a['raw_standard_error_pa']
        if b['mitigated'] is not None:
            values['corrected'] = b['mitigated']['pressure_pa']
            errors['corrected'] = b['mitigated']['standard_error_pa']
        metrics_path = directory/(prefix+'_metrics.json')
        if not metrics_path.exists():
            metrics_path = directory/(prefix+'_recovered')/(prefix+'_metrics.json')
        seconds = None
        if metrics_path.exists():
            seconds = json.loads(metrics_path.read_text()).get('usage', {}).get('quantum_seconds')
            if seconds is not None and (not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0):
                seconds = None
        usage.append(seconds)
        rounds.append({'repetition': repetition+1, 'job_id': submitted['job_id'],
                       'reported_qpu_seconds': seconds, 'calibration': b['calibration'],
                       'calibration_checks_passed': b['calibration_checks_passed'],
                       'pressure_pa': values, 'standard_error_pa': errors,
                       'rmse_vs_ideal_pa': {k: rmse(np.asarray(v).reshape(-1), ideal) for k, v in values.items()}})
        payloads.append((records, counts))
    if not rounds:
        raise ValueError('No saved hardware counts; no report or placeholder results generated')
    accepted = all(row['calibration_checks_passed'] for row in rounds)
    variants = raw_variants+(['corrected'] if accepted else [])
    means = {key: np.mean([np.asarray(row['pressure_pa'][key]).reshape(-1) for row in rounds], axis=0) for key in variants}
    se = {key: np.sqrt(np.sum([np.asarray(row['standard_error_pa'][key]).reshape(-1)**2 for row in rounds], axis=0))/len(rounds) for key in variants}
    scores = {key: {'rmse_vs_ideal_pa': rmse(mean, ideal), 'rmse_vs_classical_pa': rmse(mean, classical)} for key, mean in means.items()}
    resources = {}
    for variant in raw_variants:
        wave_records = [r for r in plan['wave_records'] if r['variant'] == variant]
        resources[variant] = {
            'wave_circuits_per_repetition': len(wave_records),
            'two_qubit_gates_per_repetition': sum(r['two_qubit_gates'] for r in wave_records),
            'maximum_circuit_depth': max(r['depth'] for r in wave_records),
            'readout_qubits': sorted({r['readout_qubit'] for r in wave_records})}
    calibration_count = sum(r['kind'] == 'calibration' for r in plan['records_by_repetition'][0])
    resources['readout_calibration'] = {
        'circuits_per_repetition': calibration_count,
        'shots_per_repetition': calibration_count*plan['shots_per_circuit'],
        'additional_corrected_wave_shots': 0,
        'timing': 'Shared job metrics do not isolate calibration QPU time'}
    comparisons, uncertainty_failure = {}, None
    if accepted:
        try:
            rng = np.random.default_rng(plan['seed'])
            samples = [pressure_draws(records, counts, plan['shots_per_circuit'], problem.trace_shape,
                                     rng, plan['uncertainty_samples']) for records, counts in payloads]
            distribution = {key: np.sqrt(np.mean((np.mean([s[key] for s in samples], axis=0)-ideal)**2, axis=1)) for key in variants}
            for key in variants:
                scores[key]['rmse_interval_95_pa'] = np.quantile(distribution[key], [0.025, 0.975]).tolist()
            for name, before, after in (('compilation_and_construction', 'original', 'optimized'),
                                        ('readout_mitigation', 'optimized', 'corrected')):
                if before not in variants:
                    continue
                delta = distribution[before]-distribution[after]
                lower, upper = np.quantile(delta, [0.025, 0.975])
                initial = scores[before]['rmse_vs_ideal_pa']
                gain = 100*(initial-scores[after]['rmse_vs_ideal_pa'])/initial if initial > 1e-12 else None
                gain_interval = (np.quantile(100*delta/distribution[before], [0.025, 0.975]).tolist()
                                 if np.all(distribution[before] > 1e-12) else None)
                verdict = 'improvement supported' if lower > 0 else 'worsening supported' if upper < 0 else 'inconclusive'
                comparisons[name] = {'error_reduction_percent': gain, 'error_reduction_interval_95_percent': gain_interval,
                                     'rmse_reduction_interval_95_pa': [float(lower), float(upper)],
                                     'assessment_under_iid_model': verdict}
        except ValueError as error:
            uncertainty_failure = str(error)
    result = {'backend': plan['backend'], 'parameters': plan['parameters'],
              'complete': len(rounds) == plan['repetitions'], 'planned_repetitions': plan['repetitions'],
              'calibration_checks_passed': accepted, 'rounds': rounds, 'metrics': scores,
              'comparisons': comparisons, 'uncertainty_failure': uncertainty_failure,
              'circuit_resources': resources,
              'reported_qpu_seconds': sum(usage) if all(s is not None for s in usage) else None,
              'pressure_pa': {k: v.tolist() for k, v in means.items()},
              'standard_error_pa': {k: v.tolist() for k, v in se.items()},
              'ideal_pressure_pa': ideal.tolist(), 'classical_pressure_pa': classical.tolist(),
              'trotter_rmse_pa': rmse(ideal, classical),
              'reference_source_sha256': reference['source_sha256'],
              'uncertainty': 'Approximate 95% smoothed parametric bootstrap intervals, with paired raw/corrected '
                             'wave counts and shared calibration uncertainty. Assumes independent shots and '
                             'stable assignment errors; excludes drift, preparation bias and correlated device errors.',
              'scope': plan['scope']}
    (directory/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_display(directory, problem, result)
    return result


def write_display(directory, problem, result):
    keys = list(result['pressure_pa'])
    with (directory/'pressure.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source', 'time_s', 'receiver', 'ideal_circuit_pa', 'classical_pa',
                         *[name for key in keys for name in (key+'_pa', key+'_se_pa')]])
        for index, (si, ti, ri) in enumerate(np.ndindex(problem.trace_shape)):
            writer.writerow([int(problem.sources[si]), float(problem.times[ti]), int(problem.receivers[ri]),
                result['ideal_pressure_pa'][index], result['classical_pressure_pa'][index],
                *[value for key in keys for value in (result['pressure_pa'][key][index],
                                                     result['standard_error_pa'][key][index])]])
    points = len(result['ideal_pressure_pa'])
    values = result['ideal_pressure_pa']+result['classical_pressure_pa']
    for key in keys:
        values += [v+d*1.96*s for v, s in zip(result['pressure_pa'][key], result['standard_error_pa'][key]) for d in (-1, 1)]
    low, high = min(values), max(values)
    padding = max((high-low)*0.12, 0.05)
    low, high = low-padding, high+padding
    x = lambda i: 75+780*i/max(points-1, 1)
    y = lambda v: 345-275*(v-low)/(high-low)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 420" role="img" aria-label="Hardware pressure against noiseless reference">',
           '<rect width="940" height="420" fill="white"/><text x="20" y="28" font-size="18">Pressure (Pa): mean across completed repetitions</text>']
    for tick in np.linspace(low, high, 5):
        svg.append(f'<path d="M65 {y(tick):.2f}H900" stroke="#ddd"/><text x="5" y="{y(tick)+4:.2f}" font-size="12">{tick:.3f}</text>')
    for index in range(points):
        svg.append(f'<text x="{x(index)-5:.2f}" y="370" font-size="12">{index+1}</text>')
        svg.append(f'<path d="M{x(index)-7:.2f} {y(result["ideal_pressure_pa"][index]):.2f}h14" stroke="{COLORS["ideal"]}" stroke-width="3"/>')
        for offset, key in zip(np.linspace(-9, 9, len(keys)), keys):
            mean, standard_error = result['pressure_pa'][key][index], result['standard_error_pa'][key][index]
            px, color = x(index)+offset, COLORS[key]
            svg.append(f'<path d="M{px:.2f} {y(mean-1.96*standard_error):.2f}V{y(mean+1.96*standard_error):.2f}" stroke="{color}"/><circle cx="{px:.2f}" cy="{y(mean):.2f}" r="4" fill="{color}"/>')
    svg.append('<text x="75" y="405" font-size="13">Observation index (source, time, receiver order); error bars: approximate 95% shot/calibration intervals</text></svg>')
    chart = ''.join(svg)
    (directory/'pressure.svg').write_text(chart)
    table_rows = ''.join(f'<tr><td>{LABELS[key]}</td><td>{result["metrics"][key]["rmse_vs_ideal_pa"]:.6g}</td><td>{result["metrics"][key]["rmse_vs_classical_pa"]:.6g}</td></tr>' for key in keys)
    resources = result['circuit_resources']
    resource_rows = ''.join(f'<tr><td>{LABELS[key]}</td><td>{resources[key]["wave_circuits_per_repetition"]}</td>'
                           f'<td>{resources[key]["two_qubit_gates_per_repetition"]}</td>'
                           f'<td>{resources[key]["maximum_circuit_depth"]}</td></tr>'
                           for key in ('original', 'optimized') if key in resources)
    comparisons = ''.join(f'<li>{escape(name.replace("_", " "))}: {entry["error_reduction_percent"]:.2f}% lower RMSE; {escape(entry["assessment_under_iid_model"])}. '
                          f'95% RMSE-reduction interval: {entry["rmse_reduction_interval_95_pa"][0]:.4g} to {entry["rmse_reduction_interval_95_pa"][1]:.4g} Pa.</li>'
                          for name, entry in result['comparisons'].items() if entry['error_reduction_percent'] is not None)
    rounds = ''.join(f'<tr><td>{row["repetition"]}</td><td>{escape(row["job_id"])}</td><td>{row["calibration_checks_passed"]}</td>'
                     +''.join(f'<td>{row["rmse_vs_ideal_pa"].get(key, "unavailable")}</td>' for key in keys)+'</tr>' for row in result['rounds'])
    round_headers = ''.join(f'<th>{LABELS[key]} RMSE</th>' for key in keys)
    original_note = ('Original versus optimized combines construction and compilation choices.'
                     if 'original' in resources else 'The original deep circuit was not included; no measured compilation improvement is claimed.')
    legend = ' · '.join(f'<span style="color:{COLORS[key]}">{LABELS[key]}</span>' for key in keys)+' · Black: noiseless circuit'
    status = 'Complete' if result['complete'] else 'Partial — do not present as the completed campaign'
    usage = 'unavailable' if result['reported_qpu_seconds'] is None else f'{result["reported_qpu_seconds"]:.2f} seconds'
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>IBM wave experiment</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#17212b}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}svg{{width:100%}}p{{line-height:1.6}}code{{overflow-wrap:anywhere}}</style>
<h1>Wave propagation on IBM hardware</h1><p><strong>{status}</strong> · Backend: {escape(result['backend'])} · Reported QPU usage: {usage}</p>
<p>{escape(result['scope'])}. Parameters: {escape(str(result['parameters']))}. Calibration checks passed: {result['calibration_checks_passed']}.</p>
<p>{legend}</p>{chart}<p>Each point identifies a transmitter/time/receiver observation; see pressure.csv for coordinates.</p>
<table><tr><th>Method</th><th>RMSE vs noiseless circuit (Pa)</th><th>RMSE vs classical wave reference (Pa)</th></tr>{table_rows}</table>
<ul>{comparisons}</ul><p>{escape(result['uncertainty'])}</p><p>Uncertainty analysis error: {escape(str(result['uncertainty_failure']))}. Passing calibration checks alone does not demonstrate improvement.</p>
<h2>Circuit resources</h2><table><tr><th>Method</th><th>Wave circuits per repetition</th><th>Total two-qubit gates per repetition</th><th>Maximum circuit depth</th></tr>{resource_rows}</table>
<p>Readout mitigation adds {resources['readout_calibration']['circuits_per_repetition']} calibration circuits and {resources['readout_calibration']['shots_per_repetition']} calibration shots per repetition, with no additional wave shots. Shared job metrics do not isolate calibration QPU time.</p>
<h2>All completed repetitions</h2><table><tr><th>Repetition</th><th>IBM job ID</th><th>Calibration accepted</th>{round_headers}</tr>{rounds}</table>
<p>{original_note} Raw versus corrected uses identical optimized wave counts. No points are clipped or selected for favorable error reduction. The noiseless reference is never used to fit the correction.</p></html>'''
    (directory/'experiment.html').write_text(html)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run-directory', required=True)
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args()
    if not args.execute:
        cli.error('Nothing executed; offline reporting requires --execute')
    generate_report(args.run_directory)


if __name__ == '__main__':
    main()
