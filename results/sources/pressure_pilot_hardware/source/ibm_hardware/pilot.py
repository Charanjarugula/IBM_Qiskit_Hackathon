"""One forward evaluation, locally selected compilation and repeated readout calibration.

No inversion: the nonlinear calibration estimator is not the unbiased raw-sampling
estimator expected by the existing optimizer. Raw counts are always retained.
Pilot resource ceilings are engineering limits, not certified accuracy thresholds.
"""
import numpy as np
from .artifacts import save_circuits
from .compilation import select_compilation
from .readout import calibration_circuits, analyze_pilot


def estimate_job_seconds(circuits, backend, shots):
    rep_delay = float(backend.default_rep_delay)
    used = {circuit.find_bit(q).index for circuit in circuits
            for instruction in circuit.data for q in instruction.qubits}
    reset_time = max(float(backend.target['reset'][(q,)].duration) for q in used)
    durations = [float(circuit.estimate_duration(backend.target, unit='s')) for circuit in circuits]
    if not np.isfinite([rep_delay, reset_time, *durations]).all() or min(rep_delay, reset_time, *durations) < 0:
        raise ValueError('Missing or invalid backend timing information')
    return 2+shots*sum(duration+rep_delay+reset_time for duration in durations)


def prepare_pilot(problem, parameters, backend, evidence, limits, seed, initial_layout):
    logical, wave, records = select_compilation(
        problem, parameters, backend, limits, seed, initial_layout, evidence)
    qubits = {record['readout_qubit'] for record in records}
    first, first_records = calibration_circuits(backend, qubits, repeat=0)
    last, last_records = calibration_circuits(backend, qubits, repeat=1)
    # Fixed shuffle avoids placing every longest wave circuit together in the payload.
    order = np.random.default_rng(seed).permutation(len(wave))
    circuits = first+[wave[i] for i in order]+last
    records = first_records+[dict(records[i], kind='wave') for i in order]+last_records
    limits.check_totals(len(circuits), 1)
    if len(circuits) > limits.circuits_per_job:
        raise ValueError('Pilot including calibrations must fit in one job')
    estimate = estimate_job_seconds(circuits, backend, limits.shots)
    if estimate > min(limits.max_execution_seconds_per_job, limits.max_total_execution_seconds):
        raise ValueError('Estimated pilot time exceeds the execution ceiling; no jobs submitted')
    logical_hash = save_circuits(evidence, 'pilot_logical.qpy', logical)
    isa_hash = save_circuits(evidence, 'pilot_isa.qpy', circuits)
    evidence.write('pilot_circuits.json', {
        'parameters': np.asarray(parameters).tolist(), 'records': records,
        'logical_qpy_sha256': logical_hash, 'isa_qpy_sha256': isa_hash})
    evidence.write('resource_plan.json', {
        'mode': 'pilot', 'forward_calls': 1, 'wave_circuits': len(wave),
        'calibration_circuits': len(first)+len(last), 'circuits': len(circuits),
        'shots': len(circuits)*limits.shots, 'shots_per_circuit': limits.shots, 'jobs': 1,
        'estimated_qpu_seconds': estimate,
        'execution_seconds_ceiling': min(limits.max_execution_seconds_per_job,
                                       limits.max_total_execution_seconds),
        'timing_note': 'Local estimate; backend overhead may differ. Runtime limit is enforced.',
        'comparison': 'Raw and corrected pressure use the same wave counts; no reference fitted'})
    return circuits, records


def run_pilot(solver, parameters):
    circuits, records = prepare_pilot(solver.problem, parameters, solver.backend,
        solver.evidence, solver.limits, solver.seed, solver.initial_layout)
    if not solver.submit_hardware:
        solver.evidence.finish({'status': 'pilot_prepared_only', 'jobs_submitted': 0})
        return
    counts = solver.submit_batch(circuits, records)
    report = analyze_pilot(records, counts, solver.limits.shots, solver.problem.trace_shape)
    solver.evidence.write('pilot_pressure.json', report)
    statistics = solver.statistics()
    statistics.update(forward_calls=1, error_mitigation=report['calibration_checks_passed'],
                      mitigation_method='single_qubit_assignment_calibration',
                      uncertainty='See pilot_pressure.json; shared calibration covariance included',
                      status='pilot_complete' if report['calibration_checks_passed'] else 'calibration_rejected')
    solver.evidence.finish(statistics)
