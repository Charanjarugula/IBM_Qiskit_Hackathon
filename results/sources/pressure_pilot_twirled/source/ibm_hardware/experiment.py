"""Three fixed repetitions of optimized raw and readout-corrected pressure.

An optional original arm includes both circuit construction and compilation differences.
Optimized raw -> corrected uses identical wave counts and isolates readout processing.
No tissue reconstruction or quantum advantage is claimed by this experiment.
"""
from dataclasses import replace
import numpy as np
from .artifacts import save_circuits
from .compilation import compile_call, select_compilation
from .pilot import estimate_job_seconds
from .readout import calibration_circuits

REPETITIONS = 3


def prepare_experiment(solver, parameters, reference, include_original=False):
    p, limits = solver.problem, solver.limits
    old_logical, original, old_records = [], [], []
    if include_original:
        # Explicit opt-in: the deeper baseline may exceed ceilings. Never silently
        # drop an arm after its inclusion has been requested, or relax the limits.
        old_logical, original, old_records = compile_call(
            p, parameters, solver.backend, limits, solver.seed, solver.initial_layout,
            optimization_level=1, direct_control=False)
    # Keep the optimized arm inside the pilot's stricter gate ceilings.
    optimized_limits = replace(limits, max_compiled_gates=min(limits.max_compiled_gates, 2000),
        max_depth=min(limits.max_depth, 1000), max_two_qubit_gates=min(limits.max_two_qubit_gates, 300))
    new_logical, optimized, new_records = select_compilation(
        p, parameters, solver.backend, optimized_limits, solver.seed, solver.initial_layout,
        solver.evidence, extra_wave_circuits=len(original), repetitions=REPETITIONS)
    fields = ('source_index', 'time_index', 'receiver_index', 'evolution_time_s', 'physical_scale_pa')
    if (len(optimized) != reference['wave_circuits'] or (include_original and
            (len(original) != len(optimized)
             or any(tuple(a[k] for k in fields) != tuple(b[k] for k in fields)
                    for a, b in zip(old_records, new_records))))):
        raise ValueError('Original and optimized arms do not describe the same wave observations')
    qubits = {record['readout_qubit'] for record in new_records}
    first, first_records = calibration_circuits(solver.backend, qubits, repeat=0)
    last, last_records = calibration_circuits(solver.backend, qubits, repeat=1)
    waves = original+optimized
    wave_records = [dict(record, kind='wave') for record in old_records+new_records]
    job_circuits = len(waves)+len(first)+len(last)
    limits.check_totals(REPETITIONS*job_circuits, REPETITIONS)
    if job_circuits > limits.circuits_per_job:
        raise ValueError('Each complete comparison must fit in one job')
    estimate = estimate_job_seconds(first+waves+last, solver.backend, limits.shots)
    if estimate > limits.max_execution_seconds_per_job or REPETITIONS*estimate > limits.max_total_execution_seconds:
        raise ValueError('Estimated experiment time exceeds the execution ceiling; no jobs submitted')
    batches, batch_records, hashes = [], [], []
    for repetition in range(REPETITIONS):
        # Same ISA circuits in every repetition; shuffled payload order is not an
        # assertion about IBM's scheduling or the time separation between jobs.
        order = np.random.default_rng(solver.seed+repetition).permutation(len(waves))
        circuits = first+[waves[i] for i in order]+last
        records = first_records+[wave_records[i] for i in order]+last_records
        batches.append(circuits)
        batch_records.append(records)
        hashes.append(save_circuits(solver.evidence, f'experiment_{repetition:02d}_isa.qpy', circuits))
    logical_hash = save_circuits(solver.evidence, 'experiment_logical.qpy', old_logical+new_logical)
    solver.evidence.write('reference.json', reference)
    solver.evidence.write('experiment_plan.json', {
        'backend': solver.backend.name, 'parameters': list(parameters),
        'include_original_arm': include_original,
        'repetitions': REPETITIONS, 'shots_per_circuit': limits.shots, 'seed': solver.seed,
        'records_by_repetition': batch_records, 'wave_records': wave_records,
        'isa_qpy_sha256': hashes, 'logical_qpy_sha256': logical_hash,
        'primary_comparison': 'Optimized raw versus readout-corrected RMSE against noiseless circuit',
        'secondary_comparison': ('Original raw versus optimized raw; combines construction and compilation'
                                 if include_original else None),
        'uncertainty_samples': 2000,
        'scope': 'Reduced forward wave propagation; no reconstruction or speedup claim'})
    solver.evidence.write('resource_plan.json', {
        'mode': 'experiment', 'repetitions': REPETITIONS, 'jobs': REPETITIONS,
        'circuits_per_job': job_circuits, 'circuits': REPETITIONS*job_circuits,
        'shots_per_circuit': limits.shots, 'shots': REPETITIONS*job_circuits*limits.shots,
        'estimated_qpu_seconds': REPETITIONS*estimate,
        'execution_seconds_ceiling': REPETITIONS*limits.max_execution_seconds_per_job,
        'timing_note': 'Local estimate; service overhead may differ. Separate Runtime limits apply.'})
    return batches, batch_records


def run_experiment(solver, parameters, reference, include_original=False):
    batches, records = prepare_experiment(solver, parameters, reference, include_original)
    if not solver.submit_hardware:
        solver.evidence.finish({'status': 'experiment_prepared_only', 'jobs_submitted': 0})
        return
    from .report import generate_report
    for circuits, batch_records in zip(batches, records):
        solver.submit_batch(circuits, batch_records)
        solver.calls += 1
        report = generate_report(solver.evidence.path)
        # Never stop early because an improvement looks favorable. Stop on failed
        # calibration quality checks instead of spending the remaining QPU budget.
        if not report['calibration_checks_passed']:
            break
    statistics = solver.statistics()
    statistics.update(status='experiment_complete' if report['complete'] else 'experiment_incomplete',
                      error_mitigation=report['calibration_checks_passed'],
                      mitigation_method='single_qubit_assignment_calibration',
                      report='experiment.html', uncertainty='See experiment.json')
    solver.evidence.finish(statistics)
