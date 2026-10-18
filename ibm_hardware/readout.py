"""One-qubit assignment calibration and pressure uncertainty; no postselection.

Assumes stable, state-independent readout errors. Calibration preparation errors
and spectator-dependent readout are not separately identifiable by this model.
"""
import numpy as np
from simulator.readout import ancilla_statistics


def calibration_circuits(backend, qubits, repeat):
    from qiskit import QuantumCircuit
    circuits, records = [], []
    for qubit in sorted(qubits):
        for state in (0, 1):
            for operation in (('x', 'measure') if state else ('measure',)):
                if not backend.target.instruction_supported(operation_name=operation, qargs=(qubit,)):
                    raise ValueError(f'Calibration operation {operation} unsupported on {qubit}')
            circuit = QuantumCircuit(backend.num_qubits, 1)
            if state:
                circuit.x(qubit)
            circuit.measure(qubit, 0)
            circuits.append(circuit)
            records.append({'kind': 'calibration', 'readout_qubit': qubit,
                            'prepared_state': state, 'repeat': repeat,
                            'classical_register': circuit.cregs[0].name})
    return circuits, records


def error_estimate(counts, state, shots):
    ancilla_statistics(counts, shots)  # Enforce the raw one-bit count contract.
    errors = counts.get(str(1-state), 0)
    estimate = errors/shots
    smoothed = (errors+1)/(shots+2)
    return estimate, smoothed*(1-smoothed)/shots


def analyze_pilot(records, counts, shots, trace_shape):
    if len(records) != len(counts):
        raise ValueError('Records and results differ in length')
    qubits = sorted({r['readout_qubit'] for r in records if r['kind'] == 'wave'})
    calibration = {}
    for record, raw in zip(records, counts):
        ancilla_statistics(raw, shots)
        if record['kind'] == 'calibration':
            key = (record['readout_qubit'], record['prepared_state'], record['repeat'])
            if key in calibration:
                raise ValueError('Duplicate calibration record')
            calibration[key] = raw
        elif record['kind'] != 'wave':
            raise ValueError('Unknown pilot record')
    models, diagnostics = {}, []
    for qubit in qubits:
        estimates, variances, changes = [], [], []
        for state in (0, 1):
            first = calibration[(qubit, state, 0)]
            second = calibration[(qubit, state, 1)]
            a, va = error_estimate(first, state, shots)
            b, vb = error_estimate(second, state, shots)
            pooled = {bit: first.get(bit, 0)+second.get(bit, 0) for bit in ('0', '1')}
            estimate, variance = error_estimate(pooled, state, 2*shots)
            estimates.append(estimate)
            variances.append(variance)
            changes.append({'state': state, 'absolute_change': abs(a-b),
                            'three_sigma_threshold': 3*float(np.sqrt(va+vb)),
                            'consistent': bool(abs(a-b) <= 3*np.sqrt(va+vb))})
        e0, e1 = estimates
        v0, v1 = variances
        contrast = 1-e0-e1
        # Reject ill-conditioned inversion, including uncertainty in calibration.
        contrast_lower = contrast-3*np.sqrt(v0+v1)
        accepted = bool(contrast_lower > 0.5 and all(c['consistent'] for c in changes))
        models[qubit] = (e0, e1, v0, v1)
        diagnostics.append({'physical_qubit': qubit, 'p_1_given_0': e0, 'p_0_given_1': e1,
                            'contrast': contrast, 'contrast_lower_3sigma': float(contrast_lower),
                            'repeat_checks': changes, 'accepted': accepted})
    size = int(np.prod(trace_shape))
    raw_mean, raw_var = np.zeros(size), np.zeros(size)
    corrected, shot_var = np.zeros(size), np.zeros(size)
    derivatives = {qubit: np.zeros((size, 2)) for qubit in qubits}
    accepted = bool(qubits) and all(d['accepted'] for d in diagnostics)
    for record, raw in zip(records, counts):
        if record['kind'] != 'wave':
            continue
        index = np.ravel_multi_index((record['source_index'], record['time_index'],
                                     record['receiver_index']), trace_shape)
        overlap, se, _ = ancilla_statistics(raw, shots)
        scale = record['physical_scale_pa']
        raw_mean[index] += scale*overlap
        raw_var[index] += (scale*se)**2
        if accepted:
            qubit = record['readout_qubit']
            e0, e1, _, _ = models[qubit]
            contrast = 1-e0-e1
            mean = (overlap+e0-e1)/contrast
            corrected[index] += scale*mean
            shot_var[index] += (scale*se/contrast)**2
            derivatives[qubit][index] += scale*np.array([1+mean, mean-1])/contrast
    result = {
        'trace_shape': list(trace_shape), 'flatten_order': 'source,time,receiver; C order',
        'raw_pressure_pa': raw_mean.reshape(trace_shape).tolist(),
        'raw_standard_error_pa': np.sqrt(raw_var).reshape(trace_shape).tolist(),
        'calibration': diagnostics, 'calibration_checks_passed': accepted,
        'mitigated': None,
        'limitations': 'Repeat checks do not certify stationarity. Submission order does not '
                      'guarantee execution timing. No correction of gate/decoherence error.'}
    if accepted:
        covariance = np.diag(shot_var)
        for qubit, derivative in derivatives.items():
            _, _, v0, v1 = models[qubit]
            covariance += (derivative*np.array([v0, v1])) @ derivative.T
        result['mitigated'] = {
            'pressure_pa': corrected.reshape(trace_shape).tolist(),
            'standard_error_pa': np.sqrt(np.diag(covariance)).reshape(trace_shape).tolist(),
            'covariance_pa2': covariance.tolist(),
            'uncertainty': 'First-order propagation of shot and shared calibration uncertainty; '
                           'excludes model bias, gate errors and unobserved drift',
            'estimator': 'Unclipped assignment-matrix inverse; finite calibration can introduce bias'}
    return result
