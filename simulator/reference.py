"""Verify noiseless 1D/4x4 circuits on the simulator machine, never on a QPU.

python -m simulator.reference --config classical/configs/small_1d.json \
    --parameters 1 1.1 --output runs/reference.json --execute
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ('classical/_core/grid.py', 'classical/_core/problem.py',
             'classical/reference.py', 'simulator/circuits.py', 'simulator/readout.py',
             'ibm_hardware/readout.py', 'simulator/reference.py')
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def load_reference(path, problem, parameters):
    import numpy as np
    reference = json.loads(Path(path).read_text())
    if (reference.get('schema') != 1 or reference.get('source_sha256') != source_hashes()
            or reference.get('acquisition_signature') != problem.acquisition_signature
            or reference.get('parameters') != list(parameters)
            or reference.get('trotter_step_s') != problem.config['trotter_step_s']
            or reference.get('base_speed_ratio') != problem.base.tolist()
            or reference.get('region_ids') != problem.regions.tolist()
            or reference.get('verified') is not True
            or reference.get('readout_verified') is not True):
        raise ValueError('Reference is unverified, stale, or describes a different wave problem')
    for key in ('ideal_pressure_pa', 'classical_pressure_pa'):
        array = np.asarray(reference[key], dtype=float)
        if array.shape != problem.trace_shape or not np.isfinite(array).all():
            raise ValueError('Invalid reference pressure array')
    checks = np.asarray([reference['operator_max_abs_error'], reference['relative_trotter_error'],
                         reference['relative_trotter_error_half_step']], dtype=float)
    if (not np.isfinite(checks).all() or np.any(checks < 0) or checks[0] > 1e-10
            or checks[1] > 0.01 or checks[2] > checks[1]+1e-10):
        raise ValueError('Reference fails equivalence or 1% Trotter-error requirement')
    return reference


def verify_rotations(n, terms):
    """Check each controlled Pauli building block against its analytic unitary.

    Composition preserves equivalence. Avoid constructing the enormous generic
    controlled circuit's full unitary for every propagation duration in 2D.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Pauli, Statevector
    from .circuits import append_rotation, evolution, pressure_interferometer
    maximum = 0.0
    angle = 0.219
    for label in sorted({label for label, _ in terms} | {'I'*n, 'Y'+'I'*(n-1)}):
        circuit = QuantumCircuit(n+1)
        append_rotation(circuit, label, angle, controlled=True)
        expected = np.eye(2**(n+1), dtype=complex)
        expected[1::2, 1::2] = (np.cos(angle)*np.eye(2**n)
                                -1j*np.sin(angle)*Pauli(label).to_matrix())
        maximum = max(maximum, float(np.max(np.abs(Operator(circuit).data-expected))))
    # Independently check the signed-amplitude interferometer, including negatives
    # and a controlled identity phase, without evolving a large wave problem.
    for terms_probe in ([('Y', -0.29)], [('I', 0.17), ('Y', 0.29)]):
        u = evolution(1, terms_probe, 0.9, 0.3)
        controlled = evolution(1, terms_probe, 0.9, 0.3, controlled=True)
        matrix = Operator(u).data
        for source in (0, 1):
            for receiver in (0, 1):
                circuit = pressure_interferometer(controlled, source, receiver, controlled=True)
                state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
                probabilities = state.probabilities(qargs=[0])
                maximum = max(maximum, abs(float(probabilities[0]-probabilities[1])
                                           -float(matrix[receiver, source].real)))
    return maximum


def verify_readout():
    """Deterministic assignment-channel checks, executed only with the reference."""
    import numpy as np
    from ibm_hardware.readout import analyze_pilot
    records, counts = [], []
    for repeat in (0, 1):
        for state, raw in ((0, {'0': 1800, '1': 200}), (1, {'0': 400, '1': 1600})):
            records.append({'kind': 'calibration', 'readout_qubit': 0,
                            'prepared_state': state, 'repeat': repeat})
            counts.append(raw)
    for receiver, raw in enumerate(({'0': 1450, '1': 550}, {'0': 750, '1': 1250})):
        records.append({'kind': 'wave', 'source_index': 0, 'time_index': 0,
                        'receiver_index': receiver, 'readout_qubit': 0, 'physical_scale_pa': 1.0})
        counts.append(raw)
    result = analyze_pilot(records, counts, 2000, (1, 1, 2))
    corrected = result['mitigated']
    if (corrected is None or not np.allclose(result['raw_pressure_pa'], [[[0.45, -0.25]]])
            or not np.allclose(corrected['pressure_pa'], [[[0.5, -0.5]]])
            or corrected['covariance_pa2'][0][1] <= 0):
        raise ValueError('Readout inverse or shared-calibration uncertainty check failed')
    bad = [{'0': 1200, '1': 800}, {'0': 800, '1': 1200}]*2+counts[4:]
    if analyze_pilot(records, bad, 2000, (1, 1, 2))['mitigated'] is not None:
        raise ValueError('Ill-conditioned calibration was not rejected')
    return True


def build_reference(problem, parameters):
    import numpy as np
    from scipy.linalg import expm
    from qiskit.quantum_info import Operator, Statevector
    from classical.reference import ClassicalForward
    from .circuits import pauli_terms, evolution
    n, terms = pauli_terms(problem.hamiltonian(parameters), max_data_qubits=6)
    count = len(problem.sources)*len(problem.receivers)*sum(len(problem.impulses(t)) for t in problem.times)
    if not 1 <= count <= 64:
        raise ValueError('Reference is limited to 1–64 small pressure circuits')
    speeds = problem.speeds(parameters)
    pressure, finer = np.zeros(problem.trace_shape), np.zeros(problem.trace_shape)
    # Identity terms require a relative control phase. Include Y to check signs.
    probe = evolution(1, [('I', 0.17), ('Y', -0.29)], 0.9, 0.3, controlled=True)
    expected = np.eye(4, dtype=complex)
    expected[np.ix_([1, 3], [1, 3])] = expm(-0.9j*(0.17*np.eye(2)-0.29*np.array([[0, -1j], [1j, 0]])))
    cache, operator_error = {}, max(float(np.max(np.abs(Operator(probe).data-expected))),
                                    verify_rotations(n, terms))
    readout_verified = verify_readout()
    for si, source in enumerate(problem.sources):
        for ti, time in enumerate(problem.times):
            for duration, increment in problem.impulses(time):
                if duration not in cache:
                    step = problem.config['trotter_step_s']
                    # The shipped 4x4 case has 992 terms and a 3,523,584
                    # conservative operation bound; keep a finite ceiling.
                    if 4*np.ceil(duration/step)*len(terms)*(6*n+1) > 4000000:
                        raise ValueError('Reference construction exceeds the small-circuit budget')
                    current = evolution(n, terms, duration, step, controlled=True)
                    half = evolution(n, terms, duration, step/2, controlled=True)
                    if n <= 3:
                        old = evolution(n, terms, duration, step).to_gate().control(1)
                        error = float(np.max(np.abs(Operator(old).data-Operator(current).data)))
                        operator_error = max(operator_error, error)
                    cache[duration] = current, half, evolution(n, terms, duration, step)
                current, half, uncontrolled = cache[duration]
                # Control=1, source basis state. Read signed amplitudes directly
                # for the noiseless reference; actual shot runs use interferometers.
                initial = Statevector.from_int(2*int(source)+1, 2**(n+1))
                state = initial.evolve(current).data
                direct = Statevector.from_int(int(source), 2**n).evolve(uncontrolled).data
                operator_error = max(operator_error, float(np.max(np.abs(state[1::2]-direct))),
                                     float(np.max(np.abs(state[0::2]))))
                finer_state = initial.evolve(half).data
                scale = increment*speeds[problem.receivers]/speeds[source]
                pressure[si, ti] += scale*state[2*problem.receivers+1].real
                finer[si, ti] += scale*finer_state[2*problem.receivers+1].real
    started = perf_counter()
    exact = ClassicalForward(problem).predict(parameters).mean
    exact_seconds = perf_counter()-started
    norm = float(np.linalg.norm(exact))
    if norm <= 1e-12:
        raise ValueError('Near-zero reference does not support relative-error validation')
    error = float(np.linalg.norm(pressure-exact)/norm)
    finer_error = float(np.linalg.norm(finer-exact)/norm)
    verified = operator_error <= 1e-10 and error <= 0.01 and finer_error <= error+1e-10
    return {'schema': 1, 'source_sha256': source_hashes(),
            'acquisition_signature': problem.acquisition_signature,
            'parameters': list(parameters), 'base_speed_ratio': problem.base.tolist(),
            'region_ids': problem.regions.tolist(), 'trotter_step_s': problem.config['trotter_step_s'],
            'data_qubits': n, 'wave_circuits': count, 'trace_shape': list(problem.trace_shape),
            'ideal_pressure_pa': pressure.tolist(), 'classical_pressure_pa': exact.tolist(),
            'operator_max_abs_error': operator_error, 'relative_trotter_error': error,
            'control_verification': 'Analytic controlled Pauli blocks, signed readout probes, source states; full generic unitary also checked for <=3 data qubits',
            'classical_forward_seconds': exact_seconds,
            'relative_trotter_error_half_step': finer_error, 'verified': bool(verified),
            'readout_verified': readout_verified,
            'scope': 'Noiseless circuit versus exact reduced discrete wave model; no hardware execution'}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', required=True)
    cli.add_argument('--parameters', type=float, nargs='+')
    cli.add_argument('--output', required=True)
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args()
    if not args.execute:
        cli.error('Nothing executed. Reference simulation requires --execute.')
    output = Path(args.output)
    if output.exists():
        cli.error('Reference output already exists')
    from classical._core.problem import Problem
    problem = Problem.load(args.config)
    parameters = problem.initial if args.parameters is None else args.parameters
    problem.speeds(parameters)
    reference = build_reference(problem, parameters)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(reference, stream, indent=2, allow_nan=False)
    if not reference['verified']:
        raise ValueError('Reference validation failed; do not submit hardware experiments')


if __name__ == '__main__':
    main()
