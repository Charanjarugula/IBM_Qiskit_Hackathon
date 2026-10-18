"""Build the shared pressure circuit, then map it to an explicit backend target."""
from math import ceil
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from simulator.circuits import pauli_terms, evolution, pressure_interferometer


def rotation_gate_bound(terms, steps):
    per_sweep = 0
    for label, _ in terms:
        active = sum(letter != 'I' for letter in label)
        if active:
            per_sweep += 2*label.count('X')+4*label.count('Y')+2*(active-1)+1
        else:
            per_sweep += 1  # Identity terms become phase gates on the control.
    return 2*steps*per_sweep


def compile_call(problem, parameters, backend, limits, seed, initial_layout=None,
                 optimization_level=1, direct_control=True):
    n, terms = pauli_terms(problem.hamiltonian(parameters), limits.max_data_qubits)
    if n+1 > backend.num_qubits:
        raise ValueError('Backend lacks data register plus readout ancilla')
    if initial_layout is not None:
        if (len(initial_layout) != n+1 or len(set(initial_layout)) != n+1
                or any(q < 0 or q >= backend.num_qubits for q in initial_layout)):
            raise ValueError('Layout must map each logical qubit, including ancilla, uniquely')
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level,
            approximation_degree=1.0, seed_transpiler=seed, initial_layout=initial_layout)
    c = problem.speeds(parameters)
    logical, compiled, records = [], [], []
    for si, source in enumerate(problem.sources):
        for ti, time in enumerate(problem.times):
            for duration, increment in problem.impulses(time):
                steps = max(1, ceil(duration/problem.config['trotter_step_s']))
                if rotation_gate_bound(terms, steps) > limits.max_logical_gates:
                    raise ValueError('Logical evolution exceeds construction budget')
                evo = evolution(n, terms, duration, problem.config['trotter_step_s'], controlled=direct_control)
                for ri, receiver in enumerate(problem.receivers):
                    qc = pressure_interferometer(evo, int(source), int(receiver), controlled=direct_control)
                    isa = pm.run(qc)
                    # Only the final ancilla measurement is allowed, with no postselection.
                    if isa.num_clbits != 1 or len(isa.cregs) != 1 or isa.cregs[0].size != 1:
                        raise ValueError('Compiled readout register differs from one-bit contract')
                    two_qubit = 0
                    measurements = []
                    used = set()
                    for instruction in isa.data:
                        name = instruction.operation.name
                        qubits = tuple(isa.find_bit(q).index for q in instruction.qubits)
                        used.update(qubits)
                        if name == 'barrier':
                            continue
                        if name == 'measure':
                            measurements.append((qubits[0], isa.find_bit(instruction.clbits[0]).index))
                        elif instruction.clbits or name in ('reset', 'if_else', 'while_loop'):
                            raise ValueError('Unexpected classical control/reset in pressure circuit')
                        if len(qubits) > 2:
                            raise ValueError('Untranslated multi-qubit gate')
                        if not backend.target.instruction_supported(operation_name=name, qargs=qubits):
                            raise ValueError(f'Instruction {name}{qubits} is not supported by backend')
                        two_qubit += int(len(qubits) == 2)
                    layout = isa.layout.final_index_layout(filter_ancillas=True)
                    if measurements != [(layout[0], 0)]:
                        raise ValueError('Ancilla measurement moved to an unexpected output bit')
                    if (isa.size() > limits.max_compiled_gates or isa.depth() > limits.max_depth
                            or two_qubit > limits.max_two_qubit_gates):
                        raise ValueError(
                            f'Compiled circuit exceeds hardware ceilings: gates={isa.size()}'
                            f'/{limits.max_compiled_gates}, depth={isa.depth()}/{limits.max_depth}, '
                            f'two_qubit={two_qubit}/{limits.max_two_qubit_gates}')
                    logical.append(qc)
                    compiled.append(isa)
                    records.append({
                        'source_index': si, 'time_index': ti, 'receiver_index': ri,
                        'source': int(source), 'receiver': int(receiver), 'time_s': float(time),
                        'evolution_time_s': float(duration),
                        'physical_scale_pa': float(increment*c[receiver]/c[source]),
                        'parameters': np.asarray(parameters).tolist(), 'data_qubits': n,
                        'readout_ancillas': 1, 'pauli_terms': len(terms), 'trotter_steps': steps,
                        'classical_register': isa.cregs[0].name, 'logical_to_physical': layout,
                        'readout_qubit': layout[0], 'optimization_level': optimization_level,
                        'seed_transpiler': seed,
                        'variant': 'optimized' if direct_control else 'original',
                        'used_physical_qubits': sorted(used), 'depth': isa.depth(),
                        'two_qubit_gates': two_qubit, 'gates': isa.size(),
                        'operations': dict(isa.count_ops())})
    return logical, compiled, records


def calibration_cost(circuit, backend):
    """Ranking proxy from gate/readout calibration, NOT predicted circuit fidelity."""
    exposure = 0.0
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == 'barrier':
            continue
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        properties = backend.target[name][qubits]
        error = None if properties is None else properties.error
        if error is None or not np.isfinite(error) or not 0 <= error < 1:
            raise ValueError(f'Missing or unusable calibration for {name}{qubits}')
        exposure -= float(np.log1p(-error))
    duration = float(circuit.estimate_duration(backend.target, unit='s'))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Missing or invalid circuit duration')
    return exposure, duration


def select_compilation(problem, parameters, backend, limits, seed, initial_layout, evidence,
                       extra_wave_circuits=0, repetitions=1):
    """Compare five exact compilations locally; no primitives or hardware jobs."""
    best, best_key, candidates = None, None, []
    for level, candidate_seed in ((1, seed), (2, seed), (3, seed), (3, seed+1), (3, seed+2)):
        row = {'optimization_level': level, 'seed_transpiler': candidate_seed}
        try:
            logical, circuits, records = compile_call(
                problem, parameters, backend, limits, candidate_seed, initial_layout, level)
            if not circuits:
                raise ValueError('Pilot needs at least one nonzero source contribution')
            costs = [calibration_cost(circuit, backend) for circuit in circuits]
            readout_qubits = {record['readout_qubit'] for record in records}
            total = len(circuits)+extra_wave_circuits+4*len(readout_qubits)
            limits.check_totals(total*repetitions, repetitions)
            if total > limits.circuits_per_job:
                raise ValueError('Pilot plus repeated calibrations must fit in one job')
            key = (sum(cost[0] for cost in costs), sum(cost[1] for cost in costs),
                   sum(record['two_qubit_gates'] for record in records))
            row.update(calibration_exposure=key[0], total_wave_duration_s=key[1],
                       two_qubit_gates=key[2], circuits_including_calibration=total)
            for record, (exposure, duration) in zip(records, costs):
                record.update(calibration_exposure=exposure, estimated_duration_s=duration)
            if best_key is None or key < best_key:
                best, best_key = (logical, circuits, records), key
        except (ValueError, QiskitError) as error:
            row['rejected'] = str(error)
        candidates.append(row)
    evidence.write('compilation_candidates.json', {
        'candidates': candidates,
        'ranking': 'Sum -log(1-error), then duration, then two-qubit gate count; heuristic only',
        'selected': None if best is None else {
            'optimization_level': best[2][0]['optimization_level'],
            'seed_transpiler': best[2][0]['seed_transpiler']}})
    if best is None:
        raise ValueError('No calibrated compilation fits the pilot limits; no jobs submitted')
    return best
