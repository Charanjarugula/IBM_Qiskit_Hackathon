"""Build the shared pressure circuit, then map it to an explicit backend target."""
from math import ceil
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from simulator.circuits import pauli_terms, evolution, pressure_interferometer


def rotation_gate_bound(terms, steps):
    per_sweep = 0
    for label, _ in terms:
        active = sum(letter != 'I' for letter in label)
        if active:
            per_sweep += 2*label.count('X')+4*label.count('Y')+2*(active-1)+1
    return 2*steps*per_sweep


def compile_call(problem, parameters, backend, limits, seed, initial_layout=None):
    n, terms = pauli_terms(problem.hamiltonian(parameters), limits.max_data_qubits)
    if n+1 > backend.num_qubits:
        raise ValueError('Backend lacks data register plus readout ancilla')
    if initial_layout is not None:
        if (len(initial_layout) != n+1 or len(set(initial_layout)) != n+1
                or any(q < 0 or q >= backend.num_qubits for q in initial_layout)):
            raise ValueError('Layout must map each logical qubit, including ancilla, uniquely')
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1,
            approximation_degree=1.0, seed_transpiler=seed, initial_layout=initial_layout)
    c = problem.speeds(parameters)
    logical, compiled, records = [], [], []
    for si, source in enumerate(problem.sources):
        for ti, time in enumerate(problem.times):
            for duration, increment in problem.impulses(time):
                steps = max(1, ceil(duration/problem.config['trotter_step_s']))
                if rotation_gate_bound(terms, steps) > limits.max_logical_gates:
                    raise ValueError('Logical evolution exceeds construction budget')
                evo = evolution(n, terms, duration, problem.config['trotter_step_s'])
                for ri, receiver in enumerate(problem.receivers):
                    qc = pressure_interferometer(evo, int(source), int(receiver))
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
                        raise ValueError('Compiled circuit exceeds hardware depth/gate ceilings')
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
                        'used_physical_qubits': sorted(used), 'depth': isa.depth(),
                        'two_qubit_gates': two_qubit, 'gates': isa.size(),
                        'operations': dict(isa.count_ops())})
    return logical, compiled, records
