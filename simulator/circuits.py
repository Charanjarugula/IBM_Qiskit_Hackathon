"""Explicit Pauli-rotation gates and Hadamard interference for signed amplitudes.

No dense exponential or UnitaryGate is used to implement evolution. Dense Pauli decomposition
is deliberately limited to small registers and is counted as classical compilation work.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def pauli_terms(hamiltonian, max_data_qubits):
    dimension = hamiltonian.shape[0]
    n = int(np.ceil(np.log2(dimension)))
    if n > max_data_qubits or n > 8:
        raise ValueError('Dense compact Pauli compilation limited to at most 8 data qubits')
    dense = np.zeros((2**n, 2**n), complex)
    dense[:dimension, :dimension] = hamiltonian.toarray()
    if not np.allclose(dense, dense.conj().T, atol=1e-12):
        raise ValueError('Non-Hermitian pressure Hamiltonian')
    op = SparsePauliOp.from_operator(dense, atol=1e-12, rtol=0)
    if np.max(np.abs(op.coeffs.imag), initial=0) > 1e-9:
        raise ValueError('Hermitian operator has complex Pauli coefficients')
    return n, [(label, float(coef.real)) for label, coef in op.to_list() if abs(coef) > 1e-12]


def append_rotation(circuit, label, angle, controlled=False):
    """Append exp(-i angle P); optional control is qubit 0, data start at 1.

    C(B^-1 RZ B) = B^-1 C(RZ) B: basis/parity gates need no controls.
    The identity Pauli's global phase becomes a relative phase on the control.
    """
    active = [(q+int(controlled), letter) for q, letter in enumerate(reversed(label))
              if letter != 'I']
    if not active:
        if controlled:
            circuit.p(-angle, 0)
        else:
            circuit.global_phase -= angle
        return
    for q, letter in active:
        if letter == 'X':
            circuit.h(q)
        elif letter == 'Y':
            circuit.sdg(q)
            circuit.h(q)
    indices = [q for q, _ in active]
    for a, b in zip(indices[:-1], indices[1:]):
        circuit.cx(a, b)
    if controlled:
        circuit.crz(2*angle, 0, indices[-1])
    else:
        circuit.rz(2*angle, indices[-1])
    for a, b in reversed(list(zip(indices[:-1], indices[1:]))):
        circuit.cx(a, b)
    for q, letter in reversed(active):
        if letter == 'X':
            circuit.h(q)
        elif letter == 'Y':
            circuit.h(q)
            circuit.s(q)


def evolution(n, terms, duration, max_step, controlled=False):
    steps = max(1, int(np.ceil(duration/max_step)))
    circuit = QuantumCircuit(n+int(controlled), name='wave_evolution')
    half_step = duration/(2*steps)
    for _ in range(steps):
        for label, coefficient in terms:
            append_rotation(circuit, label, coefficient*half_step, controlled)
        for label, coefficient in reversed(terms):
            append_rotation(circuit, label, coefficient*half_step, controlled)
    return circuit


def pressure_interferometer(evolution_circuit, source, receiver, controlled=False):
    """Ancilla <Z> = Re <receiver| U |source>; NOT |amplitude|^2.

    After first H: branch 0 prepares |receiver>, branch 1 prepares U|source>.
    The final H converts their signed real overlap to a measurable Z expectation.
    """
    n = evolution_circuit.num_qubits-int(controlled)
    if not 0 <= source < 2**n or not 0 <= receiver < 2**n:
        raise ValueError('Basis index outside encoding')
    circuit = QuantumCircuit(n+1, 1)
    circuit.h(0)
    circuit.x(0)
    for bit in range(n):
        if (receiver >> bit) & 1:
            circuit.cx(0, bit+1)
    circuit.x(0)
    for bit in range(n):
        if (source >> bit) & 1:
            circuit.cx(0, bit+1)
    gate = evolution_circuit.to_gate()
    circuit.append(gate if controlled else gate.control(1), list(range(n+1)))
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit
