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


def append_rotation(circuit, label, angle):
    """Append exp(-i * angle * P), including the global phase for P=I."""
    active = [(q, letter) for q, letter in enumerate(reversed(label)) if letter != 'I']
    if not active:
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
    circuit.rz(2*angle, indices[-1])
    for a, b in reversed(list(zip(indices[:-1], indices[1:]))):
        circuit.cx(a, b)
    for q, letter in reversed(active):
        if letter == 'X':
            circuit.h(q)
        elif letter == 'Y':
            circuit.h(q)
            circuit.s(q)


def evolution(n, terms, duration, max_step):
    steps = max(1, int(np.ceil(duration/max_step)))
    circuit = QuantumCircuit(n, name='wave_evolution')
    half_step = duration/(2*steps)
    for _ in range(steps):
        for label, coefficient in terms:
            append_rotation(circuit, label, coefficient*half_step)
        for label, coefficient in reversed(terms):
            append_rotation(circuit, label, coefficient*half_step)
    return circuit


def pressure_interferometer(evolution_circuit, source, receiver):
    """Ancilla <Z> = Re <receiver| U |source>; NOT |amplitude|^2.

    After first H: branch 0 prepares |receiver>, branch 1 prepares U|source>.
    The final H converts their signed real overlap to a measurable Z expectation.
    """
    n = evolution_circuit.num_qubits
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
    circuit.append(evolution_circuit.to_gate().control(1), list(range(n+1)))
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit
