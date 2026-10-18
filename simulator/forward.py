"""Actual Aer execution of decomposed circuits, with an auditable signed-pressure readout."""
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from classical._core.evidence import Prediction
from .readout import ancilla_statistics
from .circuits import pauli_terms, evolution, pressure_interferometer


class GateForward:
    def __init__(self, problem, evidence, shots, seed, max_circuits, max_total_shots,
                 max_data_qubits=7, max_gates=500000, noise_probability=0.0):
        if shots < 2 or max_circuits < 1 or max_total_shots < shots:
            raise ValueError('Invalid execution budget')
        if not 0 <= noise_probability <= 0.1:
            raise ValueError('Generic depolarizing probability must be in [0, 0.1]')
        n = int(np.ceil(np.log2(problem.grid.n_vertices+problem.grid.n_edges)))
        if n > min(max_data_qubits, 8):
            raise ValueError('Problem exceeds compact compilation register limit')
        noise = None
        if noise_probability:
            noise = NoiseModel()
            # Stress model only; not calibrated IBM device noise.
            noise.add_all_qubit_quantum_error(depolarizing_error(noise_probability, 1), ['h', 'x', 'rz', 'sx'])
            noise.add_all_qubit_quantum_error(depolarizing_error(noise_probability, 2), ['cx'])
        self.simulator = AerSimulator(method='statevector', noise_model=noise)
        self.problem, self.evidence = problem, evidence
        self.shots, self.seed = int(shots), int(seed)
        self.max_circuits, self.max_total_shots = int(max_circuits), int(max_total_shots)
        self.max_data_qubits, self.max_gates = max_data_qubits, max_gates
        self.calls = self.circuits = self.total_shots = 0
        self.max_observed_depth = self.max_observed_gates = 0
        self.noise_probability = noise_probability

    def predict(self, parameters):
        p = self.problem
        needed = len(p.sources)*len(p.receivers)*sum(len(p.impulses(t)) for t in p.times)
        if self.circuits+needed > self.max_circuits or self.total_shots+needed*self.shots > self.max_total_shots:
            raise RuntimeError('Forward call would exceed total circuit/shot budget; no part of this call was submitted')
        n, terms = pauli_terms(p.hamiltonian(parameters), self.max_data_qubits)
        c = p.speeds(parameters)
        output, variance = np.zeros(p.trace_shape), np.zeros(p.trace_shape)
        sampling_variance = np.zeros(p.trace_shape)
        call = self.calls
        self.calls += 1
        for si, source in enumerate(p.sources):
            for ti, time in enumerate(p.times):
                for duration, pressure_increment in p.impulses(time):
                    # Each distinct pulse contribution is separately prepared and measured.
                    evo = evolution(n, terms, duration, p.config['trotter_step_s'], controlled=True)
                    for ri, receiver in enumerate(p.receivers):
                        qc = pressure_interferometer(evo, int(source), int(receiver), controlled=True)
                        compiled = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'],
                                             optimization_level=1, seed_transpiler=self.seed)
                        if compiled.size() > self.max_gates:
                            raise RuntimeError('Compiled circuit exceeds gate limit')
                        index = self.circuits
                        run_seed = (self.seed+index) % (2**32-1)
                        self.circuits += 1
                        self.total_shots += self.shots
                        self.max_observed_depth = max(self.max_observed_depth, compiled.depth())
                        self.max_observed_gates = max(self.max_observed_gates, compiled.size())
                        result = self.simulator.run(compiled, shots=self.shots, seed_simulator=run_seed).result()
                        if not result.success:
                            raise RuntimeError(f'Aer failed: {result.status}')
                        counts = result.get_counts()
                        total = self.shots
                        overlap, se, overlap_variance = ancilla_statistics(counts, total)
                        scale = pressure_increment*c[receiver]/c[source]
                        output[si, ti, ri] += scale*overlap
                        variance[si, ti, ri] += (scale*se)**2
                        # Sample variance / N for independent +/-1 measurements.
                        # Unlike the smoothed SE, this gives an unbiased loss correction.
                        sampling_variance[si, ti, ri] += scale**2*overlap_variance
                        self.evidence.write(f'counts_{index:07d}.json', {
                            'forward_call': call, 'parameters': np.asarray(parameters).tolist(),
                            'source': int(source), 'receiver': int(receiver), 'time_s': float(time),
                            'evolution_time_s': float(duration), 'physical_scale_pa': float(scale),
                            'counts': counts, 'shots': total, 'seed_simulator': run_seed,
                            'signed_overlap': overlap, 'overlap_standard_error': float(se),
                            'unbiased_overlap_variance': float(overlap_variance),
                            'data_qubits': n, 'ancillas': 1, 'depth': compiled.depth(),
                            'gates': compiled.size(), 'operations': dict(compiled.count_ops()),
                            'pauli_terms': len(terms)})
        return Prediction(output, np.sqrt(variance), sampling_variance)

    def statistics(self):
        return {'forward_calls': self.calls, 'circuits': self.circuits, 'shots': self.total_shots,
                'maximum_depth': self.max_observed_depth, 'maximum_gates': self.max_observed_gates,
                'noise': 'ideal' if self.noise_probability == 0 else 'generic_depolarizing_stress',
                'noise_probability': self.noise_probability,
                'gradient': 'classical finite differences of gate-sampled pressure loss'}
