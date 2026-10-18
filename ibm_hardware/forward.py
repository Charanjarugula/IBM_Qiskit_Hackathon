"""Raw hardware forward calls. Construction and preparation never submit jobs."""
from math import ceil
import uuid
import numpy as np
from classical._core.evidence import Prediction
from simulator.readout import ancilla_statistics
from .artifacts import save_circuits, save_result, write_runtime_json
from .compilation import compile_call
from .policy import raw_sampler_options


class HardwareForward:
    def __init__(self, problem, backend, evidence, limits, seed=7, initial_layout=None,
                 submit_hardware=False):
        self.problem, self.backend, self.evidence, self.limits = problem, backend, evidence, limits
        self.seed, self.initial_layout, self.submit_hardware = seed, initial_layout, submit_hardware
        self.calls = self.circuits = self.jobs = 0
        self.job_ids = []
        self.sampler = None

    def prepare(self, parameters):
        logical, isa, records = compile_call(self.problem, parameters, self.backend,
                                            self.limits, self.seed, self.initial_layout)
        prefix = f'call_{self.calls:04d}'
        logical_hash = save_circuits(self.evidence, prefix+'_logical.qpy', logical)
        isa_hash = save_circuits(self.evidence, prefix+'_isa.qpy', isa)
        self.evidence.write(prefix+'_circuits.json', {
            'parameters': np.asarray(parameters).tolist(), 'records': records,
            'logical_qpy_sha256': logical_hash, 'isa_qpy_sha256': isa_hash})
        return isa, records

    def predict(self, parameters):
        if not self.submit_hardware:
            raise PermissionError('Hardware submission was not explicitly enabled')
        p, limits = self.problem, self.limits
        needed = len(p.sources)*len(p.receivers)*sum(len(p.impulses(t)) for t in p.times)
        self.limits.check_totals(self.circuits+needed,
                                self.jobs+ceil(needed/limits.circuits_per_job))
        # Compile and check the ENTIRE forward call before any of its jobs are submitted.
        circuits, records = self.prepare(parameters)
        if len(circuits) != needed:
            raise ValueError('Compiled circuit count differs from resource plan')
        output = np.zeros(p.trace_shape)
        reporting_variance, sampling_variance = np.zeros_like(output), np.zeros_like(output)
        for start in range(0, len(circuits), limits.circuits_per_job):
            batch = circuits[start:start+limits.circuits_per_job]
            batch_records = records[start:start+limits.circuits_per_job]
            counts = self.submit_batch(batch, batch_records, first_circuit=start)
            for record, raw in zip(batch_records, counts):
                overlap, se, variance = ancilla_statistics(raw, limits.shots)
                key = (record['source_index'], record['time_index'], record['receiver_index'])
                scale = record['physical_scale_pa']
                output[key] += scale*overlap
                reporting_variance[key] += (scale*se)**2
                sampling_variance[key] += scale**2*variance
        self.calls += 1
        return Prediction(output, np.sqrt(reporting_variance), sampling_variance)

    def submit_batch(self, circuits, records, first_circuit=0):
        """Submit already checked circuits; also used by the bounded readout pilot."""
        if not self.submit_hardware:
            raise PermissionError('Hardware submission was not explicitly enabled')
        limits = self.limits
        if not circuits or len(circuits) != len(records) or len(circuits) > limits.circuits_per_job:
            raise ValueError('Invalid circuit batch')
        limits.check_totals(self.circuits+len(circuits), self.jobs+1)
        if self.sampler is None:
            from qiskit_ibm_runtime.sampler import SamplerV2
            self.sampler = SamplerV2(mode=self.backend, options=raw_sampler_options(limits))
        prefix = f'job_{self.jobs:04d}'
        tag = 'pressure-'+uuid.uuid4().hex
        self.sampler.options.environment.job_tags = [tag]
        self.evidence.write(prefix+'_intent.json', {
            'tag': tag, 'backend': self.backend.name, 'forward_call': self.calls,
            'first_circuit': first_circuit, 'records': records, 'shots': limits.shots,
            'options': raw_sampler_options(limits),
            'recovery': 'If submission fails ambiguously, find this tag in IBM before retrying'})
        # Reserve before submission; an ambiguous error must never trigger an automatic retry.
        self.jobs += 1
        self.circuits += len(circuits)
        job = self.sampler.run(circuits, shots=limits.shots)
        self.job_ids.append(job.job_id())
        self.evidence.write(prefix+'_submitted.json', {'job_id': job.job_id(), 'tag': tag})
        counts = save_result(self.evidence, prefix, job.result(), records, limits.shots)
        try:
            write_runtime_json(self.evidence, prefix+'_metrics.json', job.metrics())
        except Exception as error:
            self.evidence.write(prefix+'_metrics_unavailable.json', {'type': type(error).__name__})
        return counts

    def statistics(self):
        return {'forward_calls': self.calls, 'reserved_circuits': self.circuits,
                'reserved_shots': self.circuits*self.limits.shots, 'reserved_jobs': self.jobs,
                'job_ids': self.job_ids, 'error_mitigation': False, 'error_correction': False,
                'dynamical_decoupling': False, 'twirling': False,
                'uncertainty': 'IID shot estimate only; excludes device bias and drift'}
