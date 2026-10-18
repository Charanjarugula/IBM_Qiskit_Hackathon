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
        if self.sampler is None:
            # Dedicated import avoids future top-level aliases changing primitive behavior.
            from qiskit_ibm_runtime.sampler import SamplerV2
            self.sampler = SamplerV2(mode=self.backend, options=raw_sampler_options(limits))
        output = np.zeros(p.trace_shape)
        reporting_variance, sampling_variance = np.zeros_like(output), np.zeros_like(output)
        for start in range(0, len(circuits), limits.circuits_per_job):
            batch = circuits[start:start+limits.circuits_per_job]
            batch_records = records[start:start+limits.circuits_per_job]
            prefix = f'job_{self.jobs:04d}'
            tag = 'pressure-'+uuid.uuid4().hex
            self.sampler.options.environment.job_tags = [tag]
            self.evidence.write(prefix+'_intent.json', {
                'tag': tag, 'backend': self.backend.name, 'forward_call': self.calls,
                'first_circuit': start, 'records': batch_records, 'shots': limits.shots,
                'options': raw_sampler_options(limits),
                'recovery': 'If submission fails ambiguously, find this tag in IBM before retrying'})
            # Reserve before submission. No automatic retries: an ambiguous API failure
            # might mean that a paid job already exists on the server.
            self.jobs += 1
            self.circuits += len(batch)
            job = self.sampler.run(batch, shots=limits.shots)
            self.job_ids.append(job.job_id())
            self.evidence.write(prefix+'_submitted.json', {'job_id': job.job_id(), 'tag': tag})
            result = job.result()
            counts = save_result(self.evidence, prefix, result, batch_records, limits.shots)
            for record, raw in zip(batch_records, counts):
                overlap, se, variance = ancilla_statistics(raw, limits.shots)
                key = (record['source_index'], record['time_index'], record['receiver_index'])
                scale = record['physical_scale_pa']
                output[key] += scale*overlap
                reporting_variance[key] += (scale*se)**2
                sampling_variance[key] += scale**2*variance
            # Results remain saved even if optional usage metadata is temporarily unavailable.
            try:
                write_runtime_json(self.evidence, prefix+'_metrics.json', job.metrics())
            except Exception as error:
                self.evidence.write(prefix+'_metrics_unavailable.json', {'type': type(error).__name__})
        self.calls += 1
        return Prediction(output, np.sqrt(reporting_variance), sampling_variance)

    def statistics(self):
        return {'forward_calls': self.calls, 'reserved_circuits': self.circuits,
                'reserved_shots': self.circuits*self.limits.shots, 'reserved_jobs': self.jobs,
                'job_ids': self.job_ids, 'error_mitigation': False, 'error_correction': False,
                'dynamical_decoupling': False, 'twirling': False,
                'uncertainty': 'IID shot estimate only; excludes device bias and drift'}
