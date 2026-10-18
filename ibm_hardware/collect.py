"""Recover a recorded IBM job result. Never creates a job or resumes an inversion."""
import argparse
import json
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run-directory', required=True)
    cli.add_argument('--job-index', type=int, required=True)
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args()
    if not args.execute:
        cli.error('Nothing executed; retrieval requires --execute')
    if args.job_index < 0:
        cli.error('Job index must be nonnegative')
    directory = Path(args.run_directory).resolve()
    prefix = f'job_{args.job_index:04d}'
    options = json.loads((directory/'execution_options.json').read_text())
    submitted = json.loads((directory/(prefix+'_submitted.json')).read_text())
    intent = json.loads((directory/(prefix+'_intent.json')).read_text())
    from qiskit_ibm_runtime import QiskitRuntimeService
    from classical._core.evidence import Evidence
    from .artifacts import save_result
    service = QiskitRuntimeService(channel='ibm_quantum_platform', name=options['account'],
                                  instance=options['instance'])
    job = service.job(submitted['job_id'])
    if job.backend().name != intent['backend']:
        raise ValueError('Retrieved job backend differs from recorded submission')
    output = Evidence(directory/(prefix+'_recovered'), directory/'config.json',
                      'ibm_hardware_retrieval')
    save_result(output, prefix, job.result(), intent['records'], intent['shots'])
    output.finish({'job_id': submitted['job_id'], 'new_jobs_submitted': 0})


if __name__ == '__main__':
    main()
