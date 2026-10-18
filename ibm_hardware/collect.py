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
    from .artifacts import save_result, write_runtime_json
    service = QiskitRuntimeService(channel='ibm_quantum_platform', name=options['account'],
                                  instance=options['instance'])
    job = service.job(submitted['job_id'])
    if job.backend().name != intent['backend']:
        raise ValueError('Retrieved job backend differs from recorded submission')
    output = Evidence(directory/(prefix+'_recovered'), directory/'config.json',
                      'ibm_hardware_retrieval')
    counts = save_result(output, prefix, job.result(), intent['records'], intent['shots'])
    try:
        write_runtime_json(output, prefix+'_metrics.json', job.metrics())
    except Exception as error:
        output.write(prefix+'_metrics_unavailable.json', {'type': type(error).__name__})
    if options.get('mode') == 'pilot':
        from classical._core.problem import Problem
        from .readout import analyze_pilot
        problem = Problem.load(directory/'config.json')
        output.write('pilot_pressure.json', analyze_pilot(
            intent['records'], counts, intent['shots'], problem.trace_shape))
    if options.get('mode') == 'experiment':
        from .report import generate_report
        generate_report(directory)
    output.finish({'job_id': submitted['job_id'], 'new_jobs_submitted': 0})


if __name__ == '__main__':
    main()
