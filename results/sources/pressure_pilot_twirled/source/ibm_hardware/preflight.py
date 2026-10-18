"""Compile a pilot while simulator verification is pending. This CLI cannot submit.

Use --snapshot only for offline planning against bundled IBM calibration data.
Live preparation needs a saved SDK account and an explicit --instance CRN.
Actual pilot submission still goes through run.py and requires a verified reference.
"""
import argparse


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', default='ibm_hardware/pilot_1d.json')
    cli.add_argument('--backend', required=True)
    cli.add_argument('--instance')
    cli.add_argument('--account')
    cli.add_argument('--snapshot', action='store_true')
    cli.add_argument('--parameters', type=float, nargs='+', required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--seed-transpiler', type=int, default=7)
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args(argv)
    if not args.execute:
        cli.error('Nothing executed. Compilation requires --execute.')
    if args.seed_transpiler < 0:
        cli.error('Transpiler seed must be nonnegative')
    if args.snapshot and (args.instance or args.account):
        cli.error('Offline snapshots do not use credentials or an instance')
    if not args.snapshot and (not args.instance or not args.instance.startswith('crn:')):
        cli.error('Live preparation requires an explicit instance CRN')

    from pathlib import Path
    from classical._core.problem import Problem
    from classical._core.evidence import Evidence
    from .policy import Limits
    from .pilot import prepare_pilot
    from .artifacts import write_runtime_json
    problem = Problem.load(args.config)
    problem.speeds(args.parameters)
    limits = Limits.load(Path(__file__).with_name('pilot_limits.json'))
    evidence = Evidence(args.output, args.config, 'ibm_compile_only')
    try:
        if args.snapshot:
            from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeFez, FakeMarrakesh
            choices = {'ibm_kingston': FakeKingston, 'ibm_fez': FakeFez, 'ibm_marrakesh': FakeMarrakesh}
            if args.backend not in choices:
                raise ValueError('No configured bundled snapshot for this backend')
            backend = choices[args.backend]()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(name=args.account, channel='ibm_quantum_platform',
                                          instance=args.instance)
            backend = service.backend(args.backend, instance=args.instance, use_fractional_gates=False)
            if backend.configuration().simulator or not backend.status().operational:
                raise ValueError('Live preparation needs an operational real backend')
            write_runtime_json(evidence, 'backend_status.json', backend.status().to_dict())
        properties = backend.properties()
        evidence.write('preflight.json', {
            'backend': args.backend, 'snapshot_only': args.snapshot,
            'calibration_date': str(properties.last_update_date),
            'parameters': args.parameters, 'seed_transpiler': args.seed_transpiler,
            'submission_enabled': False, 'simulation_verification': 'pending',
            'next_gate': 'Verify a matching simulator.reference output and refresh live compilation'})
        write_runtime_json(evidence, 'backend_properties.json', properties.to_dict())
        write_runtime_json(evidence, 'backend_configuration.json', backend.configuration().to_dict())
        evidence.write('hardware_limits.json', vars(limits))
        circuits, _ = prepare_pilot(problem, args.parameters, backend, evidence, limits,
                                   args.seed_transpiler, None)
        evidence.finish({'status': 'snapshot_compiled' if args.snapshot else 'live_compiled',
                         'circuits_prepared': len(circuits), 'jobs_submitted': 0,
                         'simulation_verification': 'pending'})
        print(f'Compiled {len(circuits)} circuits. No jobs submitted. Verification pending.')
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error),
                                       'jobs_submitted': 0})
        raise


if __name__ == '__main__':
    main()
