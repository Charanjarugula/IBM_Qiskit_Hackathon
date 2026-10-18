"""Prepare or explicitly submit matched raw-pressure circuits to an IBM backend.

Use python -m ibm_hardware.run from the repository root. Preparation requires
--execute; forward/invert also require --submit-hardware. Use the same reduced
classical/configs/*.json and observations as the classical and simulator runs.
For the first experiment use --mode pilot with ibm_hardware/pilot_1d.json.
Both pilot and experiment require --reference <simulator.reference output>.
While that verification is pending, ibm_hardware.preflight can compile but cannot submit.
Without --submit-hardware it only prepares the pilot and records its resource plan.
Use --mode experiment for three repeats of optimized raw versus readout-corrected
pressure. --include-original-arm additionally compares the deeper old circuit;
it must independently fit the same limits. Preparation does not submit jobs.
"""
import argparse
import hashlib
from pathlib import Path


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mode', choices=['prepare', 'pilot', 'experiment', 'forward', 'invert'], default='prepare')
    cli.add_argument('--config', required=True)
    cli.add_argument('--backend', required=True, help='Exact IBM backend name; never auto-selected')
    cli.add_argument('--instance', required=True, help='Explicit IBM service instance CRN')
    cli.add_argument('--account', help='Optional saved account name; never put a token in CLI arguments')
    cli.add_argument('--output', required=True)
    cli.add_argument('--observations', help='Required only for inversion; same file as classical/Aer')
    cli.add_argument('--reference', help='Verified simulator.reference JSON; required for pilot/experiment')
    cli.add_argument('--limits', help='Defaults to the selected mode’s limits JSON')
    cli.add_argument('--parameters', type=float, nargs='+', help='Forward/preparation speeds as ratios')
    cli.add_argument('--initial-layout', type=int, nargs='+', help='Physical qubits; ancilla is first')
    cli.add_argument('--seed-transpiler', type=int, default=7)
    cli.add_argument('--include-original-arm', action='store_true',
                     help='Experiment only: add the deep original circuit if it fits all limits')
    cli.add_argument('--execute', action='store_true', help='Authorize preparation/connection when ready')
    cli.add_argument('--submit-hardware', action='store_true', help='Separately authorize actual QPU jobs')
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    # Fail before SDK imports, credentials, compilation or backend calls.
    if not args.execute:
        cli.error('Nothing executed. --execute is required even for preparation.')
    if args.mode in ('forward', 'invert') and not args.submit_hardware:
        cli.error('Actual IBM jobs require --submit-hardware as well as --execute')
    if args.mode == 'prepare' and args.submit_hardware:
        cli.error('Preparation cannot submit hardware jobs')
    if args.include_original_arm and args.mode != 'experiment':
        cli.error('--include-original-arm is only valid for --mode experiment')
    if args.mode == 'invert' and (not args.observations or args.parameters is not None):
        cli.error('Inversion needs --observations and uses the shared configured initial guess')
    if args.mode != 'invert' and args.observations:
        cli.error('Observations are only used by inversion; pilot predictions are not fitted to a reference')
    needs_reference = args.mode in ('pilot', 'experiment')
    if needs_reference != bool(args.reference):
        cli.error('Pilot and experiment require --reference; other modes do not use it')
    if Path(args.output).exists():
        cli.error('Output directory exists; use a new path')
    if args.seed_transpiler < 0:
        cli.error('Transpiler seed must be nonnegative')
    if not args.instance.startswith('crn:'):
        cli.error('Supply an explicit IBM instance CRN; automatic selection is disabled')

    from classical._core.problem import Problem, load_observations
    from classical._core.evidence import Evidence
    from .policy import Limits, plan, raw_sampler_options
    if args.limits is None:
        limits_name = args.mode+'_limits.json' if args.mode in ('pilot', 'experiment') else 'limits.json'
        args.limits = str(Path(__file__).with_name(limits_name))
    problem, limits = Problem.load(args.config), Limits.load(args.limits)
    parameters = problem.initial if args.parameters is None else args.parameters
    problem.speeds(parameters)
    resource_plan = plan(problem, limits, args.mode, args.include_original_arm)
    observed = load_observations(args.observations, problem) if args.mode == 'invert' else None
    reference = None
    if needs_reference:
        from simulator.reference import load_reference
        reference = load_reference(args.reference, problem, parameters)

    from qiskit_ibm_runtime import QiskitRuntimeService
    from .artifacts import write_runtime_json
    from .forward import HardwareForward
    # Read saved credentials/environment through the SDK, never record or rewrite credentials.
    service = QiskitRuntimeService(channel='ibm_quantum_platform', name=args.account,
                                  instance=args.instance)
    backend = service.backend(args.backend, instance=args.instance, use_fractional_gates=False)
    if backend.configuration().simulator or not backend.status().operational:
        raise ValueError('Select an operational real IBM backend')
    evidence = Evidence(args.output, args.config, 'ibm_hardware_'+args.mode,
                        args.observations if args.mode == 'invert' else None)
    solver = HardwareForward(problem, backend, evidence, limits, args.seed_transpiler,
                             args.initial_layout, submit_hardware=args.submit_hardware)
    try:
        evidence.write('execution_options.json', vars(args))
        evidence.write('hardware_limits.json', vars(limits))
        evidence.write('resource_plan.json', resource_plan)
        evidence.write('sampler_options.json', raw_sampler_options(limits))
        evidence.write('comparison.json', {
            'inverse_problem_signature': problem.comparison_signature,
            'observations_sha256': (hashlib.sha256(Path(args.observations).read_bytes()).hexdigest()
                                    if observed is not None else None),
            'objective': ('forward_pressure_'+args.mode if args.mode in ('pilot', 'experiment')
                          else 'variance_corrected_pressure_l2_v1'),
            'readout_mitigation_requested': args.mode in ('pilot', 'experiment'), 'error_correction': False})
        write_runtime_json(evidence, 'backend_configuration.json', backend.configuration().to_dict())
        properties = backend.properties()
        write_runtime_json(evidence, 'backend_properties.json', None if properties is None else properties.to_dict())
        evidence.write('backend.json', {'name': backend.name, 'num_qubits': backend.num_qubits,
                                       'operation_names': sorted(backend.operation_names)})
        if args.mode == 'experiment':
            from .experiment import run_experiment
            run_experiment(solver, parameters, reference, args.include_original_arm)
            return
        if args.mode == 'pilot':
            from .pilot import run_pilot
            evidence.write('reference.json', reference)
            run_pilot(solver, parameters)
            return
        if args.mode == 'prepare':
            _, records = solver.prepare(parameters)
            evidence.finish({'status': 'prepared_only', 'jobs_submitted': 0,
                             'circuits_prepared': len(records),
                             'scope': 'One parameter vector; later inversion probes recompile'})
            return
        if args.mode == 'forward':
            prediction = solver.predict(parameters)
            evidence.write('pressure.json', {
                'parameters': list(parameters), 'pressure_pa': prediction.mean.tolist(),
                'standard_error_pa': prediction.standard_error.tolist(),
                'sampling_variance_pa2': prediction.sampling_variance.tolist(),
                'meaning': 'Raw hardware pressure estimate; no device-error mitigation'})
        else:
            from classical._core.inversion import invert
            parameters, _ = invert(problem, observed, solver.predict,
                                   lambda history: evidence.write('history.json', history))
            evidence.write('reconstruction.json', {
                'parameters': parameters.tolist(),
                'sound_speed_m_s': problem.speeds(parameters).reshape(problem.grid.shape).tolist(),
                'meaning': 'Raw hardware-driven sound-speed estimate; not a diagnostic image'})
        evidence.finish(solver.statistics())
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error),
                                       'execution': solver.statistics(),
                                       'recovery': 'Inspect saved job IDs/tags; never blindly rerun a submission'})
        raise


if __name__ == '__main__':
    main()
