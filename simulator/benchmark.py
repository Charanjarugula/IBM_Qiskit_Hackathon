"""Fixed-map finite-shot wave benchmark; no reconstruction or hardware access.

First create a matching simulator.reference. Run ideal and noisy benchmarks in
separate output directories, then use simulator.compare with annealing outputs.
"""
import argparse
from pathlib import Path
from time import perf_counter


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', required=True)
    cli.add_argument('--parameters', type=float, nargs='+', required=True)
    cli.add_argument('--reference', required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--repetitions', type=int, default=3)
    cli.add_argument('--shots', type=int, default=8192)
    cli.add_argument('--seed', type=int, default=7)
    cli.add_argument('--noise-probability', type=float, default=0)
    cli.add_argument('--method', choices=['auto', 'statevector', 'density_matrix'], default='auto')
    cli.add_argument('--threads', type=int, default=1)
    cli.add_argument('--max-seconds', type=float, default=1800,
                     help='Whole gate campaign; checked between circuits, not a hard interrupt')
    cli.add_argument('--max-circuits', type=int, default=192)
    cli.add_argument('--max-total-shots', type=int, default=2000000)
    cli.add_argument('--max-gates', type=int, default=500000,
                     help='Per compiled circuit; larger 2D cases may need an explicit higher ceiling')
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args(argv)
    if not args.execute:
        cli.error('Nothing executed. Simulation requires --execute.')
    if not 1 <= args.repetitions <= 10 or args.shots < 2 or not 0 <= args.seed < 2**31:
        cli.error('Use 1–10 repetitions, at least two shots and a valid seed')
    if Path(args.output).exists():
        cli.error('Use a new output directory')
    from classical._core.problem import Problem
    from classical._core.evidence import Evidence
    from classical.reference import ClassicalForward
    from .reference import load_reference
    from .forward import GateForward
    problem = Problem.load(args.config)
    problem.speeds(args.parameters)
    reference = load_reference(args.reference, problem, args.parameters)
    circuits = reference['wave_circuits']*args.repetitions
    if circuits > args.max_circuits or circuits*args.shots > args.max_total_shots:
        cli.error('Fixed-map campaign exceeds circuit/shot budget')
    evidence = Evidence(args.output, args.config, 'gate_forward_benchmark')
    evidence.write('settings.json', vars(args))
    evidence.write('reference.json', reference)
    evidence.write('comparison.json', {
        'acquisition_signature': problem.acquisition_signature, 'parameters': args.parameters,
        'base_speed_ratio': problem.base.tolist(), 'region_ids': problem.regions.tolist(),
        'scope': 'Fixed-map forward pressure; no inversion or speedup claim'})
    rows = []
    try:
        solver = GateForward(problem, evidence, args.shots, args.seed, circuits,
                             circuits*args.shots, max_data_qubits=6, max_gates=args.max_gates,
                             noise_probability=args.noise_probability, method=args.method,
                             max_parallel_threads=args.threads, max_seconds=args.max_seconds)
        for repeat in range(args.repetitions):
            start = perf_counter()
            prediction = solver.predict(args.parameters)
            elapsed = perf_counter()-start
            rows.append({'repeat': repeat, 'status': 'completed',
                         'pressure_pa': prediction.mean.tolist(),
                         'standard_error_pa': prediction.standard_error.tolist(),
                         'wall_seconds': elapsed,
                         'timing_scope': 'Circuit construction, transpilation, Aer and count recording'})
            evidence.write('repeats.json', rows)
        # Measure the same classical task separately; never feed it to the gates.
        classical_rows = []
        for repeat in range(args.repetitions):
            start = perf_counter()
            exact = ClassicalForward(problem).predict(args.parameters).mean
            classical_rows.append({'repeat': repeat, 'pressure_pa': exact.tolist(),
                                   'wall_seconds': perf_counter()-start})
        evidence.write('classical.json', classical_rows)
        evidence.finish({**solver.statistics(), 'status': 'completed',
                         'scope': 'Fixed-map forward prediction', 'quantum_hardware_jobs': 0})
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error),
                                       'completed_repetitions': len(rows)})
        raise


if __name__ == '__main__':
    main()
