"""Run matched forward-wave comparisons on a CPU, never on quantum hardware.

python -m annealing.run --config classical/configs/small_1d.json \
    --parameters 1 1.1 --output runs/annealing_small --workers 2 --execute

SQA is path-integral Monte Carlo with a nonzero transverse-field schedule.
SA is the classical annealing control. Both optimize identical fixed-point
Crank-Nicolson QUBOs with identical read/sweep budgets, not equal CPU cost.
Results concern the reduced forward solver, not full FWI or the paper's runtime.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter


def run_repeat(config, parameters, settings_dict, algorithm, seed, destination):
    from classical._core.problem import Problem
    from .forward import Settings, AnnealingForward, pressure_forward
    problem, settings = Problem.load(config), Settings(**settings_dict)
    destination = Path(destination)
    started = perf_counter()
    with destination.with_suffix('.jsonl').open('x') as stream:
        def record(row):
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
        solver = AnnealingForward(settings, algorithm, seed, record)
        try:
            pressure = pressure_forward(problem, parameters, solver.solve)
            row = {'algorithm': algorithm, 'seed': seed, 'pressure_pa': pressure.tolist(),
                   'wall_seconds': perf_counter()-started, 'bqm_calls': solver.calls,
                   'status': 'completed', 'quantum_hardware_jobs': 0}
        except Exception as error:
            row = {'algorithm': algorithm, 'seed': seed, 'status': 'failed',
                   'type': type(error).__name__, 'message': str(error),
                   'wall_seconds': perf_counter()-started, 'bqm_calls': solver.calls}
    destination.with_suffix('.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    return row


def errors(predicted, expected):
    import numpy as np
    difference = np.asarray(predicted)-np.asarray(expected)
    norm = float(np.linalg.norm(expected))
    return {'rmse_pa': float(np.sqrt(np.mean(difference**2))),
            'relative_l2': float(np.linalg.norm(difference)/norm) if norm > 1e-12 else None}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', required=True)
    cli.add_argument('--parameters', type=float, nargs='+')
    cli.add_argument('--output', required=True)
    cli.add_argument('--gate-reference', help='Optional matching verified simulator.reference JSON')
    cli.add_argument('--algorithms', choices=['sa', 'sqa'], nargs='+', default=['sa', 'sqa'])
    cli.add_argument('--repetitions', type=int, default=3)
    cli.add_argument('--workers', type=int, default=1, help='Parallel independent repeats, maximum 4')
    cli.add_argument('--seed', type=int, default=7)
    cli.add_argument('--bits', type=int, default=6)
    cli.add_argument('--refinements', type=int, default=3)
    cli.add_argument('--reads', type=int, default=16)
    cli.add_argument('--sweeps', type=int, default=200)
    cli.add_argument('--max-variables', type=int, default=512)
    cli.add_argument('--max-bqm-calls', type=int, default=2000, help='Per independent repeat')
    cli.add_argument('--max-seconds', type=float, default=600, help='Per repeat; checked between reads')
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args(argv)
    if not args.execute:
        cli.error('Nothing executed. Simulation requires --execute.')
    if (not 1 <= args.workers <= 4 or not 1 <= args.repetitions <= 10
            or len(set(args.algorithms)) != len(args.algorithms)
            or not 0 <= args.seed < 2**31-args.repetitions):
        cli.error('Use 1–4 workers, 1–10 repeats, distinct algorithms and a valid seed')
    if Path(args.output).exists():
        cli.error('Use a new output directory')
    from .forward import Settings, resource_plan, direct_cn
    from classical._core.problem import Problem
    from classical._core.evidence import Evidence
    import numpy as np
    problem = Problem.load(args.config)
    parameters = problem.initial.tolist() if args.parameters is None else args.parameters
    problem.speeds(parameters)
    gate_reference = None
    if args.gate_reference:
        from simulator.reference import load_reference
        gate_reference = load_reference(args.gate_reference, problem, parameters)
    settings = Settings(bits=args.bits, refinements=args.refinements, reads=args.reads,
                        sweeps=args.sweeps, max_variables=args.max_variables,
                        max_bqm_calls=args.max_bqm_calls, max_seconds=args.max_seconds)
    plan = resource_plan(problem, settings)
    plan.update(repetitions_per_algorithm=args.repetitions, algorithms=args.algorithms,
                workers=args.workers,
                total_bqm_calls_at_most=plan['bqm_calls_per_repeat_at_most']
                *args.repetitions*len(args.algorithms))
    evidence = Evidence(args.output, args.config, 'annealing_forward_comparison')
    if gate_reference is not None:
        evidence.write('gate_reference.json', gate_reference)
    evidence.write('settings.json', {**vars(args), 'parameters': parameters, 'annealing': asdict(settings)})
    evidence.write('resource_plan.json', plan)
    evidence.write('comparison.json', {
        'acquisition_signature': problem.acquisition_signature, 'parameters': parameters,
        'base_speed_ratio': problem.base.tolist(), 'region_ids': problem.regions.tolist(),
        'trace_axes': ['source', 'time', 'receiver'],
        'same_physics': 'Shared generator, grid, boundaries, source quadrature and pressure scaling',
        'different_numerics': 'Gate path: Strang splitting. Annealing: fixed-point Crank-Nicolson QUBO.',
        'sqa_meaning': 'Classical path-integral simulation; no quantum processor or real-time QA claim',
        'statistics': 'Independent optimization repeats; spread is not quantum shot uncertainty',
        'scope': 'Forward solver only; no tissue reconstruction or speedup claim'})
    try:
        tasks = [(str(evidence.path/'config.json'), parameters, asdict(settings), algorithm,
                  args.seed+repeat, str(evidence.path/f'{algorithm}_{repeat:02d}'))
                 for algorithm in args.algorithms for repeat in range(args.repetitions)]
        rows = []
        if args.workers == 1:
            for task in tasks:
                rows.append(run_repeat(*task))
                print(f"{rows[-1]['algorithm']} seed={rows[-1]['seed']}: {rows[-1]['status']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_repeat, *task) for task in tasks]
                for future in as_completed(futures):
                    rows.append(future.result())
                    print(f"{rows[-1]['algorithm']} seed={rows[-1]['seed']}: {rows[-1]['status']}", flush=True)
        rows.sort(key=lambda row: (row['algorithm'], row['seed']))
        evidence.write('repeats.json', rows)
        # Compute references only AFTER all sampling. Neither enters the QUBO solver.
        from classical.reference import ClassicalForward
        start = perf_counter()
        exact = ClassicalForward(problem).predict(parameters).mean
        exact_seconds = perf_counter()-start
        start = perf_counter()
        cn = direct_cn(problem, parameters)
        cn_seconds = perf_counter()-start
        report = {'exact_discrete_pressure_pa': exact.tolist(), 'cn_pressure_pa': cn.tolist(),
                  'exact_forward_seconds': exact_seconds, 'cn_forward_seconds': cn_seconds,
                  'cn_time_discretization_error': errors(cn, exact), 'algorithms': {},
                  'timing_note': 'CPU wall times on this machine; no quantum speedup measurement'}
        if gate_reference is not None:
            report['noiseless_gate_pressure_pa'] = gate_reference['ideal_pressure_pa']
            report['noiseless_gate_error_vs_exact'] = errors(gate_reference['ideal_pressure_pa'], exact)
        for algorithm in args.algorithms:
            all_rows = [row for row in rows if row['algorithm'] == algorithm]
            good = [row for row in all_rows if row['status'] == 'completed']
            for row in good:
                row['error_vs_exact'] = errors(row['pressure_pa'], exact)
                row['error_vs_cn'] = errors(row['pressure_pa'], cn)
            array = np.array([row['pressure_pa'] for row in good])
            report['algorithms'][algorithm] = {
                'completed_repeats': len(good), 'requested_repeats': args.repetitions,
                'all_repeats': all_rows,
                'mean_pressure_pa': array.mean(axis=0).tolist() if good else None,
                'repeat_standard_deviation_pa': array.std(axis=0, ddof=1).tolist() if len(good)>1 else None,
                'all_repeats_within_one_percent': len(good) == args.repetitions and all(
                    row['error_vs_exact']['relative_l2'] is not None
                    and row['error_vs_exact']['relative_l2'] <= .01 for row in good)}
        evidence.write('report.json', report)
        failed = any(row['status'] != 'completed' for row in rows)
        evidence.finish({'status': 'incomplete' if failed else 'completed',
                         'quantum_hardware_jobs': 0, 'total_repeats': len(rows),
                         'scope': 'Forward-wave experiment; inspect accuracy, including failed repeats'})
        if failed:
            raise RuntimeError('Some repeats failed; inspect report.json, not just successful runs')
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
