"""Run from the repository root with python -m simulator.run.

Use the same classical/configs/*.json and observations NPZ as the classical run.
Noise probability defaults to zero; a positive value adds generic depolarizing gate noise.
"""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Gate-simulated pressure inversion')
    parser.add_argument('--config', required=True)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--shots', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-circuits', type=int, default=10000)
    parser.add_argument('--max-total-shots', type=int, default=100000000)
    parser.add_argument('--max-data-qubits', type=int, default=7)
    parser.add_argument('--max-gates', type=int, default=500000)
    parser.add_argument('--noise-probability', type=float, default=0.0)
    args = parser.parse_args()
    if not args.execute:
        parser.error('No execution performed. Supply --execute only when authorized.')
    if (args.shots < 2 or args.seed < 0 or args.max_circuits < 1
            or args.max_total_shots < args.shots or args.max_gates < 1
            or not 1 <= args.max_data_qubits <= 8 or not 0 <= args.noise_probability <= 0.1):
        parser.error('Invalid shot, seed, resource limit or noise setting')
    from classical._core.problem import Problem, load_observations
    from classical._core.evidence import Evidence
    from classical._core.inversion import invert
    from .forward import GateForward
    problem = Problem.load(args.config)
    observed = load_observations(args.observations, problem)
    if Path(args.output).exists():
        parser.error('Output directory exists; choose a new path')
    per_call = len(problem.sources)*len(problem.receivers)*sum(len(problem.impulses(t)) for t in problem.times)
    evaluations = problem.config['optimizer']['iterations']*(1+2*len(problem.initial))+1
    circuits, shots = per_call*evaluations, per_call*evaluations*args.shots
    if circuits > args.max_circuits or shots > args.max_total_shots:
        parser.error(f'Planned inversion needs {circuits} circuits and {shots} shots; above budget')
    evidence = Evidence(args.output, args.config, 'gate_based_simulator', args.observations)
    evidence.write('execution_options.json', vars(args))
    evidence.write('comparison.json', {
        'inverse_problem_signature': problem.comparison_signature,
        'observations_sha256': hashlib.sha256(Path(args.observations).read_bytes()).hexdigest(),
        'objective': 'variance_corrected_pressure_l2_v1',
        'requirement': 'Match both signatures before pairing runs from different solvers'})
    try:
        solver = GateForward(problem, evidence, args.shots, args.seed,
                             args.max_circuits, args.max_total_shots,
                             args.max_data_qubits, args.max_gates, args.noise_probability)
        parameters, _ = invert(problem, observed, solver.predict,
                               lambda history: evidence.write('history.json', history))
        evidence.write('reconstruction.json', {
            'parameters': parameters.tolist(),
            'sound_speed_m_s': problem.speeds(parameters).reshape(problem.grid.shape).tolist(),
            'meaning': 'Estimated sound speed; not a diagnostic image'})
        evidence.finish(solver.statistics())
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
