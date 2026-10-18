"""Explicitly gated execution of the reduced classical pressure inversion."""
import argparse
import hashlib
from pathlib import Path


def main(engine='classical'):
    if engine != 'classical':
        raise ValueError('This branch contains the classical implementation only')
    parser = argparse.ArgumentParser(description='Reduced classical pressure inversion')
    parser.add_argument('--config', required=True)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('No execution performed. Supply --execute only when authorized.')
    from .problem import Problem, load_observations
    from .evidence import Evidence
    from .inversion import invert
    from classical.reference import ClassicalForward
    problem = Problem.load(args.config)
    observed = load_observations(args.observations, problem)
    if Path(args.output).exists():
        parser.error('Output directory exists; choose a new path')
    evidence = Evidence(args.output, args.config, engine, args.observations)
    evidence.write('execution_options.json', vars(args))
    evidence.write('comparison.json', {
        'inverse_problem_signature': problem.comparison_signature,
        'observations_sha256': hashlib.sha256(Path(args.observations).read_bytes()).hexdigest(),
        'objective': 'variance_corrected_pressure_l2_v1',
        'requirement': 'Match both signatures before pairing runs from different solvers'})
    try:
        solver = ClassicalForward(problem)
        parameters, _ = invert(problem, observed, solver.predict,
                               lambda history: evidence.write('history.json', history))
        speed = problem.speeds(parameters).reshape(problem.grid.shape)
        evidence.write('reconstruction.json', {'parameters': parameters.tolist(),
                       'sound_speed_m_s': speed.tolist(),
                       'meaning': 'Estimated sound speed; not a diagnostic image'})
        evidence.finish(solver.statistics())
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
