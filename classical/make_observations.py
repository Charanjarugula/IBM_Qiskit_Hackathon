"""Explicit synthetic-observation generation, separate from either optimizer."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('No data generated. --execute required when authorized.')
    import numpy as np
    from shared.problem import Problem
    from shared.evidence import Evidence
    from .reference import ClassicalForward
    p = Problem.load(args.config)
    truth = np.asarray(p.config['synthetic_truth_parameters'], dtype=float)
    p.speeds(truth)  # Validate before creating the output directory.
    evidence = Evidence(args.output, args.config, 'classical_synthetic_observations')
    try:
        solver = ClassicalForward(p)
        prediction = solver.predict(truth)
        np.savez(evidence.path/'observations.npz', pressure_pa=prediction.mean,
                 acquisition_signature=p.acquisition_signature,
                 provenance='exact_discrete_same_model_synthetic_inverse_crime')
        evidence.write('truth.json', {'parameters': truth.tolist(),
                                     'warning': 'Same-model synthetic test; not independent clinical data'})
        evidence.finish(solver.statistics())
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
