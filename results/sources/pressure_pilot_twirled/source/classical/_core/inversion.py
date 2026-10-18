"""Same projected finite-difference gradient descent for both matched forward solvers."""
import numpy as np


def invert(problem, observed, predict, checkpoint):
    cfg = problem.config['optimizer']
    count = int(cfg['iterations'])
    eps, lr, max_step = (float(cfg[k]) for k in ('difference_step', 'learning_rate', 'max_parameter_step'))
    if count < 1 or not np.isfinite([eps, lr, max_step]).all() or min(eps, lr, max_step) <= 0:
        raise ValueError('Invalid optimizer configuration')
    theta = problem.initial.copy()
    scale = float(np.sum(observed**2))
    history = []

    def loss(parameters):
        result = predict(parameters)
        if any(a.shape != observed.shape or not np.isfinite(a).all()
               for a in (result.mean, result.standard_error, result.sampling_variance)):
            raise ValueError('Invalid forward prediction')
        if np.any(result.standard_error < 0) or np.any(result.sampling_variance < 0):
            raise ValueError('Invalid sampling uncertainty')
        value = 0.5 * float(np.sum((result.mean-observed)**2)) / scale
        correction = 0.5 * float(np.sum(result.sampling_variance)) / scale
        # Unbiased for the circuit's mean-pressure objective, not for Trotter/noise bias.
        # Do not clip negative estimates: doing so would reintroduce bias.
        return value-correction, correction

    for iteration in range(count + 1):
        value, noise_floor = loss(theta)
        row = {'iteration': iteration, 'parameters': theta.tolist(),
               'relative_pressure_loss': value, 'plugin_pressure_loss': value+noise_floor,
               'estimated_sampling_loss_contribution': noise_floor}
        history.append(row)
        checkpoint(history)
        if iteration == count:
            break
        gradient = np.empty_like(theta)
        for k in range(len(theta)):
            minus, plus = theta.copy(), theta.copy()
            # Central probes when possible; otherwise a one-sided interval anchored
            # at theta. Clipping both probes asymmetrically estimates another point.
            if theta[k]-eps < problem.lower[k]:
                minus[k] = theta[k]
                plus[k] = min(problem.upper[k], theta[k]+eps)
            elif theta[k]+eps > problem.upper[k]:
                minus[k] = max(problem.lower[k], theta[k]-eps)
                plus[k] = theta[k]
            else:
                minus[k], plus[k] = theta[k]-eps, theta[k]+eps
            hi, _ = loss(plus)
            lo, _ = loss(minus)
            gradient[k] = (hi-lo)/(plus[k]-minus[k])
        step = -lr*gradient
        # One shared trust-radius rule, no truth or exact-solver access in this update.
        step *= min(1.0, max_step/max(float(np.max(np.abs(step))), 1e-30))
        row['gradient'] = gradient.tolist()
        updated = np.clip(theta+step, problem.lower, problem.upper)
        row['step'] = (updated-theta).tolist()
        theta = updated
        checkpoint(history)
    return theta, history
