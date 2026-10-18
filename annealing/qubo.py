"""Binary least-squares formulation, with signed fixed-point residual refinement.

For A x = b and x = center + W z, minimize ||A W z - (b-A center)||².
Diagonal z_i²=z_i; off-diagonal QUBO coefficients contain the factor two.
Based on the linear-equation construction in Chang et al. (2019),
https://doi.org/10.1038/s41598-019-46729-0 . The time-domain acoustic
Crank-Nicolson discretization here is our adaptation, not a paper reproduction.
"""
import numpy as np
import dimod


def encoding(dimension, bits, radius):
    if dimension < 1 or not 2 <= bits <= 10 or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid signed fixed-point encoding')
    # Two's complement includes exactly zero and both signs; range [-r, r-quantum].
    weights = radius*np.r_[-1.0, 2.0**(-np.arange(1, bits))]
    return np.kron(np.eye(dimension), weights.reshape(1, -1))


def least_squares_bqm(A, b, center, W):
    A, b, center, W = [np.asarray(x, dtype=float) for x in (A, b, center, W)]
    if (A.ndim != 2 or b.shape != (A.shape[0],) or center.shape != (A.shape[1],)
            or W.ndim != 2 or W.shape[0] != A.shape[1]
            or not all(np.isfinite(x).all() for x in (A, b, center, W))):
        raise ValueError('Invalid least-squares arrays')
    B, residual = A@W, b-A@center
    gram = B.T@B
    linear = np.diag(gram)-2*B.T@residual
    quadratic = {(i, j): 2*gram[i, j] for i in range(len(linear))
                 for j in range(i+1, len(linear)) if gram[i, j] != 0}
    return dimod.BinaryQuadraticModel(dict(enumerate(linear)), quadratic,
                                     float(residual@residual), dimod.BINARY)


def normalized_ising(bqm):
    """Normalize biases without changing minima; never claim hardware precision."""
    spin = bqm.change_vartype(dimod.SPIN, inplace=False)
    scale = max([abs(x) for x in spin.linear.values()]
                + [abs(x) for x in spin.quadratic.values()] + [1e-300])
    spin.scale(1/scale)
    return spin, scale
