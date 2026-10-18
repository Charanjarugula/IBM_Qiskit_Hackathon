# Reduced classical pressure model

Status: source and mathematical review only; numerical validation is pending.

Let B be the oriented grid incidence matrix divided by spacing, C = diag(c), K = C B,
and pi = p/c. The first-order model is

    pi_t = K z + C^(-1) s(t)
    z_t = -K^T pi
    H = [[0, i K], [-i K^T, 0]]
    w_t = -i H w + (C^(-1) s, 0), with w = (pi, z).

In the absence of forcing this gives p_tt = -C^2 B B^T p: constant-density, lossless
acoustics with zero-pressure ghost cells. These boundaries reflect waves; they are not an
absorbing surrounding medium. Density variation, attenuation and clinical instrument responses
are absent from this reduced model.

`initial_pressure` sets p(source, 0) = A Pa and z(0) = 0. `pressure_rate` drives p_t in Pa/s,
using midpoint integration of piecewise-constant bins. Neither is silently identified with the
medical paper's calibrated three-cycle tone burst. An increment Delta_p at source s yields

    p_r(t) = Delta_p * (c_r/c_s) * Re <r|exp(-i H t)|s>.

SciPy's sparse exponential action computes this discrete evolution to numerical precision.
Finite source quadrature and spatial discretization errors remain. Source and receiver scale
factors are recomputed for each candidate speed map. Pressure is signed, not an energy fraction.

The reduced optimizer minimizes

    L(theta) = ||p(theta)-d_observed||^2 / (2 ||d_observed||^2).

All source/receiver samples contribute before a model update. Each iteration evaluates a base
prediction and two parameter probes per unknown. Differences are central where possible and
one-sided near bounds. The shared update limits parameter displacement, then projects onto speed
bounds. The recorded displacement includes that projection. Convergence is not guaranteed.
The reference has zero sampling variance. The shared Prediction interface also supports a
finite-sampling variance correction for future sampled solvers; it is zero for this baseline.

The Stride implementation instead uses the upstream acoustic PDE and adjoint with frequency
stages and gradient processing. It is a separate larger classical baseline, not numerically
identical to this reduced reference or to the original paper's commercial Fullwave3D solver.

Later validation must cover independent second-order pressure evolution, normalization, source
quadrature, acquisition consistency, gradient updates and convergence. Synthetic same-model
observations verify implementation consistency but cannot establish robustness to real data.
