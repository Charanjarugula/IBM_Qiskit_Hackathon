# Scope and authorization

This is the classical stage of a sequential submission: classical PR, then simulator PR, then
IBM hardware PR. Each must be approved by the user before merging. Create subsequent branches
from the updated main. Do not publish simulator/hardware source in the classical PR.

Prepare and inspect source only. Do not run simulations, tests, imports, notebooks, compilation
or dependency installations until explicitly authorized. No hardware jobs are authorized.
There are no results or clinical datasets in this repository. Preserve that distinction in claims.

# Architecture

- `classical/stride_baseline.py` delegates 2D/3D propagation and adjoint FWI to pinned Stride.
  Its boundary/source conventions differ from the small reference. It has no AWI setup.
- `classical/reference.py` uses sparse exact propagation of a reduced constant-density pressure
  equation. It is the intended small matched baseline for later quantum comparisons.
- `shared/` defines acquisition, physical pressure, speed parameters, bounded finite-difference
  optimization and evidence capture. Read METHODS.md before changing the equations.
- Observations are generated once and shared across compared solvers. No ground-truth speed
  parameters enter the optimizer. Acquisition/model fingerprints must match for comparisons.
- `configs/` contains synthetic specifications, not observed results. The head generator uses a
  layered phantom with a known-skull starting model. It is not the paper's MIDA reconstruction.

# Invariants and later validation

Preserve signed pressure and model-dependent source/receiver scaling. Both source conventions
(initial pressure and midpoint-integrated pressure rate) must remain explicit. Do not substitute
energy probabilities for pressure. Keep Stride and reduced-reference datasets distinct.

Keep output directories unique, save configuration/source/version provenance, and retain failures.
Do not claim accuracy or runtime from an unexecuted test. The test suite includes independent
second-order pressure checks and optimizer regression cases; it remains unrun.

Retain the AGPL license and Anik MIT attribution. Upstream example scripts are reference material,
not project entry points; they may download data and use upstream filesystem paths.
