# Classical ultrasound waveform inversion

Classical source for a staged classical/simulator/hardware comparison. **Not executed or
numerically validated in this repository.** No simulation results or patient data are included.

## Implementations

- `classical/stride_baseline.py` reuses Stride/Devito for 2D/3D wave propagation and adjoint FWI.
- `classical/reference.py` provides exact discrete propagation for small reduced problems, with
  a shared pressure loss and classical finite-difference optimizer for later matched comparisons.

The two baselines have different discretizations, boundaries, source conventions and optimizers.
Do not compare the full Stride runtime with a tiny reduced experiment as a speedup result.
Neither is a reproduction of Guasch et al.'s full anatomical experiment. The paper's 32-hour
runtime is not a benchmark measured by this project.

## Data

The configurations describe synthetic speed maps. `make_observations` generates receiver pressure
from a known map using the same discrete model being inverted. This is an inverse-crime software
check, not independent experimental validation. The Stride input generator produces a layered
head phantom with a simulated abnormal region and a known-skull starting model. Anik's phantom
utilities are reused; none of his archived results are imported. No observations have been
generated in this fresh repository.

## Commands for later authorized execution

Use Python 3.11 and install `requirements.txt` in a dedicated environment when authorized.
Run from the repository root:

```bash
python -m classical.make_observations --config configs/small_1d.json --output runs/observations-01 --execute
python -m classical.run --config configs/small_1d.json --observations runs/observations-01/observations.npz --output runs/classical-01 --execute
```

Generate observations once and retain the same file for all compared solvers. Each run requires
a new output directory. `small_1d.json` has two unknown speed parameters; `matched_2d.json` has
four in a 4x4 grid. `pulse_1d.json` adds an explicit pressure-rate source. These are not anatomical
resolution reconstructions. Sound-speed-based tissue labels are illustrative and not diagnostic.

Use a **separate environment** with `classical/requirements-stride.txt` for Stride. Its solver and
adjoint are reused, not reimplemented. On macOS without an OpenMP compiler, use `DEVITO_LANGUAGE=C`:

```bash
python -m classical.make_stride_input --output data/head-input.npz --execute
DEVITO_LANGUAGE=C python -m classical.stride_baseline forward --input data/head-input.npz --config configs/stride.json --output runs/stride-forward-01 --execute
DEVITO_LANGUAGE=C python -m classical.stride_baseline invert --input runs/stride-forward-01/inversion_input.npz --config configs/stride.json --output runs/stride-inverse-01 --execute
```

The generator defaults to 3D; `--dimension 2` selects 2D. A Mac's extra RAM does not provide CUDA.
The preserved `classical/upstream_stride/` scripts are unmodified reference examples, not these
project entry points. Some download data and assume upstream paths.

## Input contracts

Reduced observations: NPZ with `pressure_pa` shaped sources × times × receivers and an
`acquisition_signature` matching the problem. Every run also records the inverse-problem
fingerprint and observation-file hash. No trace-by-trace normalization is applied.

Stride input: NPZ with `vp_m_s` (2D/3D), `spacing_m` (per axis), `dt_s`, `locations_m`
(transducers × dimensions), `source_ids` and `wavelets` (sources × time). Inversion additionally
requires `observed_pressure` shaped sources × receivers × time. Optional fixed `rho_kg_m3`
must match the speed shape. Forward input may include `initial_vp_m_s` for its exported inversion
dataset. Source injection units, time origins, geometry and instrument response must be calibrated
before using experimental measurements. Stride traces are not silently relabelled as reduced data.

The Stride adapter disables per-shot normalization and optional time transforms, retaining
frequency filtering. It has absorbing boundaries, optional supplied density, and no attenuation
or AWI configuration. Original medical data and MIDA anatomy are not bundled.

Read [METHODS.md](METHODS.md) for equations and [SOURCES.md](SOURCES.md) for attribution.
Prepared checks can later be run with `python -m pytest -q`; they have deliberately not been run.
