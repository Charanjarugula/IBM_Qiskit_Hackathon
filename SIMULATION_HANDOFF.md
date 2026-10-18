# Overnight simulation handoff — give this file to Claude or Codex

Your task is to run the prepared experiments on the partner's machine, preserve
the evidence, and return an archive for review before any IBM hardware submission.
Proceed autonomously with the commands below. Actual numerical correctness and
runtime have not yet been validated by executing this campaign.

Repository: https://github.com/Charanjarugula/IBM_Qiskit_Hackathon

Run the code at commit `39e39569ac529e8759a71a5b55d6b7610f7beff5` (merged PR #6).
This handoff adds instructions only; it does not change that implementation.

## Scope and priorities

1. **First: IBM pilot reference and matched forward comparisons.** This is the
   highest priority result because the hardware experiment needs its verified
   reference. Use `ibm_hardware/pilot_1d.json` only as a configuration file; execute
   the `simulator` and `annealing` modules below, never an IBM execution module.
2. **Then: 1D reconstruction.** Compare the reduced classical solver and ideal /
   noisy gate simulations using exactly the same observations and optimizer.
3. **Then: longer 1D and bipolar-pulse forward comparisons.** Compare exact
   classical pressure, gate simulation without / with noise, simulated annealing
   (SA), and simulated quantum annealing (SQA).
4. **Finally: bounded 4×4 forward comparisons.** Attempt ideal / noisy gates and
   SA / SQA at the same fixed map. This is a stretch experiment; retain timeouts.

Do not run IBM jobs, use IBM credentials, install IBM account settings, or spend
QPU minutes. Do not run the full Stride / paper-scale classical reconstruction or
the 4×4 gate **inversion**. The small classical references below are necessary for
a fair comparison and are separate from that paper-scale benchmark. Do not run
every Python file in the repository.

Use a seven-hour overall deadline unless the partner provides an earlier one.
Reserve the final 15 minutes for collecting results. Hard time limits below are
budgets, not runtime estimates. No promise that every stretch run will finish.

## Machine and GPU policy

Three A100s are available, but the existing `GateForward` explicitly uses Aer’s
default **CPU** device. Its registers are only four qubits for the 1D cases and
seven for 4×4, including the measurement ancilla. Circuit construction and
transpilation also use CPUs. Do not assume three GPUs give a threefold speedup.

Start the pilot on the existing CPU implementation immediately. Run independent
tasks concurrently using available CPU cores, with at most three top-level jobs;
each annealing job below uses two CPU workers. Use `tmux`, a scheduler allocation,
or another process supervisor that survives disconnects. On a cluster, launch
inside an allocation using the partner's existing scheduler settings. Do not
invent partition names or reserve paid resources.

GPU acceleration is optional. Spend at most 15 minutes on it while the pilot
continues. If enabling it, make a separate local checkout / virtual environment:

- Add an optional `device` argument (`CPU` default, `GPU` explicit) to
  `simulator/forward.py`, and pass it through `simulator/benchmark.py` and
  `simulator/run.py`. Supply it to `AerSimulator`; check `available_devices()` and
  fail clearly if GPU was requested but is unavailable.
- Pin each independent process to one allocated GPU with `CUDA_VISIBLE_DEVICES`.
  Preserve scheduler-assigned visibility; do not access unallocated GPUs. There
  is no need to distribute one tiny state across all three A100s.
- Record the requested / actual Aer device, package versions, GPU model and
  local patch with results. Validate a small CPU / GPU run against the same
  reference and shot uncertainty before using the patch for the campaign.
- PyPI checked during preparation: `qiskit-aer-gpu-cu11==0.17.2` provides a
  Python 3.12 Linux x86_64 wheel, whereas `qiskit-aer-gpu==0.17.2` was unavailable.
  A separate GPU environment can install the former **instead of** CPU
  `qiskit-aer`, with the existing Qiskit / NumPy / SciPy pins and a compatible
  NVIDIA driver. Do not install two Aer distributions into the same environment,
  silently downgrade to an older Aer, or change the numerical model for speed.
- SA / SQA and the current noiseless reference verifier use CPUs. A GPU Aer
  installation does not accelerate those modules.

If GPU setup fails or costs more time than it saves, continue on CPUs and state
that clearly. Do not let GPU setup block the pilot or reduce evidence quality.

Primary references: [Aer installation](https://qiskit.github.io/qiskit-aer/getting_started.html),
[Aer device options](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html),
[versioned GPU wheel](https://pypi.org/project/qiskit-aer-gpu-cu11/0.17.2/).

## Setup: existing implementation, no source changes required

Use a fresh clone. Preserve any older runs; do not overwrite their output folders.
If an old multi-day noisy 2D inversion is still running, identify it and stop that
specific process before starting this campaign. Do not kill unrelated processes.

```bash
git clone https://github.com/Charanjarugula/IBM_Qiskit_Hackathon.git IBM_Qiskit_overnight
cd IBM_Qiskit_overnight
git checkout --detach 39e39569ac529e8759a71a5b55d6b7610f7beff5
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r simulator/requirements.txt -r annealing/requirements.txt
python -m pip check

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export QISKIT_PARALLEL=FALSE
export PYTHONUNBUFFERED=1
export OUT="$PWD/runs/overnight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT/logs"
git rev-parse HEAD > "$OUT/commit.txt"
git status --short > "$OUT/worktree_status.txt"
python -m pip freeze > "$OUT/packages.txt"
python -V > "$OUT/python.txt" 2>&1
uname -a > "$OUT/machine.txt"
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || true
```

Copy this handoff into `$OUT` as well. Record allocated CPU cores and RAM. Never
dump environment variables, API keys, or credentials into the result bundle.

Use GNU `timeout` on Linux to enforce a hard limit around **every** long command.
The scripts' `--max-seconds` checks happen between operations; they cannot
interrupt a single slow transpilation or solver call. Example helper:

```bash
run_job() {
  local label="$1" limit="$2" code
  shift 2
  printf '%s START %s\n' "$(date -u +%FT%TZ)" "$label"
  if timeout --signal=TERM --kill-after=20s "$limit" "$@" \
      > "$OUT/logs/$label.log" 2>&1; then
    code=0
  else
    code=$?
  fi
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" "$code" \
    >> "$OUT/job_status.tsv"
  printf '%s END %s exit=%s\n' "$(date -u +%FT%TZ)" "$label" "$code"
  return "$code"
}

run_job annealing_verify 120s python -m annealing.verify
```

Check that the verification succeeds before starting SA / SQA. A failure must
not block independent gate work. Do not use a global `set -e` that abandons the
entire campaign after one timeout. Check each prerequisite explicitly. Exit 124
normally indicates a timeout; record any forced termination too. Save partial
files, failures, commands and return codes. If resuming, skip only verified
completed tasks and use new output paths for retries.

## Matched forward experiments: reusable commands

For each case set the variables from the table, then run the commands below.
Keep each case's variables and dependencies together in a separate shell/task;
do not share mutable shell variables between simultaneous cases.

| Case | CONFIG | PARAMS (Bash array) | Reference hard limit | Gate hard / soft seconds | Annealing hard / per-repeat seconds |
|---|---|---|---|---|---|
| `pilot` — first | `ibm_hardware/pilot_1d.json` | `(1 1.1)` | `600s` | `1200s` / `1100` | `2100s` / `600` |
| `small_1d` | `classical/configs/small_1d.json` | `(1 1.1)` | `1200s` | `1800s` / `1700` | `2100s` / `600` |
| `pulse_1d` | `classical/configs/pulse_1d.json` | `(1 1.1)` | `1800s` | `2400s` / `2300` | `3000s` / `900` |
| `matched_2d` — stretch | `classical/configs/matched_2d.json` | `(1.03 1.12 1.03 1.03)` | `3600s` | `7200s` / `7000` | `6000s` / `1800` |

First case, ready to paste:

```bash
CASE=pilot
CONFIG=ibm_hardware/pilot_1d.json
PARAMS=(1 1.1)
REF_LIMIT=600s
GATE_LIMIT=1200s
GATE_SECONDS=1100
ANNEAL_LIMIT=2100s
ANNEAL_SECONDS=600
GATE_CAP=500000
mkdir -p "$OUT/$CASE"

run_job "${CASE}_reference" "$REF_LIMIT" python -m simulator.reference \
  --config "$CONFIG" --parameters "${PARAMS[@]}" \
  --output "$OUT/$CASE/reference.json" --execute
```

**Check success and both `verified` and `readout_verified` in `reference.json`
before proceeding with that case.** These checks cover signed-pressure readout,
controlled-circuit equivalence and a ≤1% Trotter-error target with half-step
checking. Never bypass a failed / stale reference. Send the verified pilot
reference back as soon as available; do not wait for the stretch experiments.

After a case's reference passes, these three tasks can run independently, subject
to the three-job overall concurrency limit:

```bash
run_job "${CASE}_gate_ideal" "$GATE_LIMIT" python -m simulator.benchmark \
  --config "$CONFIG" --parameters "${PARAMS[@]}" \
  --reference "$OUT/$CASE/reference.json" --output "$OUT/$CASE/gate_ideal" \
  --repetitions 3 --shots 8192 --seed 7 --threads 1 \
  --max-circuits 192 --max-total-shots 2000000 --max-gates "$GATE_CAP" \
  --max-seconds "$GATE_SECONDS" --execute

run_job "${CASE}_gate_noisy" "$GATE_LIMIT" python -m simulator.benchmark \
  --config "$CONFIG" --parameters "${PARAMS[@]}" \
  --reference "$OUT/$CASE/reference.json" --output "$OUT/$CASE/gate_noisy" \
  --repetitions 3 --shots 8192 --seed 7 --threads 1 \
  --noise-probability 0.001 --method density_matrix \
  --max-circuits 192 --max-total-shots 2000000 --max-gates "$GATE_CAP" \
  --max-seconds "$GATE_SECONDS" --execute

run_job "${CASE}_annealing" "$ANNEAL_LIMIT" python -m annealing.run \
  --config "$CONFIG" --parameters "${PARAMS[@]}" \
  --gate-reference "$OUT/$CASE/reference.json" --output "$OUT/$CASE/annealing" \
  --algorithms sa sqa --repetitions 3 --workers 2 --seed 7 \
  --bits 6 --refinements 3 --reads 16 --sweeps 200 \
  --max-variables 512 --max-bqm-calls 2000 \
  --max-seconds "$ANNEAL_SECONDS" --execute
```

For the other three cases, use the table's values and new case directory. Set
`GATE_CAP=2000000` only for `matched_2d`; retain `500000` for all 1D cases. This
raises a simulator resource ceiling, not circuit accuracy or a hardware limit.
The 4×4 case has 48 wave circuits per forward pass, so three repeats fit the
192-circuit / two-million-shot budgets. Do not start a long stretch task if it
would consume the remaining time needed for core results and collection.

Each completed gate benchmark already includes separately timed, matched exact
classical forward calculations. Annealing also records an exact classical and a
direct Crank–Nicolson reference. Do not substitute a different grid or the paper's
36-hour runtime into these numerical comparisons.

After all three tasks for a case finish, collect the comparison:

```bash
run_job "${CASE}_comparison" 120s python -m simulator.compare \
  --gate-runs "$OUT/$CASE/gate_ideal" "$OUT/$CASE/gate_noisy" \
  --annealing-runs "$OUT/$CASE/annealing" \
  --output "$OUT/$CASE/comparison"
```

If annealing failed, also generate a **separate** gate-only comparison with a new
output path and explicitly report annealing as failed / incomplete. If one gate
run failed, collect the other on its own. Preserve the failed run and original
logs; do not present the successful subset as the whole experiment. Do not pass
inversion outputs to `simulator.compare`.

## 1D reconstruction: the complete optimization loop

Use `reconstruction_1d.json` with 12 updates. The two-update configurations are
smoke tests, not evidence of convergence. Run this after pilot gate results are
available; it can overlap the other forward cases within the job limit.

```bash
CONFIG=classical/configs/reconstruction_1d.json
mkdir -p "$OUT/reconstruction_1d"
run_job reconstruction_data 120s python -m classical.make_observations \
  --config "$CONFIG" --output "$OUT/reconstruction_1d/data" --execute

# Start these only if observation generation completed successfully.
run_job reconstruction_classical 300s python -m classical.run \
  --config "$CONFIG" \
  --observations "$OUT/reconstruction_1d/data/observations.npz" \
  --output "$OUT/reconstruction_1d/classical" --execute

# Six independent runs. Schedule up to three concurrently; each has its own path.
for SEED in 7 17 27; do
  for CONDITION in ideal noisy; do
    NOISE=0
    METHOD=statevector
    if [ "$CONDITION" = noisy ]; then
      NOISE=0.001
      METHOD=density_matrix
    fi
    run_job "reconstruction_${CONDITION}_${SEED}" 2700s python -m simulator.run \
      --config "$CONFIG" \
      --observations "$OUT/reconstruction_1d/data/observations.npz" \
      --output "$OUT/reconstruction_1d/${CONDITION}_${SEED}" \
      --shots 8192 --seed "$SEED" --threads 1 \
      --noise-probability "$NOISE" --method "$METHOD" \
      --max-circuits 500 --max-total-shots 4100000 \
      --max-gates 500000 --max-seconds 2600 --execute
  done
done
```

The loop above is serial as pasted; the agent may schedule its six commands
independently. Do not launch all six alongside the other tasks without enforcing
the overall concurrency limit. Each reconstruction plans 488 circuits and
3,997,696 shots. Completion does not imply convergence or improved accuracy.

Write a small **offline results table** from the saved reconstruction outputs:

- Require matching `inverse_problem_signature`, `observations_sha256` and
  objective in `comparison.json` across classical and gate runs.
- Report completed iterations, initial / final parameters, parameter RMSE
  against the saved synthetic truth, and initial / final pressure loss.
- Include all three seeds, their spread, failed runs and whether the final
  parameter error improved. Do not select only the best seed.
- Use the saved `history.json` values; shot-variance-corrected loss can be
  negative. Do not clip or describe it as an exact deterministic residual.
- Record wall time, circuit / shot counts, noise probability and device. A plot
  of loss and parameter error versus iteration is useful if time remains.

Truth belongs in data generation and post-run scoring only, never optimizer
initialization, tuning or gradients. Do not change optimizer settings to force
the desired answer after looking at these results.

## Quality checks and interpretation

- Keep all raw counts, seed settings, configurations, source snapshots, logs,
  errors and partial iterations. A `completed.json` means execution finished;
  inspect accuracy flags and errors separately.
- Gate pressure estimates are signed amplitudes obtained from ancilla counts,
  not probabilities interpreted directly as pressure. `simulator.compare`
  checks the saved predictions against the raw counts.
- The 1% comparison target is a reporting threshold. Finite-shot estimates,
  generic noise or annealing can miss it. Preserve and explain those failures;
  never relax verification thresholds or silently replace output with exact
  answers. Do not label the hardware pilot ready solely because a process exited
  successfully; return its reference, counts, errors and uncertainties for review.
- `0.001` is a generic depolarizing stress model, not calibrated IBM device noise
  or a measured IBM hardware error rate. This campaign does not measure hardware
  error mitigation performance.
- SA is a classical control. SQA is CPU path-integral simulation of a transverse-
  field annealing model, not a D-Wave / IBM hardware run. Annealing here solves
  forward-wave QUBOs; it does not implement the complete reconstruction loop.
- These are reduced synthetic same-model tests, not real patient data, an
  anatomical head reconstruction, or a reproduction of the paper's 36-hour run.
- Simulator wall times are not QPU timings or evidence of quantum speedup.
  Concurrent tasks also contend for resources; label timings accordingly.

Fix only an actual setup / API / bookkeeping defect if it blocks progress. Save
the exact local diff, error and retry path. Preserve the equations, acquisition,
seeds, noise, budgets and acceptance tests. Do not push, merge or change GitHub
protection from the partner's machine. If any reference-hashed source changes,
regenerate the reference and dependent runs; never mix old and new evidence.

## Return these files

Send back the pilot's verified `reference.json` as soon as it exists. At the end,
return one archive containing the entire `$OUT` directory, including:

1. Pilot, 1D, pulse and attempted 4×4 references and every raw run directory.
2. Per-case comparison JSON / CSV, reconstruction table and any figures.
3. Logs, return codes, timeout records, source snapshots and environment metadata.
4. `STATUS.md`: completed / failed / timed-out / skipped tasks, numerical accuracy
   findings, whether GPU acceleration was actually used, and unresolved issues.
5. Any local source patch; no credentials, tokens or `.venv` directory.

Stop remaining campaign processes at the deadline, preserving their outputs.
Wait for those specific processes to stop writing before archiving. Keep the
uncompressed directory until the recipient confirms the archive is usable.

```bash
git diff > "$OUT/local_changes.patch"
git status --short > "$OUT/worktree_status_end.txt"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
sha256sum "$OUT.tar.gz" > "$OUT.tar.gz.sha256"
tar -tzf "$OUT.tar.gz" > "$OUT.archive_listing.txt"
```

Provide the archive, checksum and a short status message. Do not claim that
everything is correct, faster, or ready for hardware unless the numerical
evidence supports that conclusion. IBM submission remains a separate decision.
