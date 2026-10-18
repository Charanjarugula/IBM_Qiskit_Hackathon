# Aperion

**Quantum wave simulation for ultrasound imaging — IBM Qiskit Fall Fest hackathon.**

We investigate the wave-propagation step inside ultrasound full-waveform inversion (FWI): predict receiver signals for a proposed sound-speed map, compare them with observations, then update the map classically. The clinical motivation is [Guasch et al., *Full-waveform inversion imaging of the human brain* (2020)](https://doi.org/10.1038/s41746-020-0240-8).

This submission contains reduced synthetic experiments: classical references, gate-based simulation, real IBM gate hardware, and simulated quantum annealing. **It does not reproduce the paper's full 3D reconstruction or demonstrate quantum speedup.**

## Results

| Experiment | Observed result | Evidence |
| --- | --- | --- |
| Signed pressure on IBM Kingston | Gate twirling reduced mean RMSE **0.664 → 0.148 Pa (77.76%)**, improving all three paired comparisons. | Six jobs, 4,096 shots/circuit, identical saved circuits; [raw records](results/sources/pressure_paired_hardware), [figure](results/figures/pressure_repeats.png). |
| Wave energy on IBM Marrakesh | Post-selection reduced absolute error **74.95%** for the healthy eight-step condition in one job. Additional ZNE did not improve it. | Original 15-qubit job, six circuits × 4,000 shots; [counts and metadata](results/sources/handoff/3_ibm_job). |
| Ideal / noisy gate simulation | Error of the mean pressure trace: **0.583% / 8.854%**. Individual ideal repeats were 1.076–1.565%; the every-repeat 1% target was missed. | Three repeats per condition, 8,192 shots/circuit; [archive](results/sources/handoff/pilot_results.tar.gz). |
| SA / simulated quantum annealing | Mean-trace errors **1.576% / 1.549%**; SQA took **10.08×** the CPU wall time of SA in this test. | Same reduced forward task and annealing budgets, three repeats; [figure](results/figures/annealing_comparison.png). |
| Small inverse problems | Classical, ideal and noisy gate runs completed **two optimizer updates** on matched synthetic observations. | [Full run archive](results/sources/handoff/reconstruction_runs.tar.gz); convergence was not established. |

![Signed-pressure experiment on IBM Kingston](results/figures/pressure_repeats.png)

The hardware plot compares the same eight pressure observations in each job. Error bars on the left show between-job standard deviation; lines on the right connect matched pairs. Three pairs on one backend establish improvement in this small test, not general superiority. Readout correction worsened all six comparison jobs and is retained in the records.

**[All figures, captions, job IDs and limitations →](results/README.md)**

## Code

| Folder | Purpose |
| --- | --- |
| [`classical/`](classical) | Shared acoustic model, exact reduced reference, classical reconstruction, and a Stride baseline adapter. |
| [`simulator/`](simulator) | Qiskit Aer gate simulation with ideal or generic depolarizing noise; ancilla readout estimates signed pressure. |
| [`ibm_hardware/`](ibm_hardware) | IBM Runtime compilation, resource checks, execution and collection; raw counts are retained. |
| [`annealing/`](annealing) | Crank–Nicolson time steps encoded as QUBOs; classical SA and CPU path-integral SQA. No annealing hardware was used. |
| [`results/`](results) | Published evidence, source snapshots, figures and offline verification. |

The 15-qubit wave-energy demonstration and the four-active-qubit signed-pressure pipeline use different encodings and observables. Only the pressure pipeline is used in our small inverse problems. Its optimizer uses classical finite-difference gradients; this is not the paper's full 3D adjoint inversion.

## Verify the submission

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r results/requirements.txt
python results/reproduce.py
```

This verifies the saved data and regenerates the seven PNG/PDF/SVG figures. **It does not run simulations, connect to IBM, or submit jobs.** For verification alone, install NumPy and run `python results/reproduce.py --verify-only`.

The audit checks input checksums, **578 recorded source hashes**, **576 simulator count files**, all **nine IBM jobs**, and pressure statistics reconstructed from raw counts. The machine-readable result is [`results/verification.json`](results/verification.json). Source snapshots and package versions are saved with the runs; reconstruction runs used an archived execution patch, so current `main` is not presented as their exact source version.

For new experiments, each implementation has its own `requirements.txt` and module CLI. Start with `python -m simulator.benchmark --help`, `python -m annealing.run --help`, or `python -m ibm_hardware.run --help` after installing the corresponding dependencies. Keep new outputs separate from the published evidence.

## Scope and provenance

- All tested data are synthetic; there are no patient measurements or clinical validation.
- Larger matched 2D gate runs did not finish. Stride 2D inversion diverged and was stopped; 3D work was stopped. These are not counted as successful reconstructions.
- Recorded simulator timings are in the [runtime figure](results/figures/runtime_comparison.png). Compiler duration estimates and a fixed-shot scaling projection do **not** establish an accuracy-matched hardware speedup.
- The signed-pressure campaign used **111 charged QPU seconds** (two pilots plus six paired jobs). The earlier Marrakesh job used **9 seconds** separately.
- Simulator archives and wave-energy evidence were supplied through [Anik's repository](https://github.com/AnikS22/IBM_Qiskit_Fallfest/tree/a42140befaddeb18fd6296c829a39e045e6fd573). This repo adds the matched workflow, signed-pressure hardware campaign, verification and figures. Exact sources and public-copy exclusions are recorded in [`results/provenance.json`](results/provenance.json).

## Research and software

- [Guasch et al. (2020)](https://doi.org/10.1038/s41746-020-0240-8): brain FWI motivation and paper-scale classical baseline; not a benchmark we completed.
- [Costa, Jordan & Ostrander (2019)](https://arxiv.org/abs/1711.05394): quantum wave-equation formulation.
- [Schade et al. (2024)](https://arxiv.org/abs/2312.14747): gate-based wave propagation and its scaling assumptions.
- [Stride](https://github.com/trustimaging/stride): classical ultrasound software used by the baseline adapter.
- [Qiskit Aer](https://github.com/Qiskit/qiskit-aer), [IBM Runtime](https://github.com/Qiskit/qiskit-ibm-runtime), and [D-Wave samplers](https://github.com/dwavesystems/dwave-samplers): simulation, gate hardware execution, and SA/SQA implementations.

Third-party notices are retained in [`classical/licenses/`](classical/licenses) and beside the archived speed source.
