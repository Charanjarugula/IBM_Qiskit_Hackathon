# Aperion

Florida Qiskit Fallfest Hackathon

**Team Aperion:** Charan Jarugula, Anik Sahai, Aaron Loh

## About

Aperion explores how quantum computing could help with ultrasound imaging of the brain. The challenge is turning sound recordings into a useful image, especially when the waves have to travel through the skull. Doing this requires many wave simulations, which can make reconstruction expensive to compute.

The imaging approach uses ultrasound sensors called transducers around the head. One sends a sound pulse while the others record the waves that reach them. Different tissues change how quickly sound travels and how the waves bend and reflect. Those differences give us information about what is inside.

Reconstructing an image is a guess-and-check process. We start with a guess of the speed of sound at each location, simulate a pulse, and predict what the receiving sensors should record. We compare those predictions with the observed signals and repeat this for the other transmitters. Then we adjust the guess to reduce the mismatch and go through the process again. The aim is to build a sound-speed map that helps reveal the tissue structure.

**The classical approach:** A conventional computer uses the wave equation to calculate how each pulse travels through the guessed model. It also compares the signals and updates the sound-speed map. Every update requires more wave simulations.

**Our quantum approach:** We explore replacing the wave simulation step with a quantum circuit. The circuit represents the wave's evolution, and quantum measurements let us estimate signals at selected receivers. A classical computer still compares the signals and updates the map. We tested parts of this approach on small synthetic examples using Qiskit simulators and real IBM quantum hardware, including techniques to reduce hardware error. We also compared classical simulated annealing with simulated quantum annealing as another way to solve the wave model.

## Repository structure

| Folder | What it contains |
| --- | --- |
| [`classical/`](classical) | Classical wave simulation, sound-speed reconstruction and the Stride baseline setup. |
| [`simulator/`](simulator) | Gate-based quantum simulations in Qiskit Aer, with and without artificial noise. |
| [`ibm_hardware/`](ibm_hardware) | Code to prepare circuits, run them on IBM quantum hardware and collect measurements. |
| [`annealing/`](annealing) | Classical simulated annealing and simulated quantum annealing for the wave model. |
| [`results/`](results) | Experiment data, hardware job records, figures and scripts to verify the results. |

## How to run

Use Python 3.11 or 3.12. The small examples run on a CPU and do not need a GPU or an IBM account. Run the commands below from the repository root.

### Setup

```bash
git clone https://github.com/Charanjarugula/IBM_Qiskit_Hackathon.git
cd IBM_Qiskit_Hackathon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

### Check the submitted results

```bash
python results/reproduce.py
```

This checks the saved data and regenerates the figures in `results/figures/`. It does not run new simulations or submit hardware jobs. To check the evidence without redrawing figures, use `python results/reproduce.py --verify-only`.

If you only want to review the saved results, installing `results/requirements.txt` instead of the root requirements file is enough.

### Run a new simulator comparison

This runs the same small forward problem with a classical reference, ideal and noisy gate simulation, classical simulated annealing, and simulated quantum annealing. Each gate benchmark also records classical reference timings. All of these commands run locally.

```bash
python -m simulator.reference --config ibm_hardware/pilot_1d.json \
  --parameters 1 1.1 --output runs/demo/reference.json --execute

python -m simulator.benchmark --config ibm_hardware/pilot_1d.json \
  --parameters 1 1.1 --reference runs/demo/reference.json \
  --output runs/demo/gate_ideal --shots 8192 --repetitions 3 \
  --method statevector --threads 1 --max-seconds 900 --execute

python -m simulator.benchmark --config ibm_hardware/pilot_1d.json \
  --parameters 1 1.1 --reference runs/demo/reference.json \
  --output runs/demo/gate_noisy --shots 8192 --repetitions 3 \
  --noise-probability 0.001 --method density_matrix --threads 1 \
  --max-seconds 900 --execute

python -m annealing.run --config ibm_hardware/pilot_1d.json \
  --parameters 1 1.1 --gate-reference runs/demo/reference.json \
  --output runs/demo/annealing --algorithms sa sqa --repetitions 3 \
  --workers 2 --bits 6 --refinements 3 --reads 16 --sweeps 200 --execute

python -m simulator.compare \
  --gate-runs runs/demo/gate_ideal runs/demo/gate_noisy \
  --annealing-runs runs/demo/annealing --output runs/demo/comparison
```

Open `runs/demo/comparison/comparison.json` or `comparison.csv` for the errors and timings. Finishing a run does not mean it met the accuracy target; the report records both. These commands use shell line continuations for macOS/Linux. On PowerShell, enter each command on one line.

Use a new output path when repeating a run. Existing run folders are kept so their evidence is not overwritten.

<details>
<summary>Run a small classical and quantum reconstruction</summary>

This example generates one set of synthetic observations and uses it for both methods. It performs two map updates, not a full brain reconstruction.

```bash
python -m classical.make_observations --config classical/configs/small_1d.json \
  --output runs/reconstruction/observations --execute

python -m classical.run --config classical/configs/small_1d.json \
  --observations runs/reconstruction/observations/observations.npz \
  --output runs/reconstruction/classical --execute

python -m simulator.run --config classical/configs/small_1d.json \
  --observations runs/reconstruction/observations/observations.npz \
  --output runs/reconstruction/gate_ideal --method statevector --execute

python -m simulator.run --config classical/configs/small_1d.json \
  --observations runs/reconstruction/observations/observations.npz \
  --output runs/reconstruction/gate_noisy --method density_matrix \
  --noise-probability 0.001 --execute
```

Each method saves its map in `reconstruction.json` and its updates in `history.json`. The separate Stride adapter has its own dependencies in `classical/requirements-stride.txt`; it is not required for these small examples.

</details>

<details>
<summary>Prepare IBM circuits without submitting a job</summary>

This compiles against a bundled Kingston calibration snapshot. It does not need credentials, contact IBM, or spend QPU time. A snapshot is not a current live calibration.

```bash
python -m ibm_hardware.preflight --config ibm_hardware/pilot_1d.json \
  --backend ibm_kingston --snapshot --parameters 1 1.1 \
  --output runs/hardware_preview --execute
```

Live preparation and execution use `python -m ibm_hardware.run --help`. They require your own saved IBM SDK account, an explicit instance CRN and an accessible backend. Pilot execution also requires a matching verified reference. Actual submission requires the separate `--submit-hardware` flag.

The archived paired gate-twirling experiment has its [original submission script](results/sources/pressure_paired_hardware/submission_script.py) and [saved design](results/sources/pressure_paired_hardware/design.json). That script contains the original local paths and is retained as evidence. The general hardware CLI does not automatically rerun that paired campaign.

</details>

## License

Original Aperion code is licensed under the [MIT License](LICENSE). The Stride-derived [classical adapter](classical/stride_baseline.py), including its archived copies, remains under [AGPL-3.0](classical/licenses/Stride-AGPL-3.0.txt). Third-party code retains its existing copyright and license notices.
