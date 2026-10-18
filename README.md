# Aperion

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
