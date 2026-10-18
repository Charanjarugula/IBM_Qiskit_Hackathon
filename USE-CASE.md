# Aperion: ultrasound brain imaging

## The problem

Our long-term motivation is brain imaging that could be easier to bring to the patient. The work by [Guasch et al. (2020)](https://doi.org/10.1038/s41746-020-0240-8) explores using ultrasound and full-waveform inversion to reconstruct brain structure through the skull. That is the research direction behind Aperion, not a clinical capability we have built.

An ultrasound array sends pulses from different positions and records the signals that arrive at the other sensors. Reconstruction starts with a guessed sound-speed map and repeatedly adjusts it until simulated signals better match the recordings. The skull makes wave propagation complicated, and each update requires more simulations. Researchers developing ultrasound tomography are the immediate users of the software we are exploring. A future imaging system could benefit clinicians and patients if its performance were independently established.

## The classical baseline

Classical full-waveform inversion solves the wave equation and updates the model from the signal mismatch. We include a Stride adapter for that research workflow. Our completed comparisons use much smaller exact classical wave references and synthetic observations. We have not rerun the paper's full 3D reconstruction, so its reported runtime is not a matched benchmark for our quantum tests.

## Our quantum idea

We explore using a quantum circuit to evolve the wave while keeping the map update classical. Quantum measurements estimate selected receiver signals. This is a possible fit because the reduced wave equations can be represented as quantum evolution. A future advantage would depend on building those circuits efficiently and obtaining the needed signals with few enough measurements. Representing a large state in qubits alone does not guarantee a faster answer.

We implemented gate-based simulations, small IBM hardware experiments and a separate simulated-annealing comparison. We compare pressure or energy errors against references, retain raw measurements, and record runtime and QPU usage. The strongest hardware result is reduced pressure error with gate twirling across three paired comparisons. See the [results](results/README.md) for the measured values and conditions.

## What would make it useful

The next steps are a stable reconstruction over more updates, larger matched problems, realistic ultrasound data, and comparison against an optimized classical solver at the same accuracy. Any clinical application would also need a physical acquisition system and independent safety and diagnostic validation. The current work is a small computational experiment, not a medical device or an established speed improvement.
