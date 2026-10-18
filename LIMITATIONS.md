# Limitations

- **Small synthetic problems.** Our completed pressure comparisons use a two-cell model. The observations come from the same discrete model used by the reference. We did not collect patient measurements or build an ultrasound sensor system.
- **No completed paper-scale reconstruction.** The small inverse tests ran two optimizer updates. The larger 2D gate run stopped before producing its first forward result. Stride 2D inversion diverged and was stopped; the 3D run was also stopped.
- **Different observables.** The 15-qubit wave-energy experiment and four-active-qubit signed-pressure experiment use different encodings and readouts. An energy probability is not a signed pressure recording.
- **Accuracy targets were missed.** The ideal simulator's mean trace had less than 1% error, but its individual repeats did not meet the every-repeat 1% target. The noisy gate, SA and SQA runs also missed that target.
- **Hardware improvement has a limited scope.** Gate twirling reduced mean pressure RMSE by 77.76% across three pairs on one backend. Residual error and control drift remain. Readout correction worsened every pressure job; ZNE did not improve the wave-energy result. We did not use quantum error correction.
- **No demonstrated quantum speedup.** Simulator wall times measure classical software running quantum models. The earlier speed projection does not establish equal achieved accuracy, and its fixed shot count was not validated for its claimed output. It is retained only as supporting source material.
- **SQA is simulated.** Our simulated quantum annealing uses a classical CPU implementation. We did not run an annealing processor.

The [results guide](results/README.md) gives the settings, error definitions and source files for each experiment. The [status file](results/status.json) records completed and stopped runs.
