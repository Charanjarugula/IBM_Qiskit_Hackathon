# Sources and attribution

| Source | Revision / use |
|---|---|
| [Stride](https://github.com/trustimaging/stride) | `8733ecb9401e3889eb3e071fc766e26bc734da21`. Reuse upstream forward, adjoint, loss, optimizer and 2D/3D examples. Four example scripts are preserved unchanged. |
| [Anik's repository](https://github.com/AnikS22/IBM_Qiskit_Fallfest) | Inspected starting at `f43f743021a18731208b79fbf3738a7e28703fa1`. Grid and phantom utilities come from its locally reviewed version, with MIT notice preserved. Sparse evolution is adapted to explicit physical pressure. No archived results are copied. |
| [Guasch et al., 2020](https://doi.org/10.1038/s41746-020-0240-8) | Application and FWI methodology. The full 3D MIDA experiment, variable-density synthetic observations and AWI initialization are not reproduced here. |

Stride already supplies a mature acoustic solver and adjoint, so this project adapts its
orchestration instead of rewriting those components. Anik's JAX FWI implementation is not
duplicated. The small matrix-exponential reference remains because a reduced matched comparison
needs the same physical model as the later gate implementation.

The medical paper's own synthetic acoustic dataset is published at
[Dryad](https://doi.org/10.5061/dryad.nzs7h44n7). It has not been downloaded or imported here.
An anatomical dataset and simulated acoustic records are distinct from measured patient data.

This distribution includes Stride-derived orchestration/examples under the GNU Affero General
Public License in `LICENSE`. Upstream Stride copyright and license notices remain applicable.
Anik's MIT attribution is in `licenses/Anik-MIT.txt`. No third-party medical dataset is redistributed.
