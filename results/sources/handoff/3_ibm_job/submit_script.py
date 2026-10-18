# Runs notebook 03's RUN_ON_IBM cell verbatim. Usage: run_hw.py <project_dir> [--dry-run]
import sys, json, pathlib, getpass
root = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(root))
DRY = "--dry-run" in sys.argv
from qwave.hardware import build_circuits, run, analyse_counts
from qwave.resources import ibm_qpu_seconds

RX = 2
circs, meta = build_circuits(t=1.5, steps_list=(2, 4, 6, 8), folds=(1, 3))
keep = [i for i, m in enumerate(meta) if (not m["bleed"] and (m["fold"] == 1 or m["steps"] == 8)) or (m["bleed"] and m["steps"] == 8 and m["fold"] == 1)]
circs = [circs[i] for i in keep]; meta = [meta[i] for i in keep]
shots = 300 if DRY else 4000
print(f"{len(circs)} circuits x {shots} shots ≈ {ibm_qpu_seconds(len(circs) * shots):.0f} s QPU", flush=True)

if DRY:
    counts, _ = run(circs, mode="fake", shots=shots)
    out = root / "results" / "hardware_fake_kingston_DRYRUN.json"
else:
    key = json.load(open(pathlib.Path.home() / "Downloads/apikey.json"))["apikey"]
    getpass.getpass = lambda prompt="": key       # feed run()'s prompt without echoing the key
    counts, _ = run(circs, mode="ibm", shots=shots, dd=True, twirl=True)
    out = root / "results" / "03_hardware_results.json"

hw = [{**m, **analyse_counts(cn, RX, 15)} for m, cn in zip(meta, counts)]
json.dump(hw, open(out, "w"), indent=1, default=float)
for r in hw: print(r)
print("saved", out)
