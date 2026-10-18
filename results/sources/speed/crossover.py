"""Speed crossover: classical solve time vs IBM gate time, both at the SAME accuracy, as the 3D head grid grows.

Query (same for both): energy in every cell after the wave has crossed the head (t = n grid units), within 5% of exact.
  classical: velocity-Verlet finite differences (our fastest solver), largest dt that meets 5%, timed on this Mac.
  quantum:   unary circuit (the one run on ibm_marrakesh), fewest Trotter steps that meet 5% (exact subspace replay),
             x gate + readout time per step on a Heron chip (FakeKingston target durations; no queue/compile/reset).
Measured: accuracy-matched steps for n = 4..12, classical timings to n = 48, quantum gate time at 2^3 and 3^3 (135 qubits).
Beyond that it is a power-law projection; assumptions are written into the JSON.
Run: PYTHONPATH=. venv/bin/python3 scaling/crossover.py
"""
import json, pathlib, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeKingston

from qwave.grid import Grid
from qwave.phantom import head_phantom
from qwave.classical import laplacian, fd_receiver_energy, exact_receiver_energy
from qwave.circuits import unary_circuit
from qwave.encoding import unary_layers, unary_qubits
from qwave.subspace import trotter_receiver_energy, layer_matrices

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
TOL = 0.05
SHOTS = 4000                                  # what the hardware run used per circuit
SHOT_CYCLE_S = 6.616460800 / 24000            # ibm_marrakesh job db9smd9gv2cc73e60pm0: execution time / shots
CYCLE_OVERHEAD_S = SHOT_CYCLE_S - 6.288e-6    # minus that circuit's own 6.3 us gate time = reset/rep delay
HEAD = 440                                    # 22 cm head at 0.5 mm voxels (resources.paper_head)
backend = FakeKingston()


def problem(n):
    c, _ = head_phantom((n, n, n))
    return Grid((n, n, n)), c.ravel()


def ok(p, ex):
    return np.abs(p - ex).max() <= TOL * ex.max()


def matched_steps(n):
    """Fewest classical time steps and fewest Trotter steps that each reach TOL on every cell."""
    g, c = problem(n); t = float(n); R = list(range(g.n_vertices))
    ex = exact_receiver_energy(g, c, [t], 0, R)[0]
    lam = abs(spla.eigsh(laplacian(g, c), k=1, which="LA", return_eigenvectors=False)[0])
    lo = int(np.ceil(t / (1.9 / np.sqrt(lam))))          # stability limit
    fd = lambda s: ok(fd_receiver_energy(g, c, [t], 0, R, dt=t / s)[0], ex)
    mats = layer_matrices(g, c)
    tr = lambda s: ok(trotter_receiver_energy(g, c, t, s, 0, R, mats), ex)
    return bisect(fd, lo), bisect(tr, 1)


def bisect(good, lo):
    hi = lo
    while not good(hi): lo, hi = hi, hi * 2
    while hi - lo > max(1, hi // 50):                     # 2% resolution
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if good(mid) else (mid, hi)
    return hi


def classical_time(n, steps, reps=3):
    g, c = problem(n); t = float(n)
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fd_receiver_energy(g, c, [t], 0, [0], dt=t / steps)
        best = min(best, time.perf_counter() - t0)
    return best


def quantum_step_time(n):
    """Heron gate+readout time per Trotter step, from compiled 1- and 2-step circuits on the real chip layout."""
    g, c = problem(n); layers = unary_layers(g, c); d = []
    for s in (1, 2):
        qc = unary_circuit(g, c, 1.0, s, 0, layers); qc.measure_all()
        d.append(transpile(qc, backend, optimization_level=1, seed_transpiler=7).estimate_duration(backend.target))
    return {"qubits": unary_qubits(g), "per_step_s": d[1] - d[0], "fixed_s": max(d[0] - (d[1] - d[0]), 0.0)}


def powerfit(x, y):
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    return lambda z: np.exp(a) * np.asarray(z, float) ** b, b


if __name__ == "__main__":
    acc = {n: matched_steps(n) for n in (4, 6, 8, 10, 12)}
    for n, (sc, sq) in acc.items(): print(f"n={n}: classical {sc} steps, quantum {sq} Trotter steps", flush=True)
    an = np.array(list(acc)); Sc, sc_exp = powerfit(an, [v[0] for v in acc.values()]); Sq, sq_exp = powerfit(an, [v[1] for v in acc.values()])

    q = {n: quantum_step_time(n) for n in (2, 3)}
    print("quantum gate time", q, flush=True)
    per_step, fixed = q[3]["per_step_s"], q[3]["fixed_s"]

    # classical cost per time step vs voxels (timed at a fixed 200 steps), then x accuracy-matched step count
    sizes = [4, 6, 8, 12, 16, 24, 32, 40, 48]
    per = {n: classical_time(n, 200) / 200 for n in sizes}
    Pc, pc_exp = powerfit(np.array(sizes[3:]) ** 3, [per[n] for n in sizes[3:]])
    meas_cl = {n: classical_time(n, acc[n][0]) for n in acc}                     # fully measured points
    meas_q = {n: SHOTS * (fixed + per_step * Sq(n)) for n in (3,)}

    side = np.logspace(np.log10(3), np.log10(2000), 400)
    t_cl = Pc(side ** 3) * Sc(side)
    t_q = SHOTS * (fixed + per_step * Sq(side))
    t_q_today = t_q + SHOTS * CYCLE_OVERHEAD_S
    xing = lambda tq: float(side[np.argmax(t_cl > tq)]) if np.any(t_cl > tq) else None
    x_gate, x_today = xing(t_q), xing(t_q_today)
    h = lambda f: float(f(np.array([HEAD]))[0])
    head = {"classical_s": h(lambda s: Pc(s ** 3) * Sc(s)), "quantum_gate_s": h(lambda s: SHOTS * (fixed + per_step * Sq(s)))}
    head["quantum_todays_cycle_s"] = head["quantum_gate_s"] + SHOTS * CYCLE_OVERHEAD_S
    head["speedup_gate"] = head["classical_s"] / head["quantum_gate_s"]
    head["speedup_todays_cycle"] = head["classical_s"] / head["quantum_todays_cycle_s"]

    res = {
        "query": f"energy in every cell after the wave crosses the head (t = n), within {TOL:.0%} of exact",
        "accuracy_matched_steps": {str(n): {"classical_time_steps": a, "quantum_trotter_steps": b} for n, (a, b) in acc.items()},
        "step_scaling": {"classical_steps ~ n^": sc_exp, "quantum_trotter_steps ~ n^": sq_exp},
        "classical_cost_per_step ~ voxels^": pc_exp,
        "classical_measured_matched_s": {str(n): v for n, v in meas_cl.items()},
        "quantum_heron_gate_time": {str(n): v for n, v in q.items()},
        "quantum_3x3x3_matched_s_4000_shots": meas_q[3],
        "shots_per_answer": SHOTS, "todays_shot_cycle_s": SHOT_CYCLE_S,
        "crossover_side_gate_time": x_gate, "crossover_voxels_gate_time": x_gate and x_gate ** 3,
        "crossover_side_todays_cycle": x_today, "crossover_voxels_todays_cycle": x_today and x_today ** 3,
        f"full_head_{HEAD}^3": head,
        "assumptions": [
            "beyond 3x3x3 (135 qubits) this is a projection: unary needs ~4 qubits per voxel (~3.4e8 for the full head)",
            "per-step gate time held at the measured 3x3x3 value, i.e. hardware whose connectivity matches the grid",
            "noise-free: today's chips lose most of the signal (raw 0.013 vs exact 0.104 on ibm_marrakesh)",
            "quantum cost counts one 4,000-shot circuit; reading out many receivers/times needs more circuits (readout wall)",
            "classical = our numpy finite-difference solver on one laptop core, not an optimised GPU code"],
    }
    (OUT / "07_crossover.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.loglog(np.array(list(meas_cl)) ** 3, list(meas_cl.values()), "o", color="#c0392b", label="Classical, measured (accuracy-matched)")
    big = np.array(sizes[3:])
    ax.loglog(big ** 3, [per[n] * Sc(n) for n in big], "o", mfc="none", color="#c0392b", label="Classical, measured cost/step × matched steps")
    ax.loglog(side ** 3, t_cl, "--", color="#c0392b", alpha=.7, label="Classical projection")
    ax.loglog([27], [meas_q[3]], "s", color="#2471a3", label="Quantum, IBM Heron gate time, 135 qubits")
    ax.loglog(side ** 3, t_q, "-", color="#2471a3", label="Quantum projection, gate time")
    ax.loglog(side ** 3, t_q_today, ":", color="#2471a3", label="Quantum projection, today's shot cycle")
    if x_gate:
        ax.axvline(x_gate ** 3, color="gray", lw=.8, ls="--")
        ax.text(x_gate ** 3 / 1.3, t_cl.min() * 3, f"crossover\n~{x_gate:.0f}³ voxels", fontsize=8, ha="right")
    ax.axvline(HEAD ** 3, color="k", lw=.8); ax.text(HEAD ** 3 * 1.2, t_cl.max() / 30, f"full head\n{HEAD}³", fontsize=8)
    ax.set_xlabel("voxels in the 3D head grid"); ax.set_ylabel(f"time for one answer at {TOL:.0%} accuracy (s)")
    ax.set_title("Same accuracy, growing head: where quantum overtakes classical\n"
                 "(IBM gate time; noise-free projection beyond 135 qubits)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout()
    fig.savefig(OUT / "07_crossover.png", dpi=180)
    print("saved", OUT / "07_crossover.png")
