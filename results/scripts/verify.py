"""Audit saved evidence and recompute plotted statistics. Does not run solvers."""
from pathlib import Path
from hashlib import sha256
from math import sqrt, isclose
import json, tempfile, tarfile, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'sources'
sys.path.insert(0, str(ROOT.parent))
from ibm_hardware.readout import analyze_pilot

# Check the published raw evidence before deriving any statistics.
checksums = json.loads((ROOT/'checksums.json').read_text())
for rel, expected in checksums.items():
    p = ROOT/rel
    assert p.is_file() and sha256(p.read_bytes()).hexdigest() == expected, rel

workspace = tempfile.TemporaryDirectory(prefix='aperion-evidence-')
HAND = Path(workspace.name)
for filename, subdir in [('pilot_results.tar.gz','pilot'), ('reconstruction_runs.tar.gz','reconstructions')]:
    destination = HAND/subdir
    with tarfile.open(SRC/'handoff'/filename) as archive:
        for member in archive:
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts, member.name
            assert member.isfile() or member.isdir(), member.name
            target = destination/path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
load = lambda p: json.loads(p.read_text())
issues = []
checked_hashes = 0
for manifest in HAND.rglob('manifest.json'):
    d = load(manifest)
    for rel, expected in d.get('source_sha256', {}).items():
        p = manifest.parent/'source'/rel
        if not p.exists() or sha256(p.read_bytes()).hexdigest() != expected:
            issues.append(f'Source hash mismatch: {p.relative_to(HAND)}')
        else:
            checked_hashes += 1

hw = SRC/'handoff/3_ibm_job'
raw, saved = load(hw/'raw_counts.json'), load(hw/'analysed_results_pub_order.json')
recomputed = []
for pub, expected in zip(raw['pubs'], saved, strict=True):
    counts = pub['counts']; total = sum(counts.values())
    assert total == pub['shots'] == 4000
    assert all(len(k)==15 and set(k)<=set('01') and isinstance(v,int) and v>=0 for k,v in counts.items())
    kept = sum(v for k,v in counts.items() if k.count('1')==1)
    target = counts.get(format(1<<2,'015b'),0)
    p, q = target/total, target/kept
    row = {k:pub[k] for k in ['bleed','steps','fold']}
    row.update(raw=p,raw_se=sqrt(p*(1-p)/total),postselected=q,
               post_se=sqrt(q*(1-q)/kept),kept_fraction=kept/total,shots=total)
    for k,v in row.items(): assert isclose(v,expected[k],rel_tol=1e-12,abs_tol=1e-14), (pub['index'],k)
    recomputed.append(row)
assert recomputed == load(SRC/'hardware_ibm_marrakesh.json')


pilot = HAND/'pilot/runs'
ref = load(pilot/'pilot_reference.json')
exact = np.array(ref['classical_pressure_pa'])
report = load(pilot/'comparison/comparison.json')
rows = []
for m in report['methods']:
    predictions = np.array([r['pressure_pa'] for r in m['all_repeats']])
    mean = predictions.mean(axis=0)
    err = float(np.linalg.norm(mean-exact)/np.linalg.norm(exact))
    individual = [float(np.linalg.norm(p-exact)/np.linalg.norm(exact)) for p in predictions]
    assert np.allclose(mean,m['mean_pressure_pa'],rtol=1e-12,atol=1e-14)
    assert isclose(err,m['error_of_mean_vs_exact']['relative_l2'],rel_tol=1e-12,abs_tol=1e-14)
    assert all(isclose(e,r['error_vs_exact']['relative_l2'],rel_tol=1e-12,abs_tol=1e-14) for e,r in zip(individual,m['all_repeats']))
    assert all(e<=.01 for e in individual) == m['all_repeats_within_target']
    times = [r['wall_seconds'] for r in m['all_repeats']]
    rows.append(dict(method=m['method'],relative_l2_percent=err*100,
                     repeat_error_percent=[v*100 for v in individual],
                     seconds=float(np.mean(times)),repeat_seconds=times,
                     all_repeats_pass=m['all_repeats_within_target']))

count_files = 0
for p in HAND.rglob('counts_*.json'):
    d = load(p); counts = d['counts']; n = d['shots']
    assert sum(counts.values()) == n and set(counts)<=set(['0','1']),p
    overlap = (counts.get('0',0)-counts.get('1',0))/n
    assert isclose(overlap,d['signed_overlap'],abs_tol=1e-14),p
    count_files += 1
for name in ['gate_ideal','gate_noisy']:
    directory=pilot/name; config=load(directory/'config.json')
    assert load(directory/'comparison.json')['acquisition_signature']==ref['acquisition_signature']
    recovered=np.zeros((3,*ref['trace_shape']))
    for p in directory.glob('counts_*.json'):
        d=load(p); si=config['sources'].index(d['source']); ri=config['receivers'].index(d['receiver'])
        ti=config['times_s'].index(d['time_s'])
        recovered[d['forward_call'],si,ti,ri] += d['physical_scale_pa']*d['signed_overlap']
    assert np.allclose(recovered,[r['pressure_pa'] for r in load(directory/'repeats.json')],atol=1e-14)
assert load(pilot/'annealing/comparison.json')['acquisition_signature']==ref['acquisition_signature']

inversions=[]
for case in ['small_1d','pulse_1d']:
    base=HAND/'reconstructions/runs'/case
    names={'classical':'classical','ideal':'aer_ideal','noisy':'aer_noisy'}
    signatures=[load(base/n/'comparison.json') for n in names.values()]
    assert len({d['inverse_problem_signature'] for d in signatures})==1
    assert len({d['observations_sha256'] for d in signatures})==1
    config=load(base/'classical/config.json')
    row=dict(case=case,truth=config['synthetic_truth_parameters'],initial=config['initial_parameters'])
    for key,name in names.items():
        directory=base/name
        assert (directory/'completed.json').exists() and not (directory/'failure.json').exists()
        assert sha256((directory/'observations.npz').read_bytes()).hexdigest()==load(directory/'comparison.json')['observations_sha256']
        h=load(directory/'history.json'); result=load(directory/'reconstruction.json')
        assert result['parameters']==h[-1]['parameters'] and len(h)==3
        row[key]=result['parameters']
        if key!='classical':
            assert len(list(directory.glob('counts_*.json')))==load(directory/'completed.json')['circuits']
    inversions.append(row)

metrics=dict(evidence='Arithmetic and source-hash audit of supplied archives; not an independent rerun',
             pilot=rows,inversions=inversions,forward_repeats=3,shots=8192,inversion_updates=2,
             inversion_repeats=1,hardware_job_id=raw['job_id'],
             qpu_charge_seconds=load(hw/'job_metadata.json')['metrics']['usage']['qpu_charge_time_seconds'])
assert metrics == load(SRC/'audited_simulations.json'), 'Published simulator summary differs from raw archives'
audit=dict(source_hashes_checked=checked_hashes,ancilla_count_files_checked=count_files,
           hardware_pubs_checked=len(recomputed),hardware_shots_checked=24000,
           pilot_pressure_recovered_from_counts=True,paired_inverse_signatures_match=True,
           issues=issues,scope='Saved-data integrity and arithmetic only; no solvers or hardware jobs executed')

# Recompute signed pressure from original IBM counts, including rejected mitigation.
hardware_jobs = []
for directory in ['pressure_pilot_hardware', 'pressure_pilot_twirled', 'pressure_paired_hardware']:
    base = SRC/directory
    plan = load(base/'pilot_circuits.json')
    assert sha256((base/'pilot_isa.qpy').read_bytes()).hexdigest() == plan['isa_qpy_sha256']
    manifest = load(base/'manifest.json')
    for rel, expected in manifest['source_sha256'].items():
        assert sha256((base/'source'/rel).read_bytes()).hexdigest() == expected, (directory, rel)
        audit['source_hashes_checked'] += 1
    # The report helper used here must match the implementation recorded with the jobs.
    for rel in ['ibm_hardware/readout.py', 'simulator/readout.py']:
        assert sha256((ROOT.parent/rel).read_bytes()).hexdigest() == manifest['source_sha256'][rel]
    shape = tuple(load(base/'reference.json')['trace_shape'])
    reference = np.array(load(base/'reference.json')['ideal_pressure_pa'])
    for submitted in sorted(base.glob('job_*_submitted.json')):
        prefix = submitted.name.removesuffix('_submitted.json')
        identity = load(submitted)
        intent = load(base/(prefix+'_intent.json'))
        assert intent['records'] == plan['records']
        counts = load(base/(prefix+'_counts.json'))
        shots = intent['shots']
        assert len(counts) == len(plan['records']) == 12
        assert all(sum(c.values()) == shots for c in counts)
        report = analyze_pilot(plan['records'], counts, shots, shape)
        report_name = prefix+'_pressure.json' if directory == 'pressure_paired_hardware' else 'pilot_pressure.json'
        assert report == load(base/report_name), (directory, prefix)
        assert report['calibration_checks_passed']
        raw = np.array(report['raw_pressure_pa'])
        corrected = np.array(report['mitigated']['pressure_pa'])
        job_metrics = load(base/(prefix+'_metrics.json'))
        hardware_jobs.append(dict(
            experiment=directory, job_id=identity['job_id'],
            pair=identity.get('pair'), gate_twirling=intent['options']['twirling']['enable_gates'],
            shots_per_circuit=shots, circuits=len(counts),
            raw_rmse_pa=float(np.sqrt(np.mean((raw-reference)**2))),
            corrected_rmse_pa=float(np.sqrt(np.mean((corrected-reference)**2))),
            charged_qpu_seconds=job_metrics['usage']['qpu_charge_time_seconds']))

paired = [j for j in hardware_jobs if j['experiment'] == 'pressure_paired_hardware']
assert len(paired) == 6 and len(hardware_jobs) == 8
design = load(SRC/'pressure_paired_hardware/design.json')
assert [dict(pair=j['pair'], gate_twirling=j['gate_twirling']) for j in paired] == design['order']
mean_u, mean_t = [float(np.mean([j['raw_rmse_pa'] for j in paired if j['gate_twirling']==arm])) for arm in [False,True]]
for pair in range(3):
    u = next(j for j in paired if j['pair']==pair and not j['gate_twirling'])
    t = next(j for j in paired if j['pair']==pair and j['gate_twirling'])
    assert t['raw_rmse_pa'] < u['raw_rmse_pa']
assert all(j['corrected_rmse_pa'] > j['raw_rmse_pa'] for j in hardware_jobs)
assert sum(j['charged_qpu_seconds'] for j in hardware_jobs) == 111

# Runtime comparison uses the same archived fixed-map task. Keep both classical blocks.
timings = []
for method in load(pilot/'comparison/comparison.json')['methods']:
    times = [r['wall_seconds'] for r in method['all_repeats']]
    original = pilot/Path(method['run_directory']).name
    if method['method'] == 'classical_exact':
        original_rows = load(original/'classical.json')
    elif method['method'] in ['sa', 'sqa']:
        original_rows = [r for r in load(original/'repeats.json') if r['algorithm'] == method['method']]
    else:
        original_rows = load(original/'repeats.json')
    assert times == [r['wall_seconds'] for r in original_rows]
    assert isclose(float(np.mean(times)), method['mean_forward_wall_seconds'], rel_tol=1e-12)
    timings.append(dict(method=method['method'], repeat_seconds=times,
                        mean_seconds=float(np.mean(times)),
                        all_repeats_within_target=method['all_repeats_within_target']))

audit.update(hardware_jobs=hardware_jobs,
             pressure_hardware_count_dictionaries=sum(j['circuits'] for j in hardware_jobs),
             pressure_campaign_qpu_seconds=111,
             paired_mean_raw_rmse_pa=[mean_u,mean_t],
             paired_mean_absolute_rmse_reduction_pa=mean_u-mean_t,
             reduction_in_mean_rmse_percent=100*(1-mean_t/mean_u),
             timings=timings,
             published_input_files_verified=len(checksums),
             speed_projection_status='Original archived; sampling accuracy and claimed crossover not validated')
assert not issues, issues
(ROOT/'verification.json').write_text(json.dumps(audit, indent=2)+'\n')
workspace.cleanup()
print(json.dumps({k:v for k,v in audit.items() if k not in ['hardware_jobs','timings']}, indent=2))
