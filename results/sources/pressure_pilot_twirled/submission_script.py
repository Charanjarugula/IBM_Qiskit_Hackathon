"""One authorized diagnostic job; same saved ISA as the first pressure pilot.

Only gate twirling changes. This is an exploratory comparison across two jobs,
not an independent repeated causal estimate. No automatic resubmission.
"""
from pathlib import Path
import sys,json,hashlib,shutil,uuid
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/charanjarugula/Documents/ChatGPT/Quantum Computing/IBM_Qiskit_Hackathon')
sys.path.insert(0,str(REPO))
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2
from classical._core.evidence import Evidence
from ibm_hardware.artifacts import save_result,write_runtime_json
from ibm_hardware.readout import analyze_pilot

base=ROOT/'sources/pressure_pilot_hardware'
out=ROOT/'sources/pressure_pilot_twirled'
assert not out.exists(),'Never overwrite or blindly retry a submitted experiment'
assert json.loads((base/'compiled_ideal_check.json').read_text())['all_match']
plan=json.loads((base/'pilot_circuits.json').read_text())
assert hashlib.sha256((base/'pilot_isa.qpy').read_bytes()).hexdigest()==plan['isa_qpy_sha256']
with (base/'pilot_isa.qpy').open('rb') as f:circuits=qpy.load(f)
records=plan['records'];assert len(circuits)==len(records)==12
account='hackathon-account-2'
instance=QiskitRuntimeService.saved_accounts()[account]['instance']
service=QiskitRuntimeService(name=account,instance=instance)
usage=service.usage();assert usage['usage_remaining_seconds']>=60
backend=service.backend('ibm_kingston',instance=instance,use_fractional_gates=False)
assert backend.status().operational
options=json.loads((base/'sampler_options.json').read_text())
options['twirling'].update(enable_gates=True,num_randomizations=32,shots_per_randomization=32)
sampler=SamplerV2(mode=backend,options=options)
evidence=Evidence(out,base/'config.json','pressure_gate_twirling_diagnostic')
for name in ['pilot_isa.qpy','pilot_circuits.json','reference.json','compiled_ideal_check.json']:
    shutil.copyfile(base/name,out/name)
shutil.copyfile(Path(__file__),out/'submission_script.py')
evidence.write('sampler_options.json',options)
evidence.write('design.json',{'baseline_job':json.loads((base/'job_0000_submitted.json').read_text())['job_id'],
    'changed_factor':'gate twirling only: 32 randomizations x 32 shots',
    'shots_per_circuit':1024,'circuits':12,'execution_seconds_ceiling':60,
    'primary_metric':'raw-pressure RMSE against the saved noiseless circuit',
    'secondary_metric':'assignment-corrected pressure RMSE',
    'scope':'Exploratory one-job follow-up; cross-job drift and twirl variability not isolated',
    'runtime_docs':'https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-twirling-options'})
write_runtime_json(evidence,'backend_properties.json',backend.properties().to_dict())
tag='pressure-twirl-'+uuid.uuid4().hex;sampler.options.environment.job_tags=[tag]
evidence.write('job_0000_intent.json',{'tag':tag,'backend':'ibm_kingston','records':records,'shots':1024,
    'options':options,'recovery':'Check IBM for this tag before any retry'})
job=sampler.run(circuits,shots=1024)
evidence.write('job_0000_submitted.json',{'job_id':job.job_id(),'tag':tag})
print('Submitted',job.job_id(),flush=True)
counts=save_result(evidence,'job_0000',job.result(),records,1024)
write_runtime_json(evidence,'job_0000_metrics.json',job.metrics())
report=analyze_pilot(records,counts,1024,(2,2,2))
evidence.write('pilot_pressure.json',report)
evidence.finish({'status':'pilot_complete','job_ids':[job.job_id()],
    'gate_twirling':True,'readout_calibration_passed':report['calibration_checks_passed'],
    'dynamical_decoupling':False,'error_correction':False})
print('Collected',job.job_id(),flush=True)
