"""Three predeclared paired hardware repeats after an exploratory twirling pilot.

Same saved ISA and shot budget. Order is balanced U/T, T/U, U/T. No automatic
retries. Six jobs at a maximum 60 seconds each; 4096 shots per circuit.
"""
from pathlib import Path
import sys,json,hashlib,shutil,uuid
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/charanjarugula/Documents/ChatGPT/Quantum Computing/IBM_Qiskit_Hackathon')
sys.path.insert(0,str(REPO))
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.sampler import SamplerV2
from classical._core.evidence import Evidence
from ibm_hardware.artifacts import save_result,write_runtime_json
from ibm_hardware.readout import analyze_pilot
from ibm_hardware.pilot import estimate_job_seconds

base=ROOT/'sources/pressure_pilot_hardware';out=ROOT/'sources/pressure_paired_hardware'
assert not out.exists(),'Never overwrite or blindly retry a submitted experiment'
assert json.loads((base/'compiled_ideal_check.json').read_text())['all_match']
plan=json.loads((base/'pilot_circuits.json').read_text())
assert hashlib.sha256((base/'pilot_isa.qpy').read_bytes()).hexdigest()==plan['isa_qpy_sha256']
with (base/'pilot_isa.qpy').open('rb') as f:circuits=qpy.load(f)
records=plan['records'];assert len(circuits)==len(records)==12
account='hackathon-account-2';instance=QiskitRuntimeService.saved_accounts()[account]['instance']
service=QiskitRuntimeService(name=account,instance=instance)
usage=service.usage();assert usage['usage_remaining_seconds']>=360
backend=service.backend('ibm_kingston',instance=instance,use_fractional_gates=False)
assert backend.status().operational
estimate=estimate_job_seconds(circuits,backend,4096);assert estimate<=60
evidence=Evidence(out,base/'config.json','pressure_gate_twirling_paired_repeats')
for name in ['pilot_isa.qpy','pilot_circuits.json','reference.json','compiled_ideal_check.json']:
    shutil.copyfile(base/name,out/name)
shutil.copyfile(Path(__file__),out/'submission_script.py')
order=[(0,False),(0,True),(1,True),(1,False),(2,False),(2,True)]
evidence.write('design.json',{'written_before_submission_utc':datetime.now(timezone.utc).isoformat(),
    'order':[{'pair':pair,'gate_twirling':twirl} for pair,twirl in order],
    'shots_per_circuit':4096,'circuits_per_job':12,'maximum_jobs':6,'estimated_qpu_seconds':6*estimate,
    'execution_seconds_ceiling':360,'primary_metric':'Paired raw-pressure RMSE difference vs saved noiseless circuit',
    'secondary_metric':'Readout-corrected RMSE when calibration checks pass',
    'exclusions':'Exploratory pilots excluded from confirmatory summaries',
    'stop_rule':'Complete all three pairs irrespective of effect direction. Stop for submission/result failure or rejected calibration.',
    'uncertainty':'Three paired job repeats; randomizations within jobs are not independent job repeats',
    'limitations':'Short same-device campaign; residual drift possible; no speedup or clinical claim'})
evidence.write('usage_before.json',usage)
write_runtime_json(evidence,'backend_properties_before.json',backend.properties().to_dict())
summaries=[]
for index,(pair,twirl) in enumerate(order):
    prefix=f'job_{index:04d}'
    options=json.loads((base/'sampler_options.json').read_text())
    options['default_shots']=4096
    if twirl:options['twirling'].update(enable_gates=True,num_randomizations=32,shots_per_randomization=128)
    sampler=SamplerV2(mode=backend,options=options)
    tag='pressure-pair-'+uuid.uuid4().hex;sampler.options.environment.job_tags=[tag]
    evidence.write(prefix+'_intent.json',{'tag':tag,'backend':'ibm_kingston','pair':pair,'gate_twirling':twirl,
        'records':records,'shots':4096,'options':options,'recovery':'Check IBM tag and saved job ID before any retry'})
    try:
        job=sampler.run(circuits,shots=4096)
        evidence.write(prefix+'_submitted.json',{'job_id':job.job_id(),'tag':tag,'pair':pair,'gate_twirling':twirl})
        print('Submitted',index,pair,twirl,job.job_id(),flush=True)
        counts=save_result(evidence,prefix,job.result(),records,4096)
        write_runtime_json(evidence,prefix+'_metrics.json',job.metrics())
        report=analyze_pilot(records,counts,4096,(2,2,2))
        evidence.write(prefix+'_pressure.json',report)
        summaries.append({'index':index,'pair':pair,'gate_twirling':twirl,'job_id':job.job_id(),
                          'calibration_checks_passed':report['calibration_checks_passed']})
        evidence.write('collected.json',summaries)
        print('Collected',index,'calibration',report['calibration_checks_passed'],flush=True)
        if not report['calibration_checks_passed']:
            evidence.finish({'status':'calibration_rejected','jobs':summaries});break
    except BaseException as error:
        evidence.write('failure.json',{'type':type(error).__name__,'message':str(error),'index':index,
            'recovery':'Inspect submitted job IDs/tags before any retry. Never rerun this script blindly.'})
        raise
else:
    evidence.write('usage_after.json',service.usage())
    evidence.finish({'status':'complete','jobs':summaries,'error_correction':False,'dynamical_decoupling':False})
