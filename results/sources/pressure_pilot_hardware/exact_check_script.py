"""Small exact statevector check of the saved four-active-qubit ISA circuits."""
from pathlib import Path
import json
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Statevector

root=Path(__file__).resolve().parent
run=root/'sources/pressure_pilot_hardware'
records=json.loads((run/'pilot_circuits.json').read_text())['records']
ref=json.loads((run/'reference.json').read_text())
with (run/'pilot_isa.qpy').open('rb') as f: circuits=qpy.load(f)
rows=[]
for qc,record in zip(circuits,records,strict=True):
    used=sorted({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
    assert len(used)<=4
    mapping={q:i for i,q in enumerate(used)}
    small=QuantumCircuit(len(used));small.global_phase=qc.global_phase
    measured=[]
    for inst in qc.data:
        qs=[mapping[qc.find_bit(q).index] for q in inst.qubits]
        if inst.operation.name=='measure': measured.extend(qs)
        elif inst.operation.name!='barrier': small.append(inst.operation,qs)
    assert len(measured)==1
    state=Statevector.from_instruction(small)
    z='I'*(len(used)-1-measured[0])+'Z'+'I'*measured[0]
    observed=float(state.expectation_value(__import__('qiskit').quantum_info.Pauli(z)).real)
    if record['kind']=='wave':
        key=(record['source_index'],record['time_index'],record['receiver_index'])
        pressure=observed*record['physical_scale_pa']
        expected=float(np.array(ref['ideal_pressure_pa'])[key])
        error=abs(pressure-expected)
        rows.append({'kind':'wave','index':key,'active_physical_qubits':used,
                     'compiled_ideal_pressure':pressure,'reference_pressure':expected,
                     'absolute_error':error})
        assert error<1e-10
    else:
        expected=1-2*record['prepared_state'];assert abs(observed-expected)<1e-10
        rows.append({'kind':'calibration','prepared_state':record['prepared_state'],'ideal_z':observed})
out={'method':'Exact statevectors after dropping only idle physical wires; at most four active qubits',
     'circuits_checked':len(rows),'all_match':True,'rows':rows}
(run/'compiled_ideal_check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
