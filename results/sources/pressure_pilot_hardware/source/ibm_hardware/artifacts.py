"""Save exact circuits and raw IBM results before deriving pressure estimates."""
import hashlib
import json
from qiskit import qpy
from qiskit_ibm_runtime import RuntimeEncoder


def write_runtime_json(evidence, name, value):
    evidence.write(name, json.loads(json.dumps(value, cls=RuntimeEncoder)))


def save_circuits(evidence, name, circuits):
    path = evidence.path/name
    with path.open('xb') as stream:
        qpy.dump(circuits, stream)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_result(evidence, prefix, result, records, shots):
    # Save raw payload before parsing/validation, so unexpected outputs are recoverable.
    write_runtime_json(evidence, prefix+'_raw_result.json', result)
    if len(result) != len(records):
        raise ValueError('IBM returned an unexpected number of PUB results')
    counts = []
    for pub, record in zip(result, records):
        bits = getattr(pub.data, record['classical_register'])
        if bits.num_bits != 1 or bits.num_shots != shots:
            raise ValueError('IBM result differs from one-bit/shot contract')
        counts.append(bits.get_counts())
    evidence.write(prefix+'_counts.json', counts)
    return counts
