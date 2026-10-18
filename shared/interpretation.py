"""Illustrative speed-based labels, not validated tissue identification or diagnosis."""
import numpy as np
from .phantom import SPEEDS


def interpret(speed_m_s):
    names = list(SPEEDS)
    reference = np.array([SPEEDS[name] for name in names])
    distance = np.abs(np.asarray(speed_m_s)[..., None]-reference)
    index = np.argmin(distance, axis=-1)
    return {'label_names': names, 'label_indices': index.tolist(),
            'distance_to_reference_m_s': np.min(distance, axis=-1).tolist(),
            'warning': 'Nearest reference speed only; tissue identity is not uniquely determined by sound speed.'}
