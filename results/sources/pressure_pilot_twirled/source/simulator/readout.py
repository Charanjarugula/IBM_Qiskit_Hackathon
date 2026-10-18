"""Raw one-bit counts to a signed overlap; no device-error mitigation."""
from math import sqrt
from numbers import Integral


def ancilla_statistics(counts, shots):
    if shots < 2 or not counts or set(counts)-{'0', '1'}:
        raise ValueError('Expected raw one-bit ancilla counts and at least two shots')
    if any(not isinstance(n, Integral) or n < 0 for n in counts.values()) or sum(counts.values()) != shots:
        raise ValueError('Invalid counts or unexpected shot total')
    overlap = (counts.get('0', 0)-counts.get('1', 0))/shots
    probability_zero = (counts.get('0', 0)+1)/(shots+2)
    reporting_se = 2*sqrt(probability_zero*(1-probability_zero)/shots)
    sampling_variance = (1-overlap**2)/(shots-1)
    return overlap, reporting_se, sampling_variance
