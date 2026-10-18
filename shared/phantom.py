"""Simple layered head phantom: water | scalp | skull | brain (+ optional bleed).

Adapted from AnikS22/IBM_Qiskit_Fallfest (MIT; see licenses/Anik-MIT.txt).
Speeds are illustrative literature-scale values, not validated tissue classification thresholds.
This layered ellipsoid is not the anatomical phantom used by Guasch et al. 2020.
"""

import numpy as np

SPEEDS = {
    "water": 1500.0,
    "scalp": 1600.0,
    "skull": 2800.0,
    "brain": 1546.0,
    "blood": 1578.0,
}
C_REF = 1500.0
LABELS = {"water": 0, "scalp": 1, "skull": 2, "brain": 3, "blood": 4}


def head_phantom(
    shape,
    bleed=None,
    bleed_radius=0.0,
    scalp_frac=0.045,
    skull_frac=0.065,
    normalise=True,
):
    """Return (speed array, label array) on a grid of `shape` (1-3 dims).

    The head is an ellipsoid filling the box. Layer thicknesses are fractions of the head radius
    (22 cm head: scalp ~5 mm, skull ~7 mm) but never thinner than one cell, so tiny grids keep a skull.
    bleed: grid coordinates of the bleed centre (tuple) or None. bleed_radius in cells (0 -> single cell).
    """
    shape = tuple(shape)
    ctr = [(s - 1) / 2 for s in shape]
    rad = [s / 2 for s in shape]
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    d = np.sqrt(sum(((g - c0) / r) ** 2 for g, c0, r in zip(grids, ctr, rad)))
    r = min(rad)
    depth = (1 - d) * r  # ~cells inside the head surface
    ts = max(1, round(scalp_frac * r))
    tk = max(1, round(skull_frac * r))
    lab = np.full(shape, LABELS["brain"])
    lab[depth < ts + tk] = LABELS["skull"]
    lab[depth < ts] = LABELS["scalp"]
    lab[d > 1] = LABELS["water"]
    if bleed is not None:
        db = np.sqrt(sum((g - b) ** 2 for g, b in zip(grids, bleed)))
        lab[(db <= bleed_radius + 1e-9) & (lab == LABELS["brain"])] = LABELS["blood"]
    inv = {v: k for k, v in LABELS.items()}
    c = np.vectorize(lambda L: SPEEDS[inv[L]])(lab).astype(float)
    return (c / C_REF if normalise else c), lab


def ray_through(arr, axis=0, at=None):
    """1D line of cells through the array along `axis`, through the centre (or `at`)."""
    at = list(at) if at is not None else [s // 2 for s in arr.shape]
    sl = [a for a in at]
    sl[axis] = slice(None)
    return arr[tuple(sl)]


def transducers(lab):
    """Outermost head cells (scalp, or skull if no scalp at this resolution) = where transducers sit."""
    head = lab != LABELS["water"]
    from scipy.ndimage import binary_erosion

    surface = head & ~binary_erosion(head, border_value=0)
    return np.argwhere(surface)
