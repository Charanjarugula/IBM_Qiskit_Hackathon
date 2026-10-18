"""Regular 1D/2D/3D grids as graphs (vertices = cells, edges = links between neighbouring cells).

Reused from AnikS22/IBM_Qiskit_Fallfest; MIT notice in licenses/Anik-MIT.txt.

Boundary: every cell on the box face gets a 'ghost' edge to a fixed zero-pressure node outside the box
(Dirichlet / pressure-release). This keeps the discrete operator exactly energy-conserving, which is
what lets it run as unitary (quantum) evolution. Real heads sit in absorbing surroundings -> see README.md.
"""

import numpy as np
import scipy.sparse as sp


class Grid:
    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)
        assert 1 <= len(self.shape) <= 3
        self.ndim = len(self.shape)
        self.n_vertices = int(np.prod(self.shape))
        eu, ev, ax = (
            [],
            [],
            [],
        )  # edge endpoints (u = lower, v = upper); -1 = ghost node
        idx = np.arange(self.n_vertices).reshape(self.shape)
        for a in range(self.ndim):
            moved = np.moveaxis(idx, a, -1).reshape(
                -1, self.shape[a]
            )  # lines along axis a
            for line in moved:
                ends = np.concatenate([[-1], line, [-1]])
                for k in range(len(ends) - 1):
                    eu.append(ends[k])
                    ev.append(ends[k + 1])
                    ax.append(a)
        self.edge_u = np.array(eu)
        self.edge_v = np.array(ev)
        self.edge_axis = np.array(ax)
        self.n_edges = len(eu)

    def index(self, *coords):
        return int(np.ravel_multi_index(coords, self.shape))

    def coords(self, v):
        return np.unravel_index(v, self.shape)

    def incidence(self):
        """B (n_vertices x n_edges): +1 at the upper endpoint, -1 at the lower endpoint."""
        rows, cols, vals = [], [], []
        for e, (u, v) in enumerate(zip(self.edge_u, self.edge_v)):
            if u >= 0:
                rows.append(u)
                cols.append(e)
                vals.append(-1.0)
            if v >= 0:
                rows.append(v)
                cols.append(e)
                vals.append(1.0)
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_vertices, self.n_edges)
        )

    def edge_speed(self, c):
        """Sound speed on each edge: mean of its two cells (ghost edges take the single cell's speed)."""
        c = np.asarray(c).ravel()
        cu = np.where(self.edge_u >= 0, c[np.maximum(self.edge_u, 0)], np.nan)
        cv = np.where(self.edge_v >= 0, c[np.maximum(self.edge_v, 0)], np.nan)
        return np.nanmean(np.vstack([cu, cv]), axis=0)
