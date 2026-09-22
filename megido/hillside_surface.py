"""Linear regularized hillside-surface inversion core (Phase 5b, Task 1).

The hillside silhouette is fit as a smooth height field H(x, y) on a
rectangular grid. Each muon ray with a measured path length places a surface
point at its exit point ``o + L * d_hat``. Bilinear interpolation of the grid
at the exit point's (x, y) equals the exit point's z -- this is LINEAR in the
grid heights, unlike an iterative ray-march forward model (which is a
non-differentiable step function and stalls gradient-based optimizers on a
flat surface). This module builds the sparse bilinear sampling matrix, a
Laplacian smoothness operator, and the regularized normal-equations solve.

Grid flattening convention: for grids ``gx`` (length nx) and ``gy`` (length
ny), grid index ``(i, j)`` (i indexing gx, j indexing gy) flattens to
``flat = i * ny + j`` (i.e. ``np.meshgrid(gx, gy, indexing="ij").ravel()``).
Downstream consumers (Task 2's driver) must use this same convention.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def bilinear_matrix(xy: np.ndarray, gx: np.ndarray, gy: np.ndarray):
    """Build the sparse bilinear sampling matrix for points ``xy`` on grid (gx, gy).

    Parameters
    ----------
    xy : (N, 2) array of (x, y) sample points.
    gx : (nx,) strictly increasing grid coordinates along x.
    gy : (ny,) strictly increasing grid coordinates along y.

    Returns
    -------
    B : scipy.sparse.csr_matrix, shape (N, nx*ny)
        Row n has up to 4 nonzero bilinear weights (summing to 1) for the
        grid cell containing point n, flattened as ``i * ny + j``. Points
        outside ``[gx[0], gx[-1]] x [gy[0], gy[-1]]`` get an all-zero row.
    support : (nx*ny,) ndarray
        Column sums of B -- total data support at each grid node.
    """
    xy = np.asarray(xy, dtype=float)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    nx, ny = len(gx), len(gy)
    n = xy.shape[0]

    x = xy[:, 0]
    y = xy[:, 1]

    inside = (
        (x >= gx[0]) & (x <= gx[-1]) & (y >= gy[0]) & (y <= gy[-1])
    )

    rows = []
    cols = []
    vals = []

    # searchsorted gives the insertion index; clip so i0 in [0, nx-2].
    i0 = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, nx - 2)
    j0 = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, ny - 2)

    dx = gx[i0 + 1] - gx[i0]
    dy = gy[j0 + 1] - gy[j0]
    tx = np.where(dx > 0, (x - gx[i0]) / np.where(dx > 0, dx, 1.0), 0.0)
    ty = np.where(dy > 0, (y - gy[j0]) / np.where(dy > 0, dy, 1.0), 0.0)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    corners = (
        (i0, j0, w00),
        (i0 + 1, j0, w10),
        (i0, j0 + 1, w01),
        (i0 + 1, j0 + 1, w11),
    )
    for ci, cj, cw in corners:
        flat = ci * ny + cj
        for row_idx in np.nonzero(inside)[0]:
            rows.append(row_idx)
            cols.append(flat[row_idx])
            vals.append(cw[row_idx])

    G = nx * ny
    B = sp.coo_matrix((vals, (rows, cols)), shape=(n, G)).tocsr()
    support = np.asarray(B.sum(axis=0)).ravel()
    return B, support


def laplacian_matrix(nx: int, ny: int):
    """Interior 2nd-difference (5-point Laplacian) operator over an nx x ny grid.

    Flat index convention matches ``bilinear_matrix``: ``flat = i * ny + j``.
    Only interior nodes (1 <= i <= nx-2, 1 <= j <= ny-2) produce a row; a
    plane sampled on the grid has zero 2nd difference at every interior node.

    Returns
    -------
    scipy.sparse.csr_matrix, shape ((nx-2)*(ny-2), nx*ny) for nx, ny >= 3;
    an empty (0, nx*ny) matrix if there are no interior nodes.
    """
    G = nx * ny
    if nx < 3 or ny < 3:
        return sp.csr_matrix((0, G))

    rows = []
    cols = []
    vals = []
    row_idx = 0
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            center = i * ny + j
            xp = (i + 1) * ny + j
            xm = (i - 1) * ny + j
            yp = i * ny + (j + 1)
            ym = i * ny + (j - 1)
            for col, val in ((center, 4.0), (xp, -1.0), (xm, -1.0), (yp, -1.0), (ym, -1.0)):
                rows.append(row_idx)
                cols.append(col)
                vals.append(val)
            row_idx += 1

    return sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, G)).tocsr()


def solve_surface(B, z: np.ndarray, w: np.ndarray, lap, mu: float) -> np.ndarray:
    """Solve ``min ||W(Bh - z)||^2 + mu^2 ||lap h||^2`` for the flat grid heights h.

    Solved via the normal equations ``(B^T W B + mu^2 L^T L) h = B^T W z``.
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    B = sp.csr_matrix(B)
    lap = sp.csr_matrix(lap)

    W = sp.diags(w)
    BtW = B.T @ W
    A = (BtW @ B + (mu ** 2) * (lap.T @ lap)).tocsc()
    b = BtW @ z

    h = spla.spsolve(A, b)
    return np.asarray(h).ravel()


def variance_explained(B, z: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    """Weighted variance-explained: ``1 - sum(w*(Bh-z)^2) / sum(w*(z-zbar)^2)``."""
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    B = sp.csr_matrix(B)

    pred = B @ h
    resid = pred - z
    num = np.sum(w * resid ** 2)

    zbar = np.sum(w * z) / np.sum(w)
    denom = np.sum(w * (z - zbar) ** 2)

    return 1.0 - num / denom
