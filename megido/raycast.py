"""Sparse system matrix: mean path length of each row's ray bundle per voxel.

A row is one (position, sky direction) pair from megido.fitdata.RowIndex, and
its measurement is lambda = integral of opacity density along that ray. The
detector aperture is 38.4 cm, several voxels wide at the default 25 cm spacing,
so a row is modelled as a bundle of n_sub^2 parallel sub-rays across the
aperture rather than as a pinhole. Entries are path lengths AVERAGED over the
bundle, so a bundle and a pinhole through uniform material predict the same
optical depth.

The pose rotation is deliberately absent. Phase 2's detector_to_sky already
mapped every measurement into the world frame, so a row's direction is just
normalize(sx, sy, 1); rotating again would rotate twice.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

from megido.fitdata import RowIndex
from megido.voxels import VoxelGrid

# Bumped whenever ray-casting logic changes. It is part of the cache key: in
# Phase 1 a cache that ignored the code version silently served artifacts built
# before a reconstruction fix, and the same trap is live here.
INVERSION_VERSION = 1

_SAMPLES_PER_VOXEL = 3          # sampling step along a ray = spacing / this
_ROW_BLOCK = 512                # rows processed per vectorised block


def bundle_offsets(directions: np.ndarray, aperture_m: float,
                   n_sub: int) -> np.ndarray:
    """Sub-ray start offsets, [n_rows, n_sub**2, 3], perpendicular to each ray.

    The physical aperture is a plane in the DETECTOR frame, but a row aggregates
    exposures at different tilts, so there is no single aperture orientation to
    use. A square of side `aperture_m` perpendicular to the ray is pose-free and
    correct to first order in tilt (it overestimates the cross-section by
    1/cos(tilt), about 6% at the campaign's 20 degrees).
    """
    d = np.asarray(directions, dtype=np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)

    # An orthonormal frame perpendicular to each ray. cross(d, z) degenerates for
    # a vertical ray, so fall back to x there.
    zhat = np.array([0.0, 0.0, 1.0])
    u = np.cross(d, zhat)
    small = np.linalg.norm(u, axis=-1) < 1e-9
    u[small] = np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(d, u)

    g = (np.arange(n_sub) + 0.5) / n_sub - 0.5          # centred, in [-0.5, 0.5)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    gx, gy = gx.ravel() * aperture_m, gy.ravel() * aperture_m
    return gx[None, :, None] * u[:, None, :] + gy[None, :, None] * v[:, None, :]


def build_system_matrix(rows: RowIndex,
                        origins: dict[str, tuple[float, float, float]],
                        grid: VoxelGrid, *,
                        aperture_m: float,
                        n_sub: int = 4,
                        cache_dir: str | Path | None = None) -> sparse.csr_matrix:
    """A[row, voxel] in METRES, shape [rows.n_rows, grid.n_voxels]."""
    if cache_dir is not None:
        key = _cache_key(rows, origins, grid, aperture_m, n_sub)
        cache = Path(cache_dir) / f"A_{key}.npz"
        if cache.exists():
            return sparse.load_npz(cache)

    dirs = rows.directions()                                    # [nr, 3]
    offs = bundle_offsets(dirs, aperture_m, n_sub)              # [nr, ns, 3]
    starts = np.array([origins[rows.position_ids[i]] for i in rows.pos_of_row])
    starts = starts[:, None, :] + offs                          # [nr, ns, 3]

    origin = np.asarray(grid.origin, dtype=np.float64)
    shape = np.asarray(grid.shape, dtype=np.int64)
    z0, z1 = grid.extent(2)

    # Entry and exit of the grid's z slab along each ray, measured from its own
    # start. dz > 0 always: a row's direction is normalize(sx, sy, 1).
    dz = dirs[:, 2]
    t_in = (z0 - starts[:, :, 2].mean(axis=1)) / dz
    length = (z1 - z0) / dz
    step = grid.spacing / _SAMPLES_PER_VOXEL
    n_samp = np.maximum(np.ceil(length / step).astype(np.int64), 1)

    nr, ns = rows.n_rows, offs.shape[1]
    r_out, c_out, v_out = [], [], []
    for lo in range(0, nr, _ROW_BLOCK):
        hi = min(lo + _ROW_BLOCK, nr)
        cn = n_samp[lo:hi]
        row_rep = np.repeat(np.arange(lo, hi), cn)                        # [Nt]
        # midpoint sampling fraction along each ray's in-slab segment
        base = np.repeat(np.concatenate([[0], np.cumsum(cn[:-1])]), cn)
        frac = (np.arange(int(cn.sum())) - base + 0.5) / np.repeat(cn, cn)
        t = t_in[row_rep] + frac * length[row_rep]                        # [Nt]

        pts = starts[row_rep] + (t[:, None] * dirs[row_rep])[:, None, :]  # [Nt, ns, 3]
        idx = np.floor((pts - origin) / grid.spacing).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), axis=-1)              # [Nt, ns]
        flat = (idx[..., 0] * grid.shape[1] + idx[..., 1]) * grid.shape[2] + idx[..., 2]
        # Divide by ns: the bundle AVERAGES path length, it does not accumulate.
        dl = length[row_rep] / n_samp[row_rep] / ns

        r_out.append(np.broadcast_to(row_rep[:, None], inside.shape)[inside])
        c_out.append(flat[inside])
        v_out.append(np.broadcast_to(dl[:, None], inside.shape)[inside])

    A = sparse.coo_matrix(
        (np.concatenate(v_out) if v_out else np.zeros(0),
         (np.concatenate(r_out) if r_out else np.zeros(0, dtype=np.int64),
          np.concatenate(c_out) if c_out else np.zeros(0, dtype=np.int64))),
        shape=(nr, grid.n_voxels),
    ).tocsr()
    A.sum_duplicates()

    if cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cache, A)
    return A


def _cache_key(rows: RowIndex, origins: dict, grid: VoxelGrid,
               aperture_m: float, n_sub: int) -> str:
    h = hashlib.sha256()
    h.update(f"v{INVERSION_VERSION}|".encode())
    h.update(rows.key().encode())
    for pid in rows.position_ids:
        x, y, z = origins[pid]
        h.update(f"{pid}:{x:.6f},{y:.6f},{z:.6f};".encode())
    h.update(f"{grid.key()}|{aperture_m:.6f}|{n_sub}|{_SAMPLES_PER_VOXEL}".encode())
    return h.hexdigest()[:16]
