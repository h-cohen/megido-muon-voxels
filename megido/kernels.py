"""Point-spread functions of the DELIVERED voxel solver.

Spec: docs/superpowers/specs/2026-10-06-voxel-averaging-kernels-design.md.

A delivered voxel x_hat +- sigma estimates the truth blurred by the solver's
local resolution operator R (TV + non-negativity + damping + c_p, as
configured), not the point truth: count-level phantoms showed the bootstrap
sigma is a correct NOISE sigma that covers the point truth in only 33-55% (cafeteria 33-40%, Megiddo 42-55%) of
voxels, because the bias of the one-sided geometry dominates
(spike/sigma-calibration). This module measures columns of R by perturbation:

    PSF_j = (x_hat(lam + delta A e_j) - x_hat(lam)) / delta

re-solved along the nominal solve's path (`stop_at=best_iter`), for batches of
probes far enough apart laterally to share one solve. Columns suffice: the
intended use R t = sum_j t_j PSF_j, and rows would need the adjoint of a nonlinear
iterative solver.

CAVEAT: that superposition claim was NOT validated in the two-position
geometry (spec §10). PSFs are cell-truncated: the |PSF| metrics normalise
within the cell, and psf_lat is capped by the cell half-width. This module is
a probe tool, not a coverage claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.config import KernelConfig, Reconstruction
from megido.fitdata import FitData
from megido.forward import ForwardModel
from megido.inversion import solve
from megido.voxels import VoxelGrid

METRICS = ("psf_mass", "psf_dz", "psf_shift", "psf_lat")


@dataclass(frozen=True)
class PSF:
    """One probe's response inside its lateral cell, per unit probe density.

    values[a, b, :] is voxel (lo[0] + a, lo[1] + b, :) -- the full z column.
    """
    probe: tuple[int, int, int]
    lo: tuple[int, int]
    values: np.ndarray


def _lattice_step(grid: VoxelGrid, kc: KernelConfig) -> int:
    return max(1, int(round(kc.spacing_m / grid.spacing)))


def cell_half_width(grid: VoxelGrid, kc: KernelConfig) -> int:
    """Half-width (voxels) of a probe's cell. (sep - 1) // 2 keeps the cells of
    one batch disjoint: probes are >= sep voxels apart."""
    sep = max(1, int(round(kc.sep_m / grid.spacing)))
    return max(0, (sep - 1) // 2)


def z_level_indices(grid: VoxelGrid, kc: KernelConfig) -> np.ndarray:
    z0, h, nz = grid.origin[2], grid.spacing, grid.shape[2]
    levels = (kc.z_levels_m if kc.z_levels_m is not None
              else np.arange(z0 + kc.spacing_m / 2, z0 + nz * h, kc.spacing_m))
    k = np.floor((np.asarray(levels, dtype=float) - z0) / h).astype(int)
    return np.unique(k[(k >= 0) & (k < nz)])


def probe_batches(grid: VoxelGrid, rays3: np.ndarray, kc: KernelConfig) -> list[np.ndarray]:
    """Probe voxels grouped into solves: one z level per batch, probes >= sep_m
    apart laterally; the (sep/step)^2 offset batches of a level fill the
    spacing_m lattice. Only voxels crossed by >= 2 rays are probed: below that
    a voxel is its ray's private unknown (the noise shell)."""
    step = _lattice_step(grid, kc)
    sep = max(step, int(round(kc.sep_m / grid.spacing)))
    m = -(-sep // step)                                    # ceil: offsets per axis
    nx, ny, _ = grid.shape
    li = np.arange(step // 2, nx, step)
    lj = np.arange(step // 2, ny, step)
    batches: list[np.ndarray] = []
    for k in z_level_indices(grid, kc):
        for oa in range(m):
            for ob in range(m):
                ii, jj = np.meshgrid(li[oa::m], lj[ob::m], indexing="ij")
                p = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k)], -1)
                p = p[rays3[p[:, 0], p[:, 1], p[:, 2]] >= 2]
                if len(p):
                    batches.append(p.astype(np.int64))
    return batches


def probe_deltas(fwd: ForwardModel, data: FitData, probes: np.ndarray,
                 kc: KernelConfig) -> np.ndarray:
    """delta_j = delta_sigma * median(sigma of weighted rows crossing j) / spacing:
    the density whose vertical path through one voxel adds ~delta_sigma noise
    sigmas of opacity. NaN when no weighted row crosses j (probe skipped)."""
    A = fwd.A.tocsc()
    flat = np.ravel_multi_index(probes.T, fwd.grid.shape)
    out = np.full(len(flat), np.nan)
    for n, j in enumerate(flat):
        rows = A.indices[A.indptr[j]:A.indptr[j + 1]]
        w = data.w[rows]
        w = w[w > 0]
        if w.size:
            out[n] = kc.delta_sigma * float(np.median(1.0 / np.sqrt(w))) / fwd.grid.spacing
    return out


def point_spreads(fwd: ForwardModel, data: FitData, rc: Reconstruction,
                  x_hat: np.ndarray, best_iter: int, probes: np.ndarray,
                  deltas: np.ndarray, *, fit_offsets: bool, cell_half: int) -> list[PSF]:
    """One perturbed solve for a batch; each probe's PSF is the response in its
    own lateral cell (clipped to the grid), divided by its delta. Probes with a
    NaN delta are skipped."""
    shape = fwd.grid.shape
    ok = np.isfinite(deltas)
    probes, deltas = probes[ok], deltas[ok]
    if not len(probes):
        return []
    bump = np.zeros(fwd.A.shape[1])
    bump[np.ravel_multi_index(probes.T, shape)] = deltas
    lam2 = data.lam + fwd.A @ bump
    x2, _ = solve(fwd, FitData(lam=lam2, w=data.w, rows=data.rows), rc,
                  fit_offsets=fit_offsets, stop_at=best_iter)
    resp = (x2 - np.asarray(x_hat).ravel()).reshape(shape)
    out = []
    for (i, j, k), d in zip(probes, deltas):
        i0, j0 = max(0, i - cell_half), max(0, j - cell_half)
        i1, j1 = min(shape[0], i + cell_half + 1), min(shape[1], j + cell_half + 1)
        out.append(PSF(probe=(int(i), int(j), int(k)), lo=(i0, j0),
                       values=resp[i0:i1, j0:j1, :] / d))
    return out


def _edge_quantile(p: np.ndarray, edges: np.ndarray, q: float) -> float:
    """q-quantile of a histogram p over bins with the given edges (p sums to 1)."""
    cdf = np.concatenate([[0.0], np.cumsum(p)])
    n = int(np.searchsorted(cdf, q, side="left"))
    n = min(max(n, 1), len(p))
    frac = (q - cdf[n - 1]) / p[n - 1] if p[n - 1] > 0 else 0.0
    return float(edges[n - 1] + frac * (edges[n] - edges[n - 1]))


def psf_metrics(psf: PSF, grid: VoxelGrid) -> dict[str, float]:
    """mass recovery, depth spread (16-84% width of |PSF|'s z-marginal), depth
    shift (|PSF| z-centroid minus probe z), lateral width (2x the radius about
    the probe holding 68% of |PSF|'s xy-marginal). NaN if the response is zero."""
    v = psf.values
    w = np.abs(v)
    tot = float(w.sum())
    if not tot > 0:
        return {k: float("nan") for k in METRICS}
    h = grid.spacing
    z_edges = grid.origin[2] + h * np.arange(grid.shape[2] + 1)
    zc = grid.axis_centers(2)
    pz = w.sum((0, 1)) / tot
    dz = _edge_quantile(pz, z_edges, 0.84) - _edge_quantile(pz, z_edges, 0.16)
    shift = float((pz * zc).sum() - zc[psf.probe[2]])
    pxy = w.sum(2) / tot
    a = np.arange(v.shape[0]) + psf.lo[0] - psf.probe[0]
    b = np.arange(v.shape[1]) + psf.lo[1] - psf.probe[1]
    r = h * np.hypot(a[:, None], b[None, :]).ravel()
    order = np.argsort(r, kind="stable")
    cum = np.cumsum(pxy.ravel()[order])
    r68 = float(r[order][min(int(np.searchsorted(cum, 0.68 - 1e-12)), r.size - 1)])
    return {"psf_mass": float(v.sum()), "psf_dz": float(dz),
            "psf_shift": shift, "psf_lat": 2.0 * r68}
