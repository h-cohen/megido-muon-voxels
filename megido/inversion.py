"""Voxel inversion algorithms. One linear model, no I/O, no site knowledge.

    lambda = A x + c_p,    x >= 0

x is the voxel opacity density [1/m] and c_p is a free additive constant per
position. c_p is not a fudge: Phase 2 pinned each position's opacity gauge
independently (the median of lambda_p is zero by construction, then shifted to
put the most transparent direction at zero), so the absolute level of lambda is
not a measurement and must be refit here.

Fits are selected by zeroing weights over the shared row layout, so one cached
system matrix serves the full fit and every holdout fit alike.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from megido.config import Reconstruction
from megido.fitdata import FitData
from megido.forward import ForwardModel


def _update_offsets(resid: np.ndarray, w: np.ndarray,
                    pos_of_row: np.ndarray, n_pos: int) -> np.ndarray:
    """Closed-form weighted-mean residual per position (the c_p nuisance)."""
    c = np.zeros(n_pos)
    for i in range(n_pos):
        sel = pos_of_row == i
        sw = w[sel].sum()
        if sw > 0:
            c[i] = float((w[sel] * resid[sel]).sum() / sw)
    return c


def _scalings(A: sparse.csr_matrix, w: np.ndarray):
    """Row and column scalings of the weighted system used by SIRT."""
    Aw = A.multiply(np.sqrt(w)[:, None]).tocsr()
    row_sum = np.asarray(abs(Aw).sum(axis=1)).ravel()
    col_sum = np.asarray(abs(Aw).sum(axis=0)).ravel()
    return 1.0 / np.maximum(row_sum, 1e-12), 1.0 / np.maximum(col_sum, 1e-12)


def _named(c: np.ndarray, position_ids) -> dict[str, float]:
    return {pid: float(v) for pid, v in zip(position_ids, c)}


def sirt(fwd: ForwardModel, data: FitData, rc: Reconstruction
         ) -> tuple[np.ndarray, dict]:
    """Weighted SIRT with nonnegativity and per-position offset refinement.

    Stops at rc.chi2_target by the discrepancy principle: fitting past the noise
    floor is how SIRT turns counting statistics into structure.
    """
    A, lam, w = fwd.A, data.lam, data.w
    n_pos = len(fwd.rows.position_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_pos)
    row_inv, col_inv = _scalings(A, w)
    n_used = max(int(np.count_nonzero(w)), 1)

    history: list[float] = []
    chi2 = float("inf")
    k = -1
    for k in range(rc.n_iter):
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        c = c + _update_offsets(resid, w, data.rows.pos_of_row, n_pos)
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if k % 20 == 0:
            history.append(chi2)
        if chi2 <= rc.chi2_target:
            break
        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
    history.append(chi2)
    return x, {"offsets": _named(c, fwd.rows.position_ids),
               "chi2_history": history,
               "best_chi2": float(min(history)),
               "n_iter_used": k + 1}


def _grad3(x3: np.ndarray, z_weight: float) -> np.ndarray:
    """Anisotropic forward differences -> [3, nx, ny, nz], zero-padded at the far edge."""
    g = np.zeros((3,) + x3.shape)
    g[0, :-1] = np.diff(x3, axis=0)
    g[1, :, :-1] = np.diff(x3, axis=1)
    g[2, :, :, :-1] = np.diff(x3, axis=2) * z_weight
    return g


def _div3(p: np.ndarray, z_weight: float) -> np.ndarray:
    """Negative adjoint of _grad3."""
    d = np.zeros(p.shape[1:])
    d[:-1] += p[0, :-1]
    d[1:] -= p[0, :-1]
    d[:, :-1] += p[1, :, :-1]
    d[:, 1:] -= p[1, :, :-1]
    d[:, :, :-1] += p[2, :, :, :-1] * z_weight
    d[:, :, 1:] -= p[2, :, :, :-1] * z_weight
    return d


def _prox_tv(v: np.ndarray, gamma: float, zw: float, dual: np.ndarray,
             n_inner: int = 10) -> np.ndarray:
    """Chambolle dual projection for prox of gamma * ||grad .||_1 under x >= 0.

    `dual` is warm-started across outer iterations and mutated in place, which is
    what makes ten inner steps enough.
    """
    if gamma <= 0.0:
        return np.maximum(v, 0.0)
    step = 1.0 / (4.0 * (2.0 + zw * zw))
    for _ in range(n_inner):
        u = np.maximum(v + _div3(dual, zw), 0.0)
        np.clip(dual + step * _grad3(u, zw), -gamma, gamma, out=dual)
    return np.maximum(v + _div3(dual, zw), 0.0)


def sirt_tv(fwd: ForwardModel, data: FitData, rc: Reconstruction
            ) -> tuple[np.ndarray, dict]:
    """SIRT with a per-iteration anisotropic-TV proximal (denoising) step.

    tv_alpha is a fraction of the reconstructed scale (x's p95), so it transfers
    across datasets instead of needing a retune per campaign. Runs the full
    budget and returns the best-chi2 iterate: the TV step keeps it from
    overfitting the way plain SIRT does, so there is no discrepancy stop.
    """
    A, lam, w = fwd.A, data.lam, data.w
    shape = fwd.grid.shape
    n_pos = len(fwd.rows.position_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_pos)
    row_inv, col_inv = _scalings(A, w)
    n_used = max(int(np.count_nonzero(w)), 1)
    dual = np.zeros((3,) + shape)

    history: list[float] = []
    best = (float("inf"), x.copy(), c.copy())
    for k in range(rc.n_iter):
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        c = c + _update_offsets(resid, w, data.rows.pos_of_row, n_pos)
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if chi2 < best[0]:
            best = (chi2, x.copy(), c.copy())
        if k % 20 == 0:
            history.append(chi2)

        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
        scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 0.0
        gamma = rc.tv_alpha * max(scale, 1e-9)
        x = _prox_tv(x.reshape(shape), gamma, rc.tv_z_weight, dual).ravel()

    history.append(best[0])
    return best[1], {"offsets": _named(best[2], fwd.rows.position_ids),
                     "chi2_history": history,
                     "best_chi2": float(best[0]),
                     "n_iter_used": rc.n_iter}


SOLVERS = {"sirt": sirt, "tv": sirt_tv}


def solve(fwd: ForwardModel, data: FitData, rc: Reconstruction
          ) -> tuple[np.ndarray, dict]:
    if rc.algorithm not in SOLVERS:
        raise ValueError(f"unknown algorithm {rc.algorithm!r}; have {sorted(SOLVERS)}")
    return SOLVERS[rc.algorithm](fwd, data, rc)
