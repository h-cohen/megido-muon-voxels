"""The measurement vector the voxel inversion consumes, and its row ordering.

Phase 2 solved for lambda_p(sky direction): the opacity integrated along a ray
leaving detector position p in a world-frame direction. That IS the tomographic
measurement, so Phase 3 needs only to decide which of those directions are
trustworthy enough to invert, in what order, and with what weight.

Rows are (position, sky bin) pairs. NOT (exposure, detector bin): exposures at
the same spot with different tilts already had their counts combined by Phase 2
into one opacity map per position, and it is the position — not the tilt — that
sets where the ray starts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from megido.baseline import BaselineSolution, position_ids
from megido.config import SiteConfig


def position_origins(cfg: SiteConfig) -> dict[str, tuple[float, float, float]]:
    """World-frame ray origin of each position id, in metres.

    Mirrors megido.baseline.position_ids, which assigns pos0, pos1, ... in
    config order by TRANSLATION only. Two tilts at one spot are one position and
    share one origin.
    """
    ids = position_ids(cfg)
    out: dict[str, tuple[float, float, float]] = {}
    for exp in cfg.exposures:
        pid = ids[exp.id]
        origin = (float(exp.pose.x), float(exp.pose.y), float(exp.pose.z))
        if pid in out and out[pid] != origin:
            raise ValueError(f"position {pid} has two origins: {out[pid]} and {origin}")
        out[pid] = origin
    return out


@dataclass(frozen=True)
class RowIndex:
    """Which (position, sky direction) pairs are inverted, and in what order."""
    position_ids: tuple[str, ...]
    pos_of_row: np.ndarray      # [n_rows] index into position_ids
    sx: np.ndarray              # [n_rows] world-frame tangent, x
    sy: np.ndarray              # [n_rows] world-frame tangent, y
    sky_flat: np.ndarray        # [n_rows] flat index into the Phase 2 sky grid

    @property
    def n_rows(self) -> int:
        return int(self.pos_of_row.size)

    def mask_for(self, pid: str) -> np.ndarray:
        return self.pos_of_row == self.position_ids.index(pid)

    def t_reach(self) -> float:
        """Largest |tangent| carrying a measurement — what sizes the voxel grid."""
        return float(max(np.abs(self.sx).max(), np.abs(self.sy).max()))

    def directions(self) -> np.ndarray:
        """[n_rows, 3] unit world-frame direction of each row."""
        d = np.stack([self.sx, self.sy, np.ones_like(self.sx)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def key(self) -> str:
        h = hashlib.sha256()
        h.update(",".join(self.position_ids).encode())
        for arr in (self.pos_of_row, self.sx, self.sy, self.sky_flat):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class FitData:
    lam: np.ndarray             # [n_rows] measured optical depth
    w: np.ndarray               # [n_rows] 1/sigma^2, zero on excluded rows
    rows: RowIndex

    def restricted(self, keep: np.ndarray) -> "FitData":
        """Same rows, weights zeroed outside `keep`. The layout is shared so one
        cached system matrix serves the full fit and every holdout fit."""
        return FitData(lam=self.lam, w=np.where(keep, self.w, 0.0), rows=self.rows)


def build_fit_data(sol: BaselineSolution, cfg: SiteConfig, *,
                   sigma: dict[str, np.ndarray] | None = None,
                   max_tan: float = 1.25,
                   transparent_quantile: float = 0.05) -> FitData:
    """Flatten a baseline solution into (lam, w) over live (position, sky) rows.

    `transparent_quantile` is the gauge convention: which quantile of each
    position's opacity map is called open sky. It is a CHOICE, not a
    measurement — Phase 2 leaves one flat direction per position (spec section
    6.4) — and megido.voxuncert.systematic_map varies it to measure how much of
    the answer that choice is responsible for.

    `max_tan` caps the sky tangent. The Phase 2 sky grid deliberately runs to
    |t| = 2.5 so the tilted exposures lose almost no counts off its edge, but
    those far corners hold a handful of tracks and would stretch the voxel grid
    by a factor of four in each lateral axis. 1.25 matches the analysis grid.

    `sigma` is the per-sky-bin bootstrap uncertainty from
    megido.validate2.opacity_uncertainty. Without it every live row gets weight
    one, which is a statement that all directions are equally trusted — untrue,
    and the reason the reconstruct CLI computes sigma by default.
    """
    origins = position_origins(cfg)
    centers = sol.sky.centers
    n = sol.sky.n_bins
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    all_sx = centers[ii].ravel()
    all_sy = centers[jj].ravel()

    ordered = tuple(pid for pid in sorted(sol.opacity) if pid in origins)
    pos_idx, sx, sy, flat, lam, w = [], [], [], [], [], []

    for k, pid in enumerate(ordered):
        lam_p = sol.normalized_opacity(pid, transparent_quantile=transparent_quantile)
        live = np.isfinite(lam_p)
        live &= (np.abs(all_sx) <= max_tan) & (np.abs(all_sy) <= max_tan)

        if sigma is not None:
            s = np.asarray(sigma.get(pid, np.full(lam_p.shape, np.nan)), dtype=np.float64)
            # A zero or non-finite sigma is not "perfectly measured"; it means the
            # bootstrap said nothing about this bin. Drop it rather than hand the
            # solver an infinite weight that would pin the fit to one direction.
            live &= np.isfinite(s) & (s > 0)
            weights = np.zeros_like(lam_p)
            weights[live] = 1.0 / s[live] ** 2
        else:
            weights = np.where(live, 1.0, 0.0)

        idx = np.nonzero(live)[0]
        pos_idx.append(np.full(idx.size, k, dtype=np.int64))
        sx.append(all_sx[idx])
        sy.append(all_sy[idx])
        flat.append(idx.astype(np.int64))
        lam.append(lam_p[idx])
        w.append(weights[idx])

    rows = RowIndex(
        position_ids=ordered,
        pos_of_row=np.concatenate(pos_idx) if pos_idx else np.zeros(0, dtype=np.int64),
        sx=np.concatenate(sx) if sx else np.zeros(0),
        sy=np.concatenate(sy) if sy else np.zeros(0),
        sky_flat=np.concatenate(flat) if flat else np.zeros(0, dtype=np.int64),
    )
    if rows.n_rows == 0:
        raise ValueError(
            "no constrained sky directions: every opacity value is non-finite, "
            f"or none survives max_tan={max_tan}")
    return FitData(lam=np.concatenate(lam), w=np.concatenate(w), rows=rows)
