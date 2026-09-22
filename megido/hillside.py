"""Data-driven hillside (overburden surface) extraction from the muon flux.

Each Phase-2 sky-opacity pixel is an in-rock path length up to a density
scale; the exit point p + L*d̂ is a point on the hill surface z=H(x,y). The
P0/P1 parallax pins the scale. See docs/superpowers/specs/...-design.md §11.
The absolute height is weakly constrained by a single 2.2 m baseline; the
lateral shape is not. Every result carries that band.
"""
from __future__ import annotations

import numpy as np


def ray_dirs(tan_xy: np.ndarray) -> np.ndarray:
    """N×2 (tx,ty) -> N×3 unit sky-frame ray directions (tx,ty,1)/|.|."""
    tan_xy = np.asarray(tan_xy, dtype=np.float64)
    v = np.column_stack([tan_xy[:, 0], tan_xy[:, 1], np.ones(len(tan_xy))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def surface_points(p: np.ndarray, L: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Exit points p + L*d̂ (N×3) for detector position p and path lengths L."""
    p = np.asarray(p, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    return p[None, :] + L[:, None] * dirs


def grid_surface(points: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    """Bin points into (x,y) cells; return (H mean-z NaN-if-empty, count)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    nx, ny = len(xedges) - 1, len(yedges) - 1
    ix = np.clip(np.digitize(x, xedges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, yedges) - 1, 0, ny - 1)
    inside = (x >= xedges[0]) & (x <= xedges[-1]) & (y >= yedges[0]) & (y <= yedges[-1])
    zsum = np.zeros((nx, ny)); cnt = np.zeros((nx, ny))
    np.add.at(zsum, (ix[inside], iy[inside]), z[inside])
    np.add.at(cnt, (ix[inside], iy[inside]), 1.0)
    H = np.full((nx, ny), np.nan)
    nz = cnt > 0
    H[nz] = zsum[nz] / cnt[nz]
    return H, cnt
