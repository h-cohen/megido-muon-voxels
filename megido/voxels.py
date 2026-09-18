"""The world-frame voxel lattice the inversion solves on.

World frame = the sky frame Phase 2 already rotated into: z up, lengths in
METRES, origin at the P0 detector (configs/megido.yaml `frame.origin`). Phase 1
and megido.detector work in centimetres; nothing in this module does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.config import Volume


@dataclass(frozen=True)
class VoxelGrid:
    origin: tuple          # (x0, y0, z0) of the grid CORNER, metres
    spacing: float         # cubic voxel edge, metres
    shape: tuple           # (nx, ny, nz)

    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def axis_centers(self, axis: int) -> np.ndarray:
        return self.origin[axis] + (np.arange(self.shape[axis]) + 0.5) * self.spacing

    def extent(self, axis: int) -> tuple[float, float]:
        return self.origin[axis], self.origin[axis] + self.shape[axis] * self.spacing

    def key(self) -> str:
        return f"{self.origin}-{self.spacing}-{self.shape}"


def auto_grid(vol: Volume, origins: dict[str, tuple[float, float, float]],
              t_reach: float) -> VoxelGrid:
    """Lattice covering the union of every position's ray footprint.

    `t_reach` is the largest |tangent| that carries a constrained measurement —
    supplied by the caller from the live rows, NOT the sky grid's nominal edge.
    The sky grid runs to |t| = 2.5 to catch stray counts; sizing the voxel grid
    by that would quadruple it to hold bins nothing constrains.
    """
    if vol.z_max_m <= vol.z_min_m:
        raise ValueError(f"z_max_m ({vol.z_max_m}) must exceed z_min_m ({vol.z_min_m})")
    if not origins:
        raise ValueError("auto_grid needs at least one position origin")

    z0, z1 = float(vol.z_min_m), float(vol.z_max_m)
    if vol.xy_m is not None:
        (x0, x1), (y0, y1) = vol.xy_m
    else:
        # A ray of tangent t leaving (px, py, pz) is at px + t*(z1 - pz) by the
        # top of the grid; the bundle spreads half an aperture either side.
        pad = 0.5 * _APERTURE_PAD_M
        xs, ys = [], []
        for px, py, pz in origins.values():
            reach = t_reach * max(z1 - pz, 0.0) + pad
            xs += [px - reach, px + reach]
            ys += [py - reach, py + reach]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    sp = float(vol.spacing_m)
    nx = max(1, int(np.ceil((x1 - x0) / sp)))
    ny = max(1, int(np.ceil((y1 - y0) / sp)))
    nz = max(1, int(np.ceil((z1 - z0) / sp)))
    return VoxelGrid(origin=(float(x0), float(y0), z0), spacing=sp, shape=(nx, ny, nz))


# Half-width padding for the ray bundle. Imported here rather than taken from
# DetectorGeometry so voxels.py stays free of the centimetre-based module; the
# value is asserted equal to geom.aperture_m in tests/test_voxels.py.
_APERTURE_PAD_M = 0.384
