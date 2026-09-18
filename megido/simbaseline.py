"""Synthetic scenes with a known baseline and a known sky, for gating the solve.

The solve estimates the detector response and the rock's absorption at the same
time, from data that contains only their product. That is only credible if it
can recover a known answer, so this generator injects both and the tests assert
both come back — including a negative control where the sky is flat and the
solve must not invent structure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.acceptance import geometric_acceptance
from megido.angular import AnalysisGrid
from megido.baseline import position_ids
from megido.basis import SmoothBasis, make_smooth_basis
from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.sky import SkyGrid, detector_to_sky, flux_shape, make_sky_grid


@dataclass(frozen=True)
class SyntheticScene:
    grid: AnalysisGrid
    true_coeffs: np.ndarray
    true_opacity: dict[str, np.ndarray]
    true_norms: dict[str, float]
    true_flux_index: float


def make_synthetic_scene(cfg: SiteConfig, *, n_bins: int = 30,
                         sky: SkyGrid | None = None,
                         basis: SmoothBasis | None = None,
                         # geometric_acceptance peaks near 1475 cm^2, so this
                         # lands counts in the real data's range (median
                         # 176-429 per occupied core bin).
                         scale: float = 10.0, seed: int = 0,
                         opacity_amplitude: float = 0.4,
                         flux_index: float = 2.0) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    geom = DetectorGeometry.megiddo()
    sky = sky or make_sky_grid()
    basis = basis or make_smooth_basis()

    edges = np.linspace(-1.25, 1.25, n_bins + 1)
    grid = AnalysisGrid(edges=edges, counts={})
    tx, ty = grid.tan_mesh()
    acceptance = geometric_acceptance(tx, ty, geom)

    true_coeffs = rng.normal(0.0, 0.25, basis.n_coeff)
    response = acceptance * np.exp(basis.evaluate(true_coeffs, tx, ty))

    # A smooth, deterministic sky feature per position, so the test asserts
    # against structure rather than noise.
    exposure_position = position_ids(cfg)
    positions = sorted(set(exposure_position.values()))
    sky_cx, sky_cy = np.meshgrid(sky.centers, sky.centers, indexing="ij")
    true_opacity: dict[str, np.ndarray] = {}
    for k, pid in enumerate(positions):
        blob = np.exp(-0.5 * (((sky_cx - 0.25 * (k + 1)) / 0.45) ** 2
                              + ((sky_cy + 0.15 * (k + 1)) / 0.45) ** 2))
        true_opacity[pid] = (opacity_amplitude * blob).reshape(-1)

    true_norms = {e.norm_group: scale * (1.0 + 0.1 * i)
                  for i, e in enumerate(cfg.exposures)}

    counts: dict[str, np.ndarray] = {}
    for exp in cfg.exposures:
        sx, sy, on_sky = detector_to_sky(tx, ty, exp.pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acceptance > 0) & on_sky & in_grid

        lam = np.zeros_like(tx)
        lam[live] = true_opacity[exposure_position[exp.id]][flat[live]]

        mu = np.zeros_like(tx)
        mu[live] = (true_norms[exp.norm_group] * response[live]
                    * flux_shape(sx[live], sy[live], flux_index)
                    * np.exp(-lam[live]))
        counts[exp.id] = rng.poisson(np.clip(mu, 0.0, None)).astype(np.int64)

    return SyntheticScene(
        grid=AnalysisGrid(edges=edges, counts=counts),
        true_coeffs=true_coeffs,
        true_opacity=true_opacity,
        true_norms=true_norms,
        true_flux_index=flux_index,
    )
