"""Open-sky-ratio Phase 2 (megido.skyref) for campaigns that have a sky run."""
from dataclasses import replace

import numpy as np
import pytest

from megido.acceptance import geometric_acceptance
from megido.angular import AnalysisGrid
from megido.config import load_site_config
from megido.detector import DetectorGeometry
from megido.skyref import skyref_sigma, solve_skyref
from megido.sky import detector_to_sky, make_sky_grid

CFG = "configs/cafeteria.yaml"


def _scene(seed=0, scale=0.3, level=1e5):
    """Sky run = response x flux x level; position = scale x sky x exp(-lambda).

    The response carries strong per-bin structure (a dead-ish bar pattern),
    so a solver that ignored the sky run would print that structure into the
    opacity -- the recovery gate below fails for it.
    """
    cfg = load_site_config(CFG)
    edges = np.linspace(-1.25, 1.25, 51)
    grid0 = AnalysisGrid(edges=edges, counts={})
    tx, ty = grid0.tan_mesh()
    geom = DetectorGeometry.for_site(cfg)
    rng = np.random.default_rng(seed)
    response = geometric_acceptance(tx, ty, geom) * rng.uniform(0.3, 1.0, tx.shape)
    flux = (1.0 / np.sqrt(1 + tx**2 + ty**2)) ** 2
    n_sky = response * flux * level

    lam_true = 0.8 * np.exp(-((tx - 0.2) ** 2 + (ty + 0.1) ** 2) / 0.1)   # a lump
    counts = {"SKY": np.round(n_sky).astype(np.int64)}
    for e in cfg.exposures:
        counts[e.id] = np.round(scale * n_sky * np.exp(-lam_true)).astype(np.int64)
    return cfg, AnalysisGrid(edges=edges, counts=counts), lam_true


def test_recovers_opacity_up_to_a_constant_through_the_response():
    cfg, grid, lam_true = _scene()
    sol = solve_skyref(grid, cfg)
    sky = make_sky_grid()
    tx, ty = grid.tan_mesh()
    # pos0 has tilt 0, az 0: detector bin (i, j) -> the sky bin at the same tangent
    sx, sy, _ = detector_to_sky(tx, ty, cfg.exposure("pos0").pose)
    flat, ok = sky.bin_index(sx, sy)
    lam = sol.opacity["pos0"][flat[ok]]
    seen = np.isfinite(lam)
    diff = lam[seen] - lam_true[ok][seen]
    assert seen.sum() > 500
    assert np.std(diff) < 0.01          # exact up to integer rounding of counts
    assert np.corrcoef(lam[seen], lam_true[ok][seen])[0, 1] > 0.999


def test_scale_between_runs_is_absorbed_by_the_gauge():
    cfg, grid_a, _ = _scene(scale=0.3)
    _, grid_b, _ = _scene(scale=0.05)
    a = solve_skyref(grid_a, cfg).normalized_opacity("pos0")
    b = solve_skyref(grid_b, cfg).normalized_opacity("pos0")
    seen = np.isfinite(a) & np.isfinite(b)
    assert np.nanmax(np.abs(a[seen] - b[seen])) < 0.05


def test_sparse_sky_bins_are_not_constrained():
    cfg, grid, _ = _scene(level=30.0)
    sol = solve_skyref(grid, cfg, min_sky=25.0)
    n_sky = grid.counts["SKY"]
    sky = make_sky_grid()
    tx, ty = grid.tan_mesh()
    sx, sy, _ = detector_to_sky(tx, ty, cfg.exposure("pos0").pose)
    flat, ok = sky.bin_index(sx, sy)
    sparse = ok & (n_sky < 25)
    assert sparse.any()
    assert np.isnan(sol.opacity["pos0"][flat[sparse]]).all()


def test_poisson_sigma_matches_the_ratio_error():
    cfg, grid, _ = _scene(level=400.0)
    sig = skyref_sigma(grid, cfg)["pos0"]
    sky = make_sky_grid()
    tx, ty = grid.tan_mesh()
    sx, sy, _ = detector_to_sky(tx, ty, cfg.exposure("pos0").pose)
    flat, ok = sky.bin_index(sx, sy)
    i = np.argmax(np.where(ok, grid.counts["SKY"], 0))
    n_p, n_s = grid.counts["pos0"].flat[i], grid.counts["SKY"].flat[i]
    assert sig[flat.flat[i]] == pytest.approx(np.sqrt(1 / n_p + 1 / n_s))


def test_needs_a_sky_reference():
    cfg, grid, _ = _scene()
    with pytest.raises(ValueError, match="sky_reference"):
        solve_skyref(grid, replace(cfg, sky_reference=None))
