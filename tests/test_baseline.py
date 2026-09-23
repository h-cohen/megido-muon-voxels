import numpy as np
import pytest
import yaml

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution, solve_baseline
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.sky import make_sky_grid


@pytest.fixture
def cfg(tmp_path):
    """Two tilts at one position plus a second position, mirroring the campaign."""
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(tmp_path),
        "frame": {"origin": "P0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 500},
        "exposures": [
            {"id": "P0", "runs": "DET100001-DET100001",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
            {"id": "T20", "runs": "DET100002-DET100002",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
            {"id": "P1", "runs": "DET100003-DET100003",
             "pose": {"x": 2.2, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
        ],
    }))
    return load_site_config(p)


def _flat_grid(cfg, n=30, counts=400):
    edges = np.linspace(-1.25, 1.25, n + 1)
    return AnalysisGrid(edges=edges,
                        counts={e.id: np.full((n, n), counts, np.int64)
                                for e in cfg.exposures})


def test_solution_has_every_block(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    assert isinstance(sol, BaselineSolution)
    assert sol.coeffs.shape == (make_smooth_basis().n_coeff,)
    assert set(sol.norms) == {"P0", "T20", "P1"}
    assert set(sol.opacity) == {"pos0", "pos1"}


def test_positions_group_by_pose_not_by_exposure(cfg):
    """P0 and T20 share a position; P1 does not. Opacity is per POSITION."""
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    assert len(sol.opacity) == 2, "three exposures, two positions"


def test_likelihood_decreases_monotonically(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=8, fit_flux_index=False)
    h = np.array(sol.nll_history)
    assert len(h) >= 2
    assert np.all(np.diff(h) <= 1e-6), f"NLL rose: {h}"


def test_response_is_positive_inside_the_acceptance_and_zero_outside(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    b = sol.response()
    tx, ty = sol.grid.tan_mesh()
    inside = np.hypot(tx, ty) < 0.5
    assert np.all(b[inside] > 0)
    assert np.all(b[np.abs(tx) > 1.25] == 0)


def test_prediction_matches_observed_total_after_convergence(cfg):
    """The normalization update is a closed form, so totals must match closely
    over the bins the model actually covers.

    Some detector bins near the acceptance edge have no closed-form target at
    all: under a 20 degree tilt at an oblique azimuth (241 deg, not aligned to
    either axis), the grazing corner of the acceptance region maps to sky
    tangents that fall outside the default sky grid (confirmed numerically:
    the corner (tan_x, tan_y) = (1.208, -1.208) under this pose lands at sky
    tangent (-3.37, -1.34), far past the +-1.75 default span). Those bins are
    correctly excluded from `terms[eid].live` and contribute 0 to `.predict()`
    by design (see `test_unconstrained_sky_bins_are_nan_not_zero`), so the
    comparison must be restricted to the live domain the model is actually fit
    on, not the full synthetic grid (which fills every bin with equal counts
    regardless of true acceptance, unlike real data where that corner's
    acceptance is ~0.01% of the peak and contributes negligible counts)."""
    grid = _flat_grid(cfg)
    sol = solve_baseline(grid, cfg, n_iter=15, fit_flux_index=False)
    for eid, observed in grid.counts.items():
        predicted = sol.predict(eid)
        live = sol.terms[eid].live
        assert predicted[live].sum() == pytest.approx(observed[live].sum(), rel=0.02)


def test_opacity_image_is_square_and_matches_the_sky_grid(cfg):
    sky = make_sky_grid(t_max=1.5, n_bins=30)
    sol = solve_baseline(_flat_grid(cfg), cfg, sky=sky, n_iter=3, fit_flux_index=False)
    img = sol.opacity_image("pos0")
    assert img.shape == (30, 30)


def test_unconstrained_sky_bins_are_nan_not_zero(cfg):
    """A sky bin no exposure sees must be reported as unknown, never as zero
    opacity — zero would read as 'no rock there', which is a claim."""
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    img = sol.opacity_image("pos1")
    assert np.isnan(img).any()
    assert np.isfinite(img).any()


def test_flux_index_is_fitted_when_asked(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=5,
                         flux_index=2.0, fit_flux_index=True)
    assert 0.5 <= sol.flux_index <= 5.0


def test_solution_round_trips_through_disk(cfg, tmp_path):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    path = tmp_path / "baseline.npz"
    sol.save(path)
    back = BaselineSolution.load(path)
    assert np.allclose(back.coeffs, sol.coeffs)
    assert back.norms == pytest.approx(sol.norms)
    assert back.flux_index == pytest.approx(sol.flux_index)
    for pid in sol.opacity:
        assert np.allclose(back.opacity[pid], sol.opacity[pid], equal_nan=True)


def test_normalizations_do_not_collapse_under_long_runs(cfg):
    """The norm/opacity degeneracy is exact; without a fixed gauge the solve
    slides along it and the norms run to zero while the fit degrades."""
    grid = _flat_grid(cfg)
    short = solve_baseline(grid, cfg, n_iter=20, tol=0.0)
    long = solve_baseline(grid, cfg, n_iter=2000, tol=0.0)
    for group, value in long.norms.items():
        assert value > 0.1 * short.norms[group], (
            f"norm {group} fell from {short.norms[group]:.4g} to {value:.4g}"
        )


def test_opacity_gauge_is_pinned(cfg):
    """Each position's opacity has a fixed zero point, so runs are comparable."""
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=200, tol=0.0)
    for pid, lam in sol.opacity.items():
        finite = lam[np.isfinite(lam)]
        if finite.size:
            assert abs(np.median(finite)) < 1e-6, f"{pid} median {np.median(finite)}"


def test_longer_runs_do_not_worsen_the_fit(cfg):
    """Coordinate descent must not go backwards. It did, by 3700 units on real
    data, because each update was optimal given the others while the pair was
    free to drift."""
    grid = _flat_grid(cfg)
    short = solve_baseline(grid, cfg, n_iter=50, tol=0.0)
    long = solve_baseline(grid, cfg, n_iter=1500, tol=0.0)
    assert long.nll_history[-1] <= short.nll_history[-1] + 1e-6


def test_position_sky_counts_matches_build_terms(cfg):
    """position_sky_counts must be the same per-position, per-sky-bin summed
    live counts that _build_terms uses internally (before the MIN_SKY_COUNTS
    cut) -- it is the public helper `hillside --run` uses to rebuild
    heteroscedastic Poisson weights without duplicating the live-mask logic."""
    from megido.baseline import position_sky_counts
    from megido.detector import DetectorGeometry
    from megido.sky import make_sky_grid
    from megido.simbaseline import make_synthetic_scene

    grid = make_synthetic_scene(cfg).grid
    sky = make_sky_grid()

    psc = position_sky_counts(grid, cfg)
    assert set(psc) == {"pos0", "pos1"}
    for pid, arr in psc.items():
        assert arr.shape == (sky.flat_size,)
        assert arr.sum() > 0

    # Independent recompute (does NOT call position_sky_counts): summing live
    # counts per position, per sky bin, by hand. A broken helper -- double count,
    # dropped live mask, or wrong position grouping -- must fail this equality.
    from megido.acceptance import geometric_acceptance
    from megido.baseline import position_ids
    from megido.sky import detector_to_sky

    geom = DetectorGeometry.megiddo()
    tx, ty = grid.tan_mesh()
    acc = geometric_acceptance(tx, ty, geom)
    pids = position_ids(cfg)
    expected = {pid: np.zeros(sky.flat_size) for pid in psc}
    for eid, counts in grid.counts.items():
        pose = cfg.exposure(eid).pose
        sx, sy, on_sky = detector_to_sky(tx, ty, pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acc > 0) & on_sky & in_grid
        np.add.at(expected[pids[eid]], flat[live], counts[live].astype(float))
    for pid in psc:
        assert np.allclose(psc[pid], expected[pid])
