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
