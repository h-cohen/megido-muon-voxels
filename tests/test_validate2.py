import numpy as np
import pytest
import yaml

from megido.baseline import solve_baseline
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.simbaseline import make_synthetic_scene
from megido.sky import make_sky_grid
from megido.validate2 import (Check2, format_report2, leave_one_out,
                              nll_per_bin_check, opacity_uncertainty)


@pytest.fixture
def cfg(tmp_path):
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


@pytest.fixture
def scene(cfg):
    sky = make_sky_grid(t_max=1.6, n_bins=20)
    basis = make_smooth_basis(n_per_axis=4)
    return make_synthetic_scene(cfg, n_bins=24, sky=sky, basis=basis, seed=5), sky, basis


def test_uncertainty_is_positive_where_the_sky_is_seen(cfg, scene):
    s, sky, basis = scene
    sigma = opacity_uncertainty(s.grid, cfg, n_replicas=6, sky=sky, basis=basis,
                                n_iter=8, fit_flux_index=False)
    for lam_sigma in sigma.values():
        seen = np.isfinite(lam_sigma)
        assert seen.any()
        assert np.all(lam_sigma[seen] >= 0)


def test_uncertainty_shrinks_with_more_counts(cfg):
    """Poisson: four times the exposure should roughly halve the error."""
    sky = make_sky_grid(t_max=1.6, n_bins=20)
    basis = make_smooth_basis(n_per_axis=4)
    thin = make_synthetic_scene(cfg, n_bins=24, sky=sky, basis=basis,
                                scale=5.0e4, seed=6)
    thick = make_synthetic_scene(cfg, n_bins=24, sky=sky, basis=basis,
                                 scale=2.0e5, seed=6)
    kw = dict(sky=sky, basis=basis, n_iter=8, fit_flux_index=False)
    s_thin = opacity_uncertainty(thin.grid, cfg, n_replicas=6, **kw)
    s_thick = opacity_uncertainty(thick.grid, cfg, n_replicas=6, **kw)

    pid = sorted(s_thin)[0]
    a = np.nanmedian(s_thin[pid])
    b = np.nanmedian(s_thick[pid])
    assert b < a, f"more counts did not reduce the uncertainty ({b:.4f} vs {a:.4f})"


def test_leave_one_out_returns_a_check_per_exposure(cfg, scene):
    s, sky, basis = scene
    checks = leave_one_out(s.grid, cfg, sky=sky, basis=basis,
                           n_iter=8, fit_flux_index=False)
    assert len(checks) == 3
    assert all(isinstance(c, Check2) for c in checks)
    assert {c.name for c in checks} == {"loo.P0", "loo.T20", "loo.P1"}


def test_the_sole_exposure_at_a_position_is_skipped_not_silently_passed(cfg, scene):
    """P1 alone holds pos1. Holding it out leaves nothing to predict against,
    and saying so is honest; quietly scoring it would not be."""
    s, sky, basis = scene
    checks = {c.name: c for c in leave_one_out(s.grid, cfg, sky=sky, basis=basis,
                                               n_iter=8, fit_flux_index=False)}
    assert "skipped" in checks["loo.P1"].expected
    assert np.isnan(checks["loo.P1"].measured)
    assert not np.isnan(checks["loo.P0"].measured)


def test_leave_one_out_passes_on_a_self_consistent_scene(cfg, scene):
    """Synthetic data obeys the model exactly, so held-out prediction must work."""
    s, sky, basis = scene
    checks = leave_one_out(s.grid, cfg, sky=sky, basis=basis,
                           n_iter=12, fit_flux_index=False)
    scored = [c for c in checks if not np.isnan(c.measured)]
    assert scored, "at least one exposure must be genuinely scoreable"
    assert all(c.passed for c in scored), [c.detail for c in scored]


def test_nll_per_bin_is_near_one_for_a_correct_model(cfg, scene):
    s, sky, basis = scene
    sol = solve_baseline(s.grid, cfg, sky=sky, basis=basis,
                         n_iter=20, fit_flux_index=False)
    check = nll_per_bin_check(sol, s.grid)
    assert check.passed, check.detail
    assert 0.2 < check.measured < 3.0


def test_format_report2_renders_pass_and_fail(cfg):
    checks = [Check2("a", True, 1.0, "x", "ok"), Check2("b", False, 2.0, "y", "bad")]
    text = format_report2(checks)
    assert "PASS" in text and "FAIL" in text
    assert "1/2" in text
