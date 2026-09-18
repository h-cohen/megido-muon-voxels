import numpy as np
import pytest
import yaml

from megido.baseline import solve_baseline
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.simbaseline import make_synthetic_scene
from megido.sky import make_sky_grid


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
            {"id": "T20a", "runs": "DET100002-DET100002",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
            {"id": "T20b", "runs": "DET100003-DET100003",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
            {"id": "P1", "runs": "DET100004-DET100004",
             "pose": {"x": 2.2, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
        ],
    }))
    return load_site_config(p)


def test_scene_produces_counts_for_every_exposure(cfg):
    scene = make_synthetic_scene(cfg, seed=0)
    assert set(scene.grid.counts) == {"P0", "T20a", "T20b", "P1"}
    for v in scene.grid.counts.values():
        assert v.dtype == np.int64
        assert v.sum() > 10_000


def test_counts_are_higher_where_opacity_is_lower(cfg):
    """Sanity: absorption must actually suppress counts."""
    scene = make_synthetic_scene(cfg, opacity_amplitude=1.5, seed=1)
    p0 = scene.grid.counts["P0"].astype(float)
    n = p0.shape[0]
    assert p0[n // 2, n // 2] > 0


def test_solver_recovers_the_injected_detector_response(cfg):
    """THE GATE, method version. High counts, so it asks whether the solve
    is correct, not what campaign statistics deliver."""
    sky = make_sky_grid(t_max=1.6, n_bins=24)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 scale=1000.0, seed=2)

    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=5000, tol=0.0, fit_flux_index=False,
                         flux_index=scene.true_flux_index)

    tx, ty = scene.grid.tan_mesh()
    inside = np.hypot(tx, ty) < 0.8
    truth = basis.evaluate(scene.true_coeffs, tx, ty)[inside]
    got = basis.evaluate(sol.coeffs, tx, ty)[inside]

    # only shape is identifiable; a constant offset trades against the norms
    truth = truth - truth.mean()
    got = got - got.mean()
    assert np.corrcoef(truth, got)[0, 1] > 0.9
    assert np.std(got - truth) < 0.3 * np.std(truth) + 0.05


def test_solver_recovers_opacity_in_the_high_count_limit(cfg):
    """Method check: with noise removed, does the solve find the injected sky?"""
    sky = make_sky_grid(t_max=1.8, n_bins=36)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 scale=1000.0, opacity_amplitude=0.6, seed=3)
    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=5000, tol=0.0, flux_index=2.0, fit_flux_index=False)

    pid = sorted(sol.opacity)[0]
    truth, got = scene.true_opacity[pid], sol.opacity[pid]
    both = np.isfinite(truth) & np.isfinite(got)
    corr = np.corrcoef(truth[both] - truth[both].mean(),
                       got[both] - got[both].mean())[0, 1]
    assert corr > 0.90, f"high-count opacity correlation {corr:.3f}"


def test_opacity_recovery_at_campaign_statistics(cfg):
    """Reality check: what this campaign's actual counts deliver.

    scale=10 puts roughly 400 counts in a core bin, matching the real data
    (medians 176-429). Over the sky bins that are actually constrained, recovery
    is close to the high-count limit — the earlier much weaker figure came from
    scoring the fit on bins too sparsely observed to constrain anything, which
    MIN_SKY_COUNTS now excludes.
    """
    sky = make_sky_grid(t_max=1.8, n_bins=36)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 scale=10.0, opacity_amplitude=0.6, seed=3)
    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=5000, tol=0.0, flux_index=2.0, fit_flux_index=False)

    pid = sorted(sol.opacity)[0]
    truth, got = scene.true_opacity[pid], sol.opacity[pid]
    both = np.isfinite(truth) & np.isfinite(got)
    corr = np.corrcoef(truth[both] - truth[both].mean(),
                       got[both] - got[both].mean())[0, 1]
    assert corr > 0.75, f"campaign-statistics opacity correlation {corr:.3f}"


def test_a_flat_sky_is_recovered_as_flat_in_the_high_count_limit(cfg):
    """Negative control: with noise removed, the solve must not manufacture sky
    structure from nothing. Converged, the residual is std 0.046 — small enough
    that the earlier apparent "leakage floor" near 0.12 was an artefact of
    stopping the solver too early, not a property of the method.
    """
    sky = make_sky_grid(t_max=1.8, n_bins=36)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 scale=1000.0, opacity_amplitude=0.0, seed=4)
    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=5000, tol=0.0, flux_index=2.0, fit_flux_index=False)

    lam = sol.opacity[sorted(sol.opacity)[0]]
    lam = lam[np.isfinite(lam)]
    assert np.std(lam) < 0.10, f"manufactured sky structure, std={np.std(lam):.3f}"


def test_flat_sky_leakage_at_campaign_statistics_is_bounded(cfg):
    """Reality check on the negative control: how much structure the solve
    manufactures at campaign statistics. This is the floor below which no
    feature in the real sky map should be believed.
    """
    sky = make_sky_grid(t_max=1.8, n_bins=36)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 scale=10.0, opacity_amplitude=0.0, seed=4)
    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=5000, tol=0.0, flux_index=2.0, fit_flux_index=False)

    lam = sol.opacity[sorted(sol.opacity)[0]]
    lam = lam[np.isfinite(lam)]
    assert np.std(lam) < 0.15, f"campaign-statistics leakage std={np.std(lam):.3f}"


def test_flux_index_is_not_identifiable_which_is_why_it_is_fixed(cfg):
    """A structural property of the method, not a solver defect.

    lambda(s) is free per sky bin and Phi_n(s) is a function of the same sky
    direction, so only their product is determined and the likelihood is nearly
    flat in the index. This test exists so that if someone later makes the index
    identifiable — by constraining lambda, or adding an exposure geometry that
    breaks the tie — this failure tells them the assumption changed.
    """
    sky = make_sky_grid(t_max=1.8, n_bins=36)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 flux_index=3.0, opacity_amplitude=0.3, seed=9)

    nlls = []
    for index in (1.5, 3.0, 4.5):
        sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                             n_iter=200, tol=0.0,
                             flux_index=index, fit_flux_index=False)
        nlls.append(sol.nll_history[-1])

    spread = max(nlls) - min(nlls)
    reference = abs(nlls[1]) * 1e-4
    assert spread < reference, (
        f"NLL spread across flux indices is {spread:.1f}, which is large "
        f"relative to {reference:.1f} — the index may now be identifiable, "
        "in which case fixing it is no longer the right choice"
    )
