from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from megido.hillside_check import (
    RayCheck, _bilinear, _OnlyPositionSol, cross_position_check, ray_exit_distance,
    residual_grid, surface_ray_check,
)


def _flat(h0, lo=-30.0, hi=30.0, step=1.0):
    g = np.arange(lo, hi + step, step)
    return np.full((g.size, g.size), h0), g, g.copy()


def test_bilinear_is_exact_on_a_plane_and_nan_outside():
    g = np.arange(0.0, 5.0, 1.0)
    X, Y = np.meshgrid(g, g, indexing="ij")
    H = 2 * X + 3 * Y + 1
    v = _bilinear(H, g, g, np.array([1.5, 2.25]), np.array([0.5, 3.75]))
    np.testing.assert_allclose(v, [2 * 1.5 + 3 * 0.5 + 1, 2 * 2.25 + 3 * 3.75 + 1])
    assert np.isnan(_bilinear(H, g, g, np.array([-0.1]), np.array([1.0]))[0])
    H2 = H.copy(); H2[2, 2] = np.nan
    assert np.isnan(_bilinear(H2, g, g, np.array([1.5]), np.array([1.5]))[0])


def test_flat_slab_path_lengths_are_exact():
    H, gx, gy = _flat(8.0)
    dirs = np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 1.0], [-0.5, 0.4, 1.0]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    t = ray_exit_distance((0.0, 0.0, 0.0), dirs, H, gx, gy)
    np.testing.assert_allclose(t, 8.0 / dirs[:, 2], rtol=1e-6)


def test_ray_through_a_nan_hole_or_off_grid_is_nan_not_zero():
    up = np.array([[0.0, 0.0, 1.0]])
    far = np.array([[5.0, 0.0, 1.0]]) / np.sqrt(26.0)   # leaves the 10 m grid before z=8
    # a hole a few metres off-axis: an oblique ray heading into it is unpredictable
    H, gx, gy = _flat(8.0, lo=-10.0, hi=10.0)
    k = int(np.argmin(np.abs(gx - 4.0)))
    H[k - 1:k + 2, :] = np.nan                      # NaN band around x = 4
    into_hole = np.array([[0.5, 0.0, 1.0]]) / np.sqrt(1.25)   # exits near x = 4 at z = 8
    assert np.isnan(ray_exit_distance((0.0, 0.0, 0.0), into_hole, H, gx, gy)[0])
    assert np.isfinite(ray_exit_distance((0.0, 0.0, 0.0), up, H, gx, gy)[0])
    # detector under an unknown patch -> NaN for every ray
    H2, gx2, gy2 = _flat(8.0, lo=-10.0, hi=10.0)
    i = int(np.argmin(np.abs(gx2 - 0.0)))
    H2[i - 1:i + 2, i - 1:i + 2] = np.nan
    assert np.isnan(ray_exit_distance((0.0, 0.0, 0.0), up, H2, gx2, gy2)[0])
    # ray that leaves the grid before crossing
    Hf, gxf, gyf = _flat(8.0, lo=-10.0, hi=10.0)
    assert np.isnan(ray_exit_distance((0.0, 0.0, 0.0), far, Hf, gxf, gyf)[0])


def test_true_hill_reproduces_exact_path_lengths():
    """Load-bearing: with the TRUE surface and a=1, predicted opacity equals the
    fixture's exact path length -> the tracer is right."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg, _true_hill
    sol = _build_fake_sol_from_hill(n_bins=48)
    cfg = _fake_cfg()
    g = np.arange(-30.0, 30.0 + 0.2, 0.2)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    res = SimpleNamespace(H=_true_hill(GX, GY), gx=g, gy=g.copy(), a=1.0)
    chk = surface_ray_check(sol, cfg, res)
    assert chk.n_checked > 1000
    assert chk.ray_ve > 0.99, chk.ray_ve
    assert chk.ray_rms < 0.05, chk.ray_rms


def test_nan_residual_where_surface_unconstrained():
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg, _true_hill
    sol = _build_fake_sol_from_hill(n_bins=32)
    cfg = _fake_cfg()
    g = np.arange(-6.0, 6.0 + 0.25, 0.25)           # small grid: many rays leave it
    GX, GY = np.meshgrid(g, g, indexing="ij")
    res = SimpleNamespace(H=_true_hill(GX, GY), gx=g, gy=g.copy(), a=1.0)
    chk = surface_ray_check(sol, cfg, res)
    r = np.concatenate([v.ravel() for v in chk.residual.values()])
    p = np.concatenate([v.ravel() for v in chk.predicted.values()])
    assert np.isnan(r).any()                        # off-grid rays are NaN
    assert chk.n_checked == int(np.isfinite(r).sum())
    # the fake sol measures every direction, so a NaN residual must mean
    # "unpredicted" -- a residual of 0 for an unpredicted ray would break this
    assert np.array_equal(np.isnan(r), np.isnan(p))


def test_detector_at_or_above_surface_is_nan_not_zero():
    """Without the below-surface guard the tracer would return t* ~ 0 -- a
    finite 'no rock' answer where the truth is 'model does not apply'."""
    up = np.array([[0.0, 0.0, 1.0]])
    for h0 in (-1.0, 0.0):          # surface below / exactly at the detector
        H, gx, gy = _flat(h0, lo=-10.0, hi=10.0)
        assert np.isnan(ray_exit_distance((0.0, 0.0, 0.0), up, H, gx, gy)[0])


def test_prediction_is_nearly_scale_free():
    """lambda_pred = t*/a: the fitted surface scales ~linearly with a, so the
    predicted opacities should barely move between a=1 and a=2."""
    from megido.hillside_surface import fit_surface
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()
    r1 = fit_surface(sol, cfg, a=1.0, cell_m=0.5, n_restarts=1, max_points=400)
    r2 = fit_surface(sol, cfg, a=2.0, cell_m=0.5, n_restarts=1, max_points=400)
    c1, c2 = surface_ray_check(sol, cfg, r1), surface_ray_check(sol, cfg, r2)
    p1 = np.concatenate([c1.predicted[k].ravel() for k in sorted(c1.predicted)])
    p2 = np.concatenate([c2.predicted[k].ravel() for k in sorted(c2.predicted)])
    ok = np.isfinite(p1) & np.isfinite(p2)
    assert ok.sum() > 500
    assert np.corrcoef(p1[ok], p2[ok])[0, 1] > 0.98
    assert abs(c1.ray_ve - c2.ray_ve) < 0.1


_REAL = Path("runs/solve/baseline.npz")


@pytest.mark.skipif(not _REAL.exists(), reason="needs runs/solve/baseline.npz")
def test_real_data_check_is_finite():
    from megido.baseline import BaselineSolution
    from megido.config import load_site_config
    from megido.hillside_surface import fit_surface
    cfg = load_site_config("configs/megido.yaml")
    sol = BaselineSolution.load(_REAL)
    res = fit_surface(sol, cfg, a=8.0, cell_m=1.0, n_restarts=1, max_points=800)
    chk = surface_ray_check(sol, cfg, res)
    assert chk.n_checked > 0
    assert np.isfinite(chk.ray_ve) and np.isfinite(chk.ray_rms)


class _Shifted:
    """Wrap a sol and add a constant to ONE position's opacity: the per-position
    gauge freedom Phase 2 leaves unmeasured."""

    def __init__(self, sol, pid, c):
        self.sky, self._sol, self._pid, self._c = sol.sky, sol, pid, c

    def normalized_opacity(self, pid, transparent_quantile=0.05):
        lam = self._sol.normalized_opacity(pid, transparent_quantile=transparent_quantile)
        return lam + self._c if pid == self._pid else lam


def test_headline_metrics_are_gauge_invariant_per_position():
    """Only differences are measured: a constant added to one position's opacity
    must not move the headline VE/RMS, must be recovered as that position's
    offset, and must hurt only the raw (gauge-naive) VE."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg, _true_hill
    sol = _build_fake_sol_from_hill(n_bins=48)
    cfg = _fake_cfg()
    g = np.arange(-30.0, 30.0 + 0.2, 0.2)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    res = SimpleNamespace(H=_true_hill(GX, GY), gx=g, gy=g.copy(), a=1.0)
    base = surface_ray_check(sol, cfg, res)
    shifted = surface_ray_check(_Shifted(sol, "pos1", 0.7), cfg, res)
    assert abs(shifted.ray_ve - base.ray_ve) < 1e-6
    assert abs(shifted.ray_rms - base.ray_rms) < 1e-6
    assert abs(shifted.offsets["pos1"] - base.offsets["pos1"] - 0.7) < 1e-6
    assert abs(shifted.offsets["pos0"] - base.offsets["pos0"]) < 1e-9
    assert shifted.ray_ve_raw < base.ray_ve_raw - 0.05


def test_per_position_stats_are_reported():
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg, _true_hill
    sol = _build_fake_sol_from_hill(n_bins=48)
    cfg = _fake_cfg()
    g = np.arange(-30.0, 30.0 + 0.2, 0.2)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    chk = surface_ray_check(sol, cfg, SimpleNamespace(H=_true_hill(GX, GY), gx=g, gy=g.copy(), a=1.0))
    assert set(chk.per_position) == {"pos0", "pos1"}
    for pid, st in chk.per_position.items():
        assert st["n"] > 100
        assert st["corr"] > 0.99 and st["ve"] > 0.99
    assert sum(st["n"] for st in chk.per_position.values()) == chk.n_checked


# --- Task 1: per-ray exit points, residual grid, cross-position check -----


def test_exit_xy_matches_flat_slab_formula():
    """Flat-slab surface, detector at origin: exit xy must equal
    (h0/d_z)*d_xy exactly, NaN wherever t* is NaN (matches ray_exit_distance)."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill(n_bins=32)
    cfg = _fake_cfg()
    h0 = 8.0
    g = np.arange(-4.0, 4.0 + 0.5, 0.5)     # small enough that oblique rays leave it
    H = np.full((g.size, g.size), h0)
    res = SimpleNamespace(H=H, gx=g, gy=g.copy(), a=1.0)
    chk = surface_ray_check(sol, cfg, res)

    exit0 = chk.exit_xy["pos0"]
    assert exit0.shape == (32, 32, 2)

    centers = sol.sky.centers
    SX, SY = np.meshgrid(centers, centers, indexing="ij")
    norm = np.sqrt(1.0 + SX ** 2 + SY ** 2)
    dx, dy, dzc = SX / norm, SY / norm, 1.0 / norm
    t = h0 / dzc
    expect_x, expect_y = t * dx, t * dy

    t_pred = chk.predicted["pos0"] * res.a           # predicted is t*/a
    ok = np.isfinite(t_pred)
    assert ok.sum() > 100
    np.testing.assert_allclose(exit0[..., 0][ok], expect_x[ok], rtol=1e-6)
    np.testing.assert_allclose(exit0[..., 1][ok], expect_y[ok], rtol=1e-6)
    assert np.array_equal(np.isnan(exit0[..., 0]), np.isnan(t_pred))
    assert (~ok).any()                                # some off-grid rays are NaN


def test_residual_grid_is_mean_per_node_and_nan_where_no_ray():
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg, _true_hill
    sol = _build_fake_sol_from_hill(n_bins=48)
    cfg = _fake_cfg()
    g = np.arange(-30.0, 30.0 + 0.2, 0.2)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    res = SimpleNamespace(H=_true_hill(GX, GY), gx=g, gy=g.copy(), a=1.0)
    chk = surface_ray_check(sol, cfg, res)

    # far corners (radius > the rays' reach) get no ray -> guaranteed NaN
    gx = gy = np.arange(-20.0, 20.0 + 4.0, 4.0)
    grid = residual_grid(chk, gx, gy)
    assert grid.shape == (gx.size, gy.size)
    finite = np.isfinite(grid)
    assert finite.any()
    assert np.all(np.abs(grid[finite]) < 0.05)
    assert (~finite).any()                            # some nodes have no ray


def test_residual_grid_averages_rays_on_the_same_node():
    gx = np.array([0.0, 1.0, 2.0])
    gy = np.array([0.0, 1.0, 2.0])
    # a THIRD ray at [1,0] has a finite residual (5.0) but an exit point far
    # outside the grid -- it must be skipped (F6: exercise that branch), not
    # silently pulled into whichever node the rounding would nearest-match.
    residual = {"p0": np.array([[1.0, 2.0], [5.0, np.nan]])}
    exit_xy = {"p0": np.array([[[1.0, 1.0], [1.1, 0.9]],
                               [[50.0, 50.0], [5.0, 5.0]]])}
    chk = RayCheck(residual=residual, predicted={}, ray_ve=float("nan"),
                  ray_rms=float("nan"), n_checked=3, a=1.0, exit_xy=exit_xy)
    grid = residual_grid(chk, gx, gy)
    assert np.isclose(grid[1, 1], 1.5)                # both in-range finite rays round to (1,1)
    assert np.isnan(grid[0, 0])
    finite = np.isfinite(grid)
    assert finite.sum() == 1                          # only (1,1) populated -- the out-of-range
    assert not np.any(np.isclose(grid[finite], 5.0))  # ray never lands anywhere


def test_cross_position_check_out_of_sample_recovery():
    """A surface fit from ONE position's opacities must still predict the
    OTHER position's rays well on this synthetic (both positions see the
    same true hill), and beat the flat-slab null for that scored position."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill(n_bins=128)
    cfg = _fake_cfg()
    fit_kwargs = dict(a=1.0, cell_m=0.5, n_restarts=1, max_points=400)
    out = cross_position_check(sol, cfg, fit_kwargs=fit_kwargs, n_shuffle=1)

    assert set(out["fit_on"]) == {"pos0", "pos1"}
    c01 = out["fit_on"]["pos0"]["pos1"]["corr"]
    c10 = out["fit_on"]["pos1"]["pos0"]["corr"]
    assert c01 > 0.8, c01
    assert c10 > 0.8, c10
    assert c01 > out["null_flat"]["pos1"]["corr"]
    assert c10 > out["null_flat"]["pos0"]["corr"]


def test_shuffled_null_is_beaten_by_the_real_out_of_sample_corr():
    """Decides whether the surface shows real directional structure: the
    flat-slab null alone is too weak (a scrambled single-position fit still
    forms a dome-like upper envelope). The real out-of-sample corr must beat
    the shuffled-opacity null's WORST CASE (corr_max over seeds) in both
    directions on the synthetic hill -- if it does not, that is reported, not
    weakened away."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill(n_bins=128)
    cfg = _fake_cfg()
    fit_kwargs = dict(a=1.0, cell_m=0.5, n_restarts=1, max_points=400)
    out = cross_position_check(sol, cfg, fit_kwargs=fit_kwargs, n_shuffle=3)

    c01 = out["fit_on"]["pos0"]["pos1"]["corr"]
    c10 = out["fit_on"]["pos1"]["pos0"]["corr"]
    s01 = out["null_shuffled"]["pos0"]["pos1"]
    s10 = out["null_shuffled"]["pos1"]["pos0"]
    assert s01["n_seeds"] == 3 and s10["n_seeds"] == 3
    assert c01 > s01["corr_max"], (c01, s01)
    assert c10 > s10["corr_max"], (c10, s10)


class _ScrambledSol:
    """Wrap a sol, replacing one position's opacities with a seeded random
    permutation of themselves (same values, scrambled sky directions) --
    breaks the correspondence between direction and measured opacity so a
    surface fit from the OTHER position should fail to predict it."""

    def __init__(self, sol, pid, seed=0):
        self._sol, self._pid = sol, pid
        self.sky = sol.sky
        self._rng_seed = seed
        self._perm = None

    def normalized_opacity(self, position_id, transparent_quantile=0.05):
        lam = np.asarray(
            self._sol.normalized_opacity(position_id, transparent_quantile=transparent_quantile),
            dtype=float)
        if position_id != self._pid:
            return lam
        if self._perm is None:
            self._perm = np.random.default_rng(self._rng_seed).permutation(lam.size)
        return lam.ravel()[self._perm].reshape(lam.shape)

    def __getattr__(self, name):
        return getattr(self._sol, name)


def test_cross_position_check_negative_control_scrambled_pos1():
    """Must be able to FAIL: with pos1's opacities scrambled, a surface fit
    from pos0 (unaffected) can no longer predict pos1's (scrambled) rays,
    while pos0 scored on its own fit is unaffected. Gate raised to 0.95
    (real: 0.9995) -- a leaky wrapper that let pos1 through this same fixture
    only reaches 0.863, so 0.95 cleanly separates leaked from clean."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill(n_bins=128)
    scrambled = _ScrambledSol(sol, "pos1")
    cfg = _fake_cfg()
    fit_kwargs = dict(a=1.0, cell_m=0.5, n_restarts=1, max_points=400)
    out = cross_position_check(scrambled, cfg, fit_kwargs=fit_kwargs, n_shuffle=1)

    assert out["fit_on"]["pos0"]["pos1"]["corr"] < 0.3, out["fit_on"]["pos0"]["pos1"]
    assert out["in_sample"]["pos0"]["corr"] > 0.95, out["in_sample"]["pos0"]


def test_only_position_sol_hides_every_other_position():
    """Direct unit test of the wrapper F1 targets: the hidden position reads
    back as all-NaN (never a fabricated zero), the visible position is passed
    through exactly."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill
    sol = _build_fake_sol_from_hill(n_bins=16)
    w = _OnlyPositionSol(sol, "pos0")
    lam0 = w.normalized_opacity("pos0")
    lam1 = w.normalized_opacity("pos1")
    np.testing.assert_array_equal(lam0, sol.normalized_opacity("pos0"))
    assert np.isnan(lam1).all()


def test_in_sample_pos0_is_invariant_to_pos1_scrambling():
    """The strongest leak detector (F1): a fit on pos0 alone must not depend
    on pos1's data AT ALL. If `_OnlyPositionSol` ever let pos1's (here
    scrambled) opacities leak into the pos0-only fit, `in_sample["pos0"]`
    would move between the clean and scrambled runs; it must not, exactly."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill(n_bins=128)
    scrambled = _ScrambledSol(sol, "pos1")
    cfg = _fake_cfg()
    fit_kwargs = dict(a=1.0, cell_m=0.5, n_restarts=1, max_points=400)
    clean = cross_position_check(sol, cfg, fit_kwargs=fit_kwargs, n_shuffle=1)
    dirty = cross_position_check(scrambled, cfg, fit_kwargs=fit_kwargs, n_shuffle=1)
    assert clean["in_sample"]["pos0"] == dirty["in_sample"]["pos0"]


def test_cross_position_check_records_fit_failure_without_crashing():
    """F2: a fit that raises (here forced via an absurd min_count) must be
    recorded as {"error": ...} at fit_on/in_sample/null_shuffled, never crash
    cross_position_check itself."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill(n_bins=32)
    cfg = _fake_cfg()
    fit_kwargs = dict(a=1.0, cell_m=0.5, n_restarts=1, max_points=400,
                      min_count=10 ** 9)
    out = cross_position_check(sol, cfg, fit_kwargs=fit_kwargs, n_shuffle=1)
    assert "error" in out["fit_on"]["pos0"]
    assert "error" in out["in_sample"]["pos0"]
    assert "error" in out["null_shuffled"]["pos0"]
