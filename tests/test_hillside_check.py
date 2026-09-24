from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from megido.hillside_check import _bilinear, ray_exit_distance, surface_ray_check


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
