"""Absolute opacity gauge from a live-time-normalised open-sky run.

The flat-ceiling phantom is the gate: with a free per-position offset the
constant part of a flat slab's opacity is absorbed by c_p and only the
oblique (sec theta - 1) excess reaches the voxels, landing in the outer shell
that only oblique rays cross. With the gauge measured (offset fixed at 0) the
slab comes back flat.
"""
from dataclasses import replace

import numpy as np
import pytest
import uproot

from megido.baseline import BaselineSolution
from megido.config import Reconstruction, load_site_config
from megido.forward import build_forward_model
from megido.inversion import solve
from megido.phantom import project, sky_rows
from megido.rootingest import read_live_time
from megido.skyref import solve_skyref
from tests.test_skyref import _scene

CFG = "configs/cafeteria.yaml"
RC = Reconstruction(algorithm="tv", n_iter=150, tv_alpha=0.01, tv_z_weight=0.5)


def _flat_ceiling():
    cfg = load_site_config(CFG)
    cfg = replace(cfg, volume=replace(cfg.volume, spacing_m=0.5))
    rows = sky_rows(("pos0", "pos1"), t_max=0.9, n_bins=18)
    fwd = build_forward_model(rows, cfg, cache_dir=None)
    zc = fwd.grid.axis_centers(2)
    truth = np.zeros(fwd.grid.shape)
    truth[:, :, (zc > 6.6) & (zc < 7.4)] = 0.125
    data = project(fwd, truth.ravel())
    covered = np.asarray(abs(fwd.A).sum(axis=0)).ravel().reshape(fwd.grid.shape)
    seen = covered[:, :, (zc > 6.6) & (zc < 7.4)].sum(2) > 0
    column = float(truth[0, 0].sum() * fwd.grid.spacing)   # true vertical lambda
    return fwd, data, seen, column


def _columns(fwd, x, seen):
    col = x.reshape(fwd.grid.shape).sum(2) * fwd.grid.spacing
    nx, ny = col.shape
    centre = col[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2].mean()
    return centre, np.percentile(col[seen], 95)


def test_measured_gauge_recovers_a_flat_ceiling_flat():
    fwd, data, seen, column = _flat_ceiling()
    x, info = solve(fwd, data, RC, fit_offsets=False)
    centre, rim = _columns(fwd, x, seen)
    assert info["offsets"] == {"pos0": 0.0, "pos1": 0.0}
    assert centre == pytest.approx(column, rel=0.1)
    assert rim == pytest.approx(centre, rel=0.15)


def test_free_offset_hides_the_ceiling_and_brightens_the_rim():
    """The degeneracy the measured gauge removes (and the reason the gate above
    can fail): the free fit puts the slab into c_p."""
    fwd, data, seen, column = _flat_ceiling()
    x, info = solve(fwd, data, RC)
    centre, rim = _columns(fwd, x, seen)
    assert min(info["offsets"].values()) > 0.4 * column
    assert centre < 0.3 * column
    assert rim > 1.5 * centre


def test_live_time_normalised_sky_ratio_is_absolute():
    cfg, grid, lam_true = _scene(scale=1.0)
    # counts were made with scale 1 = equal live times; state that explicitly
    live = {"pos0": 1000.0, "pos1": 1000.0, "SKY": 1000.0}
    sol = solve_skyref(grid, cfg, live_time=live)
    assert sol.absolute
    tx, ty = grid.tan_mesh()
    from megido.sky import detector_to_sky
    sx, sy, _ = detector_to_sky(tx, ty, cfg.exposure("pos0").pose)
    flat, ok = sol.sky.bin_index(sx, sy)
    lam = sol.opacity["pos0"][flat[ok]]
    seen = np.isfinite(lam)
    assert abs(np.mean(lam[seen] - lam_true[ok][seen])) < 0.005   # no constant left
    np.testing.assert_array_equal(sol.normalized_opacity("pos0"), sol.opacity["pos0"])


def test_live_time_ratio_sets_the_scale():
    cfg, grid, lam_true = _scene(scale=0.25)      # position ran 1/4 the sky's time
    sol = solve_skyref(grid, cfg, live_time={"pos0": 250.0, "pos1": 250.0, "SKY": 1000.0})
    assert sol.norms["pos0"] == pytest.approx(0.25)
    lam = sol.opacity["pos0"]
    assert np.nanmin(lam) == pytest.approx(0.0, abs=0.01)   # lump-free edge is open sky


def test_absolute_flag_survives_save_and_load(tmp_path):
    cfg, grid, _ = _scene()
    sol = solve_skyref(grid, cfg, live_time={"pos0": 1.0, "pos1": 1.0, "SKY": 1.0})
    sol.save(tmp_path / "b.npz")
    assert BaselineSolution.load(tmp_path / "b.npz").absolute
    rel = solve_skyref(grid, cfg)
    rel.save(tmp_path / "r.npz")
    assert not BaselineSolution.load(tmp_path / "r.npz").absolute


def test_live_time_is_the_sum_of_inter_event_intervals(tmp_path):
    edges = np.linspace(0, 500, 10001)             # 0.05 s bins, as the DAQ writes
    v = np.zeros(10000)
    v[2] = 100          # 100 intervals centred at 0.125 s
    v[40] = 10          # 10 intervals centred at 2.025 s
    with uproot.recreate(tmp_path / "a.root") as f:
        f["dT"] = (v, edges)
    assert read_live_time(tmp_path / "a.root") == pytest.approx(100 * 0.125 + 10 * 2.025)
