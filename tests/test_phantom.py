from pathlib import Path

import numpy as np
import pytest

from megido.config import load_site_config
from megido.fitdata import FitData
from megido.forward import build_forward_model
from megido.inversion import solve
from megido.phantom import (anomaly_from_topography, depth_localization,
                            interface_height, load_topography, project,
                            sky_rows, surface_volume)
from megido.voxels import VoxelGrid

TOPO = Path("/home/hadar/Cloud/Work/Postdoc/01_data/raw/topography.csv")


def test_topography_loads_as_a_regular_grid():
    xs, ys, z = load_topography(TOPO)
    assert (xs.size, ys.size) == (41, 41)
    assert z.shape == (41, 41)
    assert xs[0] == pytest.approx(-80.0) and xs[-1] == pytest.approx(80.0)
    assert np.isfinite(z).all()


def test_surface_volume_fills_below_the_surface_only():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 2, 4))
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    z_surface = np.array([[2.0, 2.0], [2.0, 2.0]])
    rho = surface_volume(grid, xs, ys, z_surface, density=0.7).reshape(grid.shape)
    np.testing.assert_allclose(rho[:, :, :2], 0.7)
    np.testing.assert_allclose(rho[:, :, 2:], 0.0)


def test_sky_rows_covers_every_position_and_bin():
    rows = sky_rows(("pos0", "pos1"), t_max=1.0, n_bins=6)
    assert rows.n_rows == 2 * 36
    assert rows.mask_for("pos0").sum() == 36
    assert np.abs(rows.sx).max() < 1.0


def test_project_is_the_forward_model_plus_noise(tmp_path):
    rows = sky_rows(("pos0", "pos1"), t_max=0.8, n_bins=6)
    cfg = _cfg(tmp_path)
    fwd = build_forward_model(rows, cfg, cache_dir=None)
    truth = np.full(fwd.grid.n_voxels, 0.1)

    clean = project(fwd, truth, sigma=0.0)
    np.testing.assert_allclose(clean.lam, fwd.predict(truth))
    assert isinstance(clean, FitData)

    noisy = project(fwd, truth, sigma=0.05, seed=1)
    assert np.std(noisy.lam - clean.lam) == pytest.approx(0.05, rel=0.3)


def test_interface_height_finds_the_top_of_a_filled_column():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 2, 6))
    rho3 = np.zeros(grid.shape)
    rho3[:, :, :3] = 1.0
    np.testing.assert_allclose(interface_height(rho3, grid), 2.5)


def test_anomaly_sits_at_one_layer_and_follows_the_high_ground():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(4, 1, 6))
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0])
    surf = np.array([[1.0], [1.0], [9.0], [9.0]])   # high ground = last two columns
    vol = anomaly_from_topography(grid, xs, ys, surf, z_anomaly_m=3.0,
                                  quantile=0.5, density=1.0).reshape(grid.shape)
    # exactly one z-layer populated, the one containing z=3.0 (index 3)
    assert set(np.nonzero(vol.sum(axis=(0, 1)))[0].tolist()) == {3}
    # only the high-ground columns carry the anomaly
    col = vol.sum(axis=2)[:, 0]
    np.testing.assert_allclose(col > 0, [False, False, True, True])


def test_depth_localization_is_one_for_a_single_layer_and_small_when_smeared():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 2, 5))
    sharp = np.zeros(grid.shape)
    sharp[:, :, 2] = 1.0
    assert depth_localization(sharp, grid) == pytest.approx(1.0)

    flat = np.ones(grid.shape)
    assert depth_localization(flat, grid) == pytest.approx(1.0 / 5)


def _cfg(tmp_path):
    """25-view dense config over a +-30 m grid: generous multi-position baseline.

    This is the FAVOURABLE geometry — as many well-separated views as a phantom
    can have. It gates the CODE. It is still one-sided (every ray points up), so
    even here depth is not localized; that is asserted below, not worked around.
    """
    dets = [(x, y) for x in (-20, -10, 0, 10, 20) for y in (-20, -10, 0, 10, 20)]
    blk = "\n".join(
        f"  - id: D{i}\n    runs: DET{i}-DET{i}\n"
        f"    pose: {{x: {dx}, y: {dy}, z: 0, tilt_deg: 0, az_deg: 0}}"
        for i, (dx, dy) in enumerate(dets))
    p = tmp_path / "phantom.yaml"
    p.write_text(
        "site: phantom\ndata_dir: /tmp\n"
        "volume: {z_min_m: 0.0, z_max_m: 8.0, spacing_m: 1.0, n_aperture_sub: 2,\n"
        "         xy_m: [[-30, 30], [-30, 30]]}\n"
        "reconstruction: {algorithm: tv, n_iter: 400, tv_alpha: 0.002, tv_z_weight: 0.3}\n"
        "exposures:\n" + blk + "\n")
    return load_site_config(p)


@pytest.mark.skipif(not TOPO.exists(), reason="topography.csv not on this machine")
def test_the_code_reproduces_data_recovers_lateral_structure_and_does_not_fake_depth(
        tmp_path, capsys):
    """THE TASK 7 GATE.

    Twenty-five detectors over a +-30 m grid view a thin anomaly whose lateral
    footprint is the topography surface's high ground, placed at a single known
    height. Truth is built and projected at HALF the inversion spacing, so the
    solver never inverts its own discretisation.

    Three assertions, and each is the honest one:
      1. The reconstruction reproduces the measurements (data-space). This is
         what the forward model + solver provably must do, and it is what caught
         the detector-inside-grid ray bug during this task's development.
      2. Lateral (column-integrated) structure is recovered. This is the part of
         the scene a one-sided geometry actually constrains.
      3. Depth is NOT localized. Truth is one layer; the reconstruction must
         smear it, because muons arrive from above only. A gate that let the
         reconstruction claim sharp depth would be rewarding a lie.

    Do NOT add an interface-RMSE fidelity assertion here and do NOT loosen these
    three. If assertion 1 fails, the forward model or solver is broken — a defect
    to find and report, never a threshold to lower.
    """
    cfg = _cfg(tmp_path)
    xs, ys, z = load_topography(TOPO)
    rows = sky_rows(tuple(e.id.replace("D", "pos") for e in cfg.exposures),
                    t_max=1.2, n_bins=25)

    import dataclasses
    fine_cfg = dataclasses.replace(
        cfg, volume=dataclasses.replace(cfg.volume, spacing_m=0.5))
    fine = build_forward_model(rows, fine_cfg, cache_dir=None)
    coarse = build_forward_model(rows, cfg, cache_dir=None)

    truth = anomaly_from_topography(fine.grid, xs, ys, z, z_anomaly_m=4.0,
                                    quantile=0.5, density=1.0)
    data = project(fine, truth, sigma=0.01, seed=1)
    data = FitData(lam=data.lam, w=data.w, rows=rows)

    x, info = solve(coarse, data, cfg.reconstruction)

    pred = coarse.predict(x, info["offsets"])
    data_corr = float(np.corrcoef(pred, data.lam)[0, 1])
    rel_res = float(np.linalg.norm(pred - data.lam)
                    / np.linalg.norm(data.lam - data.lam.mean()))

    truth_c = anomaly_from_topography(coarse.grid, xs, ys, z, z_anomaly_m=4.0)
    cr = x.reshape(coarse.grid.shape).sum(axis=2)
    ct = truth_c.reshape(coarse.grid.shape).sum(axis=2)
    lateral = float(np.corrcoef(cr.ravel(), ct.ravel())[0, 1])
    depth = depth_localization(x.reshape(coarse.grid.shape), coarse.grid)

    print(f"\nTask 7 gate: data corr={data_corr:.4f} rel-res={rel_res:.4f} "
          f"lateral col-corr={lateral:.3f} depth peak/total={depth:.3f} "
          f"(truth depth=1.0)")

    assert data_corr > 0.99          # code reproduces the measurements
    assert rel_res < 0.15
    assert lateral > 0.6             # lateral structure recovered
    assert depth < 0.5               # depth NOT falsely localized
