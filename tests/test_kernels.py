"""Point-spread probes of the delivered solver (spec 2026-10-06, section 4)."""
import dataclasses

import numpy as np
import pytest

from megido.config import KernelConfig, Reconstruction, load_site_config
from megido.fitdata import FitData
from megido.forward import build_forward_model
from megido.inversion import solve
from megido.kernels import (PSF, cell_half_width, point_spreads, probe_batches,
                            probe_deltas, psf_metrics)
from megido.phantom import project, sky_rows
from megido.resolution import rays_per_voxel, views_per_voxel
from megido.voxels import VoxelGrid

SIGMA = 0.02


def _slab(grid, z0, z1, rho):
    zc = grid.axis_centers(2)
    t = np.zeros(grid.shape)
    t[:, :, (zc >= z0) & (zc < z1)] = rho
    return t


def _problem(cfg, rows, background):
    fwd = build_forward_model(rows, cfg, cache_dir=None)
    truth = background(fwd.grid)
    d = project(fwd, truth.ravel(), sigma=SIGMA, seed=11)
    data = FitData(lam=d.lam, w=d.w, rows=rows)
    x, info = solve(fwd, data, cfg.reconstruction)
    rays3 = rays_per_voxel(fwd).reshape(fwd.grid.shape)
    return fwd, data, cfg.reconstruction, x, info, rays3


def _megiddo_problem(tmp_path, positions=2):
    """The real campaign's geometry at 0.5 m: P0/T20 at the origin, P1 at 2.2 m."""
    exps = ("  - id: P0\n    runs: DET1-DET2\n"
            "    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}\n"
            "  - id: T20\n    runs: DET3-DET4\n"
            "    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}\n")
    if positions == 2:
        exps += ("  - id: P1\n    runs: DET5-DET6\n"
                 "    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}\n")
    p = tmp_path / "megiddo.yaml"
    p.write_text(
        "site: megido-kernels\ndata_dir: /tmp\n"
        "volume: {z_min_m: 1.0, z_max_m: 11.0, spacing_m: 0.5, n_aperture_sub: 2,\n"
        "         xy_m: [[-8.0, 10.0], [-8.0, 8.0]]}\n"
        "reconstruction: {algorithm: tv, n_iter: 300, tv_alpha: 0.005, tv_z_weight: 0.3}\n"
        "kernels: {spacing_m: 2.0, sep_m: 4.0, z_levels_m: [2.25, 4.25, 6.25, 8.25, 10.25]}\n"
        "exposures:\n" + exps)
    cfg = load_site_config(p)
    pids = ("pos0", "pos1") if positions == 2 else ("pos0",)
    rows = sky_rows(pids, t_max=1.0, n_bins=24)
    return cfg, _problem(cfg, rows, lambda g: _slab(g, 8.0, 9.0, 0.05))


def _generous_problem(tmp_path):
    """9 detectors at 3 m pitch: the central probe at z = 3.5 m is seen by all 9
    positions, baselines 3-8.5 m against z 3.5 m (a 10 m pitch left it seen by one)."""
    dets = [(x, y) for x in (-3, 0, 3) for y in (-3, 0, 3)]
    blk = "\n".join(
        f"  - id: D{i}\n    runs: DET{i}-DET{i}\n"
        f"    pose: {{x: {dx}, y: {dy}, z: 0, tilt_deg: 0, az_deg: 0}}"
        for i, (dx, dy) in enumerate(dets))
    p = tmp_path / "generous.yaml"
    p.write_text(
        "site: generous\ndata_dir: /tmp\n"
        "volume: {z_min_m: 0.0, z_max_m: 8.0, spacing_m: 1.0, n_aperture_sub: 2,\n"
        "         xy_m: [[-14, 14], [-14, 14]]}\n"
        "reconstruction: {algorithm: tv, n_iter: 400, tv_alpha: 0.002, tv_z_weight: 0.3}\n"
        "exposures:\n" + blk + "\n")
    cfg = load_site_config(p)
    rows = sky_rows(tuple(f"pos{i}" for i in range(9)), t_max=1.2, n_bins=25)
    return cfg, _problem(cfg, rows, lambda g: _slab(g, 6.0, 7.0, 0.05))


def test_probe_batches_cover_the_lattice_once_and_keep_probes_apart(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    kc = cfg.kernels
    batches = probe_batches(fwd.grid, rays3, kc)
    allp = np.concatenate(batches)
    assert len({tuple(p) for p in allp}) == len(allp)              # each probe once
    assert all(rays3[tuple(p)] >= 2 for p in allp)                  # covered only
    sep_vox = round(kc.sep_m / fwd.grid.spacing)
    for b in batches:
        assert len(set(b[:, 2])) == 1                               # one z level per batch
        d = np.abs(b[:, None, :2] - b[None, :, :2]).max(-1)
        np.fill_diagonal(d, 10**9)
        assert d.min() >= sep_vox                                   # laterally separated


def test_probe_with_no_weighted_rows_is_skipped(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    probes = np.concatenate(probe_batches(fwd.grid, rays3, cfg.kernels))
    dead = FitData(lam=data.lam, w=np.zeros_like(data.w), rows=data.rows)
    assert np.isnan(probe_deltas(fwd, dead, probes, cfg.kernels)).all()
    live = probe_deltas(fwd, data, probes, cfg.kernels)
    assert np.isfinite(live).all() and (live > 0).all()
    # delta = delta_sigma * median sigma / voxel spacing; sigma is SIGMA everywhere
    np.testing.assert_allclose(live, cfg.kernels.delta_sigma * SIGMA / fwd.grid.spacing)


def test_psf_metrics_of_hand_built_responses():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(5, 5, 10))
    v = np.zeros((5, 5, 10)); v[2, 2, 4] = 1.0                      # a perfect delta
    m = psf_metrics(PSF(probe=(2, 2, 4), lo=(0, 0), values=v), grid)
    assert m["psf_mass"] == pytest.approx(1.0)
    assert m["psf_dz"] == pytest.approx(0.68)                       # 16-84% of one voxel
    assert m["psf_shift"] == pytest.approx(0.0)
    assert m["psf_lat"] == pytest.approx(0.0)
    s = np.zeros((5, 5, 10)); s[2, 2, :] = 0.1                      # smeared up the column
    m = psf_metrics(PSF(probe=(2, 2, 4), lo=(0, 0), values=s), grid)
    assert m["psf_mass"] == pytest.approx(1.0)
    assert m["psf_dz"] == pytest.approx(6.8)                        # 16-84% of 10 m
    assert m["psf_shift"] == pytest.approx(5.0 - 4.5)               # centroid 5.0, probe 4.5


def test_zero_response_gives_nan_metrics_not_errors():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(3, 3, 4))
    m = psf_metrics(PSF(probe=(1, 1, 1), lo=(0, 0), values=np.zeros((3, 3, 4))), grid)
    assert all(np.isnan(m[k]) for k in ("psf_mass", "psf_dz", "psf_shift", "psf_lat"))


def test_edge_probe_cell_is_clipped_not_wrapped(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    nx, ny, _ = fwd.grid.shape
    h = cell_half_width(fwd.grid, cfg.kernels)
    covered = np.argwhere(rays3 >= 2)
    p = covered[np.argmin(covered[:, 0])]                           # lowest-i covered voxel
    probes = p[None, :]
    deltas = probe_deltas(fwd, data, probes, cfg.kernels)
    (psf,) = point_spreads(fwd, data, rc, x, info["best_iter"], probes, deltas,
                           fit_offsets=True, cell_half=h)
    assert psf.lo[0] >= 0 and psf.lo[0] + psf.values.shape[0] <= nx
    assert psf.lo[1] >= 0 and psf.lo[1] + psf.values.shape[1] <= ny
    assert np.isfinite(psf.values).all()


def test_psfs_are_deterministic(tmp_path):
    """Gate 5."""
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    b = probe_batches(fwd.grid, rays3, cfg.kernels)[3]
    d = probe_deltas(fwd, data, b, cfg.kernels)
    h = cell_half_width(fwd.grid, cfg.kernels)
    a1 = point_spreads(fwd, data, rc, x, info["best_iter"], b, d, fit_offsets=True, cell_half=h)
    a2 = point_spreads(fwd, data, rc, x, info["best_iter"], b, d, fit_offsets=True, cell_half=h)
    for p, q in zip(a1, a2):
        np.testing.assert_array_equal(p.values, q.values)


def test_well_resolved_geometry_gives_a_compact_psf(tmp_path):
    """Gate 1: 9 positions, baselines 3-8.5 m against z = 3.5 m."""
    cfg, (fwd, data, rc, x, info, rays3) = _generous_problem(tmp_path)
    kc = dataclasses.replace(cfg.kernels, spacing_m=1.0, sep_m=6.0)
    i, j = np.array(fwd.grid.shape[:2]) // 2
    k = int((3.5 - fwd.grid.origin[2]) / fwd.grid.spacing)          # probe at z = 3.5 m
    probes = np.array([[i, j, k]])
    assert rays3[i, j, k] >= 2
    views = views_per_voxel(fwd).reshape(fwd.grid.shape)
    assert views[i, j, k] >= 4                                      # parallax exists
    deltas = probe_deltas(fwd, data, probes, kc)
    (psf,) = point_spreads(fwd, data, rc, x, info["best_iter"], probes, deltas,
                           fit_offsets=True, cell_half=cell_half_width(fwd.grid, kc))
    m = psf_metrics(psf, fwd.grid)
    print(f"\ngate 1 (generous): {m}")
    h = fwd.grid.spacing
    assert m["psf_mass"] >= 0.8
    assert m["psf_dz"] <= 2 * h
    assert abs(m["psf_shift"]) <= 1 * h
