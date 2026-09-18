import json

import numpy as np
import pytest

from megido.angular import AnalysisGrid
from megido.baseline import solve_baseline
from megido.cli import main
from megido.config import load_site_config

# z 3-7 m at a 2.2 m baseline: zmid=5 m gives dz ~0.80 m, coarser than the 0.5 m
# voxel spacing, so depth is genuinely NOT resolved here. The brief's original
# fixture used z 1-3 m (zmid=2 m, dz ~0.13 m), which the closed-form resolution
# formula (megido/resolution.py) actually resolves at this baseline+spacing --
# that is the same brief-fixture bug already documented in
# tests/test_volexport.py (Task 12); campaign_resolution itself is unchanged
# and is covered separately in tests/test_resolution.py.
CONFIG = """
site: t
data_dir: {data_dir}
binning: {{t_max: 1.25, n_bins: 20}}
volume: {{z_min_m: 3.0, z_max_m: 7.0, spacing_m: 0.5, n_aperture_sub: 2}}
reconstruction: {{algorithm: tv, n_iter: 20, tv_alpha: 0.01}}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {{x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}}
  - id: T20
    runs: DET3-DET4
    pose: {{x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}}
  - id: P1
    runs: DET5-DET6
    pose: {{x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}}
"""


@pytest.fixture
def workspace(tmp_path):
    cfg_path = tmp_path / "site.yaml"
    cfg_path.write_text(CONFIG.format(data_dir=tmp_path))
    cfg = load_site_config(cfg_path)

    rng = np.random.default_rng(0)
    edges = np.linspace(-1.25, 1.25, 21)
    t = 0.5 * (edges[:-1] + edges[1:])
    shape = np.exp(-(t[:, None] ** 2 + t[None, :] ** 2))
    counts = {e.id: rng.poisson(3000 * shape).astype(np.int64) for e in cfg.exposures}

    # These are the exact keys megido.anghist.load_counts reads. Writing
    # "counts"/"edges" instead would fail inside the CLI, not in the code under
    # test.
    ingest = tmp_path / "ingest"
    ingest.mkdir()
    for eid, c in counts.items():
        np.savez_compressed(ingest / f"counts_{eid}.npz", values=c,
                            xedges=edges, yedges=edges, name=np.array("txty"))

    solve_dir = tmp_path / "solve"
    sol = solve_baseline(AnalysisGrid(edges=edges, counts=counts), cfg, n_iter=40)
    sol.save(solve_dir / "baseline.npz")
    return cfg_path, ingest, solve_dir, tmp_path


def test_reconstruct_writes_volumes_and_exits_zero(workspace, capsys):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    rc = main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
               "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    assert rc == 0
    assert (out / "volume_full.npz").exists()
    assert (out / "views.npy").exists()
    text = capsys.readouterr().out
    assert "baseline" in text.lower()
    assert "depth" in text.lower()


def test_reconstruct_writes_holdout_volumes(workspace):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
          "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    assert (out / "volume_holdout_pos0.npz").exists()
    assert (out / "volume_holdout_pos1.npz").exists()


def test_reconstruct_with_bootstrap_writes_uncertainty(workspace):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    # --rebin 1: the fixture has 20 bins, and the CLI default factor of 10 would
    # collapse the analysis grid to 2x2 and make the baseline solve degenerate.
    rc = main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
               "--run", str(ingest), "--out", str(out), "--bootstrap", "2",
               "--rebin", "1", "--iters", "30", "--no-systematic"])
    assert rc == 0
    assert (out / "uncertainty.npz").exists()


def test_bootstrap_without_a_run_directory_fails_loudly(workspace, capsys):
    cfg_path, ingest, solve_dir, tmp = workspace
    rc = main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
               "--out", str(tmp / "v"), "--bootstrap", "3", "--rebin", "1"])
    assert rc == 1
    assert "--run" in capsys.readouterr().out


def test_reconstruct_reports_a_missing_baseline(tmp_path, capsys):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG.format(data_dir=tmp_path))
    rc = main(["reconstruct", "--config", str(p), "--solve", str(tmp_path / "nope"),
               "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "baseline.npz" in capsys.readouterr().out


def test_export_produces_the_viewer_contract(workspace):
    cfg_path, ingest, solve_dir, tmp = workspace
    out = tmp / "voxels"
    main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
          "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    rc = main(["export", "--config", str(cfg_path), "--run", str(out)])
    assert rc == 0
    assert (out / "volume.npy").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert "views" in meta["layers"]
    assert meta["resolution"]["depth_resolved"] is False


def test_compare_two_runs(workspace, capsys):
    cfg_path, ingest, solve_dir, tmp = workspace
    a, b = tmp / "va", tmp / "vb"
    for out in (a, b):
        main(["reconstruct", "--config", str(cfg_path), "--solve", str(solve_dir),
              "--out", str(out), "--bootstrap", "0", "--no-systematic"])
    rc = main(["compare", "--a", str(a), "--b", str(b)])
    assert rc == 0
    assert "unchanged" in capsys.readouterr().out


def test_compare_reports_a_missing_run(tmp_path, capsys):
    rc = main(["compare", "--a", str(tmp_path / "x"), "--b", str(tmp_path / "y")])
    assert rc == 1
    assert "volume_full.npz" in capsys.readouterr().out
