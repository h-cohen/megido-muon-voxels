import json

import numpy as np
import pytest

from megido.config import load_site_config
from megido.reconstruct import VoxelSolution
from megido.volexport import compare_volumes, export_volume
from megido.voxels import VoxelGrid

CONFIG = """
site: t
data_dir: /tmp
volume: {z_min_m: 5.0, z_max_m: 15.0, spacing_m: 0.5}
exposures:
  - id: P0
    runs: DET1-DET2
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
  - id: P1
    runs: DET3-DET4
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
"""

# z 5-15 m at a 2.2 m baseline mirrors the real Megiddo campaign (see
# tests/test_resolution.py): dz at mid-range is metres-scale, far coarser than
# the 0.5 m voxel spacing, so depth is genuinely NOT resolved here. The
# original brief fixture used z 1-3 m, which the closed-form resolution
# formula (megido/resolution.py) actually resolves at this baseline+spacing;
# that was a bug in the brief's numbers, not in campaign_resolution (which is
# already covered, and agrees with this range, in test_resolution.py).
GRID = VoxelGrid(origin=(-1.0, -1.0, 5.0), spacing=0.5, shape=(4, 4, 4))


def _cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(CONFIG)
    return load_site_config(p)


def _vol(seed=0) -> VoxelSolution:
    rng = np.random.default_rng(seed)
    return VoxelSolution(rho=np.abs(rng.normal(size=GRID.n_voxels)), grid=GRID,
                         offsets={"pos0": 0.1, "pos1": -0.1},
                         position_ids=("pos0", "pos1"),
                         info={"best_chi2": 1.2, "algorithm": "tv"})


def test_export_writes_the_contract(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")

    out = export_volume(run, _cfg(tmp_path))
    assert (out.parent / "volume.npy").exists()
    meta = json.loads((out.parent / "meta.json").read_text())
    assert meta["shape"] == list(GRID.shape)
    assert meta["axis_order"] == "xyz"
    assert meta["spacing_m"] == pytest.approx(0.5)
    assert meta["origin_m"] == list(GRID.origin)


def test_meta_carries_the_detectors_with_their_tilts(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    export_volume(run, _cfg(tmp_path))

    meta = json.loads((run / "meta.json").read_text())
    ids = {d["id"] for d in meta["detectors"]}
    assert ids == {"P0", "P1"}
    assert all("tilt_deg" in d and "az_deg" in d for d in meta["detectors"])


def test_meta_carries_the_resolution_verdict(tmp_path):
    """The viewer must be able to show what the campaign cannot resolve without
    re-deriving it."""
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    export_volume(run, _cfg(tmp_path))

    meta = json.loads((run / "meta.json").read_text())
    assert meta["resolution"]["depth_resolved"] is False
    assert "max_baseline_m" in meta["resolution"]


def test_optional_layers_are_exported_when_present(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    np.savez_compressed(run / "uncertainty.npz",
                        mean=np.zeros(GRID.shape, dtype=np.float32),
                        sigma=np.ones(GRID.shape, dtype=np.float32),
                        snr=np.ones(GRID.shape, dtype=np.float32),
                        n_replicas=np.asarray(4), origin=np.asarray(GRID.origin),
                        spacing=np.asarray(GRID.spacing))
    np.save(run / "views.npy", np.full(GRID.shape, 2, dtype=np.int16))
    np.save(run / "rays.npy", np.full(GRID.shape, 7, dtype=np.int32))

    export_volume(run, _cfg(tmp_path))
    meta = json.loads((run / "meta.json").read_text())
    assert (run / "sigma.npy").exists()
    assert (run / "views.npy").exists()
    assert set(meta["layers"]) >= {"volume", "sigma", "snr", "views", "rays"}


def test_export_without_optional_layers_still_succeeds(tmp_path):
    run = tmp_path / "run"
    _vol().save(run / "volume_full.npz")
    export_volume(run, _cfg(tmp_path))
    meta = json.loads((run / "meta.json").read_text())
    assert meta["layers"] == ["volume"]


def test_compare_of_a_volume_with_itself_reports_no_change():
    v = _vol()
    r = compare_volumes(v, v)
    assert r["corr"] == pytest.approx(1.0)
    assert r["rms_delta"] == pytest.approx(0.0)
    assert r["verdict"] == "unchanged"


def test_compare_detects_a_real_change():
    a, b = _vol(0), _vol(1)
    r = compare_volumes(a, b)
    assert r["rms_delta"] > 0
    assert r["verdict"] != "unchanged"


def test_compare_rejects_mismatched_grids():
    a = _vol()
    other = VoxelSolution(rho=np.zeros(8), grid=VoxelGrid((0, 0, 0), 1.0, (2, 2, 2)),
                          offsets={}, position_ids=(), info={})
    with pytest.raises(ValueError, match="grid"):
        compare_volumes(a, other)
