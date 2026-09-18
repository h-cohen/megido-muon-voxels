import numpy as np
import pytest

from megido.config import Binning, Pose, load_site_config

CONFIG = "configs/megido.yaml"


def test_loads_four_exposures():
    cfg = load_site_config(CONFIG)
    assert [e.id for e in cfg.exposures] == ["P0", "T20a", "T20b", "P1"]
    assert cfg.site == "megido"
    assert cfg.x_axis_bearing_deg == 241


def test_run_ranges_are_inclusive():
    cfg = load_site_config(CONFIG)
    p0 = cfg.exposure("P0")
    assert p0.run_ids[0] == 200084
    assert p0.run_ids[-1] == 200104
    assert len(p0.run_ids) == 21


def test_norm_group_defaults_to_exposure_id():
    cfg = load_site_config(CONFIG)
    assert cfg.exposure("P0").norm_group == "P0"
    assert cfg.exposure("T20b").norm_group == "T20b"


def test_t20b_file_count():
    cfg = load_site_config(CONFIG)
    # T20b spans 200119-200144 with no internal gaps.
    files = cfg.files_for("T20b")
    assert len(files) == 26
    assert all(f.exists() for f in files)


def test_missing_run_ids_are_tolerated_not_errors(tmp_path):
    # Build a tiny data dir with a gap (300002 is missing) and confirm
    # files_for skips the missing id instead of raising.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "DET300001_BEAM_20260101_000000_filter.data").touch()
    (data_dir / "DET300003_BEAM_20260101_000000_filter.data").touch()

    config_path = tmp_path / "site.yaml"
    config_path.write_text(f"""
site: test
data_dir: {data_dir}
exposures:
  - id: GAP
    runs: DET300001-DET300003
    pose: {{x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 0}}
""")

    cfg = load_site_config(config_path)
    files = cfg.files_for("GAP")
    assert len(files) == 2


def test_exposure_file_counts_sum_to_seventy():
    cfg = load_site_config(CONFIG)
    counts = {e.id: len(cfg.files_for(e.id)) for e in cfg.exposures}
    assert counts == {"P0": 21, "T20a": 12, "T20b": 26, "P1": 11}
    assert sum(counts.values()) == 70


def test_zero_tilt_rotation_is_pure_yaw():
    r = Pose(0, 0, 0, tilt_deg=0, az_deg=0).rotation()
    assert np.allclose(r, np.eye(3))


def test_rotation_is_orthonormal():
    r = Pose(0, 0, 0, tilt_deg=20, az_deg=241).rotation()
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_tilt_moves_the_normal_off_zenith_by_the_stated_angle():
    r = Pose(0, 0, 0, tilt_deg=20, az_deg=0).rotation()
    normal = r @ np.array([0.0, 0.0, 1.0])
    assert np.degrees(np.arccos(normal[2])) == pytest.approx(20.0)


def test_binning_edges():
    b = Binning(t_max=1.25, n_bins=500)
    e = b.edges()
    assert len(e) == 501
    assert e[0] == pytest.approx(-1.25)
    assert e[-1] == pytest.approx(1.25)
    assert (e[1] - e[0]) == pytest.approx(0.005)


def test_site_config_supplies_volume_and_reconstruction_defaults(tmp_path):
    """Configs written before Phase 3 must still load."""
    from megido.config import load_site_config

    p = tmp_path / "old.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n"
    )
    cfg = load_site_config(p)
    assert cfg.volume.spacing_m == 0.25
    assert cfg.reconstruction.algorithm == "tv"


def test_site_config_reads_volume_and_reconstruction_blocks(tmp_path):
    from megido.config import load_site_config

    p = tmp_path / "new.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "volume: {z_min_m: 2.0, z_max_m: 9.0, spacing_m: 0.5, n_aperture_sub: 3}\n"
        "reconstruction: {n_iter: 40, tv_alpha: 0.02}\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n"
    )
    cfg = load_site_config(p)
    assert (cfg.volume.z_min_m, cfg.volume.z_max_m) == (2.0, 9.0)
    assert cfg.volume.n_aperture_sub == 3
    assert cfg.reconstruction.n_iter == 40
    assert cfg.reconstruction.tv_z_weight == 0.5   # untouched default


def test_volume_xy_box_survives_yaml_round_trip(tmp_path):
    from megido.config import load_site_config

    p = tmp_path / "box.yaml"
    p.write_text(
        "site: t\ndata_dir: /tmp\n"
        "volume: {xy_m: [[-3.0, 3.0], [-2.0, 2.0]]}\n"
        "exposures:\n"
        "  - id: P0\n    runs: DET1-DET2\n"
        "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n"
    )
    cfg = load_site_config(p)
    assert cfg.volume.xy_m == ((-3.0, 3.0), (-2.0, 2.0))
