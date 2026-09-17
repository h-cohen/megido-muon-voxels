import numpy as np
import pytest
import yaml

from megido.config import load_site_config
from megido.detector import DetectorGeometry
from megido.pipeline import exposure_key, process_exposure
from megido.sim import simulate, write_raw_file


@pytest.fixture
def fake_site(tmp_path):
    """A two-run synthetic exposure plus a config pointing at it."""
    geom = DetectorGeometry.megiddo()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for rid in (200001, 200002):
        truth = simulate(geom, n_events=3000, seed=rid)
        write_raw_file(truth, data_dir / f"DET{rid}_BEAM_20260101_000000_filter.data")

    cfg_path = tmp_path / "site.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(data_dir),
        "frame": {"origin": "E0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 100},
        "exposures": [{
            "id": "E0",
            "runs": "DET200001-DET200002",
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241},
        }],
    }))
    return cfg_path, tmp_path / "out", data_dir


def test_processes_all_files_in_an_exposure(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    r = process_exposure(cfg, "E0", out)
    assert r.n_events == 6000
    assert r.n_valid > 0
    assert r.counts_path.exists()
    assert r.tracks_path.exists()
    assert not r.cached


def test_second_run_hits_the_cache(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    first = process_exposure(cfg, "E0", out)
    second = process_exposure(cfg, "E0", out)
    assert second.cached
    assert second.key == first.key
    assert second.n_valid == first.n_valid


def test_force_bypasses_the_cache(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    process_exposure(cfg, "E0", out)
    again = process_exposure(cfg, "E0", out, force=True)
    assert not again.cached


def test_key_changes_when_binning_changes(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    k1 = exposure_key(cfg, "E0")

    raw = yaml.safe_load(cfg_path.read_text())
    raw["binning"]["n_bins"] = 200
    cfg_path.write_text(yaml.safe_dump(raw))
    k2 = exposure_key(load_site_config(cfg_path), "E0")

    assert k1 != k2, "changing binning must invalidate the artifact"


def test_key_changes_when_a_new_file_appears(fake_site):
    cfg_path, out, data_dir = fake_site
    cfg = load_site_config(cfg_path)
    k1 = exposure_key(cfg, "E0")

    geom = DetectorGeometry.megiddo()
    truth = simulate(geom, n_events=100, seed=999)
    write_raw_file(truth, data_dir / "DET200002_BEAM_20260102_000000_filter.data")

    assert exposure_key(load_site_config(cfg_path), "E0") != k1


def test_counts_total_matches_valid_tracks_in_range(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    r = process_exposure(cfg, "E0", out)
    from megido.anghist import load_counts
    h = load_counts(r.counts_path)
    assert 0 < h.total <= r.n_valid
