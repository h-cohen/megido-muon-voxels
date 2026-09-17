import pytest
import yaml

from megido.cli import main
from megido.detector import DetectorGeometry
from megido.sim import simulate, write_raw_file


@pytest.fixture
def site(tmp_path):
    geom = DetectorGeometry.megiddo()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    truth = simulate(geom, n_events=3000, seed=11)
    write_raw_file(truth, data_dir / "DET200001_BEAM_20260101_000000_filter.data")

    cfg = tmp_path / "site.yaml"
    cfg.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(data_dir),
        "frame": {"origin": "E0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 100},
        "exposures": [{
            "id": "E0",
            "runs": "DET200001-DET200001",
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241},
        }],
    }))
    return cfg, tmp_path / "out"


def test_ingest_returns_zero_and_writes_artifacts(site, capsys):
    cfg, out = site
    rc = main(["ingest", "--config", str(cfg), "--out", str(out)])
    assert rc == 0
    assert (out / "counts_E0.npz").exists()
    assert (out / "tracks_E0.parquet").exists()
    assert "E0" in capsys.readouterr().out


def test_validate_prints_a_report(site, capsys):
    cfg, out = site
    rc = main(["validate", "--config", str(cfg), "--exposure", "E0"])
    captured = capsys.readouterr().out
    assert "adjacency" in captured
    assert "acceptance_cutoff" in captured
    assert rc in (0, 1)


def test_validate_returns_one_when_a_check_fails(site, capsys, monkeypatch):
    cfg, _ = site
    from megido import validate as V
    monkeypatch.setattr(
        V, "run_all",
        lambda *a, **k: [V.Check("forced", False, 0.0, "never", "forced failure")],
    )
    assert main(["validate", "--config", str(cfg), "--exposure", "E0"]) == 1


def test_unknown_command_returns_nonzero():
    with pytest.raises(SystemExit):
        main(["nonsense"])
