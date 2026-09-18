import numpy as np
import pytest
import yaml

from megido.anghist import AngularHist, save_counts
from megido.cli import main


@pytest.fixture
def site(tmp_path):
    cfg = tmp_path / "site.yaml"
    cfg.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(tmp_path),
        "frame": {"origin": "P0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 500},
        "exposures": [
            {"id": "P0", "runs": "DET100001-DET100001",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
            {"id": "T20", "runs": "DET100002-DET100002",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
        ],
    }))
    run_dir = tmp_path / "ingest"
    edges = np.linspace(-1.25, 1.25, 501)
    rng = np.random.default_rng(0)
    for eid in ("P0", "T20"):
        values = rng.poisson(30, size=(500, 500)).astype(np.int64)
        save_counts(AngularHist(values=values, xedges=edges, yedges=edges),
                    run_dir, eid, meta={})
    return cfg, run_dir, tmp_path / "solve"


def test_solve_writes_its_artifacts_and_prints_a_report(site, capsys):
    """CLI mechanics. The fixture is structureless noise, so the physics checks
    are expected to fail — that is asserted separately below."""
    cfg, run_dir, out = site
    main(["solve", "--config", str(cfg), "--run", str(run_dir),
          "--out", str(out), "--iters", "4"])
    assert (out / "baseline.npz").exists()
    printed = capsys.readouterr().out.lower()
    assert "flux index" in printed
    assert "normalizations" in printed


def test_solve_returns_nonzero_when_the_data_fails_validation(site):
    """The fixture is uniform Poisson noise with no angular structure. A model
    that claimed to fit it would be broken, so the gate must reject it and the
    command must exit nonzero — this is the CLI's value as a gate."""
    cfg, run_dir, out = site
    assert main(["solve", "--config", str(cfg), "--run", str(run_dir),
                 "--out", str(out), "--iters", "4"]) == 1


def test_solve_prints_the_validation_report(site, capsys):
    cfg, run_dir, out = site
    main(["solve", "--config", str(cfg), "--run", str(run_dir),
          "--out", str(out), "--iters", "4"])
    printed = capsys.readouterr().out
    assert "deviance_per_bin" in printed
    assert "checks passed" in printed


def test_solve_reports_a_missing_run_directory(site, capsys):
    cfg, _, out = site
    rc = main(["solve", "--config", str(cfg), "--run", str(out / "nope"),
               "--out", str(out), "--iters", "2"])
    assert rc == 1


def test_solve_solution_reloads(site):
    from megido.baseline import BaselineSolution
    cfg, run_dir, out = site
    main(["solve", "--config", str(cfg), "--run", str(run_dir),
          "--out", str(out), "--iters", "4"])
    sol = BaselineSolution.load(out / "baseline.npz")
    assert sol.coeffs.shape[0] > 0
    assert len(sol.opacity) >= 1
