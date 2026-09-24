import json
from pathlib import Path

import numpy as np
import pytest

from megido.cli import _format_cross_position_line, main

REAL_CONFIG = Path("configs/megido.yaml")
REAL_SOLVE = Path("runs/solve")
REAL_INGEST = Path("runs/ingest")


def test_format_cross_position_line_survives_every_error_shape():
    """F2: the formatter must not crash on any of the error shapes
    cross_position_check can produce (a raised fit at fit_on[p], at
    null_flat, or at null_shuffled[p][q]), and must still render the
    successful entries."""
    xpos = {
        "a": 8.0,
        "fit_on": {
            "pos0": {"error": "too few populated cells"},
            "pos1": {"pos0": {"n": 5, "corr": 0.5, "ve": 0.1}},
        },
        "in_sample": {},
        "null_flat": {"error": "boom"},
        "null_shuffled": {
            "pos1": {"pos0": {"corr_mean": 0.1, "corr_max": 0.2,
                              "ve_mean": 0.05, "n_seeds": 2}},
        },
    }
    line = _format_cross_position_line(xpos)
    assert "fit pos0 -> ...: failed (too few populated cells)" in line
    assert "fit pos1 -> pos0: r=0.50 VE=10%" in line
    assert "shuffled null r 0.10..0.20, 2 seeds" in line
    assert "error: boom" in line


@pytest.mark.skipif(not REAL_CONFIG.exists() or not (REAL_SOLVE / "baseline.npz").exists(),
                     reason="needs the real configs/megido.yaml + runs/solve/baseline.npz")
def test_hillside_writes_surface_artifacts(tmp_path):
    out = tmp_path / "voxels"
    rc = main(["hillside", "--config", str(REAL_CONFIG), "--solve", str(REAL_SOLVE),
               "--out", str(out),
               "--surface-max-points", "150", "--surface-restarts", "1"])
    assert rc == 0

    npy_path = out / "hill_surface.npy"
    sigma_path = out / "hill_surface_sigma.npy"
    meta_path = out / "hill_surface_meta.json"
    assert npy_path.exists()
    assert sigma_path.exists()
    assert meta_path.exists()

    H = np.load(npy_path)
    sigma = np.load(sigma_path)
    assert H.ndim == 2
    assert sigma.shape == H.shape

    text = meta_path.read_text()

    def _reject_constants(c):
        raise ValueError(f"non-finite JSON constant found: {c}")

    # Must be STRICT valid JSON: bare NaN/Infinity tokens (as opposed to the
    # substring "NaN" inside the honesty `note` prose, which is fine) are not
    # valid JSON and would raise here via parse_constant.
    meta = json.loads(text, parse_constant=_reject_constants)

    for key in ("variance_explained", "coverage_frac", "scale_assumed", "note",
                "gx", "gy", "a", "coverage_radius_m", "n_rays", "detectors", "units"):
        assert key in meta, f"meta missing {key}"

    assert meta["scale_assumed"] is True
    assert isinstance(meta["gx"], list)
    assert isinstance(meta["gy"], list)
    assert len(meta["gx"]) == H.shape[0]
    assert len(meta["gy"]) == H.shape[1]
    assert 0.0 <= meta["variance_explained"] <= 1.0 or meta["variance_explained"] < 0
    assert 0.0 <= meta["coverage_frac"] <= 1.0

    for key in ("ray_ve", "ray_ve_raw", "ray_rms", "n_rays_checked",
                "ray_offsets", "ray_per_position", "ray_check_note"):
        assert key in meta, f"meta missing {key}"
    assert meta["n_rays_checked"] > 0
    assert meta["ray_rms"] is None or meta["ray_rms"] >= 0

    grid_path = out / "hill_residual_grid.npy"
    assert grid_path.exists()
    grid = np.load(grid_path)
    assert grid.shape == H.shape

    assert "ray_cross_position" in meta
    xpos = meta["ray_cross_position"]
    assert "fit_on" in xpos and "null_flat" in xpos and "null_shuffled" in xpos
    assert "ray_cross_position_note" in meta
    assert meta.get("residual_grid_file") == "hill_residual_grid.npy"
    assert "residual_grid_lim" in meta

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert (out / "hill_surface.png").exists()
    assert (out / "hill_residual.png").exists()


@pytest.mark.skipif(not REAL_CONFIG.exists() or not (REAL_SOLVE / "baseline.npz").exists()
                     or not REAL_INGEST.exists(),
                     reason="needs the real configs/megido.yaml + runs/solve/baseline.npz "
                            "+ runs/ingest counts")
def test_hillside_run_produces_heteroscedastic_meta(tmp_path):
    """--run rebuilds per-sky-bin counts from the ingest grid (baseline.npz
    itself carries no counts -- BaselineSolution.load sets counts={}
    deliberately) and turns on heteroscedastic Poisson GP noise."""
    out = tmp_path / "voxels"
    rc = main(["hillside", "--config", str(REAL_CONFIG), "--solve", str(REAL_SOLVE),
               "--out", str(out), "--run", str(REAL_INGEST),
               "--surface-max-points", "150", "--surface-restarts", "1"])
    assert rc == 0

    meta = json.loads((out / "hill_surface_meta.json").read_text())
    assert meta["heteroscedastic"] is True
    assert "length_scale" in json.dumps(meta) or "length_scale_m" in meta
    assert "length_scale_m" in meta
    assert "q_hi" in meta
    assert 0.0 < meta["q_hi"] <= 1.0


@pytest.mark.skipif(not REAL_CONFIG.exists() or not (REAL_SOLVE / "baseline.npz").exists(),
                     reason="needs the real configs/megido.yaml + runs/solve/baseline.npz")
def test_hillside_without_run_falls_back_to_geometry_only(tmp_path, capsys):
    """Omitting --run must not crash: it prints a note pointing at --run and
    writes heteroscedastic: false."""
    out = tmp_path / "voxels"
    rc = main(["hillside", "--config", str(REAL_CONFIG), "--solve", str(REAL_SOLVE),
               "--out", str(out), "--run", str(tmp_path / "no-such-ingest-dir"),
               "--surface-max-points", "150", "--surface-restarts", "1"])
    assert rc == 0

    meta = json.loads((out / "hill_surface_meta.json").read_text())
    assert meta["heteroscedastic"] is False

    captured = capsys.readouterr()
    assert "--run" in captured.out
