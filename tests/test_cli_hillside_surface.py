import json
from pathlib import Path

import numpy as np
import pytest

from megido.cli import main

REAL_CONFIG = Path("configs/megido.yaml")
REAL_SOLVE = Path("runs/solve")
REAL_INGEST = Path("runs/ingest")


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

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert (out / "hill_surface.png").exists()


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
