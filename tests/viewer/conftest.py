"""Shared fixtures for viewer Playwright tests.

`run_fixture` writes a tiny synthetic run directory (not real campaign data)
so tests are fast and self-contained; Task 10's final smoke test is the one
gate that points at the real `runs/voxels` output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from megido.viewerbuild import build

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"


@pytest.fixture(scope="session")
def dist_path() -> Path:
    return build(VIEWER_DIR)


@pytest.fixture
def run_fixture(tmp_path):
    """Write a synthetic run dir; returns its Path. shape/layers overridable."""
    def _make(shape=(6, 5, 4), layers=("volume",), spacing_m=0.5,
              origin_m=(0.0, 0.0, 1.0), depth_resolved=False, hill=False):
        run = tmp_path / "run"
        run.mkdir()
        rng = np.random.default_rng(0)
        vol = rng.random(shape, dtype=np.float32)
        np.save(run / "volume.npy", vol)
        for name in layers:
            if name == "volume":
                continue
            arr = rng.random(shape, dtype=np.float32)
            np.save(run / f"{name}.npy", arr)
        if hill:
            gx = list(np.arange(0.0, 3.5, 0.5))
            gy = list(np.arange(0.0, 3.0, 0.5))
            nx, ny = len(gx), len(gy)
            H = np.full((nx, ny), 2.0, dtype=np.float32)
            # sigma varies with i (row) only, spanning 0.1..1.0, so the fixture
            # exercises a non-degenerate robustRange without depending on j.
            sigma = np.array(
                [[0.1 + 0.9 * i / (nx - 1) for _ in range(ny)] for i in range(nx)],
                dtype=np.float32,
            )
            np.save(run / "hill_surface.npy", H)
            np.save(run / "hill_surface_sigma.npy", sigma)
            hill_meta = {
                "gx": gx,
                "gy": gy,
                "a": 8.0,
                "variance_explained": 0.81,
                "ray_ve": 0.5,
                "scale_assumed": True,
                "note": "fixture",
            }
            (run / "hill_surface_meta.json").write_text(json.dumps(hill_meta))
        meta = {
            "shape": list(shape),
            "axis_order": "xyz",
            "origin_m": list(origin_m),
            "spacing_m": spacing_m,
            "units": "opacity density [1/m]",
            "value_range": [float(vol.min()), float(vol.max())],
            "suggested_iso": [float(vol.max()) * 0.3, float(vol.max()) * 0.6],
            "run": "run",
            "layers": list(layers),
            "resolution": {
                "max_baseline_m": 2.2,
                "n_positions": 2,
                "depth_resolved": depth_resolved,
                "verdict": "depth NOT resolved: synthetic fixture verdict"
                if not depth_resolved else "depth resolved: synthetic fixture verdict",
            },
        }
        (run / "meta.json").write_text(json.dumps(meta))
        return run
    return _make
