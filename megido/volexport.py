"""The viewer data contract, and run-to-run comparison.

volume.npy + meta.json keeps Phase 4's viewer decoupled from the solver: the
viewer reads arrays and metadata, never the solver's internals, so either side
can be rebuilt without touching the other.

meta.json deliberately carries the resolution verdict. A viewer that can render
a crisp isosurface without being able to say that its height is unmeasured
would be a misleading instrument.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from megido.config import SiteConfig
from megido.reconstruct import VoxelSolution
from megido.resolution import campaign_resolution

_SKY_SIGMA_T = 0.05      # the Phase 2 sky grid's angular bin width


def _json_safe(obj):
    """Recursively strip numpy scalar/array types so json.dumps doesn't choke.

    `info` on a VoxelSolution is whatever the solver's internals happened to
    put there (chi2, iteration counts, ...), which is often numpy dtypes even
    though this module only ever produces plain Python values itself.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def export_volume(run_dir: str | Path, cfg: SiteConfig, *,
                  out_dir: str | Path | None = None) -> Path:
    """Write volume.npy + meta.json (+ any optional layers found) for the viewer."""
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)

    vol = VoxelSolution.load(run / "volume_full.npz")
    rho = vol.rho3().astype(np.float32)
    np.save(out / "volume.npy", rho)
    layers = ["volume"]

    unc = run / "uncertainty.npz"
    if unc.exists():
        with np.load(unc) as d:
            for name in ("sigma", "snr"):
                if name in d:
                    np.save(out / f"{name}.npy", d[name].astype(np.float32))
                    layers.append(name)

    for name in ("views", "rays", "systematic", "backprojection"):
        src = run / f"{name}.npy"
        if src.exists():
            if out != run:
                np.save(out / f"{name}.npy", np.load(src))
            layers.append(name)

    pos = np.maximum(rho, 0.0)
    p99 = float(np.percentile(pos, 99)) if pos.max() > 0 else 1.0
    res = campaign_resolution(cfg, sigma_t=_SKY_SIGMA_T,
                              feature_pitch_m=max(2.0, 4 * cfg.volume.spacing_m))

    meta = {
        "shape": list(rho.shape),
        "axis_order": "xyz",
        "origin_m": list(vol.grid.origin),
        "spacing_m": vol.grid.spacing,
        "units": "opacity density [1/m]",
        "value_range": [float(rho.min()), float(rho.max())],
        "suggested_iso": [round(0.3 * p99, 6), round(0.6 * p99, 6)],
        "run": run.name,
        "layers": layers,
        "inversion_version": vol.version,
        "offsets": vol.offsets,
        "fit_info": vol.info,
        "detectors": [
            {"id": e.id, "x": e.pose.x, "y": e.pose.y, "z": e.pose.z,
             "tilt_deg": e.pose.tilt_deg, "az_deg": e.pose.az_deg}
            for e in cfg.exposures
        ],
        "resolution": {
            "max_baseline_m": res["max_baseline_m"],
            "n_positions": res["n_positions"],
            "depth_resolution_m": res["depth_resolution_m"],
            "depth_resolved": res["depth_resolved"],
            "verdict": res["verdict"],
        },
    }
    (out / "meta.json").write_text(json.dumps(_json_safe(meta), indent=2) + "\n")
    return out / "volume.npy"


def compare_volumes(a: VoxelSolution, b: VoxelSolution) -> dict:
    """What changed between two runs.

    `verdict` is deliberately coarse. A delta smaller than a thousandth of the
    volume's own scale is reported as `unchanged` rather than as an improvement:
    with a limited-angle geometry, small deltas are noise far more often than
    they are news.
    """
    if a.grid != b.grid:
        raise ValueError(f"grid mismatch: {a.grid.key()} vs {b.grid.key()}")

    da, db = a.rho, b.rho
    delta = db - da
    scale = max(float(np.percentile(np.abs(da), 95)), 1e-12)
    rms = float(np.sqrt(np.mean(delta**2)))
    corr = (float(np.corrcoef(da, db)[0, 1])
            if da.std() > 0 and db.std() > 0 else float("nan"))
    mass_a, mass_b = float(da.sum()), float(db.sum())

    if rms < 1e-3 * scale:
        verdict = "unchanged"
    elif rms < 1e-1 * scale:
        verdict = "shifted slightly"
    else:
        verdict = "shifted substantially"

    return {
        "corr": corr,
        "rms_delta": rms,
        "max_abs_delta": float(np.abs(delta).max()),
        "mass_change_frac": (mass_b - mass_a) / mass_a if mass_a else float("nan"),
        "scale": scale,
        "verdict": verdict,
    }
