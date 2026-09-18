"""Exposure registry. Poses live in data (configs/*.yaml), never in code.

Adding a datapoint: drop files in data_dir, append one exposure block, re-run.

Pose coordinates are METRES; every other length in Phase 1 is centimetres.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_RUN_RANGE = re.compile(r"^DET(\d+)\s*-\s*DET(\d+)$")


@dataclass(frozen=True)
class Pose:
    x: float          # metres — detector position in the site frame
    y: float          # metres
    z: float          # metres
    tilt_deg: float   # degrees from zenith
    az_deg: float     # degrees, bearing of the detector's local +x (bar) axis

    def rotation(self) -> np.ndarray:
        """R = Rz(az) @ Ry(tilt). Tilt takes the normal off zenith; az is the
        bearing of the detector's local +x (bar) axis."""
        t = np.radians(self.tilt_deg)
        a = np.radians(self.az_deg)
        ry = np.array([[np.cos(t), 0.0, np.sin(t)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(t), 0.0, np.cos(t)]])
        rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a), np.cos(a), 0.0],
                       [0.0, 0.0, 1.0]])
        return rz @ ry


@dataclass(frozen=True)
class PoseSigma:
    xy: float = 0.05     # metres
    ang: float = 1.0     # degrees


@dataclass(frozen=True)
class Exposure:
    id: str
    run_ids: tuple[int, ...]
    pose: Pose
    pose_sigma: PoseSigma = field(default_factory=PoseSigma)
    norm_group: str = ""
    note: str = ""


@dataclass(frozen=True)
class Binning:
    t_max: float = 1.25
    n_bins: int = 500

    def edges(self) -> np.ndarray:
        return np.linspace(-self.t_max, self.t_max, self.n_bins + 1)


@dataclass(frozen=True)
class Volume:
    """The voxel lattice the Phase 3 inversion solves on.

    `spacing_m` defaults to 0.25 rather than the spec's 0.10 starting value:
    Megiddo's rock sits at roughly 5-15 m, not the cafeteria's 3 m ceiling, so a
    0.10 m lattice over that range is about 5e7 unknowns against 2e4
    measurements. megido.resolution reports what is actually resolvable, and
    this is a config field precisely so it can be retuned from that evidence.
    """
    z_min_m: float = 1.0
    z_max_m: float = 12.0
    spacing_m: float = 0.25
    xy_m: tuple | None = None       # ((x0, x1), (y0, y1)); None -> from ray footprints
    n_aperture_sub: int = 4         # sub-rays per axis across the aperture (n^2 total)


@dataclass(frozen=True)
class Reconstruction:
    algorithm: str = "tv"           # "sirt" | "tv"
    n_iter: int = 150
    nonneg: bool = True
    chi2_target: float = 1.0        # discrepancy-principle stop for plain SIRT
    tv_alpha: float = 0.01          # TV threshold as a fraction of x's p95
    tv_z_weight: float = 0.5        # anisotropic TV: relative weight of z gradients
    seed: int = 42


@dataclass(frozen=True)
class SiteConfig:
    site: str
    data_dir: Path
    frame_origin: str
    x_axis_bearing_deg: float
    exposures: tuple[Exposure, ...]
    binning: Binning
    volume: Volume = field(default_factory=Volume)
    reconstruction: Reconstruction = field(default_factory=Reconstruction)

    def exposure(self, eid: str) -> Exposure:
        for e in self.exposures:
            if e.id == eid:
                return e
        raise KeyError(f"no exposure {eid!r}; have {[e.id for e in self.exposures]}")

    def files_for(self, eid: str) -> list[Path]:
        """Existing data files for an exposure, sorted by run id.

        A run id in the configured range with no file on disk is skipped, not an
        error: the campaign has gaps (e.g. DET200117, DET200118).
        """
        out: list[Path] = []
        for rid in self.exposure(eid).run_ids:
            matches = sorted(self.data_dir.glob(f"DET{rid}_*.data"))
            out.extend(matches)
        return out


def _parse_runs(spec: str) -> tuple[int, ...]:
    m = _RUN_RANGE.match(str(spec).strip())
    if not m:
        raise ValueError(f"run spec {spec!r} must look like 'DET200084-DET200104'")
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi < lo:
        raise ValueError(f"run range {spec!r} is reversed")
    return tuple(range(lo, hi + 1))


def load_site_config(path: str | Path) -> SiteConfig:
    raw = yaml.safe_load(Path(path).read_text())
    frame = raw.get("frame", {})
    binning = Binning(**raw.get("binning", {}))

    exposures = []
    for block in raw["exposures"]:
        eid = block["id"]
        exposures.append(
            Exposure(
                id=eid,
                run_ids=_parse_runs(block["runs"]),
                pose=Pose(**block["pose"]),
                pose_sigma=PoseSigma(**block.get("pose_sigma", {})),
                norm_group=block.get("norm_group", eid),
                note=block.get("note", ""),
            )
        )
    vol_raw = dict(raw.get("volume", {}))
    if vol_raw.get("xy_m") is not None:
        vol_raw["xy_m"] = tuple(tuple(float(v) for v in pair) for pair in vol_raw["xy_m"])
    volume = Volume(**vol_raw)
    reconstruction = Reconstruction(**raw.get("reconstruction", {}))

    return SiteConfig(
        site=raw["site"],
        data_dir=Path(raw["data_dir"]),
        frame_origin=frame.get("origin", ""),
        x_axis_bearing_deg=float(frame.get("x_axis_bearing_deg", 0.0)),
        exposures=tuple(exposures),
        binning=binning,
        volume=volume,
        reconstruction=reconstruction,
    )
