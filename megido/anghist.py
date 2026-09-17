"""Angular histograms and the counts_<exp>.npz artifact.

This npz layout is the ingest seam: cafeteria_3d_modeling's
muontomo.io.load_phantom_dir reads {values, xedges, yedges, name} plus a
meta.json, so the ported solver consumes these files unchanged.

Counts stay raw int64. The Poisson bootstrap and MLEM both need un-normalised
integers; do not scale them here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.config import Binning
from megido.tracks import Tracks


@dataclass(frozen=True)
class AngularHist:
    values: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    name: str = "txty"

    @property
    def total(self) -> int:
        return int(self.values.sum())

    def __add__(self, other: "AngularHist") -> "AngularHist":
        if not np.allclose(self.xedges, other.xedges) or not np.allclose(self.yedges, other.yedges):
            raise ValueError("cannot add histograms with different binning")
        return AngularHist(values=self.values + other.values, xedges=self.xedges,
                           yedges=self.yedges, name=self.name)


def histogram_tracks(tracks: Tracks, binning: Binning) -> AngularHist:
    edges = binning.edges()
    v = tracks.valid & np.isfinite(tracks.tan_x) & np.isfinite(tracks.tan_y)
    values, xedges, yedges = np.histogram2d(
        tracks.tan_x[v], tracks.tan_y[v], bins=[edges, edges]
    )
    return AngularHist(values=values.astype(np.int64), xedges=xedges, yedges=yedges)


def save_counts(hist: AngularHist, out_dir: Path, exposure_id: str, meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"counts_{exposure_id}.npz"
    np.savez_compressed(path, values=hist.values, xedges=hist.xedges,
                        yedges=hist.yedges, name=np.array(hist.name))

    meta_path = out_dir / "meta.json"
    doc = json.loads(meta_path.read_text()) if meta_path.exists() else {"exposures": {}}
    doc.setdefault("exposures", {})[exposure_id] = dict(meta, total_counts=hist.total)
    meta_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


def load_counts(path: Path) -> AngularHist:
    with np.load(path) as d:
        return AngularHist(values=d["values"], xedges=d["xedges"],
                           yedges=d["yedges"], name=str(d["name"]))
