"""Ingest for campaigns delivered as pre-binned ROOT histograms.

Some campaigns (the TAU cafeteria run) arrive not as raw `.data` but as the
DAQ's own `txty` TH2: counts over (tan theta_x, tan theta_y), axis 0 = tx,
the same convention Phase 1's `np.histogram2d(tan_x, tan_y)` uses. This module
writes those into the exact `counts_<id>.npz` seam Phase 1 produces, so every
stage downstream runs unchanged.

The ROOT bins must coincide with the site's configured binning after cropping:
a mismatch is an error, never a silent resample.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.anghist import AngularHist, save_counts
from megido.config import SiteConfig


@dataclass(frozen=True)
class RootIngestResult:
    source_id: str
    path: Path
    total_in_file: int
    total_kept: int


def _crop_index(file_edges: np.ndarray, target: np.ndarray, axis: str) -> int:
    i0 = int(np.argmin(np.abs(file_edges - target[0])))
    sl = file_edges[i0:i0 + target.size]
    if sl.size != target.size or not np.allclose(sl, target, atol=1e-9):
        raise ValueError(
            f"ROOT {axis} bins ({file_edges.size - 1} over {file_edges[0]:g}..{file_edges[-1]:g}) "
            f"do not contain the configured binning ({target.size - 1} over "
            f"{target[0]:g}..{target[-1]:g}) on a common edge set")
    return i0


def read_root_counts(path: Path, hist: str, edges: np.ndarray) -> tuple[AngularHist, int]:
    """Load a ROOT TH2 and crop it to `edges`. Returns (hist, total counts in file)."""
    import uproot

    with uproot.open(path) as f:
        h = f[hist]
        values = np.asarray(h.values(), dtype=np.float64)
        xe = np.asarray(h.axes[0].edges(), dtype=np.float64)
        ye = np.asarray(h.axes[1].edges(), dtype=np.float64)

    if not np.allclose(values, np.round(values)) or (values < 0).any():
        raise ValueError(f"{Path(path).name}:{hist} is not a histogram of non-negative counts")
    i0 = _crop_index(xe, edges, "x")
    j0 = _crop_index(ye, edges, "y")
    n = edges.size - 1
    kept = np.round(values[i0:i0 + n, j0:j0 + n]).astype(np.int64)
    return (AngularHist(values=kept, xedges=edges.copy(), yedges=edges.copy(), name=hist),
            int(round(values.sum())))


def ingest_root(cfg: SiteConfig, out_dir: Path) -> list[RootIngestResult]:
    """Write counts_<id>.npz for every ROOT-backed exposure and the sky reference."""
    out_dir = Path(out_dir)
    edges = cfg.binning.edges()
    sources: list[tuple[str, str, str, dict]] = []
    for e in cfg.exposures:
        if not e.root_file:
            raise ValueError(f"exposure {e.id!r} has no root_file; use `ingest` for raw data")
        sources.append((e.id, e.root_file, e.root_hist,
                        {"pose": vars(e.pose), "norm_group": e.norm_group, "note": e.note}))
    if cfg.sky_reference is not None:
        s = cfg.sky_reference
        sources.append((s.id, s.root_file, s.root_hist, {"role": "open_sky_reference"}))

    results = []
    for sid, fname, hist, meta in sources:
        path = cfg.data_dir / fname
        h, total = read_root_counts(path, hist, edges)
        out = save_counts(h, out_dir, sid, dict(meta, exposure=sid, source=str(path),
                                                 root_hist=hist, total_in_file=total))
        results.append(RootIngestResult(sid, out, total, h.total))
    return results
