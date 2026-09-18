"""Analysis grid for Phase 2: Phase 1's fine histograms, coarsened for fitting.

Phase 1 emits 500x500 bins of 0.005 tan units, which is the right resolution to
preserve information but averages about 4.8 counts per bin over the campaign's
868,000 tracks. A Poisson fit needs counts, not noise, so Phase 2 rebins by 10
to 50x50 bins of 0.05, giving roughly 480 counts in each occupied bin.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.anghist import AngularHist, load_counts


def rebin(hist: AngularHist, factor: int) -> AngularHist:
    """Sum `factor` x `factor` blocks of bins. Counts are conserved exactly."""
    n = hist.values.shape[0]
    if n % factor:
        raise ValueError(f"factor {factor} must divide the bin count {n}")
    m = n // factor
    values = hist.values.reshape(m, factor, m, factor).sum(axis=(1, 3))
    edges = hist.xedges[::factor]
    return AngularHist(values=values.astype(np.int64), xedges=edges,
                       yedges=edges, name=hist.name)


@dataclass(frozen=True)
class AnalysisGrid:
    edges: np.ndarray
    counts: dict[str, np.ndarray]

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def tan_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """tx, ty as [m, m] arrays. tx varies along axis 0, matching the
        histogram convention `np.histogram2d(tan_x, tan_y)` used in Phase 1."""
        c = self.centers
        return np.meshgrid(c, c, indexing="ij")

    def occupied(self, min_counts: int = 1) -> np.ndarray:
        total = np.zeros((self.n_bins, self.n_bins), dtype=np.int64)
        for v in self.counts.values():
            total = total + v
        return total >= min_counts


def load_analysis_grid(run_dir: Path, exposure_ids: Sequence[str],
                       factor: int = 10) -> AnalysisGrid:
    run_dir = Path(run_dir)
    edges = None
    counts: dict[str, np.ndarray] = {}
    for eid in exposure_ids:
        h = rebin(load_counts(run_dir / f"counts_{eid}.npz"), factor)
        if edges is None:
            edges = h.xedges
        elif edges.shape != h.xedges.shape or not np.allclose(edges, h.xedges):
            raise ValueError(f"exposure {eid!r} has different binning from the others")
        counts[eid] = h.values
    if edges is None:
        raise ValueError("no exposures given")
    return AnalysisGrid(edges=edges, counts=counts)
