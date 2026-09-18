"""Statistical and systematic uncertainty of the voxel volume.

Spec section 7.4 names three products. Two live here:

  * a per-voxel statistical sigma from a Poisson bootstrap over the RAW COUNTS,
    pushed through the entire chain — Phase 2 baseline solve included — so the
    spread carries the baseline's own uncertainty rather than pretending the
    detector response is known;
  * the Phase 2 gauge direction as a SEPARATE systematic map. The absolute level
    of lambda is not measured (spec section 6.4), and which convention is chosen
    moves the volume. That is not a random error and must never be averaged into
    sigma.

The third, the analytic depth resolution, is geometric and lives in
megido.resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution, solve_baseline
from megido.config import SiteConfig
from megido.reconstruct import solve_voxels
from megido.voxels import VoxelGrid


@dataclass(frozen=True)
class BootstrapResult:
    mean: np.ndarray            # [nx, ny, nz]
    sigma: np.ndarray           # [nx, ny, nz]
    grid: VoxelGrid | None
    n_replicas: int
    replica_seeds: tuple[int, ...] = field(default=(), repr=False)

    def snr(self) -> np.ndarray:
        """mean / sigma, NaN where sigma is zero.

        Zero spread across replicas does not mean infinite confidence; it means
        the voxel never moved, which for a voxel no ray reaches is a statement
        about coverage, not about certainty.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.sigma > 0, self.mean / self.sigma, np.nan)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean.astype(np.float32),
            sigma=self.sigma.astype(np.float32),
            snr=self.snr().astype(np.float32),
            n_replicas=np.asarray(self.n_replicas, dtype=np.int64),
            origin=np.asarray(self.grid.origin if self.grid else (0.0, 0.0, 0.0)),
            spacing=np.asarray(self.grid.spacing if self.grid else 0.0),
        )


def voxel_bootstrap(grid: AnalysisGrid, cfg: SiteConfig, *,
                    n_replicas: int = 8, seed: int = 0,
                    cache_dir: str | Path | None = "runs/.cache",
                    solve_kwargs: dict | None = None) -> BootstrapResult:
    """Poisson-resample the counts and re-run the WHOLE chain per replica.

    Resampling counts and re-solving only the voxels would hold the baseline
    fixed and understate the error: with no open-sky run the detector response
    is fitted from the same counts, so its uncertainty is part of the answer's.
    That makes each replica a full baseline solve, which is why the default
    replica count is small and the CLI exposes it.

    The voxel LATTICE is pinned once, from the nominal (unresampled) counts,
    and reused by every replica. `solve_voxels` otherwise auto-derives its
    grid from `rows.t_reach()`, which depends on which sky bins survive
    resampling — a replica could then land on a different grid shape, and
    `np.stack` below would crash (or, worse, silently misalign with the
    nominal grid used for honest-SNR reporting downstream). Pinning removes
    that possibility rather than merely making it unlikely.
    """
    rng = np.random.default_rng(seed)
    kw = dict(solve_kwargs or {})
    stack: list[np.ndarray] = []

    # Nominal (unresampled) solve fixes the lattice; its own rho is not part
    # of the statistics, only its grid is.
    nominal_sol = solve_baseline(grid, cfg, **kw)
    vgrid: VoxelGrid = solve_voxels(nominal_sol, cfg, cache_dir=cache_dir,
                                    holdouts=False)["full"].grid

    for _ in range(n_replicas):
        counts = {eid: rng.poisson(v).astype(np.int64)
                  for eid, v in grid.counts.items()}
        sol = solve_baseline(AnalysisGrid(edges=grid.edges, counts=counts), cfg, **kw)
        fits = solve_voxels(sol, cfg, cache_dir=cache_dir, holdouts=False,
                            grid=vgrid)
        stack.append(fits["full"].rho3())

    arr = np.stack(stack)
    return BootstrapResult(mean=arr.mean(axis=0), sigma=arr.std(axis=0),
                           grid=vgrid, n_replicas=n_replicas)


def systematic_map(sol: BaselineSolution, cfg: SiteConfig, *,
                   base_quantile: float = 0.05,
                   alt_quantile: float = 0.25,
                   cache_dir: str | Path | None = "runs/.cache") -> np.ndarray:
    """How much of the volume comes from the gauge choice rather than the data.

    Phase 2 leaves one flat direction per position: adding a constant to
    lambda_p and rescaling that position's normalization changes nothing
    observable (spec section 6.4). `normalized_opacity` picks a convention — the
    5th percentile of each map is called open sky — and the inversion's
    per-position offsets absorb most, but not all, of a different choice.

    This map is the volume difference between the chosen convention and a
    deliberately different one. It is a systematic, not a sigma: it does not
    shrink with more counts, and it is reported alongside the statistical error
    rather than combined with it.

    The knob is the QUANTILE, not an additive shift. normalized_opacity
    subtracts its own quantile, so adding a constant to every opacity value
    leaves the result bit-identical — an additive probe would report a
    systematic of exactly zero and look reassuring while measuring nothing.
    """
    base = solve_voxels(sol, cfg, cache_dir=cache_dir, holdouts=False,
                        transparent_quantile=base_quantile)["full"]
    alt = solve_voxels(sol, cfg, cache_dir=cache_dir, holdouts=False,
                       transparent_quantile=alt_quantile)["full"]
    return alt.rho3() - base.rho3()
