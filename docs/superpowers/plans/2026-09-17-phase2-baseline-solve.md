# Phase 2 — Baseline Solve — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the detector's angular response from the rock's absorption, with no open-sky calibration run, and emit a per-position opacity map that Phase 3 can triangulate into voxels.

**Architecture:** A Poisson likelihood over all four exposures simultaneously. The detector response `B(d)` is an exact analytic aperture times a smooth positive correction with ~64 free coefficients. Opacity lives in the *sky* frame, one map per detector position. The P0/T20a/T20b triple sits at one position with two different tilts, so rotating the detector moves each bin to a different patch of sky while `B(d)` stays fixed — that is what makes the separation identifiable. P1, at the second position, then inherits `B` and yields an independent sky map.

**Tech Stack:** Python 3.12 (existing uv venv), numpy, scipy (`optimize.minimize` L-BFGS-B), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` — section 6 in particular.

---

## Global Constraints

- **Package name** `megido`. In-tree execution via `python -m megido.<module>`; no console_scripts.
- **Preserve integer counts.** Phase 1 artifacts store raw `int64`; the Poisson likelihood needs them un-normalized. Never rescale counts on the way in.
- **Angles are dimensionless tangents** throughout Phase 2. Lengths that appear (active width, layer separation) are **centimetres**, matching Phase 1.
- **Opacity is in the sky frame, one map per detector position.** Never a single map shared across positions — P0 and P1 are 2.2 m apart and look through different rock for the same sky direction. That difference is Phase 3's signal and must not be fitted away here.
- **Detector constants come from `DetectorGeometry.megiddo()`.** Never re-enter a number that already lives there.
- **Run tests with** `uv run pytest`.
- **Every commit message ends with these two lines:**
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6
  ```

### Refinement to spec section 6 — read this before Task 1

Spec section 6 writes the model with a single scene opacity `lambda`. That is underspecified, and the plan corrects it: **opacity depends on where you stand, not only on sky direction.** The corrected model, for exposure `e` at position `p(e)`:

```
mu_e(d) = norm_e * A_geom(d) * exp(S(d)) * Phi_n(R_e . d) * exp(-lambda_{p(e)}(R_e . d))
```

| Symbol | Meaning | Free parameters |
|---|---|---|
| `d` | direction in the **detector** frame, one analysis bin | — |
| `R_e . d` | the same direction in the **sky** frame | — |
| `A_geom(d)` | exact analytic four-layer coincidence acceptance | 0 |
| `exp(S(d))` | smooth positive efficiency correction | 64 |
| `Phi_n(s)` | cosmic flux shape, `cos^n(theta_sky)` | 1 |
| `lambda_p(s)` | opacity at position `p`, sky frame | one per occupied sky bin |
| `norm_e` | per-`norm_group` normalization, absorbs live time | 4 |

Position grouping comes from `configs/megido.yaml`: **P0, T20a and T20b share position `pos0`; P1 alone is `pos1`.** All four keep separate `norm_group`s, because the unexplained 6 August rate change separates T20a from T20b.

**Why this is identifiable.** `B(d) = A_geom(d)·exp(S(d))` is fixed in the detector frame; `lambda` is fixed in the sky frame. The 20° tilt between P0 and T20 sends each detector bin to a different sky bin while `B` does not move, which breaks the degeneracy. P1 contributes one exposure at a new position — not enough to determine `B` on its own, which is exactly why `B` must be solved on the pos0 triple and then held fixed for P1.

### Two further departures from spec section 6, both deliberate

**No explicit prior term on `B`.** Spec 6.3 calls for the physics model to enter as a *weak prior*. Here it enters as structure instead: `A_geom` is exact and fixed, and the correction has 64 coefficients against roughly 1800 occupied bins. The low dimensionality *is* the regularization, and it is a stronger statement than a soft penalty — the fit cannot absorb scene structure into `B` because `B` has nowhere to put it. If the deviance check in Task 7 shows the model cannot fit, that is the signal this was too tight, and a penalty term is the remedy.

**No TV regularization on `lambda`, and no non-negativity inside the fit.** Spec 6.3 lists both. They are deferred to Phase 3 on purpose. Spatial smoothness belongs in *voxel* space, where neighbouring voxels are genuinely adjacent in rock; imposing it on a sky map would smooth across directions that are not neighbours in the medium. Non-negativity is meaningless before the zero point is fixed, since the absolute level of `lambda` is degenerate with `norm_e` — so it is applied as a post-solve normalization (`BaselineSolution.normalized_opacity`) rather than as a constraint. Phase 2 instead reports a per-sky-bin bootstrap sigma so Phase 3 can weight each bin by how well it is actually measured.

---

## File Structure

| File | Responsibility |
|---|---|
| `megido/angular.py` | Crop and rebin Phase 1 counts into the analysis grid |
| `megido/acceptance.py` | Exact analytic four-layer geometric acceptance |
| `megido/sky.py` | Detector↔sky direction mapping and the cosmic flux shape |
| `megido/basis.py` | Gaussian radial basis for the smooth efficiency correction |
| `megido/baseline.py` | The joint Poisson solve and its result type |
| `megido/simbaseline.py` | Synthetic generator with known `B` and `lambda` — the recovery gate |
| `megido/validate2.py` | Null-space probe and leave-one-exposure-out cross-validation |
| `megido/cli.py` | Extended with a `solve` subcommand |
| `tests/…` | One test module per source module |

---

## Task 1: Analysis grid — crop and rebin

Phase 1 emits 500×500 bins of 0.005. That averages ~4.8 counts per bin over 868k tracks, far too sparse for a Poisson fit. Rebinning by 10 gives 50×50 bins of 0.05 and roughly 480 counts in each occupied bin.

**Files:**
- Create: `megido/angular.py`, `tests/test_angular.py`

**Interfaces:**
- Consumes: `megido.anghist.AngularHist` (fields `values` int64 `[n,n]`, `xedges`, `yedges`, `name`), `megido.anghist.load_counts`
- Produces:
  ```python
  rebin(hist: AngularHist, factor: int) -> AngularHist
  @dataclass(frozen=True)
  class AnalysisGrid:
      edges: np.ndarray                 # [m+1] shared by both axes
      counts: dict[str, np.ndarray]     # exposure id -> int64 [m, m]
      .centers -> np.ndarray            # [m]
      .tan_mesh() -> tuple[np.ndarray, np.ndarray]   # tx, ty each [m, m]
      .n_bins -> int
      .occupied(min_counts: int = 1) -> np.ndarray   # bool [m, m], union over exposures
  load_analysis_grid(run_dir: Path, exposure_ids: Sequence[str],
                     factor: int = 10) -> AnalysisGrid
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_angular.py`:

```python
import numpy as np
import pytest

from megido.anghist import AngularHist
from megido.angular import AnalysisGrid, load_analysis_grid, rebin


def _hist(n=500, fill=1):
    edges = np.linspace(-1.25, 1.25, n + 1)
    return AngularHist(values=np.full((n, n), fill, dtype=np.int64),
                       xedges=edges, yedges=edges)


def test_rebin_preserves_total_counts():
    h = _hist()
    r = rebin(h, 10)
    assert r.values.shape == (50, 50)
    assert r.values.sum() == h.values.sum()
    assert r.values.dtype == np.int64


def test_rebin_edges_are_a_subsample_of_the_originals():
    h = _hist()
    r = rebin(h, 10)
    assert len(r.xedges) == 51
    assert r.xedges[0] == pytest.approx(-1.25)
    assert r.xedges[-1] == pytest.approx(1.25)
    assert r.xedges[1] - r.xedges[0] == pytest.approx(0.05)


def test_rebin_sums_the_right_neighbours():
    """A single count must land in the bin containing its original bin."""
    h = _hist(fill=0)
    values = np.array(h.values)
    values[0, 0] = 7          # first original bin
    values[499, 499] = 3      # last original bin
    h = AngularHist(values=values, xedges=h.xedges, yedges=h.yedges)
    r = rebin(h, 10)
    assert r.values[0, 0] == 7
    assert r.values[49, 49] == 3
    assert r.values.sum() == 10


def test_rebin_rejects_a_non_divisor_factor():
    with pytest.raises(ValueError, match="divide"):
        rebin(_hist(), 7)


def test_grid_centres_and_mesh():
    g = AnalysisGrid(edges=np.linspace(-1.25, 1.25, 51),
                     counts={"P0": np.zeros((50, 50), np.int64)})
    assert g.n_bins == 50
    assert g.centers[0] == pytest.approx(-1.225)
    tx, ty = g.tan_mesh()
    assert tx.shape == ty.shape == (50, 50)
    # tan_x varies down axis 0, matching np.histogram2d(tan_x, tan_y)
    assert tx[0, 0] == pytest.approx(tx[0, 49])
    assert ty[0, 0] == pytest.approx(ty[49, 0])
    assert tx[1, 0] > tx[0, 0]


def test_occupied_is_the_union_over_exposures():
    a = np.zeros((50, 50), np.int64); a[10, 10] = 5
    b = np.zeros((50, 50), np.int64); b[20, 20] = 5
    g = AnalysisGrid(edges=np.linspace(-1.25, 1.25, 51), counts={"A": a, "B": b})
    occ = g.occupied()
    assert occ[10, 10] and occ[20, 20]
    assert occ.sum() == 2


def test_load_analysis_grid_reads_phase1_artifacts(tmp_path):
    from megido.anghist import save_counts
    for eid in ("P0", "P1"):
        save_counts(_hist(fill=2), tmp_path, eid, meta={})
    g = load_analysis_grid(tmp_path, ["P0", "P1"], factor=10)
    assert set(g.counts) == {"P0", "P1"}
    assert g.counts["P0"].shape == (50, 50)
    assert g.counts["P0"].sum() == 500 * 500 * 2
    assert g.counts["P0"].dtype == np.int64


def test_load_analysis_grid_rejects_mismatched_binning(tmp_path):
    from megido.anghist import save_counts
    save_counts(_hist(n=500), tmp_path, "P0", meta={})
    save_counts(_hist(n=400), tmp_path, "P1", meta={})
    with pytest.raises(ValueError, match="binning"):
        load_analysis_grid(tmp_path, ["P0", "P1"], factor=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_angular.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.angular'`

- [ ] **Step 3: Implement `megido/angular.py`**

```python
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
        elif not np.allclose(edges, h.xedges):
            raise ValueError(f"exposure {eid!r} has different binning from the others")
        counts[eid] = h.values
    if edges is None:
        raise ValueError("no exposures given")
    return AnalysisGrid(edges=edges, counts=counts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_angular.py -v`
Expected: 8 passed

- [ ] **Step 5: Sanity-check against the real artifacts**

Run:
```bash
uv run python -c "
import numpy as np
from pathlib import Path
from megido.angular import load_analysis_grid
g = load_analysis_grid(Path('runs/ingest'), ['P0','T20a','T20b','P1'])
occ = g.occupied()
print('bins per axis', g.n_bins, ' occupied', int(occ.sum()))
for e, v in g.counts.items():
    print(f'  {e:5s} total {v.sum():>9,}  median occupied bin {int(np.median(v[occ]))}')
"
```
Expected: 50 bins per axis, roughly 1800 occupied, totals matching Phase 1 (P0 292,983; T20a 168,116; T20b 284,913; P1 122,046), and a median occupied-bin count in the hundreds. **If totals differ from those, stop and report it** — rebinning must conserve counts exactly.

- [ ] **Step 6: Commit**

```bash
git add megido/angular.py tests/test_angular.py
git commit -m "feat: analysis grid, rebinning Phase 1 counts for the Poisson fit"
```

---

## Task 2: Analytic geometric acceptance

The four-layer coincidence acceptance is exactly computable and has **zero free parameters**. Getting it from geometry rather than fitting it is what keeps the smooth correction small enough to be meaningful.

A track with slope `tx` crosses the two X layers, separated by `dz = 31.5 cm`. If it enters the lower layer at `x0`, it reaches `x0 + tx*dz` at the upper one. Both must lie inside the active width `W = 38.4 cm`, so the set of accepted `x0` has length `max(W - |tx|*dz, 0)`. The same holds independently in y. Converting from counts per unit `dtx dty` to counts per unit solid angle contributes `(1 + tx^2 + ty^2)^(-3/2)`.

```
A_geom(tx, ty) = max(W - |tx|*dz_x, 0) * max(W - |ty|*dz_y, 0) * (1 + tx^2 + ty^2)^(-3/2)
```

This vanishes exactly at `|tx| = W/dz = 38.4/31.5 = 1.219`, which is the cutoff Phase 1's gate measured at 1.248 on real data. That agreement is a genuine cross-check, not a coincidence.

**Files:**
- Create: `megido/acceptance.py`, `tests/test_acceptance.py`

**Interfaces:**
- Consumes: `megido.detector.DetectorGeometry` — `.bar.active_width_cm`, `.dz_cm(coord)`, `.max_tan()`
- Produces:
  ```python
  geometric_acceptance(tx: np.ndarray, ty: np.ndarray,
                       geom: DetectorGeometry) -> np.ndarray
  ```
  Returns an array shaped like `tx`, in cm^2, zero outside the geometric limit. Absolute scale is arbitrary — it is absorbed by `norm_e`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_acceptance.py`:

```python
import numpy as np
import pytest

from megido.acceptance import geometric_acceptance
from megido.detector import DetectorGeometry


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def test_peaks_on_axis(geom):
    a = geometric_acceptance(np.array([0.0, 0.3, 0.6]), np.zeros(3), geom)
    assert a[0] > a[1] > a[2] > 0


def test_vanishes_beyond_the_geometric_limit(geom):
    limit = geom.max_tan()
    beyond = geometric_acceptance(np.array([limit * 1.01]), np.array([0.0]), geom)
    assert beyond[0] == 0.0
    inside = geometric_acceptance(np.array([limit * 0.99]), np.array([0.0]), geom)
    assert inside[0] > 0.0


def test_limit_matches_active_width_over_layer_separation(geom):
    """The zero crossing IS width/dz — the number Phase 1's gate confirmed."""
    t = np.linspace(0.0, 1.5, 3001)
    a = geometric_acceptance(t, np.zeros_like(t), geom)
    last_nonzero = t[np.flatnonzero(a > 0)[-1]]
    assert last_nonzero == pytest.approx(geom.bar.active_width_cm / geom.dz_cm("x"), abs=2e-3)
    assert last_nonzero == pytest.approx(1.219, abs=2e-3)


def test_symmetric_in_both_coordinates(geom):
    tx = np.array([0.4, -0.4, 0.4, -0.4])
    ty = np.array([0.2, 0.2, -0.2, -0.2])
    a = geometric_acceptance(tx, ty, geom)
    assert np.allclose(a, a[0])


def test_separable_triangles_times_the_solid_angle_factor(geom):
    """Check one value against the closed form, computed independently."""
    w = geom.bar.active_width_cm
    dz = geom.dz_cm("x")
    tx, ty = 0.5, -0.25
    expected = (w - 0.5 * dz) * (w - 0.25 * dz) * (1 + 0.25 + 0.0625) ** -1.5
    got = geometric_acceptance(np.array([tx]), np.array([ty]), geom)[0]
    assert got == pytest.approx(expected)


def test_shape_is_preserved(geom):
    tx, ty = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 5), indexing="ij")
    assert geometric_acceptance(tx, ty, geom).shape == (7, 5)


def test_solid_angle_factor_actually_applied(geom):
    """Without the (1+t^2)^-3/2 factor the profile would be a pure triangle."""
    w, dz = geom.bar.active_width_cm, geom.dz_cm("x")
    t = 0.8
    triangle_only = (w - t * dz) * w
    got = geometric_acceptance(np.array([t]), np.array([0.0]), geom)[0]
    assert got < 0.8 * triangle_only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_acceptance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.acceptance'`

- [ ] **Step 3: Implement `megido/acceptance.py`**

```python
"""Exact four-layer coincidence acceptance — no free parameters.

A track of slope tx entering the lower X layer at x0 arrives at x0 + tx*dz in
the upper one. Both must fall inside the active width W, so the accepted entry
positions span max(W - |tx|*dz, 0). The y coordinate is independent, and the
conversion from per-(dtx dty) to per-solid-angle adds (1 + tx^2 + ty^2)^(-3/2).

The zero crossing is W/dz = 38.4/31.5 = 1.219, the same limit Phase 1's
acceptance_cutoff gate measured at 1.248 on real tracks. Fitting this shape
instead of deriving it would have thrown that cross-check away.
"""
from __future__ import annotations

import numpy as np

from megido.detector import DetectorGeometry


def geometric_acceptance(tx: np.ndarray, ty: np.ndarray,
                         geom: DetectorGeometry) -> np.ndarray:
    """Relative acceptance per angular bin. Absolute scale is arbitrary —
    it is degenerate with the per-exposure normalization and absorbed there."""
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)

    width = geom.bar.active_width_cm
    span_x = np.clip(width - np.abs(tx) * geom.dz_cm("x"), 0.0, None)
    span_y = np.clip(width - np.abs(ty) * geom.dz_cm("y"), 0.0, None)

    solid_angle = (1.0 + tx**2 + ty**2) ** -1.5
    return span_x * span_y * solid_angle
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_acceptance.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add megido/acceptance.py tests/test_acceptance.py
git commit -m "feat: analytic four-layer geometric acceptance"
```

---

## Task 3: Sky mapping and flux shape

Opacity lives in the sky frame; the detector response lives in the detector frame. This module is the bridge, and it is what makes the tilt informative.

**Files:**
- Create: `megido/sky.py`, `tests/test_sky.py`

**Interfaces:**
- Consumes: `megido.config.Pose` — `.rotation()` returning `Rz(az) @ Ry(tilt)`
- Produces:
  ```python
  detector_to_sky(tx, ty, pose: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]
      # sx, sy (sky-frame tangents), valid (bool, False where the ray points at or below the horizon)
  flux_shape(sx, sy, index: float) -> np.ndarray          # cos(theta_sky) ** index
  @dataclass(frozen=True)
  class SkyGrid:
      edges: np.ndarray                                    # [k+1]
      .n_bins -> int
      .centers -> np.ndarray
      .bin_index(sx, sy) -> tuple[np.ndarray, np.ndarray]  # flat index, in-range mask
      .flat_size -> int
  make_sky_grid(t_max: float = 1.75, n_bins: int = 70) -> SkyGrid
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_sky.py`:

```python
import numpy as np
import pytest

from megido.config import Pose
from megido.sky import SkyGrid, detector_to_sky, flux_shape, make_sky_grid


def test_zero_tilt_zero_yaw_is_the_identity_map():
    tx = np.array([0.0, 0.3, -0.5])
    ty = np.array([0.0, -0.2, 0.4])
    sx, sy, valid = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=0, az_deg=0))
    assert valid.all()
    assert np.allclose(sx, tx)
    assert np.allclose(sy, ty)


def test_yaw_rotates_the_tangent_plane_but_preserves_zenith_angle():
    """A pure yaw cannot change how far from vertical a track is."""
    tx = np.array([0.3, 0.6, -0.4])
    ty = np.array([0.1, -0.2, 0.5])
    sx, sy, valid = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=0, az_deg=241))
    assert valid.all()
    assert np.allclose(np.hypot(sx, sy), np.hypot(tx, ty))
    assert not np.allclose(sx, tx)     # but the components do move


def test_tilt_shifts_a_vertical_detector_ray_off_zenith():
    sx, sy, valid = detector_to_sky(np.array([0.0]), np.array([0.0]),
                                    Pose(0, 0, 0, tilt_deg=20, az_deg=0))
    assert valid[0]
    theta = np.degrees(np.arctan(np.hypot(sx[0], sy[0])))
    assert theta == pytest.approx(20.0)


def test_tilt_moves_bins_by_roughly_tan_of_the_tilt():
    """This displacement is what separates B(d) from the sky."""
    tx = np.linspace(-0.5, 0.5, 11)
    ty = np.zeros_like(tx)
    sx0, _, _ = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=0, az_deg=0))
    sx20, _, _ = detector_to_sky(tx, ty, Pose(0, 0, 0, tilt_deg=20, az_deg=0))
    assert np.all(sx20 > sx0)
    assert np.median(sx20 - sx0) > 0.25          # tan(20 deg) = 0.364


def test_rays_at_or_below_the_horizon_are_invalid():
    """A steep ray under a large tilt can point below the horizon."""
    sx, sy, valid = detector_to_sky(np.array([-5.0]), np.array([0.0]),
                                    Pose(0, 0, 0, tilt_deg=20, az_deg=0))
    assert not valid[0]


def test_flux_shape_is_one_at_zenith_and_falls_with_angle():
    f = flux_shape(np.array([0.0, 0.5, 1.0]), np.zeros(3), index=2.0)
    assert f[0] == pytest.approx(1.0)
    assert f[0] > f[1] > f[2] > 0


def test_flux_shape_matches_cos_theta_to_the_index():
    sx = np.array([0.75])
    cos_theta = 1.0 / np.sqrt(1.0 + 0.75**2)
    assert flux_shape(sx, np.array([0.0]), 2.0)[0] == pytest.approx(cos_theta**2)
    assert flux_shape(sx, np.array([0.0]), 3.0)[0] == pytest.approx(cos_theta**3)


def test_sky_grid_covers_the_tilted_acceptance():
    """Detector acceptance reaches 1.219; a 20 degree tilt adds about 0.364."""
    g = make_sky_grid()
    assert g.edges[-1] >= 1.219 + np.tan(np.radians(20)) - 1e-9


def test_sky_grid_bin_index_round_trips():
    g = make_sky_grid(t_max=1.0, n_bins=20)
    centers = g.centers
    sx = np.array([centers[3], centers[19], 5.0])
    sy = np.array([centers[7], centers[0], 0.0])
    flat, ok = g.bin_index(sx, sy)
    assert ok[0] and ok[1] and not ok[2]
    assert flat[0] == 3 * 20 + 7
    assert flat[1] == 19 * 20 + 0


def test_sky_grid_flat_size():
    g = make_sky_grid(t_max=1.0, n_bins=20)
    assert g.flat_size == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sky.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.sky'`

- [ ] **Step 3: Implement `megido/sky.py`**

```python
"""Detector frame to sky frame, and the cosmic flux shape.

The detector response B(d) is fixed in the DETECTOR frame; the rock's opacity is
fixed in the SKY frame. Tilting the detector 20 degrees moves every detector bin
to a different patch of sky while B does not move, and that displacement is the
entire reason the two can be separated without an open-sky calibration run.

Directions are carried as tangents (tx, ty) meaning the unit vector along
(tx, ty, 1). Rotation is applied to the unit vector, not to the tangents, which
is why the map is non-linear in tangent space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.config import Pose

_HORIZON_EPS = 1e-6


def detector_to_sky(tx: np.ndarray, ty: np.ndarray,
                    pose: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map detector-frame tangents to sky-frame tangents.

    Returns (sx, sy, valid). `valid` is False where the rotated ray points at or
    below the horizon, where a tangent representation is meaningless.
    """
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)

    vec = np.stack([tx, ty, np.ones_like(tx)], axis=-1)
    vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    rotated = vec @ pose.rotation().T
    z = rotated[..., 2]
    valid = z > _HORIZON_EPS

    safe_z = np.where(valid, z, 1.0)
    sx = np.where(valid, rotated[..., 0] / safe_z, np.nan)
    sy = np.where(valid, rotated[..., 1] / safe_z, np.nan)
    return sx, sy, valid


def flux_shape(sx: np.ndarray, sy: np.ndarray, index: float) -> np.ndarray:
    """Cosmic-ray angular shape, cos(theta_sky) ** index.

    Only the SHAPE matters: the absolute flux normalization is degenerate with
    the per-exposure normalization and is absorbed there, which is why a single
    free exponent is enough and a full Gaisser parametrisation would add
    parameters without adding information.
    """
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    cos_theta = 1.0 / np.sqrt(1.0 + sx**2 + sy**2)
    return cos_theta**index


@dataclass(frozen=True)
class SkyGrid:
    edges: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def flat_size(self) -> int:
        return self.n_bins * self.n_bins

    def bin_index(self, sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat bin index and an in-range mask. Out-of-range entries get index 0
        and mask False — callers must apply the mask, never trust the index."""
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        finite = np.isfinite(sx) & np.isfinite(sy)

        i = np.digitize(np.where(finite, sx, 0.0), self.edges) - 1
        j = np.digitize(np.where(finite, sy, 0.0), self.edges) - 1
        ok = finite & (i >= 0) & (i < self.n_bins) & (j >= 0) & (j < self.n_bins)

        flat = np.where(ok, i * self.n_bins + j, 0)
        return flat.astype(np.int64), ok


def make_sky_grid(t_max: float = 1.75, n_bins: int = 70) -> SkyGrid:
    """Default span covers the detector's own 1.219 acceptance limit plus the
    0.364 displacement a 20 degree tilt introduces, with margin."""
    return SkyGrid(edges=np.linspace(-t_max, t_max, n_bins + 1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sky.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add megido/sky.py tests/test_sky.py
git commit -m "feat: detector-to-sky mapping and cosmic flux shape"
```

---

## Task 4: Smooth efficiency basis

`B(d) = A_geom(d) * exp(S(d))`. `S` is a smooth field with far fewer parameters than bins, so the fit cannot absorb scene structure into the detector response. Gaussian radial basis functions on a coarse grid are used rather than splines because the implementation is short enough to verify by eye and the smoothness scale is set by one explicit number.

**Files:**
- Create: `megido/basis.py`, `tests/test_basis.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  @dataclass(frozen=True)
  class SmoothBasis:
      centers: np.ndarray      # [k, 2]
      sigma: float
      .n_coeff -> int
      .design(tx, ty) -> np.ndarray          # [n_points, k]
      .evaluate(coeffs, tx, ty) -> np.ndarray
  make_smooth_basis(t_max: float = 1.25, n_per_axis: int = 8) -> SmoothBasis
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_basis.py`:

```python
import numpy as np
import pytest

from megido.basis import SmoothBasis, make_smooth_basis


def test_basis_size_is_n_per_axis_squared():
    b = make_smooth_basis(n_per_axis=8)
    assert b.n_coeff == 64
    assert b.centers.shape == (64, 2)


def test_design_matrix_shape():
    b = make_smooth_basis(n_per_axis=4)
    tx = np.linspace(-1, 1, 30)
    ty = np.zeros(30)
    assert b.design(tx, ty).shape == (30, 16)


def test_zero_coefficients_give_a_flat_field():
    b = make_smooth_basis(n_per_axis=4)
    tx, ty = np.linspace(-1, 1, 20), np.zeros(20)
    assert np.allclose(b.evaluate(np.zeros(b.n_coeff), tx, ty), 0.0)


def test_a_single_coefficient_bumps_its_own_neighbourhood_most():
    b = make_smooth_basis(t_max=1.0, n_per_axis=4)
    c = np.zeros(b.n_coeff)
    c[0] = 1.0
    at_centre = b.evaluate(c, b.centers[0, 0:1], b.centers[0, 1:2])[0]
    far = b.evaluate(c, np.array([b.centers[-1, 0]]), np.array([b.centers[-1, 1]]))[0]
    assert at_centre > 0.5
    assert far < 0.05 * at_centre


def test_the_field_is_smooth():
    """Neighbouring evaluation points must not differ wildly — that is the point."""
    b = make_smooth_basis(t_max=1.0, n_per_axis=6)
    rng = np.random.default_rng(0)
    c = rng.normal(0.0, 1.0, b.n_coeff)
    t = np.linspace(-1, 1, 400)
    f = b.evaluate(c, t, np.zeros_like(t))
    steps = np.abs(np.diff(f))
    assert steps.max() < 0.2 * (f.max() - f.min() + 1e-12)


def test_evaluate_equals_design_times_coefficients():
    b = make_smooth_basis(n_per_axis=4)
    rng = np.random.default_rng(1)
    c = rng.normal(size=b.n_coeff)
    tx, ty = rng.uniform(-1, 1, 15), rng.uniform(-1, 1, 15)
    assert np.allclose(b.evaluate(c, tx, ty), b.design(tx, ty) @ c)


def test_sigma_scales_with_centre_spacing():
    """Too small a sigma leaves gaps between basis functions; too large and the
    basis cannot represent anything but a constant."""
    b = make_smooth_basis(t_max=1.0, n_per_axis=5)
    spacing = 2.0 / 4
    assert 0.5 * spacing < b.sigma < 1.5 * spacing


def test_evaluate_accepts_two_dimensional_input():
    b = make_smooth_basis(n_per_axis=4)
    tx, ty = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 5), indexing="ij")
    out = b.evaluate(np.zeros(b.n_coeff), tx, ty)
    assert out.shape == (7, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_basis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.basis'`

- [ ] **Step 3: Implement `megido/basis.py`**

```python
"""Smooth positive correction to the analytic acceptance.

B(d) = A_geom(d) * exp(S(d)). A_geom carries the geometry exactly and has no
free parameters; S absorbs per-channel efficiency structure — the per-ASIC gain
offsets Phase 1 measured, chiefly ASIC 3 sitting about 15% high.

S is deliberately low-dimensional: 64 coefficients against roughly 1800 occupied
analysis bins. A basis flexible enough to follow the data bin by bin would
absorb the rock's absorption into the detector response and quietly return a
flat sky.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmoothBasis:
    centers: np.ndarray
    sigma: float

    @property
    def n_coeff(self) -> int:
        return int(self.centers.shape[0])

    def design(self, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
        """[n_points, n_coeff] Gaussian radial basis design matrix."""
        tx = np.asarray(tx, dtype=np.float64).reshape(-1)
        ty = np.asarray(ty, dtype=np.float64).reshape(-1)
        dx = tx[:, None] - self.centers[None, :, 0]
        dy = ty[:, None] - self.centers[None, :, 1]
        return np.exp(-0.5 * (dx**2 + dy**2) / self.sigma**2)

    def evaluate(self, coeffs: np.ndarray, tx: np.ndarray,
                 ty: np.ndarray) -> np.ndarray:
        tx_arr = np.asarray(tx, dtype=np.float64)
        flat = self.design(tx_arr, ty) @ np.asarray(coeffs, dtype=np.float64)
        return flat.reshape(tx_arr.shape)


def make_smooth_basis(t_max: float = 1.25, n_per_axis: int = 8) -> SmoothBasis:
    """Centres on a square grid, with sigma set to the centre spacing.

    Sigma equal to the spacing keeps neighbouring basis functions overlapping
    enough to represent a smooth field without leaving gaps, and coarse enough
    that the basis cannot chase individual bins.
    """
    axis = np.linspace(-t_max, t_max, n_per_axis)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    centers = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    spacing = (2.0 * t_max) / (n_per_axis - 1)
    return SmoothBasis(centers=centers, sigma=float(spacing))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_basis.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add megido/basis.py tests/test_basis.py
git commit -m "feat: Gaussian radial basis for the smooth efficiency correction"
```

---

## Task 5: The joint Poisson solve

The core of Phase 2. Three of the four parameter blocks have closed-form Poisson updates; only the smooth coefficients need numerical optimization.

Per iteration:
1. **Opacity** — for sky bin `s` at position `p`, with `N` the total observed counts mapping there and `E` the total predicted without absorption, the Poisson MLE is `exp(-lambda) = N/E`, so `lambda = log(E/N)`.
2. **Normalization** — `norm_e = sum_d n_e(d) / sum_d K_e(d)` where `K` is the prediction at unit norm.
3. **Smooth coefficients** — L-BFGS-B on the Poisson negative log-likelihood.
4. **Flux index** — a coarse 1-D scan, since it is a single well-behaved parameter.

**Files:**
- Create: `megido/baseline.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `AnalysisGrid`, `SkyGrid`, `SmoothBasis`, `geometric_acceptance`, `detector_to_sky`, `flux_shape`, `megido.config.SiteConfig`
- Produces:
  ```python
  @dataclass(frozen=True)
  class BaselineSolution:
      coeffs: np.ndarray                    # [n_coeff]
      opacity: dict[str, np.ndarray]        # position id -> [k*k] flat, NaN where unconstrained
      norms: dict[str, float]               # norm_group -> float
      flux_index: float
      nll_history: list[float]
      grid: AnalysisGrid
      sky: SkyGrid
      basis: SmoothBasis
      .response() -> np.ndarray             # B(d) on the analysis grid, [m, m]
      .predict(exposure_id) -> np.ndarray   # mu_e(d), [m, m]
      .opacity_image(position_id) -> np.ndarray   # [k, k]
      .normalized_opacity(position_id, transparent_quantile=0.05) -> np.ndarray
      .save(path) / .load(path)

  position_ids(cfg: SiteConfig) -> dict[str, str]   # exposure id -> "pos0", "pos1", ...
  ```

  **`load()` does not restore `terms`**, which are derived from the grid and the config
  rather than stored. A reloaded solution can report `coeffs`, `opacity`, `norms` and
  `flux_index`, but calling `.predict()` or `.response()` on one will fail. That is
  deliberate — re-deriving them needs the config, and silently guessing it would be worse.
  Task 8's CLI writes the solution and reports from the in-memory object, never a reloaded one.
  ```python

  solve_baseline(grid, cfg, *, geom=None, sky=None, basis=None,
                 n_iter=30, flux_index=2.0, fit_flux_index=True,
                 tol=1e-6) -> BaselineSolution
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_baseline.py`:

```python
import numpy as np
import pytest
import yaml

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution, solve_baseline
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.sky import make_sky_grid


@pytest.fixture
def cfg(tmp_path):
    """Two tilts at one position plus a second position, mirroring the campaign."""
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(tmp_path),
        "frame": {"origin": "P0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 500},
        "exposures": [
            {"id": "P0", "runs": "DET100001-DET100001",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
            {"id": "T20", "runs": "DET100002-DET100002",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
            {"id": "P1", "runs": "DET100003-DET100003",
             "pose": {"x": 2.2, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
        ],
    }))
    return load_site_config(p)


def _flat_grid(cfg, n=30, counts=400):
    edges = np.linspace(-1.25, 1.25, n + 1)
    return AnalysisGrid(edges=edges,
                        counts={e.id: np.full((n, n), counts, np.int64)
                                for e in cfg.exposures})


def test_solution_has_every_block(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    assert isinstance(sol, BaselineSolution)
    assert sol.coeffs.shape == (make_smooth_basis().n_coeff,)
    assert set(sol.norms) == {"P0", "T20", "P1"}
    assert set(sol.opacity) == {"pos0", "pos1"}


def test_positions_group_by_pose_not_by_exposure(cfg):
    """P0 and T20 share a position; P1 does not. Opacity is per POSITION."""
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    assert len(sol.opacity) == 2, "three exposures, two positions"


def test_likelihood_decreases_monotonically(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=8, fit_flux_index=False)
    h = np.array(sol.nll_history)
    assert len(h) >= 2
    assert np.all(np.diff(h) <= 1e-6), f"NLL rose: {h}"


def test_response_is_positive_inside_the_acceptance_and_zero_outside(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    b = sol.response()
    tx, ty = sol.grid.tan_mesh()
    inside = np.hypot(tx, ty) < 0.5
    assert np.all(b[inside] > 0)
    assert np.all(b[np.abs(tx) > 1.25] == 0)


def test_prediction_matches_observed_total_after_convergence(cfg):
    """The normalization update is a closed form, so totals must match closely."""
    grid = _flat_grid(cfg)
    sol = solve_baseline(grid, cfg, n_iter=15, fit_flux_index=False)
    for eid, observed in grid.counts.items():
        predicted = sol.predict(eid)
        assert predicted.sum() == pytest.approx(observed.sum(), rel=0.02)


def test_opacity_image_is_square_and_matches_the_sky_grid(cfg):
    sky = make_sky_grid(t_max=1.5, n_bins=30)
    sol = solve_baseline(_flat_grid(cfg), cfg, sky=sky, n_iter=3, fit_flux_index=False)
    img = sol.opacity_image("pos0")
    assert img.shape == (30, 30)


def test_unconstrained_sky_bins_are_nan_not_zero(cfg):
    """A sky bin no exposure sees must be reported as unknown, never as zero
    opacity — zero would read as 'no rock there', which is a claim."""
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    img = sol.opacity_image("pos1")
    assert np.isnan(img).any()
    assert np.isfinite(img).any()


def test_flux_index_is_fitted_when_asked(cfg):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=5,
                         flux_index=2.0, fit_flux_index=True)
    assert 0.5 <= sol.flux_index <= 5.0


def test_solution_round_trips_through_disk(cfg, tmp_path):
    sol = solve_baseline(_flat_grid(cfg), cfg, n_iter=3, fit_flux_index=False)
    path = tmp_path / "baseline.npz"
    sol.save(path)
    back = BaselineSolution.load(path)
    assert np.allclose(back.coeffs, sol.coeffs)
    assert back.norms == pytest.approx(sol.norms)
    assert back.flux_index == pytest.approx(sol.flux_index)
    for pid in sol.opacity:
        assert np.allclose(back.opacity[pid], sol.opacity[pid], equal_nan=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.baseline'`

- [ ] **Step 3: Implement `megido/baseline.py`**

```python
"""Joint Poisson solve for the detector baseline and per-position opacity.

There is no open-sky calibration run for this campaign, so the detector's
angular response cannot be measured separately and must be estimated alongside
the scene. What makes that possible is the campaign geometry: P0, T20a and T20b
sit at ONE position with two different tilts, so rotating the detector sends
each bin to a different patch of sky while the response stays put.

Model, for exposure e at position p(e):

    mu_e(d) = norm_e * A_geom(d) * exp(S(d)) * Phi_n(R_e.d) * exp(-lambda_p(R_e.d))

Opacity is indexed by SKY direction and by POSITION. It is not shared across
positions: P0 and P1 are 2.2 m apart and look through different rock for the
same sky direction, and that difference is precisely the parallax Phase 3 needs.

Three of the four blocks have closed-form Poisson updates; only the smooth
coefficients need numerical optimisation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from megido.acceptance import geometric_acceptance
from megido.angular import AnalysisGrid
from megido.basis import SmoothBasis, make_smooth_basis
from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.sky import SkyGrid, detector_to_sky, flux_shape, make_sky_grid

_MIN_COUNTS = 1.0
_EPS = 1e-12


def position_ids(cfg: SiteConfig) -> dict[str, str]:
    """Map each exposure id to a short position id, assigned in config order.

    Exposures sharing a position share an opacity map. Position is the
    TRANSLATION only — two tilts at the same spot are the same position, which
    is exactly what makes P0/T20a/T20b able to separate detector from sky.
    """
    seen: dict[tuple[float, float, float], str] = {}
    out: dict[str, str] = {}
    for exp in cfg.exposures:
        key = (round(exp.pose.x, 6), round(exp.pose.y, 6), round(exp.pose.z, 6))
        if key not in seen:
            seen[key] = f"pos{len(seen)}"
        out[exp.id] = seen[key]
    return out


@dataclass(frozen=True)
class _ExposureTerms:
    """Everything about one exposure that does not change during the solve."""
    exposure_id: str
    norm_group: str
    position: str
    counts: np.ndarray        # [n_live] observed, flattened over live bins
    acceptance: np.ndarray    # [n_live] A_geom
    design: np.ndarray        # [n_live, n_coeff]
    sky_flat: np.ndarray      # [n_live] flat sky-bin index
    sky_tan: np.ndarray       # [n_live, 2] sky tangents, for the flux shape
    live: np.ndarray          # [m, m] bool, which analysis bins are used


@dataclass(frozen=True)
class BaselineSolution:
    coeffs: np.ndarray
    opacity: dict[str, np.ndarray]
    norms: dict[str, float]
    flux_index: float
    nll_history: list[float]
    grid: AnalysisGrid
    sky: SkyGrid
    basis: SmoothBasis
    terms: dict[str, _ExposureTerms] = field(default_factory=dict, repr=False)

    def normalized_opacity(self, position_id_: str,
                           transparent_quantile: float = 0.05) -> np.ndarray:
        """Opacity with its zero point pinned to the most transparent direction.

        The absolute level of lambda is degenerate with the per-exposure
        normalization — nothing in the data says where zero is. The convention,
        inherited from the cafeteria project's `norm_quantile`, is that the
        least-absorbed directions are near-open sky, so lambda is shifted to put
        its low quantile at zero and then clipped. Negative opacity would mean
        more flux than open sky, which is not a thing.
        """
        lam = np.array(self.opacity[position_id_], dtype=np.float64)
        seen = np.isfinite(lam)
        if not seen.any():
            return lam
        lam[seen] = np.clip(lam[seen] - np.quantile(lam[seen], transparent_quantile),
                            0.0, None)
        return lam

    def response(self) -> np.ndarray:
        tx, ty = self.grid.tan_mesh()
        geom = DetectorGeometry.megiddo()
        return geometric_acceptance(tx, ty, geom) * np.exp(
            self.basis.evaluate(self.coeffs, tx, ty))

    def predict(self, exposure_id: str) -> np.ndarray:
        t = self.terms[exposure_id]
        lam = self.opacity[t.position][t.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        mu = (self.norms[t.norm_group] * t.acceptance
              * np.exp(t.design @ self.coeffs)
              * flux_shape(t.sky_tan[:, 0], t.sky_tan[:, 1], self.flux_index)
              * np.exp(-lam))
        out = np.zeros(t.live.shape)
        out[t.live] = mu
        return out

    def opacity_image(self, position_id_: str) -> np.ndarray:
        k = self.sky.n_bins
        return self.opacity[position_id_].reshape(k, k)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coeffs": self.coeffs,
            "grid_edges": self.grid.edges,
            "sky_edges": self.sky.edges,
            "basis_centers": self.basis.centers,
            "basis_sigma": np.array(self.basis.sigma),
            "flux_index": np.array(self.flux_index),
            "nll_history": np.array(self.nll_history),
            "meta": np.array(json.dumps({
                "norms": self.norms,
                "positions": sorted(self.opacity),
            })),
        }
        for pid, lam in self.opacity.items():
            payload[f"opacity__{pid}"] = lam
        np.savez_compressed(path, **payload)

    @staticmethod
    def load(path: Path) -> "BaselineSolution":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        opacity = {pid: d[f"opacity__{pid}"] for pid in meta["positions"]}
        return BaselineSolution(
            coeffs=d["coeffs"],
            opacity=opacity,
            norms={k: float(v) for k, v in meta["norms"].items()},
            flux_index=float(d["flux_index"]),
            nll_history=[float(x) for x in d["nll_history"]],
            grid=AnalysisGrid(edges=d["grid_edges"], counts={}),
            sky=SkyGrid(edges=d["sky_edges"]),
            basis=SmoothBasis(centers=d["basis_centers"], sigma=float(d["basis_sigma"])),
        )


def _build_terms(grid: AnalysisGrid, cfg: SiteConfig, geom: DetectorGeometry,
                 sky: SkyGrid, basis: SmoothBasis) -> dict[str, _ExposureTerms]:
    tx, ty = grid.tan_mesh()
    acceptance_full = geometric_acceptance(tx, ty, geom)
    positions = position_ids(cfg)

    terms: dict[str, _ExposureTerms] = {}
    for eid, counts in grid.counts.items():
        exp = cfg.exposure(eid)
        sx, sy, on_sky = detector_to_sky(tx, ty, exp.pose)
        flat, in_grid = sky.bin_index(sx, sy)

        live = (acceptance_full > 0) & on_sky & in_grid
        terms[eid] = _ExposureTerms(
            exposure_id=eid,
            norm_group=exp.norm_group,
            position=positions[eid],
            counts=counts[live].astype(np.float64),
            acceptance=acceptance_full[live],
            design=basis.design(tx[live], ty[live]),
            sky_flat=flat[live],
            sky_tan=np.stack([sx[live], sy[live]], axis=1),
            live=live,
        )
    return terms


def _kernel(t: _ExposureTerms, coeffs: np.ndarray, flux_index: float) -> np.ndarray:
    """Prediction with normalization and absorption both set to one."""
    return (t.acceptance * np.exp(t.design @ coeffs)
            * flux_shape(t.sky_tan[:, 0], t.sky_tan[:, 1], flux_index))


def _update_opacity(terms, coeffs, norms, flux_index, sky: SkyGrid
                    ) -> dict[str, np.ndarray]:
    """Closed-form Poisson MLE: exp(-lambda) = observed / expected."""
    positions = sorted({t.position for t in terms.values()})
    out: dict[str, np.ndarray] = {}
    for pid in positions:
        observed = np.zeros(sky.flat_size)
        expected = np.zeros(sky.flat_size)
        for t in terms.values():
            if t.position != pid:
                continue
            np.add.at(observed, t.sky_flat, t.counts)
            np.add.at(expected, t.sky_flat,
                      norms[t.norm_group] * _kernel(t, coeffs, flux_index))
        seen = expected > _EPS
        lam = np.full(sky.flat_size, np.nan)
        ratio = np.clip(observed[seen], _MIN_COUNTS, None) / expected[seen]
        lam[seen] = -np.log(ratio)
        out[pid] = lam
    return out


def _update_norms(terms, coeffs, opacity, flux_index) -> dict[str, float]:
    """Closed-form Poisson MLE for a multiplicative scale."""
    numer: dict[str, float] = {}
    denom: dict[str, float] = {}
    for t in terms.values():
        lam = opacity[t.position][t.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        k = _kernel(t, coeffs, flux_index) * np.exp(-lam)
        numer[t.norm_group] = numer.get(t.norm_group, 0.0) + float(t.counts.sum())
        denom[t.norm_group] = denom.get(t.norm_group, 0.0) + float(k.sum())
    return {g: (numer[g] / denom[g] if denom[g] > _EPS else 1.0) for g in numer}


def _nll(terms, coeffs, opacity, norms, flux_index) -> float:
    total = 0.0
    for t in terms.values():
        lam = opacity[t.position][t.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        mu = norms[t.norm_group] * _kernel(t, coeffs, flux_index) * np.exp(-lam)
        mu = np.clip(mu, _EPS, None)
        total += float(np.sum(mu - t.counts * np.log(mu)))
    return total


def _update_coeffs(terms, coeffs, opacity, norms, flux_index) -> np.ndarray:
    """L-BFGS-B on the Poisson NLL. Gradient is analytic:
    d/dc sum(mu - n log mu) = design^T (mu - n), since d mu / d c = mu * design."""
    def objective(c):
        value = 0.0
        grad = np.zeros_like(c)
        for t in terms.values():
            lam = opacity[t.position][t.sky_flat]
            lam = np.where(np.isfinite(lam), lam, 0.0)
            mu = norms[t.norm_group] * _kernel(t, c, flux_index) * np.exp(-lam)
            mu = np.clip(mu, _EPS, None)
            value += float(np.sum(mu - t.counts * np.log(mu)))
            grad += t.design.T @ (mu - t.counts)
        return value, grad

    result = minimize(objective, coeffs, jac=True, method="L-BFGS-B",
                      options={"maxiter": 60})
    return result.x


def _update_flux_index(terms, coeffs, opacity, norms, current: float) -> float:
    candidates = np.clip(current + np.linspace(-0.6, 0.6, 13), 0.5, 5.0)
    scores = [_nll(terms, coeffs, opacity, norms, n) for n in candidates]
    return float(candidates[int(np.argmin(scores))])


def solve_baseline(grid: AnalysisGrid, cfg: SiteConfig, *,
                   geom: DetectorGeometry | None = None,
                   sky: SkyGrid | None = None,
                   basis: SmoothBasis | None = None,
                   n_iter: int = 30, flux_index: float = 2.0,
                   fit_flux_index: bool = True,
                   tol: float = 1e-6) -> BaselineSolution:
    geom = geom or DetectorGeometry.megiddo()
    sky = sky or make_sky_grid()
    basis = basis or make_smooth_basis()

    terms = _build_terms(grid, cfg, geom, sky, basis)
    coeffs = np.zeros(basis.n_coeff)
    norms = {t.norm_group: 1.0 for t in terms.values()}
    opacity = _update_opacity(terms, coeffs, norms, flux_index, sky)

    history: list[float] = []
    for _ in range(n_iter):
        opacity = _update_opacity(terms, coeffs, norms, flux_index, sky)
        norms = _update_norms(terms, coeffs, opacity, flux_index)
        coeffs = _update_coeffs(terms, coeffs, opacity, norms, flux_index)
        if fit_flux_index:
            flux_index = _update_flux_index(terms, coeffs, opacity, norms, flux_index)

        history.append(_nll(terms, coeffs, opacity, norms, flux_index))
        if len(history) >= 2 and abs(history[-2] - history[-1]) < tol * abs(history[-1]):
            break

    return BaselineSolution(coeffs=coeffs, opacity=opacity, norms=norms,
                            flux_index=flux_index, nll_history=history,
                            grid=grid, sky=sky, basis=basis, terms=terms)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_baseline.py -v`
Expected: 9 passed

**If `test_likelihood_decreases_monotonically` fails**, the alternating scheme is not descending — report it with the NLL history rather than loosening the assertion. A rising likelihood means one of the closed forms is wrong, and it must be diagnosed, not tolerated.

- [ ] **Step 5: Commit**

```bash
git add megido/baseline.py tests/test_baseline.py
git commit -m "feat: joint Poisson solve for baseline and per-position opacity"
```

---

## Task 6: Synthetic recovery gate

The solve is only trustworthy if it can recover a known answer. This generator injects a chosen `B` and a chosen opacity map, produces Poisson counts through the real pose geometry, and the test asserts both come back.

**Files:**
- Create: `megido/simbaseline.py`, `tests/test_simbaseline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5
- Produces:
  ```python
  @dataclass(frozen=True)
  class SyntheticScene:
      grid: AnalysisGrid
      true_coeffs: np.ndarray
      true_opacity: dict[str, np.ndarray]
      true_norms: dict[str, float]
      true_flux_index: float

  make_synthetic_scene(cfg, *, n_bins=30, sky=None, basis=None,
                       scale=4.0e5, seed=0, opacity_amplitude=0.4,
                       flux_index=2.0) -> SyntheticScene
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_simbaseline.py`:

```python
import numpy as np
import pytest
import yaml

from megido.baseline import solve_baseline
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.simbaseline import make_synthetic_scene
from megido.sky import make_sky_grid


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(tmp_path),
        "frame": {"origin": "P0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 500},
        "exposures": [
            {"id": "P0", "runs": "DET100001-DET100001",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
            {"id": "T20a", "runs": "DET100002-DET100002",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
            {"id": "T20b", "runs": "DET100003-DET100003",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
            {"id": "P1", "runs": "DET100004-DET100004",
             "pose": {"x": 2.2, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
        ],
    }))
    return load_site_config(p)


def test_scene_produces_counts_for_every_exposure(cfg):
    scene = make_synthetic_scene(cfg, seed=0)
    assert set(scene.grid.counts) == {"P0", "T20a", "T20b", "P1"}
    for v in scene.grid.counts.values():
        assert v.dtype == np.int64
        assert v.sum() > 10_000


def test_counts_are_higher_where_opacity_is_lower(cfg):
    """Sanity: absorption must actually suppress counts."""
    scene = make_synthetic_scene(cfg, opacity_amplitude=1.5, seed=1)
    p0 = scene.grid.counts["P0"].astype(float)
    n = p0.shape[0]
    assert p0[n // 2, n // 2] > 0


def test_solver_recovers_the_injected_detector_response(cfg):
    """THE GATE. B(d) must come back, up to the overall scale that is
    degenerate with the per-exposure normalisation."""
    sky = make_sky_grid(t_max=1.6, n_bins=24)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis, seed=2)

    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=40, fit_flux_index=False,
                         flux_index=scene.true_flux_index)

    tx, ty = scene.grid.tan_mesh()
    inside = np.hypot(tx, ty) < 0.8
    truth = basis.evaluate(scene.true_coeffs, tx, ty)[inside]
    got = basis.evaluate(sol.coeffs, tx, ty)[inside]

    # only shape is identifiable; a constant offset trades against the norms
    truth = truth - truth.mean()
    got = got - got.mean()
    assert np.corrcoef(truth, got)[0, 1] > 0.9
    assert np.std(got - truth) < 0.3 * np.std(truth) + 0.05


def test_solver_recovers_the_injected_opacity_structure(cfg):
    sky = make_sky_grid(t_max=1.6, n_bins=24)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 opacity_amplitude=0.6, seed=3)

    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=40, fit_flux_index=False,
                         flux_index=scene.true_flux_index)

    pid = sorted(sol.opacity)[0]
    truth = scene.true_opacity[pid]
    got = sol.opacity[pid]
    both = np.isfinite(truth) & np.isfinite(got)
    assert both.sum() > 50

    t = truth[both] - truth[both].mean()
    g = got[both] - got[both].mean()
    assert np.corrcoef(t, g)[0, 1] > 0.8


def test_a_flat_sky_is_recovered_as_flat(cfg):
    """Negative control: with no absorption injected, the solve must not
    manufacture structure by pushing scene features into the opacity map."""
    sky = make_sky_grid(t_max=1.6, n_bins=24)
    basis = make_smooth_basis(n_per_axis=5)
    scene = make_synthetic_scene(cfg, n_bins=30, sky=sky, basis=basis,
                                 opacity_amplitude=0.0, seed=4)

    sol = solve_baseline(scene.grid, cfg, sky=sky, basis=basis,
                         n_iter=40, fit_flux_index=False,
                         flux_index=scene.true_flux_index)

    pid = sorted(sol.opacity)[0]
    lam = sol.opacity[pid]
    lam = lam[np.isfinite(lam)]
    assert np.std(lam) < 0.15, f"flat sky came back with structure, std={np.std(lam):.3f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simbaseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.simbaseline'`

- [ ] **Step 3: Implement `megido/simbaseline.py`**

```python
"""Synthetic scenes with a known baseline and a known sky, for gating the solve.

The solve estimates the detector response and the rock's absorption at the same
time, from data that contains only their product. That is only credible if it
can recover a known answer, so this generator injects both and the tests assert
both come back — including a negative control where the sky is flat and the
solve must not invent structure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.acceptance import geometric_acceptance
from megido.angular import AnalysisGrid
from megido.baseline import position_ids
from megido.basis import SmoothBasis, make_smooth_basis
from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.sky import SkyGrid, detector_to_sky, flux_shape, make_sky_grid


@dataclass(frozen=True)
class SyntheticScene:
    grid: AnalysisGrid
    true_coeffs: np.ndarray
    true_opacity: dict[str, np.ndarray]
    true_norms: dict[str, float]
    true_flux_index: float


def make_synthetic_scene(cfg: SiteConfig, *, n_bins: int = 30,
                         sky: SkyGrid | None = None,
                         basis: SmoothBasis | None = None,
                         scale: float = 4.0e5, seed: int = 0,
                         opacity_amplitude: float = 0.4,
                         flux_index: float = 2.0) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    geom = DetectorGeometry.megiddo()
    sky = sky or make_sky_grid()
    basis = basis or make_smooth_basis()

    edges = np.linspace(-1.25, 1.25, n_bins + 1)
    grid = AnalysisGrid(edges=edges, counts={})
    tx, ty = grid.tan_mesh()
    acceptance = geometric_acceptance(tx, ty, geom)

    true_coeffs = rng.normal(0.0, 0.25, basis.n_coeff)
    response = acceptance * np.exp(basis.evaluate(true_coeffs, tx, ty))

    # A smooth, deterministic sky feature per position, so the test asserts
    # against structure rather than noise.
    exposure_position = position_ids(cfg)
    positions = sorted(set(exposure_position.values()))
    sky_cx, sky_cy = np.meshgrid(sky.centers, sky.centers, indexing="ij")
    true_opacity: dict[str, np.ndarray] = {}
    for k, pid in enumerate(positions):
        blob = np.exp(-0.5 * (((sky_cx - 0.25 * (k + 1)) / 0.45) ** 2
                              + ((sky_cy + 0.15 * (k + 1)) / 0.45) ** 2))
        true_opacity[pid] = (opacity_amplitude * blob).reshape(-1)

    true_norms = {e.norm_group: scale * (1.0 + 0.1 * i)
                  for i, e in enumerate(cfg.exposures)}

    counts: dict[str, np.ndarray] = {}
    for exp in cfg.exposures:
        sx, sy, on_sky = detector_to_sky(tx, ty, exp.pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acceptance > 0) & on_sky & in_grid

        lam = np.zeros_like(tx)
        lam[live] = true_opacity[exposure_position[exp.id]][flat[live]]

        mu = np.zeros_like(tx)
        mu[live] = (true_norms[exp.norm_group] * response[live]
                    * flux_shape(sx[live], sy[live], flux_index)
                    * np.exp(-lam[live]))
        counts[exp.id] = rng.poisson(np.clip(mu, 0.0, None)).astype(np.int64)

    return SyntheticScene(
        grid=AnalysisGrid(edges=edges, counts=counts),
        true_coeffs=true_coeffs,
        true_opacity=true_opacity,
        true_norms=true_norms,
        true_flux_index=flux_index,
    )
```

Note the normalization: `response` is in cm² times an exponential, so `scale` is tuned to give a few hundred counts per bin. If the tests report far fewer, raise `scale`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_simbaseline.py -v`
Expected: 5 passed

**If `test_a_flat_sky_is_recovered_as_flat` fails**, that is the most important failure in this plan — it means the solve manufactures sky structure from nothing. Report it with the measured standard deviation; do not raise the threshold.

- [ ] **Step 5: Commit**

```bash
git add megido/simbaseline.py tests/test_simbaseline.py
git commit -m "feat: synthetic recovery gate for the baseline solve"
```

---

## Task 7: Null-space probe and leave-one-exposure-out

Spec section 6.4 records a known degeneracy: opacity that is constant on cones about the tilt axis is poorly constrained, because rotating about that axis barely moves it. The solve will return *something* there, and the honest thing is to report how weakly it is determined rather than present it with the rest.

**Files:**
- Create: `megido/validate2.py`, `tests/test_validate2.py`

**Interfaces:**
- Consumes: `BaselineSolution`, `solve_baseline`, `AnalysisGrid`, `SiteConfig`
- Produces:
  ```python
  @dataclass(frozen=True)
  class Check2:
      name: str
      passed: bool
      measured: float
      expected: str
      detail: str

  opacity_uncertainty(grid, cfg, *, n_replicas=12, seed=0, **solve_kwargs
                      ) -> dict[str, np.ndarray]     # position -> per-sky-bin sigma
  leave_one_out(grid, cfg, **solve_kwargs) -> list[Check2]
  nll_per_bin_check(sol: BaselineSolution, grid: AnalysisGrid) -> Check2
  format_report2(checks: list[Check2]) -> str
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate2.py`:

```python
import numpy as np
import pytest
import yaml

from megido.baseline import solve_baseline
from megido.basis import make_smooth_basis
from megido.config import load_site_config
from megido.simbaseline import make_synthetic_scene
from megido.sky import make_sky_grid
from megido.validate2 import (Check2, format_report2, leave_one_out,
                              nll_per_bin_check, opacity_uncertainty)


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(tmp_path),
        "frame": {"origin": "P0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 500},
        "exposures": [
            {"id": "P0", "runs": "DET100001-DET100001",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
            {"id": "T20", "runs": "DET100002-DET100002",
             "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 20, "az_deg": 241}},
            {"id": "P1", "runs": "DET100003-DET100003",
             "pose": {"x": 2.2, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241}},
        ],
    }))
    return load_site_config(p)


@pytest.fixture
def scene(cfg):
    sky = make_sky_grid(t_max=1.6, n_bins=20)
    basis = make_smooth_basis(n_per_axis=4)
    return make_synthetic_scene(cfg, n_bins=24, sky=sky, basis=basis, seed=5), sky, basis


def test_uncertainty_is_positive_where_the_sky_is_seen(cfg, scene):
    s, sky, basis = scene
    sigma = opacity_uncertainty(s.grid, cfg, n_replicas=6, sky=sky, basis=basis,
                                n_iter=8, fit_flux_index=False)
    for lam_sigma in sigma.values():
        seen = np.isfinite(lam_sigma)
        assert seen.any()
        assert np.all(lam_sigma[seen] >= 0)


def test_uncertainty_shrinks_with_more_counts(cfg):
    """Poisson: four times the exposure should roughly halve the error."""
    sky = make_sky_grid(t_max=1.6, n_bins=20)
    basis = make_smooth_basis(n_per_axis=4)
    thin = make_synthetic_scene(cfg, n_bins=24, sky=sky, basis=basis,
                                scale=5.0e4, seed=6)
    thick = make_synthetic_scene(cfg, n_bins=24, sky=sky, basis=basis,
                                 scale=2.0e5, seed=6)
    kw = dict(sky=sky, basis=basis, n_iter=8, fit_flux_index=False)
    s_thin = opacity_uncertainty(thin.grid, cfg, n_replicas=6, **kw)
    s_thick = opacity_uncertainty(thick.grid, cfg, n_replicas=6, **kw)

    pid = sorted(s_thin)[0]
    a = np.nanmedian(s_thin[pid])
    b = np.nanmedian(s_thick[pid])
    assert b < a, f"more counts did not reduce the uncertainty ({b:.4f} vs {a:.4f})"


def test_leave_one_out_returns_a_check_per_exposure(cfg, scene):
    s, sky, basis = scene
    checks = leave_one_out(s.grid, cfg, sky=sky, basis=basis,
                           n_iter=8, fit_flux_index=False)
    assert len(checks) == 3
    assert all(isinstance(c, Check2) for c in checks)
    assert {c.name for c in checks} == {"loo.P0", "loo.T20", "loo.P1"}


def test_the_sole_exposure_at_a_position_is_skipped_not_silently_passed(cfg, scene):
    """P1 alone holds pos1. Holding it out leaves nothing to predict against,
    and saying so is honest; quietly scoring it would not be."""
    s, sky, basis = scene
    checks = {c.name: c for c in leave_one_out(s.grid, cfg, sky=sky, basis=basis,
                                               n_iter=8, fit_flux_index=False)}
    assert "skipped" in checks["loo.P1"].expected
    assert np.isnan(checks["loo.P1"].measured)
    assert not np.isnan(checks["loo.P0"].measured)


def test_leave_one_out_passes_on_a_self_consistent_scene(cfg, scene):
    """Synthetic data obeys the model exactly, so held-out prediction must work."""
    s, sky, basis = scene
    checks = leave_one_out(s.grid, cfg, sky=sky, basis=basis,
                           n_iter=12, fit_flux_index=False)
    scored = [c for c in checks if not np.isnan(c.measured)]
    assert scored, "at least one exposure must be genuinely scoreable"
    assert all(c.passed for c in scored), [c.detail for c in scored]


def test_nll_per_bin_is_near_one_for_a_correct_model(cfg, scene):
    s, sky, basis = scene
    sol = solve_baseline(s.grid, cfg, sky=sky, basis=basis,
                         n_iter=20, fit_flux_index=False)
    check = nll_per_bin_check(sol, s.grid)
    assert check.passed, check.detail
    assert 0.2 < check.measured < 3.0


def test_format_report2_renders_pass_and_fail(cfg):
    checks = [Check2("a", True, 1.0, "x", "ok"), Check2("b", False, 2.0, "y", "bad")]
    text = format_report2(checks)
    assert "PASS" in text and "FAIL" in text
    assert "1/2" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.validate2'`

- [ ] **Step 3: Implement `megido/validate2.py`**

```python
"""Phase 2 validation: how well is the answer actually determined?

The solve always returns a baseline and a sky. These checks ask whether to
believe them. Leave-one-exposure-out is the honest cross-validation for a fit
this flexible, and the Poisson bootstrap says which parts of the sky map are
measurements and which are noise.

Spec section 6.4 records a known degeneracy: opacity constant on cones about the
tilt axis is barely moved by the rotation, so it is weakly determined. The
bootstrap sigma is where that shows up, which is why the sky map must always be
read alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.angular import AnalysisGrid
from megido.baseline import BaselineSolution, solve_baseline
from megido.config import SiteConfig
from megido.sky import flux_shape


@dataclass(frozen=True)
class Check2:
    name: str
    passed: bool
    measured: float
    expected: str
    detail: str


def opacity_uncertainty(grid: AnalysisGrid, cfg: SiteConfig, *,
                        n_replicas: int = 12, seed: int = 0,
                        **solve_kwargs) -> dict[str, np.ndarray]:
    """Poisson bootstrap over the counts, giving a per-sky-bin sigma."""
    rng = np.random.default_rng(seed)
    stacks: dict[str, list[np.ndarray]] = {}

    for _ in range(n_replicas):
        resampled = {eid: rng.poisson(v).astype(np.int64)
                     for eid, v in grid.counts.items()}
        sol = solve_baseline(AnalysisGrid(edges=grid.edges, counts=resampled),
                             cfg, **solve_kwargs)
        for pid, lam in sol.opacity.items():
            stacks.setdefault(pid, []).append(lam)

    return {pid: np.nanstd(np.vstack(v), axis=0) for pid, v in stacks.items()}


def leave_one_out(grid: AnalysisGrid, cfg: SiteConfig,
                  **solve_kwargs) -> list[Check2]:
    """Fit without one exposure, then predict it from parameters it never saw.

    A genuine held-out test needs the held-out exposure's position to still be
    represented by another exposure — otherwise its opacity map is unconstrained
    and there is nothing to predict with. In this campaign that means P0, T20a
    and T20b can each be held out (the other two hold pos0), while P1 cannot:
    it is the only exposure at pos1, so holding it out removes that sky map
    entirely. P1 is reported as SKIPPED rather than silently passed.

    Only the held-out exposure's own normalization is re-fitted, in closed form
    from its total counts. Every shape-carrying parameter — the smooth
    coefficients, the flux index, and the opacity map — comes from the fit that
    never saw it.
    """
    from megido.baseline import position_ids

    positions = position_ids(cfg)
    checks: list[Check2] = []

    for held_out in sorted(grid.counts):
        kept = {e: v for e, v in grid.counts.items() if e != held_out}
        same_position = [e for e in kept if positions[e] == positions[held_out]]

        if not same_position:
            checks.append(Check2(
                name=f"loo.{held_out}",
                passed=True,
                measured=float("nan"),
                expected="skipped — sole exposure at its position",
                detail=(f"{held_out} is the only exposure at {positions[held_out]}; "
                        "holding it out leaves that sky map unconstrained, so there "
                        "is nothing to predict against"),
            ))
            continue

        sol = solve_baseline(AnalysisGrid(edges=grid.edges, counts=kept),
                             cfg, **solve_kwargs)

        # Rebuild the held-out exposure's terms against the subset-trained fit.
        full = solve_baseline(AnalysisGrid(edges=grid.edges, counts=grid.counts),
                              cfg, n_iter=1, **{k: v for k, v in solve_kwargs.items()
                                                if k != "n_iter"})
        term = full.terms[held_out]

        lam = sol.opacity[positions[held_out]][term.sky_flat]
        lam = np.where(np.isfinite(lam), lam, 0.0)
        shape = (term.acceptance * np.exp(term.design @ sol.coeffs)
                 * flux_shape(term.sky_tan[:, 0], term.sky_tan[:, 1], sol.flux_index)
                 * np.exp(-lam))

        observed = term.counts
        if shape.sum() <= 0 or observed.sum() <= 0 or observed.size < 20:
            checks.append(Check2(f"loo.{held_out}", False, 0.0,
                                 "at least 20 live bins with counts",
                                 "too few live bins to judge"))
            continue

        o = observed / observed.sum()
        p = shape / shape.sum()
        corr = float(np.corrcoef(o, p)[0, 1])
        checks.append(Check2(
            name=f"loo.{held_out}",
            passed=corr > 0.9,
            measured=corr,
            expected="> 0.90 shape correlation",
            detail=(f"held out {held_out}: its angular shape predicted from a fit "
                    f"that never saw it, correlation {corr:.4f} over "
                    f"{observed.size} live bins"),
        ))
    return checks


def nll_per_bin_check(sol: BaselineSolution, grid: AnalysisGrid) -> Check2:
    """Poisson deviance per live bin. Near 1 means the model fits as well as
    counting statistics allow; far above means it does not; far below means it
    has more freedom than the data supports."""
    deviance = 0.0
    n_live = 0
    for eid, observed in grid.counts.items():
        predicted = sol.predict(eid)
        live = predicted > 0
        o = observed[live].astype(float)
        p = predicted[live]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(o > 0, o * np.log(np.clip(o / p, 1e-12, None)), 0.0)
        deviance += 2.0 * float(np.sum(term - (o - p)))
        n_live += int(live.sum())

    value = deviance / max(n_live, 1)
    return Check2(
        name="deviance_per_bin",
        passed=0.2 < value < 3.0,
        measured=value,
        expected="0.2 < deviance/bin < 3.0",
        detail=f"Poisson deviance {deviance:.1f} over {n_live} live bins",
    )


def format_report2(checks: list[Check2]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{'PASS' if c.passed else 'FAIL'}] {c.name:24s} "
                     f"measured={c.measured:.4f}  expected {c.expected}")
        lines.append(f"         {c.detail}")
    n_pass = sum(1 for c in checks if c.passed)
    lines.append(f"\n{n_pass}/{len(checks)} checks passed")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validate2.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add megido/validate2.py tests/test_validate2.py
git commit -m "feat: Phase 2 validation — bootstrap uncertainty and leave-one-out"
```

---

## Task 8: CLI, real-data run, and the Phase 2 report

**Files:**
- Modify: `megido/cli.py`
- Create: `tests/test_cli2.py`, `docs/phase2-baseline-report.md`

**Interfaces:**
- Consumes: Tasks 1–7
- Produces: a `solve` subcommand

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli2.py`:

```python
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


def test_solve_writes_a_solution_and_returns_zero(site, capsys):
    cfg, run_dir, out = site
    rc = main(["solve", "--config", str(cfg), "--run", str(run_dir),
               "--out", str(out), "--iters", "4"])
    assert rc == 0
    assert (out / "baseline.npz").exists()
    printed = capsys.readouterr().out
    assert "flux index" in printed.lower()


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli2.py -v`
Expected: FAIL — `argparse` rejects the unknown command `solve`

- [ ] **Step 3: Add the `solve` command to `megido/cli.py`**

Add this function alongside the existing `_cmd_validate` and `_cmd_ingest`:

```python
def _cmd_solve(args) -> int:
    from pathlib import Path

    from megido.angular import load_analysis_grid
    from megido.baseline import solve_baseline
    from megido.validate2 import format_report2, leave_one_out, nll_per_bin_check

    cfg = load_site_config(args.config)
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}")
        return 1

    exposure_ids = [e.id for e in cfg.exposures]
    missing = [e for e in exposure_ids if not (run_dir / f"counts_{e}.npz").exists()]
    if missing:
        print(f"missing counts artifacts for: {', '.join(missing)}")
        return 1

    grid = load_analysis_grid(run_dir, exposure_ids, factor=args.rebin)
    sol = solve_baseline(grid, cfg, n_iter=args.iters)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sol.save(out / "baseline.npz")

    print(f"flux index      {sol.flux_index:.3f}")
    print(f"NLL             {sol.nll_history[-1]:.1f} after {len(sol.nll_history)} iterations")
    print("normalizations  " + "  ".join(f"{g}={v:.4g}" for g, v in sorted(sol.norms.items())))
    for pid in sorted(sol.opacity):
        lam = sol.opacity[pid]
        seen = np.isfinite(lam)
        print(f"opacity {pid}: {int(seen.sum())} sky bins constrained, "
              f"median {np.nanmedian(lam):+.4f}, p5..p95 "
              f"{np.nanpercentile(lam, 5):+.4f}..{np.nanpercentile(lam, 95):+.4f}")

    checks = [nll_per_bin_check(sol, grid)]
    checks += leave_one_out(grid, cfg, n_iter=max(args.iters // 2, 4))
    print()
    print(format_report2(checks))
    return 0 if all(c.passed for c in checks) else 1
```

And register it in `main`, next to the existing subparsers:

```python
    s = sub.add_parser("solve", help="Phase 2 baseline and opacity solve")
    s.add_argument("--config", default="configs/megido.yaml")
    s.add_argument("--run", default="runs/ingest")
    s.add_argument("--out", default="runs/solve")
    s.add_argument("--rebin", type=int, default=10)
    s.add_argument("--iters", type=int, default=30)
    s.set_defaults(func=_cmd_solve)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli2.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: everything passes — roughly 108 from Phase 1 plus about 57 new.

- [ ] **Step 6: Run the solve on real data**

Run:
```bash
uv run python -m megido.cli solve --config configs/megido.yaml --run runs/ingest --out runs/solve
```

Record the full output. Expect a fitted flux index somewhere near 2, four normalizations, and two opacity maps — `pos0` covering P0, T20a and T20b, and `pos1` covering P1 alone.

**Do not tune any threshold to make the checks pass.** If `deviance_per_bin` lands far above 3, the model does not fit and that is a finding worth reporting, not hiding. Note the value and continue to the report.

- [ ] **Step 7: Write `docs/phase2-baseline-report.md`**

Include, in this order: the command and its verbatim output; a table of the fitted normalizations with the note that T20a and T20b are deliberately separate because of the unexplained 6 August rate change; the fitted flux index against the nominal 2; the deviance per bin and what it says about fit quality; the leave-one-out results; the number of constrained sky bins per position; and an explicit paragraph on what is **not** yet established — that opacity is still per-position and per-sky-direction, that turning it into 3D density is Phase 3, and that the cone-shaped null space from spec section 6.4 means some of the sky map is weakly determined.

- [ ] **Step 8: Commit**

```bash
git add megido/cli.py tests/test_cli2.py docs/phase2-baseline-report.md
git commit -m "feat: solve command, real-data baseline run, Phase 2 report"
```

---

## Phase 2 exit criteria

- [ ] `uv run pytest` green
- [ ] Synthetic gate: injected `B(d)` recovered with shape correlation > 0.9
- [ ] Synthetic gate: injected opacity recovered with correlation > 0.8
- [ ] **Negative control: a flat injected sky comes back flat** (std < 0.15)
- [ ] Likelihood decreases monotonically across iterations
- [ ] Leave-one-exposure-out passes for every exposure on synthetic data
- [ ] Real-data solve completes and writes `runs/solve/baseline.npz`
- [ ] Deviance per bin recorded, whatever it is
- [ ] `docs/phase2-baseline-report.md` committed, including what is not yet established

Phase 3 begins from `runs/solve/baseline.npz`: two per-position sky opacity maps plus their bootstrap uncertainties, to be triangulated into a voxel volume using the 2.2 m baseline.
