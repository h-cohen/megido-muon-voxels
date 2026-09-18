# Phase 3 reconstruction report

## Four things to know before reading any number below

1. **This geometry cannot localize depth, and the reconstruction does not pretend
   to.** Every muon ray points upward, so the inverse problem is one-sided and
   limited-angle. With a single 2.2 m baseline between the two positions, the
   closed-form depth resolution is **dz ≈ 1.36 m at mid-range — about 5× the
   0.25 m voxel spacing.** The same baseline sets an alias period of **5.91 m**
   (for an assumed 2 m feature pitch): structure separated by roughly that
   distance along the depth axis is indistinguishable from structure at a
   different depth. Lateral (map-view) structure is measured; the *height* of
   that structure is set by the smoothness prior, not by the data. The volume's
   own `meta.json` carries this verdict (`depth_resolved: false`) so no downstream
   viewer can present a crisp depth surface without also carrying the caveat.
2. **The headline "well-measured" fraction is 70% of a small region, not 88% of
   the grid.** The CLI once printed "591479 of 675840 voxels above SNR 3" (87.5%).
   That number is meaningless: ~71% of the grid is edge padding that no ray ever
   crosses, and those voxels get stable near-zero prior-driven values that score
   high SNR while measuring nothing. The honest figure is **69.5% above SNR 3
   within the 18% of the grid seen by both positions** (the depth-informative
   region).
3. **The absolute opacity scale is convention-dependent by ~40%.** There is no
   open-sky run, so the zero point of opacity is a chosen convention (Phase 2's
   transparency quantile). Varying that convention moves the reconstructed density
   by up to **40.2% of its own peak.** This is a systematic, reported separately
   from the statistical bootstrap sigma, and it does not shrink with more counts.
4. **The physically meaningful opacity numbers are those inside the viewed
   region.** The all-grid median density (0.0025 1/m) is dominated by unviewed
   near-zero voxels and should be ignored. Within the two-position region the
   median is 0.078 1/m, p95 0.170, max 1.44.

Each is expanded below.

## 1. What Phase 3 does

Phase 2 produced, for each detector position, the muon opacity λ integrated along
every constrained sky direction — a line integral of rock density along each ray.
That is exactly a tomographic measurement. Phase 3 inverts it into a 3D field:

```
S3  forward model   poses  -> sparse path-length matrix A over a voxel grid
S4  inversion       λ, A   -> voxel opacity density x >= 0, plus uncertainty
    export                 -> volume.npy + meta.json for the viewer
```

The system is `λ = A x + c_p` with `x ≥ 0`, where `c_p` is a free additive
constant per position (Phase 2 pins each position's opacity gauge independently,
so the absolute level of λ is not a measurement and is refit here). The solver is
SIRT with an anisotropic total-variation proximal step; the pose rotation is
absent by design, because Phase 2 already mapped every measurement into the world
frame. Alongside the full fit, the run produces one single-position holdout fit
per position, a per-voxel Poisson bootstrap, the gauge systematic, a per-voxel
view count, and a model-free backprojection.

## 2. Real-campaign result

Full run over the four exposures (P0, T20a, T20b at one position; P1 at 2.2 m),
every number below taken from `runs/phase3_real.log`:

| Quantity | Value |
|---|---|
| Voxel grid | 128 × 120 × 44 at 0.25 m, origin (−14.89, −14.89, 1.0) m |
| Rows inverted | 2915, over 2 positions |
| χ² (weighted residual per used row) | 0.4029 |
| Per-position offsets (gauge nuisance) | pos0 +0.697, pos1 +0.092 |
| Opacity density, two-position region | median 0.078, p95 0.170, max 1.44 1/m |
| Opacity density, whole grid | median 0.0025 1/m — *dominated by unviewed voxels; do not use* |
| Gauge systematic | max \|Δρ\| 0.95 1/m = **40.2% of the peak** |
| Backprojection anchor | plane at z = 6 m, 58 × 50 at 0.30 m pitch |
| Depth resolution / alias period | dz_mid 1.36 m (~5× voxel spacing); alias period 5.91 m at 2.2 m baseline |

**Bootstrap uncertainty (6 replicas, full chain re-run per replica):**

| Region | Voxels | % of grid | Above SNR 3 |
|---|---|---|---|
| Viewed by ≥1 position | 197067 | 29.2% | 62.5% |
| Viewed by both positions (depth-informative) | 121616 | 18.0% | **69.5%** |
| Whole grid | 675840 | 100% | 87.5% — *meaningless; see caveat 2* |

The bootstrap re-runs the entire chain — including the Phase 2 baseline solve —
on each Poisson-resampled replica, so the spread carries the detector-response
uncertainty, not only the inversion's. Six replicas is indicative, not a precise
error bar; the SNR fractions are stable coverage estimates, not tight confidence
intervals.

## 3. What the two phantom gates establish

Two gates run the same code, and they answer two different questions.

- **Task 7 — is the code correct, under generous geometry?** Twenty-five
  detectors over a ±30 m grid, a topography-shaped thin anomaly. Result: data
  reproduced at correlation **0.997**, lateral column structure recovered at
  **0.80**, depth localization **0.19** (a single-layer truth smears across z).
  This licenses one claim only: the forward model and solver are correct. It says
  nothing about the real campaign — its geometry is far more generous than
  anything Megiddo has.
- **Task 9 — what can the real campaign claim?** The actual two-position geometry
  with the single 2.2 m baseline and the 20° tilt. Result: data reproduced at
  **0.995**, lateral recovery **0.66** (weaker, as two positions must be), depth
  localization **0.086**. This is the honest picture of the real result: lateral
  structure is measured; depth is not.

Neither gate asserts a surface-fidelity (interface-RMSE) bound, because a
one-sided geometry cannot recover a filled surface's depth and a gate that
demanded it would be demanding a physical impossibility.

## 4. What would change the answer

**The most promising next analysis is a different observable, not a better
solver.** The detector sits under a small hill in a long cavern, looking up. The
overburden is thick toward the crown and thins toward the flanks, so a line of
sight that grazes out through the hillside crosses from heavy absorption to
near-open sky over a narrow angular range — a **sharp edge in the muon flux, the
hill's silhouette.** That edge is a *lateral* feature, exactly the class this
geometry resolves well (lateral correlation 0.66 against depth localization
0.09). Two consequences worth pursuing (recorded in spec §11):

1. The edge can be located per position by a gradient detector on the
   reconstructed sky flux, rather than by trusting per-voxel depth.
2. A sharp edge gives far better parallax than a diffuse field, so the shift of
   the flux edge between P0 and P1 can constrain the hillside surface where the
   voxel field cannot — turning the campaign's one weakness (depth) into a
   tractable one-dimensional edge-disparity problem built on its one strength
   (lateral contrast).

**On the instrument side:** a second *translated* position would add a genuine
baseline and directly improve depth resolution (dz scales as 1/baseline). A
larger *tilt* at the same spot would not — tilt adds sky coverage, not parallax.

## 5. Open items carried forward

- **Rock material** — needed to re-derive the multiple-Coulomb-scattering
  systematic; do not inherit a concrete radiation length from the cafeteria
  project.
- **Surveyed baseline** — the 2.2 m P0↔P1 separation sets the absolute depth
  scale and should be independently confirmed; angles alone fix only ratios.
- **Structure scale** — now has phantom evidence (spec §11): lateral structure is
  recovered, depth is regulariser-set, and `megido.resolution.campaign_resolution`
  reports the closed-form number for any configured volume.
