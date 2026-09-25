# TAU cafeteria campaign through the megido pipeline

Detached from the Megiddo analysis: its own config (`configs/cafeteria.yaml`),
its own run folder (`runs/cafeteria/`), branch `worktree-cafeteria-run`. The
Megiddo config, artifacts and code paths are unchanged (no `detector:` /
`sky_reference:` block → exactly the old behaviour, tested).

## What had to change, and why

| Megiddo | Cafeteria | Change |
|---|---|---|
| raw `.data`, Phase 1 builds `counts_*.npz` | DAQ ROOT `txty` TH2 (800×800, tan ±2) | `megido/rootingest.py` crops to the site binning and writes the same `counts_*.npz` seam (bins must coincide; no resampling) |
| no open-sky run → tilt-based joint solve | clear-sky roof run of the same detector | `megido/skyref.py`: `λ = −ln(max(n_pos/(s·n_sky), 0.05))` in the detector frame, scattered to the sky grid by pose; gauge-pinned exactly like the joint solve |
| Detector 3 constants (width 38.4 cm, dz 31.5 cm, tan edge 1.22) | fails here: sky acceptance edge is tan 0.91 | optional `detector:` override, **measured from the sky run**: width 35.375 cm (hit positions, all 4 layers), dz 38.9 cm (acceptance edge) — `DetectorGeometry.for_site` |
| bootstrap re-runs the joint solve | re-runs the sky ratio, sky counts resampled too | `solver=` hook on `opacity_uncertainty` / `voxel_bootstrap` |

Without `--bootstrap`, `reconstruct --run` uses the analytic Poisson sigma
`sqrt(1/n_pos + 1/n_sky)`.

## Commands

See the header of `configs/cafeteria.yaml`. The hillside stage was run with
`--surface-a 48` (see below).

## The bright outer shell (diagnosed 2026-09-24)

Symptom: the outermost voxels outshine the interior and mask the beams.
Two causes, established with phantoms through the real geometry:

1. **Fitted opacity gauge (FIXED).** With a relative gauge the per-position
   offset `c_p` absorbs the constant part of a flat ceiling (phantom: c_p =
   0.0996 of a true 0.100) and only the oblique `(sec θ − 1)` excess reaches
   the voxels, in the outer shell. Pinning c_p at its true value restores the
   slab flat (centre 0.100, rim 0.099). The cafeteria gauge is now
   **measured**: live time per run = Σ `dT` (rate ratios cross-checked
   against the `dT` slope and the `rate` profile to 0.3%), λ absolute, c_p
   fixed at 0 (`BaselineSolution.absolute`, `solve(..., fit_offsets=False)`;
   gate: `tests/test_absolute_gauge.py`). Real data: inner column opacity
   0.065 → 0.084 (vertical λ ≈ 0.08, σ 0.007); negative-λ bins are all
   within 3σ (acceptance-edge noise). A free fit would take c_p = +0.03 /
   +0.02. Residual systematic: flux difference roof vs cafeteria epoch,
   shown as the +3% flux-scale systematic map (7% of peak).
2. **Noise overfitted into exclusive voxels (handled: `tv_alpha` 0.03 +
   viewer coverage gate at 6 rays).** Stronger TV alone is the wrong tool:
   beam modulation (column profile across y≈0, peaks x = 0.1/1.7/3.5 m vs
   troughs) falls 0.99 → 0.72 → 0.55 → 0.34 at `tv_alpha` 0.01/0.03/0.05/0.08,
   and 0.2 removes the shell but blurs the beams away. The shell is instead
   hidden by `rays.npy` (rows crossing each voxel, written by `reconstruct`)
   and the viewer's "hide voxels crossed by fewer than N rays" gate (on by
   default when the layer exists, N = 6; display-only, hidden = not
   constrained). Edge/centre p99 density with the gate at 6: 3.3× → 1.8× at
   0.01 → 0.03. 0.03 is where χ² ≈ 1 under the pipeline's bootstrap σ
   (~1.45× tighter than analytic). Details of the diagnosis:
   306 k voxels vs 2586 rows: near the grid edge every oblique ray has voxels
   no other ray crosses. Under non-negativity, positive noise becomes mass
   there and negative noise can only erode the shared interior. Flat-ceiling
   phantom + real-σ noise, measured gauge: shell/covered density 17.5× at
   `tv_alpha` 0.01 (χ² 0.26), 3.3× at 0.2; noiseless 1.1×. Early stopping
   removes the shell only by stopping before the ceiling is reconstructed
   (χ² < 1 after 2 iterations; column 45% of truth). Real-data TV sweep:
   shell 5.2× / 2.8× / 2.2× / 1.2× at `tv_alpha` 0.01 / 0.03 / 0.08 / 0.2,
   χ² 0.31 → 2.66; out-of-sample r best at 0.2 in both directions
   (0.13 / 0.22), weak everywhere.

3. **Real oblique opacity, placed by the depth null space (not an
   artifact; not removable by the voxel solve).** Measured λ exceeds a flat
   ceiling (λ₀·sec θ, λ₀ from |t| < 0.2) by +0.013…+0.019 (pos0) and
   +0.019…+0.055 (pos1) between 22° and 45°, 6–22σ: walls / neighbouring
   structure. Near-vertical rays pin the central columns low, so the solver
   can only put that excess where oblique rays alone go — the outer, upper
   grid. At ceiling height centre and edge voxels have the SAME ray count
   (4–5 at z 6.7–7.5 m; rays land ~0.35 m apart there, wider than the voxels),
   so no ray-count gate separates them; a gate strong enough to remove this
   part also removes the beams. The beams are sharpest in 2D (column opacity,
   backprojection at 7 m); in 3D the remedy is a different unknown (a layer
   at the independently measured ~7 m ceiling), not a better voxel solve.

## Beams in 3D (2026-09-24)

The 3D voxels DO localize the beams in height: two positions line up sharp
features only at their true height. Spike (phantom beams at 6.6–7.0 m +
ceiling slab + side walls, real geometry, real-σ noise): per-height beam
contrast (profile across y≈0, peaks at the beam x vs troughs) is ≈0 from
1.9 to 5.5 m and peaks at 6.7 m for every solver variant tried. Real data
peaks at 6.3–7.9 m. What changed:

- `tv_z_weight` 0 (was 0.5): z-smoothing fights the parallax. Phantom
  contrast 2.64 → 3.16 (truth 4.67); real focus 7.5 → 7.1 m (cafeteria
  project's model-free beam parallax: 7.0–7.1 m), contrast 1.91 → 2.46.
  Rejected in the same spike: an L1 sparsity prior (contrast 2.84 at 0.05,
  collapses at 0.15, χ² 4.9) and 0.4 m voxels (worse contrast, larger shell).
- Viewer SNR gate at 3: the corners of the square acceptance are the noisiest
  directions and leave diagonal streaks the ray gate keeps; at SNR ≥ 3 the
  streaks fall to 0.21 of beam brightness while the beams keep their value.

Previous (relative-gauge) results are kept in `runs/cafeteria/solve_relgauge`
and `voxels_relgauge`; `voxels/gauge_before_after.png` compares them.

## Results (2026-09-24)

- Ingest: pos0 6.33 M, pos1 2.41 M, sky 28.5 M tracks, 100% inside ±1.25.
- Opacity: 1293 sky bins constrained per position.
- Voxels: grid 90×85×40 at 0.20 m, 2586 rows, χ² 0.36; 8-replica bootstrap:
  58% of the voxels seen by both positions are above SNR 3; gauge systematic
  11% of peak.
- Lateral structure: line features along y at roughly 1.7 m pitch in x (the
  ceiling beams the cafeteria project found) plus a transverse band near
  y ≈ 4 m. They appear both in the inverted column opacity and in the
  model-free backprojection at z = 7 m (`cafeteria_overview.png`).
- Depth: **not resolved** (dz ≈ 0.9 m at z = 5 m on the 1.92 m baseline); mass
  piles toward the grid top — regulariser, not data. Same honest limit as
  Megiddo.
- Hillside surface: the uniform-solid "hill" model does **not** describe this
  overburden (a ceiling with beams, not a hill). At a = 48 (chosen so the
  surface sits near the ~7 m ceiling; a = 8, Megiddo's rock default, puts it
  ~1 m up and each single-position fit misses the other detector entirely):
  per-ray VE −9%, out-of-sample r 0.08 / 0.32, inside the shuffled nulls.
  Keep it as a display overlay only.

## Caveats

- pos1 pose (1.775, 0.720) m is the cafeteria project's self-calibration,
  never surveyed; absolute lateral scale rides on it.
- The 0.20 m voxel spacing is coarse for 1.7 m-pitch beams (the cafeteria
  project needed 0.08 m for clean beam lines); raising resolution is a config
  change (`volume.spacing_m`) at a cost in solve time.
- The viewer title still reads "Megiddo" (hardcoded); the data shown is the
  loaded run.
