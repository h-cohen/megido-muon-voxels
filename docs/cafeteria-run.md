# TAU cafeteria campaign through the megido pipeline

A second campaign run through the same pipeline, detached from the Megiddo
analysis: its own config (`configs/cafeteria.yaml`) and run folder
(`runs/cafeteria/`). Every new capability is opt-in from the site config; a
config without the new blocks (Megiddo) behaves exactly as before.

## Running it

```bash
C=configs/cafeteria.yaml; R=runs/cafeteria
uv run python -m megido.cli ingest      --config $C --out $R/ingest        # ROOT txty + live times
uv run python -m megido.cli solve       --config $C --run $R/ingest --out $R/solve
uv run python -m megido.cli reconstruct --config $C --solve $R/solve --run $R/ingest \
    --out $R/voxels --cache $R/.cache --bootstrap 8 --backproject-z 7.0
uv run python -m megido.cli export      --config $C --run $R/voxels
uv run python -m megido.cli hillside    --config $C --surface-a 48 --solve $R/solve \
    --run $R/ingest --out $R/voxels                                           # display overlay only
uv run python -m megido.cli view        # Load run -> runs/cafeteria/voxels
```

In the viewer: **Render → Voxel cubes** for block voxels (threshold and
cube size under it); clip z to ≈ 0.66–0.86 (6–8 m) and use the Top preset to
read the ceiling beams.

## How this campaign differs from Megiddo, and what that required

| Megiddo | Cafeteria | Change (all opt-in) |
|---|---|---|
| raw `.data`; Phase 1 builds `counts_*.npz` | DAQ ROOT `txty` TH2 (800×800, tan ±2) | `megido/rootingest.py`: crop to the site binning into the same `counts_*.npz` seam (bins must coincide; never resampled); live time per run = Σ `dT` |
| no open-sky run → tilt-based joint solve | clear-sky roof run of the same detector | `megido/skyref.py`: λ = −ln(max(n_pos / (t_pos/t_sky · n_sky), 0.05)) in the detector frame, scattered to the sky grid by pose |
| opacity zero point degenerate → fitted offset `c_p` | zero point **measured** by the live-time ratio | `BaselineSolution.absolute`; `c_p` fixed at 0 (`solve(..., fit_offsets=False)`); the gauge systematic becomes a ±3% flux-scale systematic |
| Detector 3 constants (width 38.4 cm, tan edge 1.22) | fail here: acceptance edge tan 0.91 | `detector:` override **measured from the sky run**: width 35.375 cm (hit positions, all 4 layers), dz 38.9 cm (acceptance edge) — `DetectorGeometry.for_site` |
| bootstrap re-runs the joint solve | re-runs the sky ratio, sky counts resampled too | `solver=` hook on `opacity_uncertainty` / `voxel_bootstrap` |

Reconstruction settings (`configs/cafeteria.yaml`, each justified inline
there): `tv_alpha` 0.03, `tv_z_weight` 0, `coverage_damping` 0.05, 0.20 m
voxels, display crop x −5…7, y −5…5 m.

## Current result

- Ingest: pos0 6.33 M, pos1 2.41 M, sky 28.5 M tracks; live times 875 175 /
  334 074 / 3 579 978 s. 1293 sky bins constrained per position.
- Voxels: 90×85×40 at 0.20 m, 2586 rows, χ² 0.99 under the 8-replica
  bootstrap σ; 37% of the voxels both positions see are above SNR 3;
  flux-scale (+3%) systematic 0.024 1/m max.
- The ceiling beams (line features along y at ≈ 1.7 m pitch in x, plus a
  cross beam near y ≈ 3 m) are localized **in height**: per-height beam
  contrast is ≈ 0 from 1.9 to 5.5 m and peaks at 7.1 m, matching the
  cafeteria project's independent, model-free beam parallax (7.0–7.1 m).
  Depth in general is still **not** resolved (dz ≈ 0.9 m at 5 m); sharp
  features localize, broad ones do not.
- The sides stay brighter than the beams because they are more opaque:
  directions at 22–45° carry 15–55% more opacity than a flat ceiling
  (6–22σ; walls / neighbouring structure). The beams are the sharpest
  regular structure, not the most opaque.
- Hillside surface: the uniform-solid "hill" model does not describe a
  ceiling (per-ray VE 2.5%; out-of-sample r 0.02 / 0.39 against shuffled
  nulls 0.06–0.14 / 0.15–0.21 at a = 48). Display overlay only.

## Why the reconstruction is set up this way — the bright outer shell

The first reconstruction's outer shell outshone the interior and hid the
beams. Established with phantoms through the real geometry and real-σ noise:

1. **Fitted opacity gauge (fixed by measuring it).** A free `c_p` absorbs the
   constant part of a flat ceiling (phantom: 0.0996 of a true 0.100) and only
   the oblique (sec θ − 1) excess reaches the voxels — in the shell. Measured
   gauge restores the slab flat (centre 0.100, rim 0.099;
   `tests/test_absolute_gauge.py`). Live-time rate ratios agree with the `dT`
   slope and the `rate` profile to 0.3%.
2. **Noise parked in poorly covered voxels (fixed by coverage damping).**
   Voxels one or two rays cross are those rays' private unknowns; SIRT hands
   them their whole residual and non-negativity keeps the positive noise.
   A phantom with only beams + ceiling + noise reproduces the real shell
   (side p99 3.80× beam; real 3.85×). Compared at comparable χ²:

   | solver (real data unless noted) | χ² | beam contrast | side p99 / beam | phantom false shell / beam | phantom mass < 4 m (truth 20%) |
   |---|---|---|---|---|---|
   | TV 0.03, zw 0, no damping | 0.67 | 2.46 | 3.85 | 0.220 | 6% |
   | stronger TV 0.06 / 0.1 | 1.14 / 1.74 | 1.84 / 1.09 | 3.78 / 4.04 | 0.236 / 0.257 | — |
   | step weighting γ 0.5 | 1.80 | 2.43 | 1.88 | 0.157 | — |
   | **coverage damping μ 0.05 (adopted)** | 0.68 | 3.28 | 1.60 | 0.118 | 22% |
   | damping μ 0.2 | 0.91 | 3.74 | 1.16 | 0.096 | 31% (mass piles above the detectors) |
   | + depth prior β 1…30 (rejected) | 0.84…1.54 | 2.00…0.91 | 1.65…1.38 | 0.97…0.94 | a wrong band captures 24–74% |

3. **`tv_z_weight` 0:** z-smoothing fights the parallax. Phantom beam
   contrast 2.64 → 3.16 (truth 4.67); real focus 7.5 → 7.1 m.

## Tried and rejected (with evidence)

- **Stronger TV** to hide the shell: blurs the beams (contrast 0.99 → 0.34
  from `tv_alpha` 0.01 to 0.08; 0.2 removes them).
- **Soft depth prior** toward the 6.3–7.9 m band: real beam contrast
  3.28 → 0.91 and a deliberately wrong band captured 24–74% of the mass.
- **Cropping the solve box** (as the cafeteria project did): oblique rays
  exit through its sides and their opacity lands on the box walls at 2–5 m.
  Cropping the *display* (`volume.viewer_crop_xy_m`) has no such cost.
- **L1 sparsity prior**, **0.4 m voxels**, **early stopping**: each worse on
  contrast or fit. An angle cut |t| ≤ 1.0 is a no-op (acceptance ends at
  0.91 per axis). 0.1 m voxels: contrast 2.46 → 2.84 at 4× cost, χ² 0.44.
- **Viewer coverage gate at 6 rays**: at ceiling height every voxel has only
  4–5 rays, so it hid 82% of the beam voxels; the default is 2.

## Compared with cafeteria_3d_modeling

Its clean-looking viewer product was a thin sheet pinned at 7 m, cropped,
blurred and drawn as iso-surfaces — not a 3D solve. Its genuine 3D solve
(`runs/full3d`) peaks its beams at 8.5 m (contrast 1.77) with side p99 6.98×
its beam mean; ours peaks at 7.1 m (2.48 before damping, 3.28 after).

## Caveats

- pos1 pose (1.775, 0.720) m is the cafeteria project's self-calibration,
  never surveyed; absolute lateral scale rides on it.
- Real flux differences between the roof run and the cafeteria epoch enter
  λ as a constant; shown as the ±3% flux-scale systematic map.
- The viewer title reads "Megiddo" (hardcoded); the data shown is the loaded
  run.
