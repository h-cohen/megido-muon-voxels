# Upper-Envelope Hillside Surface — Design

**Status:** proposed
**Date:** 2026-09-23
**Builds on:** `docs/superpowers/specs/2026-09-22-hillside-gp-surface-design.md`
(the Matérn-5/2 GP surface). This changes *what the GP fits*, not the GP core.
**Parent spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` §11.

## 1. Motivation — the detector-spot dip is a real defect

The fitted hillside surface has a ~2.4 m **dip** at each detector's own (x, y).
Checked against ground truth (`topography.csv`, test-only), that region is a
**peak**, not a dip: the detectors sit under the crown with the thickest
overburden straight up (true surface ≈ 50 m over the detectors, ≈ 49 m a few
metres out). The current fit **inverts** the true shape exactly there.

**Root cause (diagnosed, not assumed).** Each exit point sits at
`o + a·λ·d̂`. A high-opacity ray (thick overburden — the crown) is pushed
*outward* by its large `a·λ·sinθ` and lands in the surrounding ring; a
low-opacity ray stays near the detector and exits low. Worse, Phase 2's gauge
pins the transparent-quantile opacity to zero, manufacturing spurious short,
low-`λ` rays whose exit points pile up at the detector footprint. The result:
**the exit-point placement sorts rays radially by opacity**, so a detector's own
column is dominated by the lowest exit points and the mean-like GP through all
points is biased low — by construction.

**Why flux weighting alone is insufficient.** The opacity solve already corrects
flux (`cos(θ)^index`, index fits to 2.0 = the known cos²θ). Adding a `cos²θ`
weight to the surface fit barely moves the dip: even a fit using *only*
near-vertical rays still dips ~1.5 m, because the radial-sorting artifact is in
the *placement*, not the angular sampling.

**Validated fix.** The true surface is the **maximum overburden per column**; the
low exit points are artifacts. Fitting the per-column **upper quantile** (~0.85)
of exit-z lifts the detector-footprint height to crown level (~9–11 m at the
convention scale), largely erasing the dip. This is the change.

## 2. Scope

**In scope:**
- Change the GP training data from raw exit points to a **per-cell,
  flux-weighted upper-quantile** reduction of exit-z (the "upper envelope").
- Fold the cosmic-ray flux weight (`dz^index`, the known angular distribution)
  into that reduction — up-weighting the high-flux near-vertical rays.
- A load-bearing **dip gate** using real data + `topography.csv` ground truth
  (permitted in tests only), plus no-regression on the synthetic recovery and
  calibration gates.
- Keep the GP core (`megido/hillside_gp.py`) and `SurfaceResult` unchanged;
  viewer/CLI unchanged.

**Out of scope (explicit):**
- The opacity gauge zero-point (transparent-quantile pinning). It is the
  upstream contributor to the spurious low rays and is **noted, deferred** — a
  known Phase-2 property (absolute level unmeasured), revisited separately if
  needed. Nothing here modifies `megido/baseline.py`'s solve.
- The ray-intersection forward model (rejected: non-linear, stalls optimizers).
- Robust/asymmetric-loss GP (rejected in favour of the simpler, validated
  quantile reduction).
- Absolute-scale recovery — still assumed via `a`; only shape is data-driven.
- `topography.csv` in the deliverable path — it stays TEST-ONLY ground truth,
  never read by `megido/` or the CLI. `calibration_open_sky.csv` never read.

## 3. The estimator

### 3.1 Exit points (unchanged placement, `dz` re-exposed)
`exit_points(sol, cfg, a)` returns `(pts, pos_index, dz, sky_flat)` — `dz` (the
sky-frame vertical direction cosine `1/√(1+sx²+sy²)` = cos(sky zenith)) is
re-added, needed for the flux weight. (This is unrelated to the earlier
double-`dz` *noise* bug: the height-space noise still uses `|z − pose.z|`, never
`dz` again.)

### 3.2 Per-cell upper-envelope reduction (new)
On the footprint grid `(gx, gy)` (from `_footprint_grid`, unchanged), assign each
in-grid exit point to its nearest cell. For each cell with at least `min_count`
points (default 8):

- **flux weight** per point `w = dz^index`, `index = sol.flux_index` (the known
  cos²θ at index 2.0), clipped to a small floor to stay finite.
- **target** `y_cell` = the **flux-weighted upper quantile** at level `q_hi`
  (default 0.85) of the cell's exit-z values. Weighted quantile: sort exit-z,
  take the value where the cumulative flux weight reaches `q_hi` of the total.
  This tracks the maximum overburden (crown) and up-weights high-flux vertical
  rays simultaneously.
- **noise** `σ_cell²` from the spread of the cell's *upper band* (exit-z at or
  above `y_cell`'s neighbourhood): `σ_cell = (weighted IQR or std of the upper
  band)/√n_eff + rel_floor·|y_cell − pose.z|`, `n_eff` = effective (flux-weighted)
  count. Cells with tiny `n_eff` get larger noise. This replaces the per-ray
  Poisson noise; `sky_counts`/`--run` still feed `n_eff` when available
  (heteroscedastic), else a geometry-only floor is used (fallback note).

Cells below `min_count` produce no target (they become NaN via the coverage
mask, never 0).

### 3.3 GP fit (unchanged core)
Fit the Matérn-5/2 GP by ML-II to the cell centroids `X_cell` and targets
`y_cell` with per-cell noise `σ_cell²` (`megido.hillside_gp.fit_hyperparams` /
`predict`, unchanged). The training set is now ≤ `nx·ny` cells — far smaller than
the ~3k raw rays, so the `max_points` subsample is largely unnecessary but kept
as a guard.

### 3.4 Prediction, coverage, honesty (unchanged)
Predict posterior mean `H` + std `sigma` on the grid; `covered = sigma <
cov_tau·signal_std`; uncovered → NaN (never 0). `support = 1 − sigma/signal_std`.
`variance_explained` computed on the cell targets. `scale_assumed = True`; note
updated to state: upper-envelope (q_hi), flux-weighted to the known cos²θ,
absolute scale assumed via `a`, gauge zero-point deferred.

## 4. Interfaces

### 4.1 `megido/hillside_surface.py`
```python
def exit_points(sol, cfg, a=1.0, transparent_quantile=0.05)
    -> (pts, pos_index, dz, sky_flat)      # dz re-added

def fit_surface(sol, cfg, *, a=8.0, cell_m=1.0, sky_counts=None,
                cov_tau=0.9, n_restarts=3, max_points=1000,
                q_hi=0.85, min_count=8) -> SurfaceResult
```
- New params `q_hi`, `min_count`. `SurfaceResult` fields unchanged
  (`n_rays` = total in-grid rays used across cells; add nothing).
- New private helper `_upper_envelope_cells(X, z, dz, gx, gy, index, q_hi,
  min_count, sky_counts_per_ray) -> (X_cell, y_cell, noise_cell)`.

### 4.2 CLI / viewer
No interface change. `hillside` gains optional `--surface-qhi` (default 0.85) and
`--surface-min-count` (default 8) for tuning, alongside the existing
`--surface-a` / `--surface-cell` / `--run` / cost flags. Meta JSON adds `q_hi`.
Viewer unchanged (same npy/meta).

## 5. Testing

TDD, red before green. Assert physical relationships; a gate must be able to fail.

1. **Weighted-quantile unit test:** `_upper_envelope_cells` on a hand-built cell
   returns the flux-weighted `q_hi` value; a cell with `< min_count` points is
   dropped; higher `dz` points pull the target up (flux weight sign).
2. **Dip gate (load-bearing, real data + ground truth):** sk-if `runs/solve` +
   `topography.csv` present. Fit real data (fast settings). Confirm truth has NO
   dip at the detectors (truth_det ≥ truth_ring − ε — a sanity check on the
   region), then assert the FITTED footprint is not a spurious minimum:
   `H_det ≥ ring_mean − tol` (tol small, e.g. 0.5 m at convention scale), and the
   near-detector fitted-vs-truth shape correlation exceeds a baseline. This gate
   FAILS on the old mean estimator (dip ≈ 2.4 m below ring) and passes on the
   upper-envelope fit. Thresholds set from the measured fix, documented as a
   ruling, never loosened to pass.
3. **No regression — synthetic recovery:** the synthetic-hill gate (corr > 0.9)
   and calibration (1σ in [0.6, 0.95]) still hold with the upper-envelope
   estimator on the clean synthetic fixture (where no dip artifact exists, the
   upper envelope must not distort the recovered smooth hill).
4. **Flux weight applied:** a fit with the flux weight vs a flat weight differs
   in the expected direction (more vertical influence), and `index` comes from
   `sol.flux_index`.
5. **Honesty invariants:** `scale_assumed is True`; note states upper-envelope +
   flux + assumed scale; unconstrained cells → NaN not 0; height scales ~linearly
   with `a`.
6. **Real regen + full suite** green; record real VE, coverage, and the
   before/after detector-footprint heights.

## 6. Risks & rulings

- **Upper quantile could bias the whole surface high**, not just the detector.
  The no-regression synthetic gate guards this: on a clean hill the q_hi fit must
  still recover the true smooth shape (corr > 0.9), not an inflated envelope.
  Ruling at implementation: tune `q_hi` against the synthetic recovery AND the
  real dip removal jointly; if they conflict, favour the honest synthetic
  recovery and report the residual real dip rather than over-tuning `q_hi`.
- **Coarser training set (cells not rays)** reduces effective N; the GP posterior
  std may widen. That is honest (fewer independent constraints) and acceptable.
- **VE may shift** from the current 0.33; the calibration + dip gates are the
  primary quality bars, VE secondary (documented, as before).
- **The gauge remains the upstream cause** of spurious low rays; the upper
  envelope is robust to them but does not fix the gauge. Noted for a future pass.

## 7. Deliverable

Same artifacts; the detector dips replaced by a crown-consistent surface, flux
weighted to the known cos²θ. Viewer unchanged. `topography.csv` used only to gate
the fix in tests, never in the shipped path.
