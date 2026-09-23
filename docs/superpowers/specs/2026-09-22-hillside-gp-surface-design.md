# Gaussian-Process Hillside Surface — Design

**Status:** proposed
**Date:** 2026-09-22
**Supersedes:** the linear (bilinear + Laplacian) surface inversion in
`megido/hillside_surface.py` (Phase 5b). That code documented a negative-to-positive
result and is retained in git history; this design **replaces** it as the shipped
hillside deliverable.
**Parent spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md`
§11 "Data-driven hillside extraction from the flux edge."

## 1. Motivation

The hillside surface `H(x, y)` is the project's crucial scientific deliverable. The
current linear fit (bilinear sampling of exit points onto a grid + Laplacian
smoothness, normal-equations solve) achieves variance-explained ≈ 0.40 on the real
campaign, with two weaknesses:

1. **Smoothness is hand-tuned.** The regulariser weight `mu` is a magic constant
   (0.3); nothing ties it to the data's actual length-scale.
2. **Uncertainty is a convention-proxy, not an error bar.** `sigma` comes from
   bootstrapping the opacity zero-point (`transparent_quantile`), which measures
   sensitivity to a gauge convention, not the statistical uncertainty of the height.

A Gaussian-process (GP) regression on the exit points fixes both: the smoothness
length-scale is **learned from the data** (marginal-likelihood / ML-II), and the GP
posterior standard deviation is a **genuine per-cell height error bar** that grows
away from data — turning the coverage edge into calibrated uncertainty rather than a
hard mask.

**What does NOT change (honesty invariants).** The GP does not create parallax
information. The absolute height scale remains set by the assumed inverse-density
`a` (opacity-units per metre); only the fitted SHAPE at that scale is data-driven.
The honesty note in `SurfaceResult` is preserved and updated to describe the GP.
This is a one-baseline campaign; depth/absolute-scale is still not resolved.

## 2. Scope

**In scope:**
- A new GP surface core (`megido/hillside_gp.py`): Matérn-5/2 kernel, ML-II
  hyperparameter fit, heteroscedastic observation noise, prediction of posterior
  mean + std on the footprint grid.
- Rework `megido.hillside_surface.fit_surface` to call the GP core and return the
  **same `SurfaceResult` dataclass** (unchanged field names/types), so the CLI and
  viewer need no interface changes.
- Heteroscedastic noise from Poisson counts per sky bin + geometric z-leverage.
  This needs the ingested counts, which `baseline.npz` does not carry; the `hillside`
  CLI gains an optional `--run` flag (mirroring `reconstruct --bootstrap`'s
  precedent). Without `--run`, fall back to geometry-only leverage weights and print
  a note.
- Retire the linear solver internals (`bilinear_matrix` used only for support →
  replaced; `laplacian_matrix`, `solve_surface`, `_solve_with_ridge`). `exit_points`
  and `_footprint_grid` are RETAINED and reused.
- Tests and an honest exit gate.

**Out of scope (explicitly):**
- Voxel rendering upgrades and voxel reconstruction changes (separate threads).
- The viewer's display-only smoothing slider stays display-only and unchanged.
- Any new heavy dependency. GP is hand-rolled on numpy + scipy.
- Ground-truth validation against `topography.csv` in the deliverable path
  (fit-only, per standing project decision). `topography.csv` may be used ONLY in
  tests as synthetic ground truth.

## 3. The GP model

### 3.1 Observations
`exit_points(sol, cfg, a)` (retained, unchanged) yields N exit points
`(x_i, y_i, z_i)` — one per finite, positive-opacity sky pixel per position
(N ≈ 3174 on the current campaign: 1972 at P0, 1202 at P1). We regress the height
observation `z_i` on the location `(x_i, y_i)`.

### 3.2 Prior
`H(x, y) ~ GP(m, k)` with:
- **Mean** `m`: constant, set to the weighted mean of `z_i` (a constant mean keeps
  extrapolation honest — the posterior reverts to the data mean, not a trend, away
  from data).
- **Kernel** `k`: **Matérn ν=5/2**, isotropic, with signal variance `σ_f²` and
  length-scale `ℓ`:
  `k(r) = σ_f² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(−√5 r/ℓ)`, `r = ‖(x,y)−(x',y')‖`.
  Matérn-5/2 is the geostatistics/kriging default for terrain: twice
  mean-square differentiable (smooth hillside) without the unphysical infinite
  smoothness of the RBF kernel.

### 3.3 Heteroscedastic noise
Each observation has its own variance `σ_i²` on the diagonal of the noise matrix:

`σ_i² = (a · dz_i · σ_λ,i)² + (η · σ_f)²`

- `dz_i = 1/√(sx_i² + sy_i² + 1)` — the ray's vertical direction cosine. Converts
  opacity-path noise into height noise (retained from `exit_points`' geometry).
- `σ_λ,i` — opacity uncertainty for the sky bin. From Poisson counting: with
  `N_i` = total observed counts in that sky bin at that position
  (`position_sky_counts[pid][flat_i]`, rebuilt from the ingest grid), the relative
  opacity error scales as `1/√N_i`, so `σ_λ,i = λ_i / √(max(N_i, 1))` with a small
  relative floor to avoid zero-noise on very high-count bins.
- `η` — a learned homogeneous noise-floor fraction (nugget), so the model is
  identifiable even where counts are large. Fitted by ML-II alongside `σ_f`, `ℓ`.
- **Fallback (`--run` absent):** counts unavailable → drop the Poisson term and use
  `σ_i² = (a · dz_i · c)² + (η σ_f)²` for a constant `c` (pure geometric leverage),
  and set `SurfaceResult.note` to say the fit is homoscedastic-in-opacity.

### 3.4 Hyperparameter fit (ML-II)
Fit `θ = (log ℓ, log σ_f, log η)` by minimising the negative log marginal
likelihood

`nll(θ) = ½ zᵀ Kθ⁻¹ z + ½ log|Kθ| + (N/2) log 2π`,   `Kθ = k(X,X) + diag(σ_i²)`

via `scipy.optimize.minimize` (L-BFGS-B) with a Cholesky factorisation of `Kθ`
(N≈3k → one factorisation ≈ 1–3 s; a handful of optimiser iterations is seconds).
Bounds keep `ℓ` within [cell, footprint], `σ_f`, `η` positive. Multiple restarts
(≈3) guard against local minima.

### 3.5 Prediction
On the footprint grid nodes `X*` (`_footprint_grid`, retained), the posterior is
- mean `H = m + k(X*, X) Kθ⁻¹ (z − m)`
- covariance diag → `sigma = sqrt(k(X*,X*) − k(X*,X) Kθ⁻¹ k(X,X*))` (predictive std
  of the latent height; the honest per-cell error bar).

### 3.6 Coverage / NaN masking
Repo rule: an unconstrained node is NaN, never 0. With a GP the natural, honest
criterion is the **posterior std relative to the prior std**: a node whose predictive
`sigma ≥ τ · σ_f` (default `τ = 0.9`, i.e. the data barely reduced the prior there)
is unconstrained → `H` and `sigma` set NaN. `coverage_frac` = fraction of grid nodes
below `τ`. This replaces the bilinear support count and removes `min_support`.
`coverage_radius_m` = 95th percentile of in-grid data-point radius from the detector
centroid (retained computation).

## 4. Interfaces

### 4.1 `megido/hillside_gp.py` (new)
```python
def matern52(X1, X2, length_scale, signal_var) -> np.ndarray: ...
    # (n1, n2) kernel matrix

def nll(theta, X, z, noise_var, mean) -> float: ...
    # negative log marginal likelihood; theta = (log ℓ, log σ_f, log η_floor)

def fit_hyperparams(X, z, noise_base, mean, *, n_restarts=3) -> GPHypers: ...
    # ML-II; noise_base = the count/geometry part of σ_i² (η added inside)

def predict(X_train, z, noise_var, mean, Xstar, hypers) -> tuple[np.ndarray, np.ndarray]:
    ...  # returns (mean_star, std_star)
```
`GPHypers`: frozen dataclass `(length_scale, signal_std, noise_floor, nll)`.

### 4.2 `megido/hillside_surface.py` (reworked `fit_surface`)
Signature changes minimally; **`SurfaceResult` is unchanged**:
```python
def fit_surface(sol, cfg, *, a=8.0, cell_m=1.0,
                sky_counts: dict[str, np.ndarray] | None = None,
                cov_tau=0.9, n_restarts=3) -> SurfaceResult: ...
```
- `sky_counts`: per-position `position_sky_counts` (from the ingest grid) enabling
  heteroscedastic Poisson weights; `None` → geometric-leverage fallback.
- Removed params: `mu`, `min_support`, `ridge`, `n_boot`, `quantiles` (all
  linear-solver / bootstrap-sigma specific).
- `SurfaceResult` fields keep names/types. `support` is repurposed to carry the
  per-node data-influence proxy `1 − sigma/σ_f` (still a "how constrained" map for
  the viewer/PNG); `variance_explained` = weighted VE of GP mean vs `z`;
  `scale_assumed` stays `True`; `note` updated.

### 4.3 CLI `hillside` (`megido/cli.py`)
- Add `--run` (default `runs/ingest`, like `solve`/`reconstruct`): when the ingest
  counts are present, build `position_sky_counts` and pass as `sky_counts`.
- If `--run` counts are missing/unreadable, proceed with fallback + printed note.
- Writes are unchanged: `hill_surface.npy`, `hill_surface_sigma.npy`,
  `hill_surface_meta.json`, `hill_surface.png`. `--surface-a` (default 8.0) and
  `--surface-cell` (default 1.0) unchanged. Add hypers (`length_scale`,
  `signal_std`, `noise_floor`) and `heteroscedastic: true/false` to the meta JSON.

### 4.4 Viewer
No code change required — same `SurfaceResult` → same npy/meta shapes. The existing
"Hillside surface" toggle, `sigma` mesh, and display-only smoothing slider all keep
working. The `note` and VE shown in the dock update automatically from the meta.

## 5. Rebuilding per-sky-bin counts

`solve` builds the counts grid from the ingest histograms; `BaselineSolution.load`
deliberately drops counts (`counts={}`). The `hillside` path will rebuild them with
the same loader `solve` uses (the ingest angular histograms under `--run`), grouped
to `position_sky_counts[pid]` exactly as `baseline._build_terms` does (live mask +
`MIN_SKY_COUNTS`). Factor that grouping into a small reusable helper if it is not
already callable without running the full solve; do not duplicate the masking logic.

## 6. Testing

TDD, red before green. Assert physical relationships, not implementation.

1. **Kernel PSD & shape:** `matern52(X,X,ℓ,σf)` is symmetric positive-definite
   (Cholesky succeeds) and `k(r=0)=σf²`, decreasing in r.
2. **ML-II recovers a known length-scale:** sample a GP draw (or a sinusoid) at a
   known `ℓ`, add noise; `fit_hyperparams` recovers `ℓ` within a tolerance. Gate must
   fail for a stub returning a fixed `ℓ`.
3. **Heteroscedastic weighting works:** two identical geometries, one with a
   high-noise (low-count) subset; assert the fit follows the low-noise points more
   closely than an unweighted fit would (residual asymmetry check).
4. **Posterior std grows away from data:** predictive `sigma` at a grid node far from
   all exit points is ≥ that at a well-surrounded node, and → σ_f in the empty limit.
5. **Coverage mask:** nodes above `cov_tau·σ_f` are NaN in `H` and `sigma`; a
   well-covered node is finite. NaN, never 0.
6. **Honesty invariants:** `scale_assumed is True`; `note` states the assumed scale;
   changing `a` rescales `H` about the detector height but leaves the normalised
   SHAPE (correlation with a fixed reference) unchanged.
7. **Synthetic-hill recovery (exit gate, mirrors the Phase 5b spike):** build a
   synthetic smooth hill from `topography.csv` (TEST-ONLY ground truth), forward it
   through the real geometry to opacities, run the full GP `fit_surface`, and assert
   recovered-vs-truth shape correlation ≥ 0.9 over the covered region AND calibration:
   ≈ 60–90 % of covered truth points fall within ±1σ of the GP mean (uncertainty is
   honest, not just small). A gate that cannot fail for a broken fit tests nothing.
8. **Real-data smoke:** `fit_surface` on `runs/solve` returns finite VE, a covered
   region, and VE ≥ the linear baseline's ~0.40 (or, if slightly lower, a documented
   ruling that better-calibrated uncertainty is the win — decided against real
   numbers, not assumed).

## 7. Risks & rulings

- **VE might not beat 0.40.** GP optimises marginal likelihood, not VE; a smoother,
  better-calibrated fit can trade a little VE for honest uncertainty. Ruling at
  implementation time against real numbers; the calibration gate (7) is the primary
  quality bar, VE the secondary.
- **N³ cost.** N≈3k is fine exactly. If a future campaign grows N past ~8k, add
  inducing points; out of scope now, noted so it is not rediscovered as a bug.
- **Count plumbing.** If rebuilding `position_sky_counts` cleanly proves to need the
  full grid build, the fallback (geometry-only weights) keeps the deliverable working
  while the Poisson term is wired; ship fallback first if needed, heteroscedastic
  behind it.

## 8. Deliverable

Same artifacts, better numbers and honest error bars. `megido hillside` writes the
GP surface; the viewer shows it with no change. The linear surface code is removed
from the active path and lives in git history.
