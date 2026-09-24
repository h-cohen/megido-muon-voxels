# Surface Ray-Check and Viewer Upgrades — Design

**Status:** proposed
**Date:** 2026-09-24
**Builds on:** `2026-09-23-hillside-upper-envelope-design.md` (the GP upper-envelope
surface) and the Phase 4 viewer.

## 1. Motivation

Two gaps remain after the first major revision:

1. **The surface has no ray-space goodness-of-fit.** Its headline VE (≈81 %) is
   computed on the per-cell upper-envelope targets, not on the measurements. No
   test currently asks the physical question: *if the fitted surface were a
   uniform solid, would it reproduce the opacity each ray actually measured?*
2. **The viewer under-reports what the analysis knows.** The surface's real
   posterior σ is written to disk but never loaded (the mesh is one flat colour);
   the volume uses NEAREST sampling with no lighting, which reads as blocks with
   no depth cues; and density above the fitted surface is drawn as though it were
   rock.

### Rejected alternative (recorded)

A **hard "zero density above H" constraint inside the voxel inversion** was
considered and rejected. `H` rides on the assumed scale `a`: at `a = 8` its
median (12.1 m) sits at the voxel grid top (12.0 m) and the constraint would zero
≈10 % of the mass; at `a = 4` it would cut the grid in half; at `a = 16` it does
nothing. The voxels' vertical structure would then be set by an assumption, not by
data — a prior that shapes depth, which CLAUDE.md "Honest limits" forbids. The
same visual benefit is delivered honestly as a **display-only clip** (§4.3).

## 2. Scope

**In scope**
- A. `megido/hillside_check.py`: ray-trace every measured direction through the
  fitted surface treated as a uniform solid of density `1/a`, compare predicted
  with measured opacity, and report ray-space VE, RMS, count, and per-position
  residual sky maps. Wired into `hillside` CLI (meta fields + `hill_residual.png`).
- B1. Viewer: load `hill_surface_sigma.npy`; colour the surface mesh by σ with a
  legend; caption shows both per-cell and per-ray VE.
- B2. Viewer: trilinear sampling (hardware `LINEAR` when `OES_texture_float_linear`
  exists, manual 8-tap fallback otherwise) and headlight gradient shading, each
  with a toggle (both default on).
- B3. Viewer: display-only "clip volume above surface" toggle (default off),
  mirrored in hover picking.

**Out of scope**
- Any change to the voxel inversion, its cache key, or `INVERSION_VERSION`.
- Isosurface / MIP render modes.
- Using the residual map to *fit* anything (it is a diagnostic only).
- `topography.csv` in the deliverable path (test-only, unchanged rule).

## 3. A — Ray-space surface check

### 3.1 Model
A detector at origin `o` looks along unit `d̂ = normalize(sx, sy, 1)`. Under the
surface model, rock fills `z < H(x, y)` with uniform density `ρ = 1/a`. The ray
leaves rock at the first `t > 0` with `o_z + t·d_z ≥ H(o_xy + t·d_xy)`; that exit
distance `t*` is the in-rock path, so the predicted opacity is `λ_pred = t*/a`.
Residual `r = λ_meas − λ_pred` with `λ_meas = normalized_opacity(pid)`.

Because `H` scales ≈ linearly with `a` about each detector (tested in the
upper-envelope work), `t*` scales by `a` and `λ_pred` is scale-free **in the
ideal case** — on the synthetic hill the prediction correlates 0.98 between
`a = 1` and `a = 2`. **On real data it is not**: the fitted surface does not scale
cleanly with `a` (per-cell ray density falls as `1/a²` at fixed `cell_m`, so
coverage and shape shift), and the ray VE moves strongly with `a` (fast fit:
0.02 / −0.35 / −0.73 at `a = 4 / 8 / 16`). **Ray VE is never quoted without its
`a` and fit settings.**

**Amendment (2026-09-24, gauge).** Phase 2 pins each position's opacity level
independently and never measures it, so scoring the absolute `t*/a` against the
gauge-pinned `λ_meas` punishes an unmeasurable constant (on real data the raw VE
was −60%, mostly pos1 sitting ~0.78 below the prediction). The check is therefore
**gauge-invariant per position**: each position's additive offset
`c_p = mean(λ_meas − λ_pred)` is fitted, reported, and removed before scoring —
the role `c_p` plays in the voxel solve. The offset also absorbs any constant
per-position model bias (e.g. from the upper envelope). No multiplicative scale
is absorbed; slope misfit stays in the residual.

### 3.2 Ray tracing
`H` is sampled bilinearly on its node grid `(gx, gy)`; a sample is **NaN** if the
point lies outside the grid or any bilinear corner is NaN. March `t` from 0 in
steps of `step = min(Δgx, Δgy)/4` up to `t_max`; the first step where the ray is
at or above the surface brackets `t*`, refined by 30 bisection iterations. If the
ray reaches a NaN surface sample (unconstrained or off-grid) before crossing, or
never crosses by `t_max`, `t* = NaN`. If the detector is not below the surface at
`t = 0`, `t* = NaN`. **Unpredicted rays are NaN, never 0**, and are excluded from
every metric.

### 3.3 Outputs
`surface_ray_check(sol, cfg, result, *, a) -> RayCheck` (frozen dataclass):
- `residual: dict[pid, (n_bins, n_bins)]` — NaN where unmeasured or unpredicted.
- `ray_ve` — gauge-invariant: `1 − Σ_p Σ(r − c_p)² / Σ_p Σ(λ_meas − mean_p λ_meas)²`
  over checked rays (each position gets one constant in the model and in the null).
- `ray_ve_raw` — the gauge-naive `1 − Σr²/Σ(λ_meas − mean)²`; secondary only.
- `offsets` — `{pid: c_p}`; `per_position` — `{pid: {n, corr, ve}}` after `c_p`.
- `residual` maps are shown **after** removing `c_p`.
- `ray_rms` — `sqrt(mean r²)`.
- `n_checked` — rays with finite measurement and prediction.
- `a`.

Directions checked: every finite `normalized_opacity` direction (including the
gauge-zero ones — the check should see where the surface fails, not just where it
was fit).

### 3.4 CLI
`hillside` runs the check after the surface fit, prints
`ray check  VE=..% (per ray, gauge-invariant, a=..)  RMS(after offset)=..  over N rays`
plus one line per position (`n`, shape `r`, VE, offset), and adds `ray_ve`,
`ray_ve_raw`, `ray_offsets`, `ray_per_position`, `ray_check_note`, `ray_rms`,
`n_rays_checked` to `hill_surface_meta.json`, and writes `hill_residual.png`
(one panel per position, diverging colormap centred on 0 with symmetric limits,
axes in sky tangent; skipped with a note if matplotlib is absent).

## 4. B — Viewer

### 4.1 σ-coloured surface (B1)
- Load `hill_surface_sigma.npy` when present (same `(nx, ny)` flattening as `H`).
- Pure helper `surfaceVertexColors(sigma, lo, hi)` in `surfacemesh.mjs` → per-vertex
  RGB `Float32Array` on a single-hue sequential ramp: pale warm (low σ, confident)
  → deep burnt orange (high σ). NaN σ → the mid colour (such vertices are never
  referenced by a finite triangle anyway).
- `lo, hi` = 5th/95th percentile of finite σ (robust to outliers), shown in a
  legend: gradient bar + `σ lo – hi m (posterior std, assumed scale)`.
- New mesh shader pair with a per-vertex colour attribute; translucency unchanged.
- Toggle "Colour by uncertainty" (default on when σ loaded; off → flat colour).
- Display smoothing moves geometry only; σ is not smoothed (and the legend says
  "raw posterior σ").
- Caption: `Hillside surface — assumed scale, 81% VE per cell · NN% per ray`
  (per-ray part shown only when `ray_ve` is in the meta).

### 4.2 Trilinear + shading (B2)
- At init request `OES_texture_float_linear`. If present, volume textures use
  `LINEAR`; if absent, keep `NEAREST` and set `uManualTrilinear` so the shader does
  an 8-tap trilinear with `texelFetch`.
- Toggle "Smooth sampling" (default on): off → `NEAREST`, manual path off
  (today's look).
- Toggle "Shading" (default on): at each contributing sample, gradient by central
  differences one voxel apart per axis; headlight Lambert
  `shade = 0.35 + 0.65·|dot(n, −dir)|`, applied only where `|∇| > ε`; `c.rgb *= shade`.
- Sampling/shading change display only; hover readout stays nearest-voxel.

### 4.3 Clip volume above surface (B3)
- The currently displayed (display-smoothed) `H` is uploaded as a 2D R32F texture;
  NaN → sentinel `1e6` (never clips unknown ground).
- Shader: when `uSurfClip` is on and the sample's `xy` lies inside the surface
  grid, a manual bilinear surface height is computed; the sample is skipped if
  `z > H`. Outside the grid or at a sentinel corner → not clipped.
- Pure JS twin `surfaceHeightAt(H, gx, gy, x, y)` (NaN outside / NaN corner) used by
  hover picking so a clipped voxel is never reported.
- Toggle "Clip volume above surface" (default **off**, disabled until a surface
  loads) with caption "display only — hides density the surface model calls air".

## 5. Testing

TDD. Physical relationships, gates that can fail.

**A (Python)**
1. Flat slab `H ≡ h0`, detector at `z = 0`: vertical `t* = h0`; oblique
   `t* = h0/d_z` (within 1e-6 relative).
2. Known synthetic hill (the upper-envelope test fixture's Gaussian), `H` = the
   true hill sampled on a fine grid, `a = 1`, fake sol whose opacity is the true
   path length: `ray_ve > 0.99` — the tracer reproduces exact path lengths.
3. NaN discipline: a surface with a NaN hole → rays exiting through it are NaN in
   `residual`, not 0, and excluded from `n_checked`; a ray that leaves the grid →
   NaN.
4. Scale near-invariance: the fitted surface on the synthetic fixture at `a = 1`
   vs `a = 2` gives residual maps with correlation > 0.9 over rays finite in both.
5. Real-data smoke (skip if `runs/solve` absent): finite `ray_ve`, `n_checked > 0`.
6. CLI: meta has `ray_ve`, `ray_rms`, `n_rays_checked`; PNG written when
   matplotlib is available.

**B (viewer)**
7. Node: `surfaceVertexColors` maps `lo` → ramp start, `hi` → ramp end, clamps
   outside; `surfaceHeightAt` matches hand bilinear, NaN outside grid and at NaN
   corner.
8. Playwright (synthetic fixture extended with an optional small surface + σ +
   meta): colour toggle changes the render and the legend shows the σ range;
   smooth and shading toggles each change the render; clip toggle reduces the
   drawn volume pixel count; zero console errors throughout.
9. Real-run smoke extended: all new toggles operable on `runs/voxels`.

## 6. Risks & rulings

- **Ray-space VE may be low** (the upper envelope deliberately ignores low rays,
  and the gauge makes ~5 % of directions zero). That is the honest number and is
  reported as such; it is *not* a gate threshold on real data.
- **Manual trilinear is ~8× the fetches**; only used when the extension is
  missing. If it makes the fallback unusably slow, halve steps on that path and
  note it.
- **Shading can hide faint structure**; it is a toggle, and the default can flip
  to off if the real-run screenshot reads worse.
