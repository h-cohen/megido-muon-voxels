# Phase 5b — Hillside surface fit (regularized inversion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the hill's overburden surface H(x,y) as the PRIMARY deliverable — a proper regularized least-squares fit of a smooth height field to all muon rays jointly (P0+T20+P1) — with honest coverage, an assumed-scale caveat, and a "fraction of opacity variance explained" metric, displayed as a real surface in the viewer.

**Architecture:** Each ray with measured normalized opacity λ places a surface point at `exit = o + a·λ·d̂` (a = 1/ρ, an assumed scale). Fitting a smooth grid H(x,y) so that bilinear-interpolation of H at each exit's (x,y) equals its z is **LINEAR in the grid values** → a sparse, well-conditioned regularized least-squares (this is the key: the earlier ray-march forward model was a non-differentiable step function that stalled the optimizer; the exit-point form is linear and robust). A feasibility spike already confirmed this geometry recovers a known synthetic hill at shape-correlation 0.97. Add gauge-bootstrap uncertainty, a coverage mask, a variance-explained honesty metric, a CLI, and a 3D surface layer in the viewer.

**Tech Stack:** NumPy + SciPy sparse least-squares (`scipy.sparse` + `lsqr`/normal equations), Matplotlib (dev, contour PNG), the WebGL2 viewer (surface mesh via the existing second GL line/tri program), pytest, Node `node:test` + Playwright.

**Spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` §11 (hillside from the flux). Supersedes the naive point-cloud `fit_hillside` and makes the surface (not the silhouette) primary. The load-bearing inversion (Tasks 1–2) gets an ADDITIONAL controller (Opus) numerics review beyond the Sonnet task review.

## The physics / math (every task assumes this)

- Detector positions (world m): distinct viewpoints from `baseline.position_ids(cfg)` → pos0 at (0,0,0) [P0+T20a+T20b share it], pos1 at (2.2,0,0). Tilt gives extra sky coverage at pos0, not a new location.
- Per position: λ = `sol.normalized_opacity(pid)` on the 100×100 sky grid (open-sky pinned to 0, hill high); pixel tangents from `sol.sky.centers` (1-D per-axis; build (sx,sy) via `meshgrid(centers,centers,indexing="ij")` matching the `i*n_bins+j` flat order). Ray dir `d̂ = (sx,sy,1)/|.|`.
- **Exit point** of ray i: `exit_i = o_i + a·λ_i·d̂_i`, using only finite λ>0 pixels. `a` = assumed inverse density scale (convention; default `a=1.0` meaning 1 opacity-unit = 1 m). exit_xy is where the ray leaves the rock; exit_z is the surface height there.
- **Inversion:** unknown H on a coarse grid (cell size `cell_m`, e.g. 1.0 m) over the exit-point footprint. For each ray, bilinear-interpolate H at exit_xy → predicted z; residual `w_i·(H(exit_xy_i) − exit_z_i)`. Add 2nd-difference smoothness `μ·∇²H`. Solve the sparse linear least-squares for the grid values (interpolation weights are LINEAR in H). This is `min_H ||W(B H − z)||² + μ²||L H||²`, B the bilinear-sample matrix, L the Laplacian.
- **Coverage mask:** a cell is constrained only where enough exit points land near it (data support ≥ a threshold); elsewhere H is NaN (never fabricate beyond coverage — the spike showed exits reach ~12 m radius).
- **Honesty metrics:** `variance_explained` = 1 − Σw(H(exit)−exit_z)² / Σw(exit_z − z̄)² (how much of the exit-height scatter the smooth surface captures — low ⇒ real opacity has non-geometric structure/noise). `scale_assumed` flag + note (absolute height ∝ 1/ρ, not from the 2.2 m parallax). Coverage fraction + radius.
- **Uncertainty:** bootstrap over `normalized_opacity` transparent_quantile (the gauge systematic) → refit H per replica → per-cell σ.

## Global Constraints

- **Honesty (non-negotiable):** every artifact/plot states the assumed scale (absolute height ρ-dependent, not data-fit), the coverage mask (no surface beyond data support), and the variance-explained fraction. A crisp surface without these is a defect. Keep the voxel viewer's depth-NOT-resolved banner intact.
- **No ground truth in the deliverable:** `topography.csv` is NOT read by any megido/ code or the CLI (it MAY appear only inside a *test* as an independent cross-check — but the synthetic gate here self-generates its hill; no task reads topography.csv). `calibration_open_sky.csv` NEVER read.
- **Consume Phase-2 artifacts:** `BaselineSolution.load(runs/solve/baseline.npz)`, `normalized_opacity`, `sol.sky`, `baseline.position_ids(cfg)`; detector poses from `cfg.exposures[*].pose`.
- **Linear inversion, not ray-march:** the forward model MUST be the linear exit-point form (bilinear-sample matrix), not an iterative ray-march (which is non-differentiable and stalled the earlier attempt). Use scipy sparse.
- **Viewer contract (Phase 4b/5):** single self-contained `viewer/dist/index.html` from `megido/viewerbuild.py`; ZERO deps/network/`file://`; do NOT modify the raymarch `FRAGMENT_SRC`/`VERTEX_SRC` or remove `preserveDrawingBuffer`; the surface mesh uses the SEPARATE GL program added in Phase 5 (markers/silhouette) — extend it for triangles if needed, do not touch the raymarch shader; keep `initViewer(root)`/`window.__viewerState`/`state.ready`; `viewer/src/*.mjs` export-only; localStorage try/catch; Playwright loads via `set_input_files(str(run_dir))`, waits on `state.ready`; the real-data smoke gate stays green.
- **Matplotlib** dev-only, lazy import; JSON/npy artifacts always written even if matplotlib absent.
- Tests: `uv run pytest tests/test_hillside_surface.py -v`; viewer `node --test viewer/test/*.mjs`, `uv run pytest tests/viewer -v --browser chromium`.
- Commit to `main`; messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6`.

## File structure
- `megido/hillside_surface.py` — new: exit points, sparse bilinear-sample matrix, Laplacian, regularized solve, coverage mask, variance_explained, bootstrap, `fit_surface(sol,cfg,...) -> SurfaceResult` (Tasks 1–2).
- `tests/test_hillside_surface.py` — synthetic-hill recovery gate (through real geometry) + unit tests.
- `megido/cli.py` — extend `hillside` to also run fit_surface and write hill_surface.npy + sigma + meta + contour PNG (Task 3).
- `viewer/src/surfacemesh.mjs` (or extend markers.mjs) — triangulated heightmesh vertices from H + coverage mask (Task 4).
- `viewer/src/app.mjs`, `viewer/shell.html` — surface layer toggle + draw (Task 4).
- Existing `megido/hillside.py` (naive, superseded) and `megido/silhouette.py` stay.

---

### Task 1: Surface inversion core (linear regularized fit)

**Files:** Create `megido/hillside_surface.py`, `tests/test_hillside_surface.py`.

**Interfaces (produce):**
- `exit_points(sol, cfg, a) -> (pts N×3, pos_index N)` — all rays' exit points `o + a·λ·d̂` over finite λ>0, plus which position each came from. Uses `position_ids` grouping + `sol.sky` centers.
- `bilinear_matrix(xy N×2, gx, gy) -> (scipy.sparse M (N×G), support G)` — each row has ≤4 weights (bilinear) into the flattened (len(gx)×len(gy)) grid; `support` = column-sum of weights (data support per cell). Points outside [gx,gy] are dropped (row all-zero / excluded).
- `laplacian_matrix(nx, ny) -> scipy.sparse` — 2nd-difference operator over the grid (for smoothness).
- `solve_surface(B, z, w, lap, mu) -> h` — solve `min ||W(Bh−z)||² + mu²||lap h||²` (normal equations or lsqr). Returns flat grid heights.
- `variance_explained(B, z, w, h) -> float`.

- [ ] **Step 1: failing tests**

`tests/test_hillside_surface.py` (transcribe; key gates):
```python
import numpy as np
from megido.hillside_surface import bilinear_matrix, laplacian_matrix, solve_surface, variance_explained

def test_bilinear_matrix_partition_of_unity():
    gx=np.linspace(0,10,11); gy=np.linspace(0,10,11)
    B,support=bilinear_matrix(np.array([[2.5,3.5],[0.0,0.0]]),gx,gy)
    assert abs(B.sum(axis=1)[0,0]-1.0)<1e-9      # weights sum to 1
    assert B.shape[1]==121

def test_solve_recovers_a_plane():
    gx=np.linspace(0,10,11); gy=np.linspace(0,10,11)
    GX,GY=np.meshgrid(gx,gy,indexing='ij'); Htrue=(2*GX+3*GY+1).ravel()
    # sample the plane at random points
    rng=np.random.default_rng(0); xy=rng.uniform(0.5,9.5,(500,2))
    z=2*xy[:,0]+3*xy[:,1]+1
    B,support=bilinear_matrix(xy,gx,gy); lap=laplacian_matrix(11,11)
    h=solve_surface(B,z,np.ones(len(z)),lap,mu=0.01)
    m=support>0.5
    assert np.corrcoef(h[m],Htrue[m])[0,1]>0.999   # plane recovered where supported
    assert variance_explained(B,z,np.ones(len(z)),h)>0.99

def test_laplacian_penalises_curvature_not_planes():
    lap=laplacian_matrix(6,6); GX,GY=np.meshgrid(np.arange(6),np.arange(6),indexing='ij')
    plane=(3*GX+2*GY).ravel().astype(float)
    assert np.allclose(lap@plane,0,atol=1e-9)      # a plane has zero 2nd-difference (interior)
```

- [ ] **Step 2: implement** `megido/hillside_surface.py` — exit_points, bilinear_matrix (scipy.sparse.coo/csr, ≤4 weights/row, drop out-of-grid), laplacian_matrix (interior 2nd differences in x and y), solve_surface (build `(BᵀW B + mu²·LᵀL) h = BᵀW z`, solve with `scipy.sparse.linalg.spsolve` or `lsqr`), variance_explained. Real code, no placeholders.
- [ ] **Step 3: run tests green.** `uv run pytest tests/test_hillside_surface.py -v`
- [ ] **Step 4: commit** (`feat(hillside-surface): linear regularized height-field inversion core`).

---

### Task 2: fit_surface driver + coverage + variance + bootstrap + synthetic gate

**Files:** Modify `megido/hillside_surface.py`, `tests/test_hillside_surface.py`.

**Interfaces:** `SurfaceResult` dataclass: `H` (nx×ny, NaN off-coverage), `sigma` (nx×ny bootstrap σ, NaN off-coverage), `support` (nx×ny), `xedges/yedges` (or gx/gy + cell_m), `a` (assumed scale), `variance_explained` (float), `coverage_frac`, `coverage_radius_m`, `n_rays`, `detectors`, `scale_assumed` (True), `note` (honesty string). `fit_surface(sol, cfg, *, a=1.0, cell_m=1.0, mu=0.3, min_support=0.5, n_boot=8, quantiles=None) -> SurfaceResult`: builds exit points at scale a; grid spanning the exit footprint (robust percentile bounds, e.g. 2–98% of exit xy) at cell_m; bilinear + laplacian; solve; mask cells with support<min_support to NaN; variance_explained; bootstrap over transparent_quantile → per-cell σ; note states assumed scale + coverage + variance_explained.

- [ ] **Step 1: failing synthetic-hill recovery gate (load-bearing)** — a fake sol whose `normalized_opacity` is generated from a KNOWN hill H_true through the REAL sky sampling (use `megido.sky.make_sky_grid`) at the real two positions (0,0,0)&(2.2,0,0); with the correct scale `a`, assert `fit_surface` recovers shape corr(H,H_true) > 0.9 over covered cells, variance_explained > 0.9, crown location within ~a couple cells, and that `scale_assumed` is True and the note mentions the assumed scale. (The spike measured 0.97; gate at 0.9.) Also assert a partial coverage mask exists (not all cells filled). Add a bootstrap-nonzero test (fake sol varying with quantile → σ>0 somewhere).
- [ ] **Step 2: implement `fit_surface` + `SurfaceResult`.** Real code.
- [ ] **Step 3: tests green.** Controller numerics-review checkpoint: report the recovery numbers.
- [ ] **Step 4: real-data run** (report output, not a gate): `fit_surface` on runs/solve → print variance_explained, coverage_frac, coverage_radius, H relief. (Honest real-data number — expected variance_explained well below the synthetic 0.9 because real opacity has non-geometric structure; that's the point.)
- [ ] **Step 5: commit** (`feat(hillside-surface): fit_surface driver, coverage, variance-explained, bootstrap`).

---

### Task 3: CLI + artifacts + contour plot

**Files:** Modify `megido/cli.py`; `tests/test_cli_hillside_surface.py` (new).

**Interfaces:** extend the `hillside` subcommand (keep silhouette output) to ALSO run `fit_surface` and write to --out: `hill_surface.npy` (H), `hill_surface_sigma.npy`, `hill_surface_meta.json` (gx/gy or origin+cell_m, a, variance_explained, coverage_frac/radius, scale_assumed, note, detectors — NaN→null via the existing `_json_nan_to_null`), and `hill_surface.png` (a filled contour of H over the covered footprint with detector markers, a colorbar, and a caption stating assumed-scale + variance_explained + coverage; matplotlib lazy, skip PNG if absent). Print a summary.

- [ ] Steps: failing CLI test (artifacts + meta keys incl variance_explained, scale_assumed, note); implement; run real command `uv run python -m megido.cli hillside --config configs/megido.yaml --solve runs/solve --out runs/voxels` (writes both silhouette AND surface artifacts); confirm hill_surface.npy + hill_surface_meta.json (+png) written and valid JSON; commit.

---

### Task 4: Viewer — hillside surface layer

**Files:** Create `viewer/src/surfacemesh.mjs`, `viewer/test/surfacemesh.test.mjs`; modify `viewer/src/app.mjs`, `viewer/shell.html`, `megido/viewerbuild.py` (_MODULE_ORDER).

**Interfaces:** `surfaceMesh(H, sigma, gx, gy) -> {positions:Float32Array, indices:Uint32Array}` — triangulated heightmesh over grid cells where H is finite (skip NaN cells; world coords x,y from gx/gy, z=H). Export-only. app.mjs: load `hill_surface.npy` + `hill_surface_meta.json` from the run dir (valid JSON, null off-coverage → NaN in the Float32 array; skip). Draw the mesh with the second GL program (extend to triangles: a tri fill and/or wireframe), semi-transparent, tinted; a `#toggle-hill-surface` checkbox in the dock; disabled/hidden if the artifact is absent. Show a small label with the assumed-scale + variance_explained caveat. This is the PRIMARY hillside display — make it prominent (its own toggle, on by default when present is acceptable).

- [ ] Steps: failing surfaceMesh test (triangulation, NaN cells skipped); implement mesh + GL draw + toggle + artifact load; add surfacemesh.mjs to _MODULE_ORDER; extend the real-data smoke test to toggle the surface (assert render change + zero console errors, all prior honesty asserts intact); visual self-check screenshot (the hill surface renders over the covered footprint with detectors); commit.
