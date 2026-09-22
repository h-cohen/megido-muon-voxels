# Phase 5 — Data-driven hillside extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the overburden surface `H(x,y)` — the hill top above the cavern — directly from the two-position muon flux, as a fitted height map with an honest uncertainty band, and plot it (2D contour + a 3D "lid" in the viewer) with the detector positions marked.

**Architecture:** A new pure-Python analysis module `megido/hillside.py` turns each Phase-2 sky-opacity pixel into a rock path length, then a 3D surface point at `p + L·d̂`; it fits the single density scale (and one inter-position offset) by requiring the P0 and P1 point clouds to agree on one height field, grids the reconciled cloud into `H(x,y)`, and bootstraps the gauge for the uncertainty band. A CLI `hillside` subcommand writes the artifacts + a contour PNG. The viewer gains detector-position markers and a 3D surface-lid layer, both via a second GL program (the raymarch shader is untouched).

**Tech Stack:** NumPy + SciPy (fit), Matplotlib (dev-only, the contour PNG), the existing WebGL2 viewer (a new small GL program for markers + mesh), Python `pytest`, Node `node:test` + Playwright for the viewer additions.

**Spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` §11 "Data-driven hillside extraction from the flux edge" (this plan implements that open item). The controller worked the physics/numerics design; the load-bearing fit math (Tasks 2–3) is reviewed by the controller personally in addition to the task reviewer.

## The physics (every task assumes this)

- Detector at world position `p` (metres): **P0 `(0,0,0)`, P1 `(2.2,0,0)`**, cavern floor `z=0`. Reconstructed rock starts at `z≈1 m`.
- Phase 2 gives, per position, `λ_p(tx,ty) = BaselineSolution.normalized_opacity(pid)` on the 100×100 sky grid — opacity clipped to ≥0 with **open sky pinned at 0** and thick overburden high. Pixel tangents are `SkyGrid.centers()` → `(tx,ty)`; the sky-frame ray is `d̂ = (tx,ty,1)/√(tx²+ty²+1)` (points up, toward that sky patch).
- Opacity is an in-rock **path length** up to scale: `L_p = a·λ_p + b_p`, with `a = 1/ρ > 0` a single shared density scale, `b_P0 = 0` fixed (gauge anchor) and `b_P1 = Δb` a lone relative offset (absorbs any residual per-position gauge mismatch). The muon travels through rock from the detector until it exits the surface, so the **surface point is `s = p + L_p·d̂`** (`z = L_p/√(tx²+ty²+1)`); an open-sky pixel (`λ≈0`) gives `L≈0`, a point at floor level → no hill there.
- **Parallax pins the scale:** P0 and P1 view overlapping surface patches from 2.2 m apart; the value of `a` that makes their two point clouds agree on one `H(x,y)` in the overlap is the fit. A single 2.2 m baseline constrains `a` (hence absolute height) **weakly** — so lateral shape is crisp, absolute height carries a wide band. This is the same depth limitation as the voxel field; keep it explicit, never a crisp height claim.
- Height-field assumption (approved): the overburden is single-valued `z=H(x,y)` (no overhangs) — valid for a hill.

## Global Constraints

- **No ground-truth in the deliverable.** The fit is data-driven from the muon flux only. `01_data/raw/topography.csv` is **not** read by `hillside.py` or the CLI. It MAY be used only inside a *test* as an independent cross-check if a task explicitly says so (none here do — tests use a self-generated synthetic hill). `calibration_open_sky.csv` is **never** read anywhere.
- **Honesty:** every artifact and plot carries the absolute-height uncertainty. `hill_meta.json` records the fitted `a`, `Δb`, the data residual, the P0-only-vs-P1-only overlap agreement, and a `height_confidence` verdict string. A plot that shows a crisp surface without its band is a defect.
- **Consume Phase-2 artifacts, don't recompute:** read `runs/solve/baseline.npz` via `BaselineSolution.load`; use `normalized_opacity(pid)`, `opacity_image(pid)`, `sol.sky` (`SkyGrid`, `.centers()`, `.n_bins`), positions `"pos0"`/`"pos1"`. Detector world poses come from `SiteConfig` (`cfg.exposures[i].pose.x/y/z`) / `meta.json`'s `detectors`.
- **Viewer constraints unchanged from Phase 4b:** single self-contained `viewer/dist/index.html` built by `megido/viewerbuild.py`; ZERO runtime deps / ZERO network / `file://`; `viewer/src/*.mjs` export-only (no default export, no dynamic import), new modules added to `_MODULE_ORDER` before `app.mjs`; **do NOT modify `FRAGMENT_SRC`/`VERTEX_SRC` or remove `preserveDrawingBuffer`** — the detector markers and the surface lid use a SEPARATE new GL program; keep `initViewer(root)`, `window.__viewerState`, `state.ready`; localStorage in try/catch; Playwright loads via `set_input_files(str(run_dir))` and waits on `state.ready`; the real-data smoke gate `tests/viewer/test_smoke.py` stays green.
- **Matplotlib is dev-only:** add it under `[project.optional-dependencies].dev`, import it lazily inside the CLI plot function; if unavailable, the CLI still writes the `.npy`/`.json` artifacts and skips the PNG with a printed note (never a hard failure).
- **Cache/versioning:** the hillside fit is cheap and re-run on demand; no cache-version key needed, but the CLI must re-read `baseline.npz` fresh each run.
- Python tests: `uv run pytest tests/test_hillside.py -v`. Viewer: `node --test viewer/test/*.mjs`, `uv run pytest tests/viewer -v --browser chromium`.
- Commit to `main`; each message ends with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6`.

## File structure
- `megido/hillside.py` — new: geometry primitives + the fit + uncertainty (Tasks 1–3).
- `tests/test_hillside.py` — new: synthetic-hill recovery + unit gates.
- `megido/cli.py` — modify: add `hillside` subcommand (Task 4).
- `viewer/src/markers.mjs` — new: build detector-marker + surface-mesh vertex data (pure) (Tasks 5–6).
- `viewer/src/app.mjs`, `viewer/shell.html` — modify: second GL program + overlay controls (Tasks 5–6).
- `megido/viewerbuild.py` — modify: `_MODULE_ORDER` += `markers.mjs` (Task 5).

---

### Task 1: Hillside geometry primitives

**Files:**
- Create: `megido/hillside.py`
- Create: `tests/test_hillside.py`

**Interfaces:**
- Produces: `ray_dirs(tan_xy) -> np.ndarray` (N×3 unit vectors from an N×2 array of `(tx,ty)`); `surface_points(p, L, dirs) -> np.ndarray` (N×3, `p + L[:,None]*dirs`); `grid_surface(points, xedges, yedges) -> tuple[np.ndarray, np.ndarray]` returning `(H, count)` where `H[i,j]` is the mean z of points whose (x,y) fall in cell (i,j) (NaN where count==0), `count[i,j]` the number of contributing points. All pure.

- [ ] **Step 1: Write failing tests**

`tests/test_hillside.py`:
```python
import numpy as np
from megido.hillside import ray_dirs, surface_points, grid_surface


def test_ray_dirs_unit_and_upward():
    d = ray_dirs(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, -0.5]]))
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d[0], [0, 0, 1])          # zenith
    assert (d[:, 2] > 0).all()                    # all upward


def test_surface_points_place_along_ray():
    p = np.array([2.2, 0.0, 0.0])
    dirs = ray_dirs(np.array([[0.0, 0.0]]))       # straight up
    L = np.array([5.0])
    s = surface_points(p, L, dirs)
    assert np.allclose(s[0], [2.2, 0.0, 5.0])     # 5 m straight up from P1


def test_grid_surface_bins_mean_z():
    pts = np.array([[0.1, 0.1, 2.0], [0.15, 0.12, 4.0], [0.9, 0.9, 9.0]])
    xe = np.array([0.0, 0.5, 1.0]); ye = np.array([0.0, 0.5, 1.0])
    H, cnt = grid_surface(pts, xe, ye)
    assert cnt[0, 0] == 2 and cnt[1, 1] == 1
    assert np.isclose(H[0, 0], 3.0)               # mean of 2.0, 4.0
    assert np.isclose(H[1, 1], 9.0)
    assert np.isnan(H[0, 1])                        # empty cell -> NaN
```

- [ ] **Step 2: Run, confirm fail** — `uv run pytest tests/test_hillside.py -v` (module missing).

- [ ] **Step 3: Implement the primitives**

`megido/hillside.py`:
```python
"""Data-driven hillside (overburden surface) extraction from the muon flux.

Each Phase-2 sky-opacity pixel is an in-rock path length up to a density
scale; the exit point p + L*d̂ is a point on the hill surface z=H(x,y). The
P0/P1 parallax pins the scale. See docs/superpowers/specs/...-design.md §11.
The absolute height is weakly constrained by a single 2.2 m baseline; the
lateral shape is not. Every result carries that band.
"""
from __future__ import annotations

import numpy as np


def ray_dirs(tan_xy: np.ndarray) -> np.ndarray:
    """N×2 (tx,ty) -> N×3 unit sky-frame ray directions (tx,ty,1)/|.|."""
    tan_xy = np.asarray(tan_xy, dtype=np.float64)
    v = np.column_stack([tan_xy[:, 0], tan_xy[:, 1], np.ones(len(tan_xy))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def surface_points(p: np.ndarray, L: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Exit points p + L*d̂ (N×3) for detector position p and path lengths L."""
    p = np.asarray(p, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    return p[None, :] + L[:, None] * dirs


def grid_surface(points: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    """Bin points into (x,y) cells; return (H mean-z NaN-if-empty, count)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    nx, ny = len(xedges) - 1, len(yedges) - 1
    ix = np.clip(np.digitize(x, xedges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, yedges) - 1, 0, ny - 1)
    inside = (x >= xedges[0]) & (x <= xedges[-1]) & (y >= yedges[0]) & (y <= yedges[-1])
    zsum = np.zeros((nx, ny)); cnt = np.zeros((nx, ny))
    np.add.at(zsum, (ix[inside], iy[inside]), z[inside])
    np.add.at(cnt, (ix[inside], iy[inside]), 1.0)
    H = np.full((nx, ny), np.nan)
    nz = cnt > 0
    H[nz] = zsum[nz] / cnt[nz]
    return H, cnt
```

- [ ] **Step 4: Run, confirm pass.** — `uv run pytest tests/test_hillside.py -v`

- [ ] **Step 5: Commit**
```bash
git add megido/hillside.py tests/test_hillside.py
git commit -m "feat(hillside): ray/surface/grid geometry primitives"
```

---

### Task 2: Parallax scale fit

**Files:**
- Modify: `megido/hillside.py`
- Modify: `tests/test_hillside.py`

**Interfaces:**
- Consumes Task 1 primitives.
- Produces: `overlap_disagreement(a, db, clouds, xedges, yedges) -> float` — RMS of `(H_P0 − H_P1)` over cells both positions populate, given scale `a` and P1 offset `db`, where `clouds` is a dict `{"pos0": (tan_xy, lam), "pos1": (tan_xy, lam, p)}` (see signature below); `fit_scale(images, xedges, yedges, *, a0=..., db0=0.0) -> dict` returning `{"a", "db", "disagreement", "n_overlap"}`. `images` is `{"pos0": {"tan": N×2, "lam": N, "p": (3,)}, "pos1": {...}}`.

- [ ] **Step 1: Failing test — recover a known scale from two synthetic clouds**

Add to `tests/test_hillside.py`:
```python
from megido.hillside import fit_scale

def _synthetic_hill(tx, ty):
    # a smooth bump surface H(x,y), metres
    return 8.0 * np.exp(-((tx) ** 2 + (ty) ** 2) / (2 * 6.0 ** 2))

def _forward(p, a_true, xy_extent=12.0, n=40):
    # emit sky pixels that hit a known hill; return (tan, lam) with lam=L/a_true
    g = np.linspace(-1.2, 1.2, n)
    tx, ty = np.meshgrid(g, g); tan = np.column_stack([tx.ravel(), ty.ravel()])
    d = ray_dirs(tan)
    # solve for L so that (p + L d)_z == H((p+Ld)_xy): fixed-point iterate
    L = np.full(len(tan), 5.0)
    for _ in range(50):
        s = p[None, :] + L[:, None] * d
        Ht = _synthetic_hill(s[:, 0], s[:, 1])
        L = np.clip(Ht / d[:, 2], 0.1, 40.0)
    lam = L / a_true          # opacity = L/a_true (a_true = 1/rho)
    return tan, lam, L

def test_fit_scale_recovers_density_from_parallax():
    a_true = 0.5
    xe = np.linspace(-14.0, 14.0, 57); ye = np.linspace(-14.0, 14.0, 57)
    t0, l0, _ = _forward(np.array([0.0, 0.0, 0.0]), a_true)
    t1, l1, _ = _forward(np.array([2.2, 0.0, 0.0]), a_true)
    images = {"pos0": {"tan": t0, "lam": l0, "p": np.array([0.0, 0, 0])},
              "pos1": {"tan": t1, "lam": l1, "p": np.array([2.2, 0, 0])}}
    out = fit_scale(images, xe, ye, a0=1.0)
    assert abs(out["a"] - a_true) / a_true < 0.15   # scale recovered within 15%
    assert out["n_overlap"] > 20
```

- [ ] **Step 2: Confirm fail, then implement**

Add to `megido/hillside.py`:
```python
from scipy.optimize import minimize


def _cloud_H(img, a, b, xedges, yedges):
    d = ray_dirs(img["tan"])
    L = a * np.asarray(img["lam"], dtype=np.float64) + b
    ok = np.isfinite(L) & (L > 0)
    pts = surface_points(np.asarray(img["p"], float), L[ok], d[ok])
    return grid_surface(pts, xedges, yedges)


def overlap_disagreement(a, db, images, xedges, yedges):
    H0, c0 = _cloud_H(images["pos0"], a, 0.0, xedges, yedges)
    H1, c1 = _cloud_H(images["pos1"], a, db, xedges, yedges)
    both = (c0 > 0) & (c1 > 0)
    if both.sum() == 0:
        return np.inf, 0
    diff = H0[both] - H1[both]
    return float(np.sqrt(np.mean(diff ** 2))), int(both.sum())


def fit_scale(images, xedges, yedges, *, a0=1.0, db0=0.0):
    def obj(params):
        a, db = params
        if a <= 1e-6:
            return 1e6
        rms, n = overlap_disagreement(a, db, images, xedges, yedges)
        if n < 5:
            return 1e6            # too little overlap: reject
        return rms
    res = minimize(obj, [a0, db0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    a, db = res.x
    rms, n = overlap_disagreement(a, db, images, xedges, yedges)
    return {"a": float(a), "db": float(db), "disagreement": rms, "n_overlap": n}
```

- [ ] **Step 3: Confirm pass** — `uv run pytest tests/test_hillside.py -v`
- [ ] **Step 4: Commit**
```bash
git add megido/hillside.py tests/test_hillside.py
git commit -m "feat(hillside): parallax scale fit by P0/P1 overlap consistency"
```

---

### Task 3: Full fit + uncertainty + self-consistency

**Files:**
- Modify: `megido/hillside.py`
- Modify: `tests/test_hillside.py`

**Interfaces:**
- Consumes Tasks 1–2.
- Produces: `fit_hillside(sol, cfg, *, footprint_m=None, cell_m=0.5, n_boot=8, quantiles=None) -> HillsideResult` where `sol` is a `BaselineSolution`, `cfg` a `SiteConfig`. `HillsideResult` (dataclass) fields: `H` (nx×ny mean surface, NaN off-hill), `sigma` (nx×ny 1-σ from the bootstrap, NaN where H is), `count` (nx×ny), `xedges`, `yedges`, `a`, `db`, `disagreement`, `data_residual` (float), `overlap_agreement` (float, P0-only vs P1-only RMS over overlap), `height_confidence` (str verdict), `detectors` (list of `{id,x,y,z}`). It reads `sol.normalized_opacity("pos0"/"pos1")`, `sol.sky.centers()`, and the per-position detector world xy from `cfg`.
- Bootstrap re-runs `normalized_opacity` at varied `transparent_quantile` (the gauge systematic, mirroring Phase 3) and re-fits; `sigma` = per-cell std of `H` across replicas.

- [ ] **Step 1: Failing synthetic-hill recovery gate (the load-bearing physics test)**

Add to `tests/test_hillside.py`:
```python
from megido.hillside import fit_hillside, HillsideResult

class _FakeSky:
    n_bins = 40
    def centers(self):
        g = np.linspace(-1.2, 1.2, 40)
        tx, ty = np.meshgrid(g, g)
        return np.column_stack([tx.ravel(), ty.ravel()])

class _FakeSol:
    """Minimal BaselineSolution stand-in emitting a known hill's opacity."""
    def __init__(self, a_true):
        self.sky = _FakeSky()
        self._a = a_true
        self._lam = {}
        for pid, p in (("pos0", [0, 0, 0]), ("pos1", [2.2, 0, 0])):
            _, lam, _ = _forward(np.array(p, float), a_true, n=40)
            self._lam[pid] = lam
    def normalized_opacity(self, pid, transparent_quantile=0.05):
        return self._lam[pid]

class _FakeCfg:
    class _E:
        def __init__(s, pid, x): s.position, s.pose = pid, type("P", (), {"x": x, "y": 0.0, "z": 0.0})
    exposures = [_E("pos0", 0.0), _E("pos1", 2.2)]

def test_fit_hillside_recovers_lateral_shape():
    a_true = 0.5
    res = fit_hillside(_FakeSol(a_true), _FakeCfg(), footprint_m=14.0, cell_m=0.5, n_boot=4)
    assert isinstance(res, HillsideResult)
    truth = _synthetic_hill(*np.meshgrid(
        0.5 * (res.xedges[:-1] + res.xedges[1:]),
        0.5 * (res.yedges[:-1] + res.yedges[1:]), indexing="ij"))
    m = np.isfinite(res.H)
    # lateral SHAPE recovered (correlation), the honest claim
    corr = np.corrcoef(res.H[m], truth[m])[0, 1]
    assert corr > 0.85
    # the fit reproduces its own data
    assert res.data_residual < 0.2
    # uncertainty is populated where the surface is
    assert np.isfinite(res.sigma[m]).all() and (res.sigma[m] >= 0).all()
    assert "height" in res.height_confidence.lower()
```

- [ ] **Step 2: Confirm fail, then implement `fit_hillside` + `HillsideResult`**

Add to `megido/hillside.py` (dataclass + the driver). Key steps: build `xedges/yedges` from `footprint_m` (± around the detector centroid) at `cell_m`; assemble `images` from `normalized_opacity` + `sky.centers()` + detector xy from `cfg`; `fit_scale`; build the combined-cloud `H`/`count`; compute `data_residual` (predicted `λ` from `H` vs measured, normalized); compute `overlap_agreement` (P0-only H vs P1-only H RMS over overlap ÷ H scale); bootstrap over `quantiles` (default e.g. `[0.02,0.05,0.1,0.2]` cycled/`n_boot`) → per-cell `sigma`; set `height_confidence` from the bootstrap band vs H range (e.g. "absolute height uncertain to ±X m (Y% of relief); lateral shape well constrained"). Include real code — no placeholders — following the signatures above. Provide the combined `H` from BOTH clouds (not just overlap).

- [ ] **Step 3: Confirm pass** — `uv run pytest tests/test_hillside.py -v` (all, incl. the recovery gate).
- [ ] **Step 4: Controller numerics review checkpoint** — the implementer notes in its report that Tasks 2–3 math is ready for the controller's personal review (the controller reviews the fit before Task 4 builds on it).
- [ ] **Step 5: Commit**
```bash
git add megido/hillside.py tests/test_hillside.py
git commit -m "feat(hillside): full surface fit, bootstrap band, self-consistency gates"
```

---

### Task 4: CLI `hillside` + artifacts + contour plot

**Files:**
- Modify: `megido/cli.py`
- Modify: `pyproject.toml` (matplotlib dev dep)
- Create: `tests/test_cli_hillside.py`

**Interfaces:**
- Consumes `fit_hillside`. Produces the `hillside` subcommand: `--config`, `--solve runs/solve`, `--out runs/voxels` (default), writing `hill_surface.npy` (H), `hill_sigma.npy`, `hill_count.npy`, `hill_meta.json` (xedges/yedges, a, db, data_residual, overlap_agreement, height_confidence, detectors, footprint, cell_m), and `hill_surface.png` (contour, if matplotlib present).

- [ ] **Step 1: Failing CLI test**

`tests/test_cli_hillside.py`: build a tiny `BaselineSolution` (or reuse a fixture/the real `runs/solve` if present is not guaranteed — construct a minimal synthetic `baseline.npz` via `BaselineSolution(...).save`, or monkeypatch), run the subcommand's function, assert `hill_surface.npy` + `hill_meta.json` written and meta has the required keys (`a`, `data_residual`, `height_confidence`, `detectors`). (Follow the pattern in the existing `tests/test_cli*.py`.)

- [ ] **Step 2: Confirm fail, then implement**

- `pyproject.toml`: add `matplotlib>=3.7` to `[project.optional-dependencies].dev`.
- `megido/cli.py`: add `_cmd_hillside` mirroring `_cmd_reconstruct`'s structure (load config, `BaselineSolution.load(Path(args.solve)/"baseline.npz")`, call `fit_hillside`, save arrays + `_json_safe`(meta)). The contour PNG in a helper that does `import matplotlib` lazily inside a `try/except ImportError` — on failure print `"matplotlib not installed; skipped hill_surface.png"` and continue. Register the subparser like the others (`set_defaults(func=...)`, `-> int`, `return 0`). Print a short summary (a, data_residual, overlap_agreement, height_confidence).

- [ ] **Step 3: Confirm pass** — `uv run pytest tests/test_cli_hillside.py -v`
- [ ] **Step 4: Run on real data**

Run: `uv run python -m megido.cli hillside --config configs/megido.yaml --solve runs/solve --out runs/voxels`
Expected: writes `runs/voxels/hill_surface.npy` + meta + PNG; prints the fit summary. Confirm `data_residual` is small and `height_confidence` states the band. (This is a real-data checkpoint, not a gate that can pass by construction.)

- [ ] **Step 5: Commit**
```bash
git add megido/cli.py pyproject.toml tests/test_cli_hillside.py uv.lock
git commit -m "feat(hillside): CLI subcommand, artifacts, contour plot"
```

---

### Task 5: Viewer — detector position markers

**Files:**
- Create: `viewer/src/markers.mjs`
- Create: `viewer/test/markers.test.mjs`
- Modify: `viewer/shell.html` (a "Detectors" toggle in the Camera section)
- Modify: `viewer/src/app.mjs` (a second GL program drawing world-space markers)
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER` += `markers.mjs`)
- Create: `tests/viewer/test_markers.py`

**Interfaces:**
- Produces `viewer/src/markers.mjs`: `markerVertices(detectors) -> Float32Array` — from `meta.detectors` (each `{x,y,z}`), build a small 3-axis cross (6 line endpoints × 3 coords) per detector in world space, returned as a flat XYZ line-list; `MARKER_HALF_M` const (cross arm length, e.g. 0.6). Pure.
- app.mjs: a SEPARATE minimal GL program (its own vertex+fragment shaders for flat colored lines, NOT the raymarch shader) that draws the marker line-list with the SAME `viewProj` render() computes, so markers sit correctly in the scene. A `#toggle-detectors` checkbox controls it; draw at the end of render() when enabled and a run is loaded.

- [ ] **Step 1: Failing marker-geometry test**

`viewer/test/markers.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { markerVertices, MARKER_HALF_M } from '../src/markers.mjs';

test('markerVertices builds a 3-axis cross per detector', () => {
  const v = markerVertices([{ x: 2.2, y: 0, z: 0 }]);
  assert.equal(v.length, 6 * 3);                 // 3 axes × 2 endpoints × 3 coords
  // x-arm endpoints straddle the centre on x
  assert.ok(Math.abs(v[0] - (2.2 - MARKER_HALF_M)) < 1e-6);
  assert.ok(Math.abs(v[3] - (2.2 + MARKER_HALF_M)) < 1e-6);
});

test('two detectors -> two crosses', () => {
  const v = markerVertices([{ x: 0, y: 0, z: 0 }, { x: 2.2, y: 0, z: 0 }]);
  assert.equal(v.length, 2 * 6 * 3);
});
```

- [ ] **Step 2: Confirm fail, implement `markers.mjs`**, then the app wiring (second GL program). Add `markers.mjs` to `_MODULE_ORDER`. Store `meta.detectors` into `state.detectors` on load; build the marker buffer once per load. Draw with a tiny program (`gl.LINES`), colored with the accent, guarded by `#toggle-detectors`.

- [ ] **Step 3: Playwright test**

`tests/viewer/test_markers.py`: load the real (or fixture-with-detectors) run; enable `#toggle-detectors`; assert the canvas render changes vs disabled (dataURL diff) and zero console errors. (run_fixture's meta includes no detectors by default — extend the fixture or point this test at runs/voxels which has them; use the same REAL_RUN pattern as test_smoke if needed.)

- [ ] **Step 4: Run + commit**
```bash
git add viewer/src/markers.mjs viewer/test/markers.test.mjs viewer/shell.html viewer/src/app.mjs megido/viewerbuild.py tests/viewer/test_markers.py
git commit -m "feat(viewer): detector position markers"
```

---

### Task 6: Viewer — 3D hillside surface "lid" + top-down contour

**Files:**
- Modify: `viewer/src/markers.mjs` (add surface-mesh builder)
- Modify: `viewer/test/markers.test.mjs`
- Modify: `viewer/shell.html` (a "Hillside surface" toggle)
- Modify: `viewer/src/app.mjs` (load `hill_surface.npy`; draw the mesh with the second GL program)
- Modify: `tests/viewer/test_smoke.py` (exercise the toggle when hill_surface is present)

**Interfaces:**
- Consumes: `hill_surface.npy` + `hill_meta.json` in the run dir (Task 4 output). The loader parses `hill_surface.npy` via the existing `parseNpy`, and reads xedges/yedges + footprint from `hill_meta.json` (add these to the run's meta load, OR fetch `hill_meta.json` from the picked directory alongside the others).
- Produces `viewer/src/markers.mjs`: `surfaceMesh(H, xedges, yedges) -> {positions: Float32Array, indices: Uint16Array|Uint32Array}` — a triangulated heightmesh over the (x,y) grid, skipping NaN cells, z = H (world metres). Pure.
- app.mjs: draw the mesh (semi-transparent, accent-tinted, backface-friendly) with the second GL program used by Task 5, behind/over the voxels as a "lid"; `#toggle-hillside` controls it. If no `hill_surface.npy` in the loaded run, the toggle is disabled/hidden.

- [ ] **Step 1: Failing mesh test**

Add to `viewer/test/markers.test.mjs`:
```js
import { surfaceMesh } from '../src/markers.mjs';
test('surfaceMesh triangulates a grid and skips NaN cells', () => {
  const H = new Float32Array([1, 2, 3, 4]);          // 2×2 grid of heights
  const xe = [0, 1, 2], ye = [0, 1, 2];
  const m = surfaceMesh(H, xe, ye, 2, 2);            // (H, xedges, yedges, nx, ny)
  assert.ok(m.positions.length === 4 * 3);           // 4 vertices, xyz
  assert.ok(m.indices.length === 6);                 // one quad -> 2 triangles
});
```
(Define the exact `surfaceMesh` signature in the implementation — pass nx,ny explicitly or derive from edges; keep it consistent with the test.)

- [ ] **Step 2: Confirm fail, implement `surfaceMesh` + the app loader/draw.** Load `hill_surface.npy`/`hill_meta.json` from the picked directory (extend `loadRun`'s file map to look for them; absent → hillside toggle hidden). Draw the mesh semi-transparent with the second GL program. Wire `#toggle-hillside`.

- [ ] **Step 3: Extend the real-data smoke test**

In `tests/viewer/test_smoke.py`: after loading `runs/voxels` (which now has `hill_surface.npy` from Task 4), assert `#toggle-hillside` is present+enabled, toggle it, assert the render changes and console stays clean. Keep every existing hard assertion (banner "depth NOT resolved", non-blank, console_errors==[]).

- [ ] **Step 4: Run + commit**

Run: `node --test viewer/test/*.mjs && uv run pytest tests/viewer -v --browser chromium && uv run pytest -q`
```bash
git add viewer/src/markers.mjs viewer/test/markers.test.mjs viewer/shell.html viewer/src/app.mjs tests/viewer/test_smoke.py
git commit -m "feat(viewer): 3D hillside surface lid layer"
```
