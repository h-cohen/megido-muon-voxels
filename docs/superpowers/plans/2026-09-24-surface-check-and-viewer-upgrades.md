# Surface Ray-Check and Viewer Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a ray-space goodness-of-fit check for the fitted hillside surface, and upgrade the viewer with σ-coloured surface, trilinear + shaded volume rendering, and a display-only clip-above-surface toggle.

**Architecture:** A new pure-numpy module `megido/hillside_check.py` ray-traces each measured direction through the fitted surface treated as a uniform solid (density `1/a`) and compares predicted with measured opacity; the `hillside` CLI reports it. Viewer changes are display-only: new pure helpers in `viewer/src/surfacemesh.mjs` (unit-tested), shader and wiring changes in `viewer/src/app.mjs`, new controls in `viewer/shell.html`.

**Tech Stack:** Python (numpy, matplotlib optional), JavaScript ES modules, WebGL2 GLSL ES 3.00, node:test, pytest-playwright.

**Spec:** `docs/superpowers/specs/2026-09-24-surface-check-and-viewer-upgrades-design.md`

## Global Constraints

- No change to the voxel inversion, `INVERSION_VERSION`, or any reconstruction cache key.
- `np.nan` means "not constrained": unpredicted rays are NaN, never 0, and excluded from every metric.
- Absolute height/scale is ASSUMED via `a`; nothing here may claim depth or scale is measured.
- `topography.csv` is TEST-ONLY; never read by `megido/` or the CLI.
- Viewer stays zero-dependency; all new rendering is display-only (no data is changed).
- `.npy` volumes stay numpy C-order; any new 3D layer upload goes through `makeVolumeTexture` (which calls `reorderForTexture`).
- TDD: red before green. Never loosen a gate to pass; if a gate cannot pass, stop and report the measured numbers.
- Implementers run ONLY their task's test files, not the full suite.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6
  ```

---

### Task 1: Ray-space surface check module

**Files:**
- Create: `megido/hillside_check.py`
- Test: `tests/test_hillside_check.py`

**Interfaces:**
- Consumes: `megido.hillside_surface._positions(cfg)`; a `SurfaceResult`-like object with `.H (nx,ny)`, `.gx`, `.gy`, `.a`; a sol with `.sky.centers` and `.normalized_opacity(pid)`.
- Produces:
  - `_bilinear(H, gx, gy, x, y) -> np.ndarray` — NaN outside the grid or if any corner is NaN.
  - `ray_exit_distance(origin, dirs, H, gx, gy, *, t_max=None, step=None, n_bisect=30) -> np.ndarray` — `(N,)` in-rock path length per unit direction, NaN if unpredictable.
  - `RayCheck` frozen dataclass: `residual: dict[str, np.ndarray]`, `predicted: dict[str, np.ndarray]` (both `(n_bins, n_bins)`), `ray_ve: float`, `ray_rms: float`, `n_checked: int`, `a: float`.
  - `surface_ray_check(sol, cfg, result, *, a=None) -> RayCheck`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hillside_check.py
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from megido.hillside_check import _bilinear, ray_exit_distance, surface_ray_check


def _flat(h0, lo=-30.0, hi=30.0, step=1.0):
    g = np.arange(lo, hi + step, step)
    return np.full((g.size, g.size), h0), g, g.copy()


def test_bilinear_is_exact_on_a_plane_and_nan_outside():
    g = np.arange(0.0, 5.0, 1.0)
    X, Y = np.meshgrid(g, g, indexing="ij")
    H = 2 * X + 3 * Y + 1
    v = _bilinear(H, g, g, np.array([1.5, 2.25]), np.array([0.5, 3.75]))
    np.testing.assert_allclose(v, [2 * 1.5 + 3 * 0.5 + 1, 2 * 2.25 + 3 * 3.75 + 1])
    assert np.isnan(_bilinear(H, g, g, np.array([-0.1]), np.array([1.0]))[0])
    H2 = H.copy(); H2[2, 2] = np.nan
    assert np.isnan(_bilinear(H2, g, g, np.array([1.5]), np.array([1.5]))[0])


def test_flat_slab_path_lengths_are_exact():
    H, gx, gy = _flat(8.0)
    dirs = np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 1.0], [-0.5, 0.4, 1.0]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    t = ray_exit_distance((0.0, 0.0, 0.0), dirs, H, gx, gy)
    np.testing.assert_allclose(t, 8.0 / dirs[:, 2], rtol=1e-6)


def test_ray_through_a_nan_hole_or_off_grid_is_nan_not_zero():
    up = np.array([[0.0, 0.0, 1.0]])
    far = np.array([[5.0, 0.0, 1.0]]) / np.sqrt(26.0)   # leaves the 10 m grid before z=8
    # a hole a few metres off-axis: an oblique ray heading into it is unpredictable
    H, gx, gy = _flat(8.0, lo=-10.0, hi=10.0)
    k = int(np.argmin(np.abs(gx - 4.0)))
    H[k - 1:k + 2, :] = np.nan                      # NaN band around x = 4
    into_hole = np.array([[0.5, 0.0, 1.0]]) / np.sqrt(1.25)   # exits near x = 4 at z = 8
    assert np.isnan(ray_exit_distance((0.0, 0.0, 0.0), into_hole, H, gx, gy)[0])
    assert np.isfinite(ray_exit_distance((0.0, 0.0, 0.0), up, H, gx, gy)[0])
    # detector under an unknown patch -> NaN for every ray
    H2, gx2, gy2 = _flat(8.0, lo=-10.0, hi=10.0)
    i = int(np.argmin(np.abs(gx2 - 0.0)))
    H2[i - 1:i + 2, i - 1:i + 2] = np.nan
    assert np.isnan(ray_exit_distance((0.0, 0.0, 0.0), up, H2, gx2, gy2)[0])
    # ray that leaves the grid before crossing
    Hf, gxf, gyf = _flat(8.0, lo=-10.0, hi=10.0)
    assert np.isnan(ray_exit_distance((0.0, 0.0, 0.0), far, Hf, gxf, gyf)[0])


def test_true_hill_reproduces_exact_path_lengths():
    """Load-bearing: with the TRUE surface and a=1, predicted opacity equals the
    fixture's exact path length -> the tracer is right."""
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg, _true_hill
    sol = _build_fake_sol_from_hill(n_bins=48)
    cfg = _fake_cfg()
    g = np.arange(-30.0, 30.0 + 0.2, 0.2)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    res = SimpleNamespace(H=_true_hill(GX, GY), gx=g, gy=g.copy(), a=1.0)
    chk = surface_ray_check(sol, cfg, res)
    assert chk.n_checked > 1000
    assert chk.ray_ve > 0.99, chk.ray_ve
    assert chk.ray_rms < 0.05, chk.ray_rms


def test_nan_residual_where_surface_unconstrained():
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg, _true_hill
    sol = _build_fake_sol_from_hill(n_bins=32)
    cfg = _fake_cfg()
    g = np.arange(-6.0, 6.0 + 0.25, 0.25)           # small grid: many rays leave it
    GX, GY = np.meshgrid(g, g, indexing="ij")
    res = SimpleNamespace(H=_true_hill(GX, GY), gx=g, gy=g.copy(), a=1.0)
    chk = surface_ray_check(sol, cfg, res)
    r = np.concatenate([v.ravel() for v in chk.residual.values()])
    assert np.isnan(r).any()                        # off-grid rays are NaN
    assert chk.n_checked == int(np.isfinite(r).sum())
    assert not np.any(r[np.isnan(r)] == 0.0)


def test_prediction_is_nearly_scale_free():
    """lambda_pred = t*/a: the fitted surface scales ~linearly with a, so the
    predicted opacities should barely move between a=1 and a=2."""
    from megido.hillside_surface import fit_surface
    from tests.test_hillside_surface import _build_fake_sol_from_hill, _fake_cfg
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()
    r1 = fit_surface(sol, cfg, a=1.0, cell_m=0.5, n_restarts=1, max_points=400)
    r2 = fit_surface(sol, cfg, a=2.0, cell_m=0.5, n_restarts=1, max_points=400)
    c1, c2 = surface_ray_check(sol, cfg, r1), surface_ray_check(sol, cfg, r2)
    p1 = np.concatenate([c1.predicted[k].ravel() for k in sorted(c1.predicted)])
    p2 = np.concatenate([c2.predicted[k].ravel() for k in sorted(c2.predicted)])
    ok = np.isfinite(p1) & np.isfinite(p2)
    assert ok.sum() > 500
    assert np.corrcoef(p1[ok], p2[ok])[0, 1] > 0.98
    assert abs(c1.ray_ve - c2.ray_ve) < 0.1


_REAL = Path("runs/solve/baseline.npz")


@pytest.mark.skipif(not _REAL.exists(), reason="needs runs/solve/baseline.npz")
def test_real_data_check_is_finite():
    from megido.baseline import BaselineSolution
    from megido.config import load_site_config
    from megido.hillside_surface import fit_surface
    cfg = load_site_config("configs/megido.yaml")
    sol = BaselineSolution.load(_REAL)
    res = fit_surface(sol, cfg, a=8.0, cell_m=1.0, n_restarts=1, max_points=800)
    chk = surface_ray_check(sol, cfg, res)
    assert chk.n_checked > 0
    assert np.isfinite(chk.ray_ve) and np.isfinite(chk.ray_rms)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_hillside_check.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `megido/hillside_check.py`**

```python
"""Ray-space check of the fitted hillside surface.

Treat the fitted surface H(x, y) as the top of a uniform solid of density 1/a
(the same assumption that placed its exit points), ray-trace every measured
sky direction from each detector to where it leaves the rock, and compare the
predicted opacity t*/a with what was measured. A goodness-of-fit in the space
the data actually lives in; the per-cell VE of the surface fit is not that.

Because H scales ~linearly with a about each detector, t* scales by a and the
prediction is ~scale-free: this checks the surface's SHAPE (measured), not its
assumed scale. Unpredictable rays (surface unknown / off-grid / never crossed)
are NaN, never 0, and excluded from every metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.hillside_surface import _positions


def _bilinear(H, gx, gy, x, y) -> np.ndarray:
    H = np.asarray(H, float)
    gx = np.asarray(gx, float); gy = np.asarray(gy, float)
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = H.shape
    out = np.full(x.shape, np.nan)
    inside = (x >= gx[0]) & (x <= gx[-1]) & (y >= gy[0]) & (y <= gy[-1])
    if not inside.any():
        return out
    xi, yi = x[inside], y[inside]
    i0 = np.clip(np.searchsorted(gx, xi, side="right") - 1, 0, nx - 2)
    j0 = np.clip(np.searchsorted(gy, yi, side="right") - 1, 0, ny - 2)
    tx = (xi - gx[i0]) / (gx[i0 + 1] - gx[i0])
    ty = (yi - gy[j0]) / (gy[j0 + 1] - gy[j0])
    h00, h10 = H[i0, j0], H[i0 + 1, j0]
    h01, h11 = H[i0, j0 + 1], H[i0 + 1, j0 + 1]
    # a NaN corner poisons the sample even at zero weight (0*NaN = NaN): intended
    out[inside] = ((1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10
                   + (1 - tx) * ty * h01 + tx * ty * h11)
    return out


def ray_exit_distance(origin, dirs, H, gx, gy, *, t_max=None, step=None,
                      n_bisect: int = 30) -> np.ndarray:
    o = np.asarray(origin, float)
    d = np.atleast_2d(np.asarray(dirs, float))
    gx = np.asarray(gx, float); gy = np.asarray(gy, float)
    H = np.asarray(H, float)
    n = d.shape[0]
    t_out = np.full(n, np.nan)
    h_here = _bilinear(H, gx, gy, np.array([o[0]]), np.array([o[1]]))[0]
    if not np.isfinite(h_here) or o[2] >= h_here:
        return t_out
    if step is None:
        step = 0.25 * min(float(np.min(np.diff(gx))), float(np.min(np.diff(gy))))
    if t_max is None:
        span = float(np.hypot(gx[-1] - gx[0], gy[-1] - gy[0]))
        top = float(np.nanmax(H)) if np.isfinite(H).any() else o[2]
        t_max = span + abs(top - o[2]) + step
    active = np.ones(n, bool)
    for k in range(1, int(np.ceil(t_max / step)) + 1):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        t = k * step
        p = o[None, :] + t * d[idx]
        h = _bilinear(H, gx, gy, p[:, 0], p[:, 1])
        unknown = ~np.isfinite(h)
        active[idx[unknown]] = False            # stays NaN: surface unknown
        cross = np.isfinite(h) & (p[:, 2] >= h)
        ci = idx[cross]
        if ci.size:
            lo = np.full(ci.size, (k - 1) * step)
            hi = np.full(ci.size, t)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                pm = o[None, :] + mid[:, None] * d[ci]
                hm = _bilinear(H, gx, gy, pm[:, 0], pm[:, 1])
                above = pm[:, 2] >= hm          # NaN -> False -> treat as below
                hi = np.where(above, mid, hi)
                lo = np.where(above, lo, mid)
            t_out[ci] = hi
            active[ci] = False
    return t_out


@dataclass(frozen=True)
class RayCheck:
    residual: dict
    predicted: dict
    ray_ve: float
    ray_rms: float
    n_checked: int
    a: float


def surface_ray_check(sol, cfg, result, *, a=None) -> RayCheck:
    a = float(result.a if a is None else a)
    H = np.asarray(result.H, float)
    gx = np.asarray(result.gx, float); gy = np.asarray(result.gy, float)
    centers = np.asarray(sol.sky.centers, float)
    nb = centers.size
    SX, SY = np.meshgrid(centers, centers, indexing="ij")
    sx, sy = SX.ravel(), SY.ravel()
    norm = np.sqrt(1.0 + sx ** 2 + sy ** 2)
    dirs = np.stack([sx / norm, sy / norm, 1.0 / norm], axis=1)

    residual, predicted, meas, pred = {}, {}, [], []
    for pid, pose in sorted(_positions(cfg).items()):
        lam = np.asarray(sol.normalized_opacity(pid), float)
        live = np.isfinite(lam)
        t = np.full(lam.shape, np.nan)
        if live.any():
            t[live] = ray_exit_distance((pose.x, pose.y, pose.z), dirs[live], H, gx, gy)
        p = t / a
        r = lam - p
        ok = np.isfinite(r)
        residual[pid] = np.where(ok, r, np.nan).reshape(nb, nb)
        predicted[pid] = np.where(np.isfinite(p), p, np.nan).reshape(nb, nb)
        meas.append(lam[ok]); pred.append(p[ok])

    m = np.concatenate(meas) if meas else np.zeros(0)
    q = np.concatenate(pred) if pred else np.zeros(0)
    if m.size == 0:
        return RayCheck(residual, predicted, float("nan"), float("nan"), 0, a)
    rr = m - q
    denom = float(np.sum((m - m.mean()) ** 2))
    ve = float(1.0 - np.sum(rr ** 2) / denom) if denom > 0 else float("nan")
    return RayCheck(residual, predicted, ve, float(np.sqrt(np.mean(rr ** 2))),
                    int(m.size), a)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_hillside_check.py -q`
Expected: PASS. If `test_true_hill_reproduces_exact_path_lengths` fails, the tracer is wrong — debug it (do not loosen). If only `test_prediction_is_nearly_scale_free` fails, print the measured correlation and |Δve| in the report and STOP (return BLOCKED) — it is a physics claim, not a threshold to tune.

- [ ] **Step 5: Commit**

```bash
git add megido/hillside_check.py tests/test_hillside_check.py
git commit -m "feat(hillside): ray-space check of the fitted surface as a uniform solid"
```
(append the two attribution lines)

---

### Task 2: Wire the ray check into the `hillside` CLI

**Files:**
- Modify: `megido/cli.py` (`_cmd_hillside_surface`)
- Test: `tests/test_cli_hillside_surface.py`

**Interfaces:**
- Consumes: `surface_ray_check`, `RayCheck` from Task 1.
- Produces: meta keys `ray_ve`, `ray_rms`, `n_rays_checked` in `hill_surface_meta.json`; file `hill_residual.png`.

- [ ] **Step 1: Failing test** — in the existing `test_hillside_writes_surface_artifacts` (it already runs the CLI with fast flags), add after the meta is parsed:

```python
    for key in ("ray_ve", "ray_rms", "n_rays_checked"):
        assert key in meta, f"meta missing {key}"
    assert meta["n_rays_checked"] > 0
    assert meta["ray_rms"] is None or meta["ray_rms"] >= 0
```
and inside its matplotlib-present branch: `assert (out / "hill_residual.png").exists()`.

- [ ] **Step 2: Run to verify fail**: `uv run pytest tests/test_cli_hillside_surface.py -q` → FAIL (keys missing).

- [ ] **Step 3: Implement** in `_cmd_hillside_surface`, right after `result = fit_surface(...)`:

```python
    from megido.hillside_check import surface_ray_check
    check = surface_ray_check(sol, cfg, result)
    print(f"ray check       VE={check.ray_ve:.1%} (per ray)  RMS={check.ray_rms:.3f}  "
          f"over {check.n_checked} rays (surface as uniform solid, density 1/a)")
```
Add to the meta dict (next to `variance_explained`):
```python
        "ray_ve": check.ray_ve,
        "ray_rms": check.ray_rms,
        "n_rays_checked": check.n_checked,
```
(The existing `_json_nan_to_null` keeps NaN valid JSON.) In the matplotlib branch that writes `hill_surface.png`, also write the residual figure via a new module-level helper:

```python
def _write_residual_png(check, sol, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pids = sorted(check.residual)
    c = np.asarray(sol.sky.centers, float)
    finite = np.concatenate([v[np.isfinite(v)] for v in check.residual.values()]) \
        if pids else np.zeros(0)
    lim = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    lim = lim if lim > 0 else 1.0
    fig, axes = plt.subplots(1, max(len(pids), 1), figsize=(5 * max(len(pids), 1), 4.4),
                             squeeze=False)
    im = None
    for ax, pid in zip(axes[0], pids):
        im = ax.imshow(check.residual[pid].T, origin="lower",
                       extent=[c[0], c[-1], c[0], c[-1]],
                       cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{pid}: measured − predicted opacity")
        ax.set_xlabel("sky tangent x"); ax.set_ylabel("sky tangent y")
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(),
                     label="Δλ  (red: more rock than a uniform hill; blue: less)")
    fig.suptitle(f"Surface ray check, a={check.a:g}: VE={check.ray_ve:.0%} per ray, "
                 f"RMS={check.ray_rms:.3f}, N={check.n_checked}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```
called as `_write_residual_png(check, sol, out / "hill_residual.png")` and included in the "written ..." print line.

- [ ] **Step 4: Run**: `uv run pytest tests/test_cli_hillside_surface.py -q` → PASS.

- [ ] **Step 5: Commit** `feat(hillside): CLI reports ray-space VE/RMS and writes hill_residual.png` (+ attribution).

---

### Task 3: Viewer — σ-coloured surface, legend, per-ray VE caption

**Files:**
- Modify: `viewer/src/surfacemesh.mjs` (add `SIGMA_RAMP`, `surfaceVertexColors`, `robustRange`)
- Modify: `viewer/src/app.mjs` (mesh shader pair, colour buffer, sigma loading, legend, caption)
- Modify: `viewer/shell.html` (toggle + legend markup + CSS)
- Modify: `tests/viewer/conftest.py` (`run_fixture(..., hill=False)`)
- Test: `viewer/test/surfacemesh.test.mjs` (create if absent; else extend), `tests/viewer/test_surface_sigma.py`

**Interfaces:**
- Produces (surfacemesh.mjs):
  - `SIGMA_RAMP = [[0.96, 0.72, 0.36], [0.45, 0.16, 0.03]]` (low σ → high σ).
  - `surfaceVertexColors(sigma, lo, hi) -> Float32Array` (length `3*sigma.length`); `t=(σ-lo)/(hi-lo)` clamped to [0,1]; non-finite σ → `t=0.5`; `hi<=lo` → treat span as 1.
  - `robustRange(values, pLo=0.05, pHi=0.95) -> [lo, hi]` over finite values; `[NaN, NaN]` if none.
- Produces (fixture): `run_fixture(..., hill=True)` additionally writes `hill_surface.npy` (7×6 grid, `gx=0..3 step 0.5`, `gy=0..2.5 step 0.5`, H ≡ 2.0), `hill_surface_sigma.npy` (values 0.1..1.0 varying with i), `hill_surface_meta.json` (`gx`, `gy`, `a: 8.0`, `variance_explained: 0.81`, `ray_ve: 0.5`, `scale_assumed: true`, `note: "fixture"`).
- Produces (DOM): `#toggle-hill-sigma` checkbox, `#hill-sigma-legend` (with `#hill-sigma-range` text); `window.__viewerState.hillSigmaColor` bool.

- [ ] **Step 1: Failing node tests** (`viewer/test/surfacemesh.test.mjs`, match the style of the existing `grid.test.mjs`):

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { surfaceVertexColors, robustRange, SIGMA_RAMP } from '../src/surfacemesh.mjs';

test('surfaceVertexColors maps lo/hi to ramp ends and clamps', () => {
  const c = surfaceVertexColors(Float32Array.from([0, 1, 2, -5, 9]), 0, 2);
  const [a, b] = SIGMA_RAMP;
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(c[k] - a[k]) < 1e-6);          // sigma=lo -> start
    assert.ok(Math.abs(c[6 + k] - b[k]) < 1e-6);      // sigma=hi -> end
    assert.ok(Math.abs(c[9 + k] - a[k]) < 1e-6);      // below lo clamps
    assert.ok(Math.abs(c[12 + k] - b[k]) < 1e-6);     // above hi clamps
    assert.ok(Math.abs(c[3 + k] - (a[k] + b[k]) / 2) < 1e-6); // midpoint
  }
});

test('surfaceVertexColors gives NaN sigma the mid colour', () => {
  const c = surfaceVertexColors(Float32Array.from([NaN]), 0, 1);
  const [a, b] = SIGMA_RAMP;
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(c[k] - (a[k] + b[k]) / 2) < 1e-6);
});

test('robustRange ignores NaN and returns 5th/95th percentiles', () => {
  const v = Float32Array.from([...Array(101).keys()].map(Number).concat([NaN]));
  const [lo, hi] = robustRange(v);
  assert.ok(Math.abs(lo - 5) < 1e-6 && Math.abs(hi - 95) < 1e-6);
  const [n1, n2] = robustRange(Float32Array.from([NaN]));
  assert.ok(Number.isNaN(n1) && Number.isNaN(n2));
});
```

- [ ] **Step 2:** `cd viewer && node --test 'test/*.test.mjs'` → FAIL (exports missing).

- [ ] **Step 3: Implement helpers** in `surfacemesh.mjs`:

```js
// Sequential single-hue ramp for the surface's posterior sigma:
// warm (confident) -> deep burnt orange (uncertain). Readable on both themes.
export const SIGMA_RAMP = [[0.96, 0.72, 0.36], [0.45, 0.16, 0.03]];

export function surfaceVertexColors(sigma, lo, hi) {
  const out = new Float32Array(sigma.length * 3);
  const [a, b] = SIGMA_RAMP;
  const span = hi > lo ? hi - lo : 1;
  for (let k = 0; k < sigma.length; k++) {
    const s = sigma[k];
    let t = Number.isFinite(s) ? (s - lo) / span : 0.5;
    t = Math.min(1, Math.max(0, t));
    for (let c = 0; c < 3; c++) out[k * 3 + c] = a[c] + (b[c] - a[c]) * t;
  }
  return out;
}

// Linear-interpolated percentiles over finite values.
export function robustRange(values, pLo = 0.05, pHi = 0.95) {
  const v = Array.from(values).filter(Number.isFinite).sort((x, y) => x - y);
  if (!v.length) return [NaN, NaN];
  const q = (p) => {
    const pos = p * (v.length - 1), i = Math.floor(pos), f = pos - i;
    return i + 1 < v.length ? v[i] + (v[i + 1] - v[i]) * f : v[i];
  };
  return [q(pLo), q(pHi)];
}
```

- [ ] **Step 4: Wire into `app.mjs`:**
  1. Add a mesh shader pair (leave `MARKER_*` and `FILL_FRAGMENT_SRC` untouched):
     ```js
     const MESH_VERTEX_SRC = `#version 300 es
     layout(location = 0) in vec3 aPos;
     layout(location = 1) in vec3 aColor;
     uniform mat4 uMarkerViewProj;
     out vec3 vColor;
     void main() { vColor = aColor; gl_Position = uMarkerViewProj * vec4(aPos, 1.0); }`;
     const MESH_FRAGMENT_SRC = `#version 300 es
     precision highp float;
     in vec3 vColor;
     uniform float uAlpha;
     out vec4 outColor;
     void main() { outColor = vec4(vColor, uAlpha); }`;
     ```
  2. Link `meshProgram` from it; uniforms `uMarkerViewProj`, `uAlpha`. Add `hillSurfaceColorBuffer` to `hillSurfaceVao` at attribute location 1 (3 floats).
  3. In `rebuildHillSurfaceBuffer()`: after positions, upload colours — `state.hillSigmaColor && surf.sigma` ? `surfaceVertexColors(surf.sigma, lo, hi)` : a flat array of `HILL_SURFACE_COLOR[0..2]` per vertex.
  4. In the hill draw block use `meshProgram` with `uAlpha = HILL_SURFACE_COLOR[3]` instead of `fillProgram`/`uFillColor`.
  5. In the loadRun hill block: if `byName.get('hill_surface_sigma.npy')` exists and its length equals `H.length`, parse into `state.hillSurface.sigma`, compute `state.hillSigmaRange = robustRange(sigma)`, enable `#toggle-hill-sigma` (checked), set `state.hillSigmaColor = true`, fill `#hill-sigma-range` with `` `σ ${lo.toFixed(2)} – ${hi.toFixed(2)} m · raw posterior std, assumed scale` `` and unhide `#hill-sigma-legend`; else disable/uncheck the toggle and hide the legend. Reset both on every load.
  6. Caption: `` `Hillside surface — assumed scale, ${pctCell}% VE per cell` `` + (`Number.isFinite(hillMeta.ray_ve)` ? `` ` · ${pctRay}% per ray` `` : '').
  7. Toggle listener: `state.hillSigmaColor = checked; rebuildHillSurfaceBuffer(); render();`
  8. Expose `hillSigmaColor` on `window.__viewerState` (it is the `state` object already).
- Add to `shell.html` under `#hill-surface-caveat`:
  ```html
  <label><input id="toggle-hill-sigma" type="checkbox" style="width:auto" disabled> Colour by uncertainty</label>
  <div id="hill-sigma-legend" hidden>
    <div class="sigma-bar" aria-hidden="true"></div>
    <span id="hill-sigma-range" class="dim mono"></span>
  </div>
  ```
  CSS: `.sigma-bar{height:8px;border-radius:4px;background:linear-gradient(90deg,rgb(245,184,92),rgb(115,41,8));margin:4px 0 2px}`.

- [ ] **Step 5: Fixture + Playwright test.** Extend `run_fixture._make` with `hill=False` writing the three files described in Interfaces. New `tests/viewer/test_surface_sigma.py`:

```python
def test_sigma_colour_toggle_changes_render_and_shows_legend(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    assert page.locator("#hill-sigma-legend").is_visible()
    assert "σ" in page.locator("#hill-sigma-range").inner_text()
    assert page.evaluate("() => window.__viewerState.hillSigmaColor") is True
    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#toggle-hill-sigma").uncheck()
    page.wait_for_timeout(200)
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after
    assert "per ray" in page.locator("#hill-surface-caveat").inner_text()
    assert not errors, errors
```
(If the `Hillside surface` dock section is collapsed by default, expand it first the way `test_dock.py` does.)

- [ ] **Step 6: Run**: `cd viewer && node --test 'test/*.test.mjs'` and `uv run pytest tests/viewer/test_surface_sigma.py tests/viewer/test_smoke.py -q` → PASS.

- [ ] **Step 7: Commit** `feat(viewer): colour hillside surface by posterior sigma, legend, per-ray VE` (+ attribution).

---

### Task 4: Viewer — trilinear sampling + gradient shading

**Files:**
- Modify: `viewer/src/app.mjs`, `viewer/shell.html`
- Test: `tests/viewer/test_render_quality.py`

**Interfaces:**
- Produces: `#toggle-smooth` and `#toggle-shading` checkboxes (both default checked); `state.smoothSampling`, `state.shading`, `state.floatLinear` (bool, extension present).

- [ ] **Step 1: Failing Playwright test** `tests/viewer/test_render_quality.py`:

```python
def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")


def _png(page):
    return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")


def test_smooth_and_shading_toggles_each_change_the_render(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    st = page.evaluate("() => ({s: window.__viewerState.smoothSampling, h: window.__viewerState.shading})")
    assert st == {"s": True, "h": True}
    base = _png(page)
    page.locator("#toggle-shading").uncheck(); page.wait_for_timeout(150)
    no_shade = _png(page)
    assert no_shade != base
    page.locator("#toggle-smooth").uncheck(); page.wait_for_timeout(150)
    nearest = _png(page)
    assert nearest != no_shade
    assert not errors, errors
```

- [ ] **Step 2:** `uv run pytest tests/viewer/test_render_quality.py -q` → FAIL (no toggles).

- [ ] **Step 3: Implement.**
  1. At GL init: `const floatLinear = !!gl.getExtension('OES_texture_float_linear');` store `state.floatLinear = floatLinear`; `state.smoothSampling = true; state.shading = true;`.
  2. Add `function applyVolumeFilter()` that binds each live 3D texture (`state.volumeTex`, `state.sigmaTex` if set) and sets MIN/MAG filter to `(state.smoothSampling && state.floatLinear) ? gl.LINEAR : gl.NEAREST`. Call it after every `makeVolumeTexture` assignment and on toggle. (Keep `makeVolumeTexture` creating NEAREST; `applyVolumeFilter` upgrades.)
  3. Fragment shader additions (volume program only):
     ```glsl
     uniform bool uManualTrilinear;
     uniform bool uShading;
     uniform vec3 uVolSize;   // float(nx, ny, nz)

     float sampleDensity(vec3 tex) {
       if (!uManualTrilinear) return texture(uVolume, tex).r;
       vec3 p = tex * uVolSize - 0.5;
       vec3 f = fract(p);
       ivec3 mx = ivec3(uVolSize) - 1;
       ivec3 a = clamp(ivec3(floor(p)), ivec3(0), mx);
       ivec3 b = clamp(ivec3(floor(p)) + 1, ivec3(0), mx);
       float c000 = texelFetch(uVolume, ivec3(a.x, a.y, a.z), 0).r;
       float c100 = texelFetch(uVolume, ivec3(b.x, a.y, a.z), 0).r;
       float c010 = texelFetch(uVolume, ivec3(a.x, b.y, a.z), 0).r;
       float c110 = texelFetch(uVolume, ivec3(b.x, b.y, a.z), 0).r;
       float c001 = texelFetch(uVolume, ivec3(a.x, a.y, b.z), 0).r;
       float c101 = texelFetch(uVolume, ivec3(b.x, a.y, b.z), 0).r;
       float c011 = texelFetch(uVolume, ivec3(a.x, b.y, b.z), 0).r;
       float c111 = texelFetch(uVolume, ivec3(b.x, b.y, b.z), 0).r;
       return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
                  mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
     }
     ```
     Replace `float density = texture(uVolume, tex).r;` with `float density = sampleDensity(tex);`. After the LUT lookup and before `c.rgb *= c.a;`:
     ```glsl
     if (uShading && c.a > 0.0) {
       vec3 h = 1.0 / uVolSize;   // one voxel per axis; voxels are cubic, so the
                                  // tex-space difference is proportional to the world gradient
       vec3 g = vec3(
         sampleDensity(tex + vec3(h.x, 0.0, 0.0)) - sampleDensity(tex - vec3(h.x, 0.0, 0.0)),
         sampleDensity(tex + vec3(0.0, h.y, 0.0)) - sampleDensity(tex - vec3(0.0, h.y, 0.0)),
         sampleDensity(tex + vec3(0.0, 0.0, h.z)) - sampleDensity(tex - vec3(0.0, 0.0, h.z)));
       float gm = length(g);
       if (gm > 1e-6) {
         float lambert = abs(dot(-g / gm, -dir));
         c.rgb *= 0.35 + 0.65 * lambert;
       }
     }
     ```
  4. Per frame set `uManualTrilinear = state.smoothSampling && !state.floatLinear`, `uShading = state.shading`, `uVolSize = meta.shape` as floats (register the uniform names in the existing uniform-location list).
  5. `shell.html`: in the Transfer (or Camera & View) section add
     ```html
     <label><input id="toggle-smooth" type="checkbox" style="width:auto" checked> Smooth sampling</label>
     <label><input id="toggle-shading" type="checkbox" style="width:auto" checked> Shading</label>
     ```
     Listeners set the state flags, call `applyVolumeFilter()` (smooth only) and `render()`.

- [ ] **Step 4: Run**: `uv run pytest tests/viewer/test_render_quality.py tests/viewer/test_raymarch.py tests/viewer/test_smoke.py -q` → PASS.

- [ ] **Step 5: Commit** `feat(viewer): trilinear sampling and gradient shading with toggles` (+ attribution).

---

### Task 5: Viewer — display-only clip above the surface

**Files:**
- Modify: `viewer/src/surfacemesh.mjs` (add `surfaceHeightAt`, `surfaceTextureData`)
- Modify: `viewer/src/app.mjs`, `viewer/shell.html`
- Test: `viewer/test/surfacemesh.test.mjs`, `tests/viewer/test_surface_clip.py`

**Interfaces:**
- Produces (surfacemesh.mjs):
  - `surfaceHeightAt(H, gx, gy, x, y) -> number` — bilinear on the node grid (`H[i*ny+j]`); NaN outside `[gx0,gxN]×[gy0,gyN]` or if any used corner is NaN.
  - `surfaceTextureData(H, nx, ny, sentinel = 1e6) -> Float32Array` — x-fastest layout for a `nx × ny` 2D texture: `out[j*nx + i] = H[i*ny + j]`, NaN → `sentinel`.
- Produces (DOM/state): `#toggle-surf-clip` (default unchecked, disabled until a surface loads), `#surf-clip-caveat`; `state.surfClip`, `state.hillDisplayH` (the displayed, possibly smoothed H).

- [ ] **Step 1: Failing node tests** (append to `viewer/test/surfacemesh.test.mjs`):

```js
import { surfaceHeightAt, surfaceTextureData } from '../src/surfacemesh.mjs';

test('surfaceHeightAt is exact on a plane, NaN outside or at a NaN corner', () => {
  const gx = [0, 1, 2], gy = [0, 1, 2, 3];
  const H = new Float32Array(12);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) H[i * 4 + j] = 2 * gx[i] + 3 * gy[j] + 1;
  assert.ok(Math.abs(surfaceHeightAt(H, gx, gy, 0.5, 2.25) - (1 + 6.75 + 1)) < 1e-5);
  assert.ok(Number.isNaN(surfaceHeightAt(H, gx, gy, -0.1, 1)));
  assert.ok(Number.isNaN(surfaceHeightAt(H, gx, gy, 1, 3.5)));
  const H2 = Float32Array.from(H); H2[1 * 4 + 1] = NaN;
  assert.ok(Number.isNaN(surfaceHeightAt(H2, gx, gy, 0.5, 0.5)));
});

test('surfaceTextureData transposes to x-fastest and sentinels NaN', () => {
  const nx = 3, ny = 2;
  const H = Float32Array.from([0, 1, 10, 11, NaN, 21]); // H[i*ny+j] = 10*i + j
  const t = surfaceTextureData(H, nx, ny, 1e6);
  // out[j*nx+i] = H[i*ny+j]
  assert.deepEqual(Array.from(t), [0, 10, 1e6, 1, 11, 21]);
});
```

- [ ] **Step 2:** `cd viewer && node --test 'test/*.test.mjs'` → FAIL.

- [ ] **Step 3: Implement helpers:**

```js
export function surfaceHeightAt(H, gx, gy, x, y) {
  const nx = gx.length, ny = gy.length;
  if (!(x >= gx[0] && x <= gx[nx - 1] && y >= gy[0] && y <= gy[ny - 1])) return NaN;
  let i = 0; while (i < nx - 2 && x > gx[i + 1]) i++;
  let j = 0; while (j < ny - 2 && y > gy[j + 1]) j++;
  const tx = (x - gx[i]) / (gx[i + 1] - gx[i]);
  const ty = (y - gy[j]) / (gy[j + 1] - gy[j]);
  const h00 = H[i * ny + j], h10 = H[(i + 1) * ny + j];
  const h01 = H[i * ny + j + 1], h11 = H[(i + 1) * ny + j + 1];
  if (![h00, h10, h01, h11].every(Number.isFinite)) return NaN;
  return (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11;
}

export function surfaceTextureData(H, nx, ny, sentinel = 1e6) {
  const out = new Float32Array(nx * ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const v = H[i * ny + j];
      out[j * nx + i] = Number.isFinite(v) ? v : sentinel;
    }
  }
  return out;
}
```

- [ ] **Step 4: Wire into `app.mjs`:**
  1. In `rebuildHillSurfaceBuffer()`, keep the displayed field: `state.hillDisplayH = H;` and upload it to a 2D texture `state.surfTex` (create once): `gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, nx, ny, 0, gl.RED, gl.FLOAT, surfaceTextureData(H, nx, ny))`, NEAREST filters, CLAMP_TO_EDGE. So the clip always matches the drawn (smoothed) surface.
  2. Volume fragment shader:
     ```glsl
     uniform bool uSurfClip;
     uniform sampler2D uSurfTex;
     uniform vec2 uSurfMin;     // (gx[0], gy[0])
     uniform vec2 uSurfStep;    // (gx[1]-gx[0], gy[1]-gy[0])  (the fit grid is uniform)
     uniform ivec2 uSurfSize;   // (nx, ny)

     bool aboveSurface(vec3 world) {
       vec2 q = (world.xy - uSurfMin) / uSurfStep;
       if (q.x < 0.0 || q.y < 0.0 || q.x > float(uSurfSize.x - 1) || q.y > float(uSurfSize.y - 1))
         return false;
       ivec2 i0 = min(ivec2(floor(q)), uSurfSize - 2);
       vec2 f = q - vec2(i0);
       float h00 = texelFetch(uSurfTex, i0, 0).r;
       float h10 = texelFetch(uSurfTex, i0 + ivec2(1, 0), 0).r;
       float h01 = texelFetch(uSurfTex, i0 + ivec2(0, 1), 0).r;
       float h11 = texelFetch(uSurfTex, i0 + ivec2(1, 1), 0).r;
       if (max(max(h00, h10), max(h01, h11)) > 1.0e5) return false;   // unknown ground: never clip
       float h = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
       return world.z > h;
     }
     ```
     In the march loop: `clipped = clipped || (uSurfClip && aboveSurface(pos));` (`pos` is the world-space sample). Bind `state.surfTex` to a free texture unit for `uSurfTex`; set `uSurfClip = state.surfClip && !!state.surfTex`.
  3. `castHoverRay`: after the clip-plane check,
     ```js
     if (state.surfClip && state.hillSurface && state.hillDisplayH) {
       const hs = surfaceHeightAt(state.hillDisplayH, state.hillSurface.gx, state.hillSurface.gy, world[0], world[1]);
       if (Number.isFinite(hs) && world[2] > hs) continue;
     }
     ```
  4. `shell.html` (under the sigma legend):
     ```html
     <label><input id="toggle-surf-clip" type="checkbox" style="width:auto" disabled> Clip volume above surface</label>
     <div id="surf-clip-caveat" class="dim">display only — hides density the surface model calls air</div>
     ```
     Enable on surface load (unchecked); disable + uncheck + `state.surfClip=false` when absent. Listener sets `state.surfClip` and renders.

- [ ] **Step 5: Playwright test** `tests/viewer/test_surface_clip.py`:

```python
_COUNT_LIT = """() => {
  const src = document.querySelector('#gl-canvas');
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const bg = [d[0], d[1], d[2]];
  let n = 0;
  for (let k = 0; k < d.length; k += 4) {
    if (Math.abs(d[k]-bg[0]) + Math.abs(d[k+1]-bg[1]) + Math.abs(d[k+2]-bg[2]) > 24) n++;
  }
  return n;
}"""


def test_clip_above_surface_hides_volume_and_is_off_by_default(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    assert page.locator("#toggle-surf-clip").is_enabled()
    assert page.locator("#toggle-surf-clip").is_checked() is False
    page.locator("#toggle-hill-surface").uncheck()      # count volume pixels only
    page.wait_for_timeout(150)
    lit_off = page.evaluate(_COUNT_LIT)
    page.locator("#toggle-surf-clip").check(); page.wait_for_timeout(150)
    lit_on = page.evaluate(_COUNT_LIT)
    assert lit_on < lit_off, (lit_off, lit_on)
    assert not errors, errors
```
(The fixture's flat surface at z = 2.0 cuts the z ∈ [1, 3] volume in half, so clipping must remove drawn pixels. Pixel [0,0] is background in the default framing; if not, sample a corner that is.)

- [ ] **Step 6: Run**: `cd viewer && node --test 'test/*.test.mjs'`; `uv run pytest tests/viewer/test_surface_clip.py tests/viewer/test_gate_and_hover.py tests/viewer/test_smoke.py -q` → PASS.

- [ ] **Step 7: Commit** `feat(viewer): display-only clip of the volume above the fitted surface` (+ attribution).

---

### Task 6: Regenerate, verify, document (controller)

- [ ] Regenerate: `uv run python -m megido.cli hillside --solve runs/solve --run runs/ingest --out runs/voxels`; record ray VE/RMS/N.
- [ ] Rebuild viewer (`uv run python -m megido.viewerbuild`), screenshot the real run: σ-coloured surface, shading on/off, clip on.
- [ ] Extend `tests/viewer/test_smoke.py` to operate the new toggles on the real run with zero console errors.
- [ ] Full suite `uv run pytest -q` + node tests.
- [ ] CLAUDE.md: add the ray check to Phase 5 / Running-it, and the viewer's new display options (all display-only) under Phase 4.
- [ ] Commit `docs: ray-space surface check and viewer upgrades` (+ attribution).

---

## Self-Review

1. **Spec coverage:** §3 ray check → T1 (model, tracing, NaN, outputs) + T2 (CLI, meta, PNG). §4.1 → T3. §4.2 → T4. §4.3 → T5. §5 tests 1-6 → T1/T2; 7-8 → T3/T4/T5; 9 → T6. §6 risks: ray VE reported not gated (T2/T6); manual trilinear fallback (T4); shading default revisited at T6 screenshot.
2. **Placeholders:** none.
3. **Type consistency:** `RayCheck.residual/predicted/ray_ve/ray_rms/n_checked/a` used identically in T1 tests, T2 CLI, T2 PNG. JS helper names (`surfaceVertexColors`, `robustRange`, `SIGMA_RAMP`, `surfaceHeightAt`, `surfaceTextureData`) consistent across T3/T5 tests and wiring. DOM ids consistent between shell.html snippets and Playwright tests.
