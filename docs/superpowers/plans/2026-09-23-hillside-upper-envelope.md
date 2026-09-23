# Upper-Envelope Hillside Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the spurious height dip at the two detector spots by fitting the hillside GP to a per-cell flux-weighted upper-quantile of exit-z (the upper envelope) instead of all raw exit points.

**Architecture:** Exit-point placement `o + a·λ·d̂` sorts rays radially by opacity, so gauge-manufactured low-opacity rays pile at each detector footprint and a mean-like GP inverts the crown into a dip. A new `_upper_envelope_cells` reduction takes, per grid cell, the flux-weighted (`dz^index`) upper quantile of exit-z; the existing Matérn-5/2 GP then fits those cell targets. GP core and `SurfaceResult` are unchanged.

**Tech Stack:** Python, numpy, scipy (no new deps). `megido.hillside_gp` unchanged.

**Spec:** `docs/superpowers/specs/2026-09-23-hillside-upper-envelope-design.md`

## Global Constraints

- No new runtime dependency; numpy + scipy only.
- Metres throughout (Phase 3). `Pose.x/y/z` metres.
- `np.nan` means "not constrained" — never 0. Uncovered cells → NaN.
- Absolute height is ASSUMED via `a`; only shape is data-driven. `scale_assumed=True`; honesty note preserved and updated.
- Feed the fit `BaselineSolution.normalized_opacity` (via `exit_points`), never raw `.opacity`.
- `topography.csv` is TEST-ONLY ground truth (path `/home/hadar/Cloud/Work/Postdoc/01_data/raw/topography.csv`); never read by `megido/` or the CLI. `calibration_open_sky.csv` never read.
- `SurfaceResult` field names/types unchanged so the viewer/CLI need no change.
- The opacity gauge zero-point is OUT OF SCOPE (noted, deferred) — do not modify `megido/baseline.py`'s solve.
- TDD: red before green. A gate must be able to fail for a broken implementation; never loosen a threshold to force green.
- Commit messages end with the two attribution lines (shown in each Commit step).

---

### Task 1: Upper-envelope reduction helpers (pure functions)

**Files:**
- Modify: `megido/hillside_surface.py` (add `_weighted_quantile`, `_upper_envelope_cells`; no change yet to `exit_points`/`fit_surface`)
- Test: `tests/test_hillside_surface.py` (add unit tests)

**Interfaces:**
- Produces:
  - `_weighted_quantile(v, w, q) -> float` — weighted quantile of values `v` with weights `w` at level `q` (midpoint convention).
  - `_upper_envelope_cells(X, z, dz, gx, gy, *, index=2.0, q_hi=0.85, min_count=8, ray_counts=None) -> (X_cell, y_cell, noise_cell)` — per-cell flux-weighted upper-quantile reduction. `X` is `(N,2)` exit xy, `z` `(N,)` exit-z, `dz` `(N,)` sky vertical cosine. Returns cell centroids `(M,2)`, targets `(M,)`, per-cell noise variance `(M,)`. Cells with `< min_count` points are dropped.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hillside_surface.py  (add near the other helpers)
from megido.hillside_surface import _weighted_quantile, _upper_envelope_cells


def test_weighted_quantile_matches_unweighted_when_flat():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones(5)
    # midpoint-convention weighted quantile ~ numpy quantile for uniform weights
    assert abs(_weighted_quantile(v, w, 0.5) - 3.0) < 1e-9
    assert _weighted_quantile(v, w, 0.9) > 4.0


def test_weighted_quantile_upweights_high_weight_values():
    v = np.array([0.0, 10.0])
    # nearly all weight on the high value -> quantile near the high value
    assert _weighted_quantile(v, np.array([0.01, 100.0]), 0.5) > 9.0
    assert _weighted_quantile(v, np.array([100.0, 0.01]), 0.5) < 1.0


def test_upper_envelope_takes_high_quantile_and_flux_upweights_vertical():
    # one cell (all points inside), z has a low cluster and a high cluster; the
    # high cluster is near-vertical (dz~1, high flux weight) -> target must land
    # in the HIGH cluster, not the (more numerous) low cluster.
    gx = np.array([0.0, 2.0]); gy = np.array([0.0, 2.0])
    rng = np.random.default_rng(0)
    low_xy = rng.uniform(0.2, 1.8, (40, 2)); low_z = rng.normal(2.0, 0.2, 40)
    low_dz = np.full(40, 0.6)                      # oblique, low flux
    hi_xy = rng.uniform(0.2, 1.8, (12, 2)); hi_z = rng.normal(9.0, 0.2, 12)
    hi_dz = np.full(12, 0.99)                      # vertical, high flux
    X = np.vstack([low_xy, hi_xy]); z = np.concatenate([low_z, hi_z])
    dz = np.concatenate([low_dz, hi_dz])
    Xc, yc, nc = _upper_envelope_cells(X, z, dz, gx, gy, index=2.0, q_hi=0.85, min_count=8)
    assert Xc.shape[0] == 1                         # one populated cell
    assert yc[0] > 6.0                              # tracks the high (crown) cluster
    assert nc[0] > 0


def test_upper_envelope_drops_sparse_cells():
    gx = np.array([0.0, 2.0, 4.0]); gy = np.array([0.0, 2.0, 4.0])
    X = np.array([[0.5, 0.5], [0.6, 0.6]])          # 2 points, below min_count
    z = np.array([1.0, 2.0]); dz = np.array([0.9, 0.9])
    Xc, yc, nc = _upper_envelope_cells(X, z, dz, gx, gy, min_count=8)
    assert Xc.shape[0] == 0
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_hillside_surface.py -k "weighted_quantile or upper_envelope" -q`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement the helpers**

Add to `megido/hillside_surface.py` (after the imports / near `_footprint_grid`):

```python
def _weighted_quantile(v, w, q: float) -> float:
    """Weighted quantile of values ``v`` with non-negative weights ``w`` at level
    ``q`` in [0,1], midpoint (Hazen-like) convention. Falls back to the unweighted
    quantile if all weights are zero."""
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    total = w.sum()
    if total <= 0:
        return float(np.quantile(v, q))
    cq = (np.cumsum(w) - 0.5 * w) / total
    return float(np.interp(q, cq, v))


def _upper_envelope_cells(X, z, dz, gx, gy, *, index: float = 2.0,
                          q_hi: float = 0.85, min_count: int = 8,
                          ray_counts=None):
    """Reduce exit points to one flux-weighted upper-quantile target per grid cell.

    The true surface is the MAXIMUM overburden per column; the low exit points are
    artifacts of radial opacity sorting (see the spec). Per cell we take the
    ``q_hi`` weighted quantile of exit-z, weighting each ray by the cosmic-ray flux
    ``dz**index`` (the known cos^index angular distribution) and, when available,
    its Poisson counts -- so the sparse high-flux near-vertical rays that actually
    sample the crown drive the target. Cells with fewer than ``min_count`` points
    are dropped (unconstrained -> NaN downstream, never 0).

    Returns (X_cell (M,2) centroids, y_cell (M,) targets, noise_cell (M,) variances).
    """
    X = np.asarray(X, float); z = np.asarray(z, float); dz = np.asarray(dz, float)
    nx, ny = len(gx), len(gy)
    ix = np.clip(np.searchsorted(gx, X[:, 0]) - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(gy, X[:, 1]) - 1, 0, ny - 1)
    cell = ix * ny + iy
    fw = np.clip(dz ** index, 1e-3, None)
    if ray_counts is not None:
        fw = fw * np.clip(np.asarray(ray_counts, float), 1.0, None)

    xs, ys, targets, noises = [], [], [], []
    for c in np.unique(cell):
        m = cell == c
        if int(m.sum()) < min_count:
            continue
        zc, wc, xc, yc = z[m], fw[m], X[m, 0], X[m, 1]
        target = _weighted_quantile(zc, wc, q_hi)
        wsum = wc.sum()
        n_eff = (wsum ** 2) / np.sum(wc ** 2)          # Kish effective count
        wmean = np.sum(wc * zc) / wsum
        wvar = np.sum(wc * (zc - wmean) ** 2) / wsum
        sigma = np.sqrt(wvar) / np.sqrt(max(n_eff, 1.0)) + 0.02 * abs(target) + 1e-3
        xs.append(np.average(xc, weights=wc))
        ys.append(np.average(yc, weights=wc))
        targets.append(target)
        noises.append(sigma ** 2)

    if not xs:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,))
    return (np.column_stack([xs, ys]), np.asarray(targets), np.asarray(noises))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_hillside_surface.py -k "weighted_quantile or upper_envelope" -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add megido/hillside_surface.py tests/test_hillside_surface.py
git commit -m "feat(hillside): flux-weighted upper-envelope cell reduction helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6"
```

---

### Task 2: Wire the upper envelope into `fit_surface` + dip gate

**Files:**
- Modify: `megido/hillside_surface.py` (`exit_points` re-adds `dz`; `fit_surface` uses `_upper_envelope_cells`; new params `q_hi`, `min_count`; note updated)
- Modify: `tests/test_hillside_surface.py` (update `exit_points` unpack sites; adapt synthetic no-regression tests; add the real-data dip gate)

**Interfaces:**
- Consumes: Task 1 helpers; `megido.hillside_gp.{fit_hyperparams, predict}`.
- Produces:
  - `exit_points(sol, cfg, a=1.0, transparent_quantile=0.05) -> (pts, pos_index, dz, sky_flat)` (4-tuple; `dz` = `1/√(1+sx²+sy²)`).
  - `fit_surface(sol, cfg, *, a=8.0, cell_m=1.0, sky_counts=None, cov_tau=0.9, n_restarts=3, max_points=1000, q_hi=0.85, min_count=8) -> SurfaceResult`.

- [ ] **Step 1: Write/adapt the failing tests**

First, every existing `exit_points(...)` unpack must become a 4-tuple. In `tests/test_hillside_surface.py` the heteroscedastic test unpacks it — change:
```python
    _, pos_index, dz, sky_flat = exit_points(sol, cfg, a=1.0)
```
Keep the existing `test_synthetic_hill_recovery_gate`, `test_gp_uncertainty_is_calibrated`, `test_absolute_scale_rides_on_assumption_a`, `test_unconstrained_nodes_are_nan_not_zero`, and `test_heteroscedastic_downweights_low_count_bins` — they must still pass (no regression) with the new estimator (they call `fit_surface` with the same signature; new params default).

Add the load-bearing dip gate:
```python
from pathlib import Path
from megido.config import load_site_config
from megido.baseline import BaselineSolution
from megido.hillside_surface import _positions

_REAL_SOLVE = Path("runs/solve")
_TOPO = Path("/home/hadar/Cloud/Work/Postdoc/01_data/raw/topography.csv")


@pytest.mark.skipif(not (_REAL_SOLVE / "baseline.npz").exists() or not _TOPO.exists(),
                     reason="needs runs/solve/baseline.npz + topography.csv ground truth")
def test_detector_footprint_is_not_a_spurious_dip():
    """Load-bearing: the fitted surface must NOT invert the crown into a dip at the
    detector footprints. Ground truth (topography.csv, TEST-ONLY) is a peak there;
    the old mean estimator produced a ~2.4 m dip below the surrounding ring. The
    upper-envelope estimator must bring the footprint back up (no spurious deep
    minimum)."""
    import numpy as np
    cfg = load_site_config("configs/megido.yaml")
    sol = BaselineSolution.load(_REAL_SOLVE / "baseline.npz")
    r = fit_surface(sol, cfg, a=8.0, cell_m=1.0, n_restarts=1, max_points=800)
    pos = _positions(cfg); pids = sorted(pos)
    gx, gy = r.gx, r.gy

    T = np.loadtxt(_TOPO, delimiter=",", skiprows=1)
    tx, ty, tz = T[:, 0], T[:, 1], T[:, 2]
    def truth(x, y, rr):
        m = np.hypot(tx - x, ty - y) < rr
        return float(tz[m].mean()) if m.any() else np.nan

    for pid in pids:
        p = pos[pid]
        # sanity: truth is NOT a dip at the detector (peak/flat)
        assert truth(p.x, p.y, 3.0) >= truth(p.x, p.y, 8.0) - 1.0
        i = int(np.argmin(np.abs(gx - p.x))); j = int(np.argmin(np.abs(gy - p.y)))
        ii, jj = np.meshgrid(np.arange(len(gx)), np.arange(len(gy)), indexing="ij")
        d = np.hypot(ii - i, jj - j)
        ring = (d >= 3) & (d <= 6) & np.isfinite(r.H)
        assert np.isfinite(r.H[i, j]), f"{pid} footprint unconstrained"
        # fitted footprint must not sit far below its ring (old dip was ~2.4 m)
        assert r.H[i, j] >= np.nanmean(r.H[ring]) - 1.0, (
            f"{pid} still dips: H_det={r.H[i,j]:.2f} ring={np.nanmean(r.H[ring]):.2f}")
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_hillside_surface.py -q`
Expected: FAIL — `exit_points` still returns 3-tuple (unpack error) and/or the dip gate fails on the current mean estimator.

- [ ] **Step 3: Implement**

In `megido/hillside_surface.py`:

(a) `exit_points` — re-add `dz` to the 4-tuple (compute `dz` list, return it; update the empty-case return to 4 elements). Its docstring lists `dz` as "the sky-frame vertical direction cosine, used for the cosmic-ray flux weight in `fit_surface`".

(b) `fit_surface` — replace the raw-point noise/fit block with the reduction:
```python
    pts, pos_index, dz, sky_flat = exit_points(sol, cfg, a=a)
    if pts.shape[0] == 0:
        raise ValueError("no finite, positive-opacity sky pixels to fit a surface from")

    gx, gy = _footprint_grid(pts[:, :2], cell_m)
    nx, ny = len(gx), len(gy)
    inside = ((pts[:, 0] >= gx[0]) & (pts[:, 0] <= gx[-1])
              & (pts[:, 1] >= gy[0]) & (pts[:, 1] <= gy[-1]))
    X = pts[inside, :2]; z = pts[inside, 2]
    dz_in = dz[inside]; flat_in = sky_flat[inside]; pos_in = pos_index[inside]

    pos_pose = _positions(cfg); pids_sorted = sorted(pos_pose)
    index = float(getattr(sol, "flux_index", 2.0))

    ray_counts = None
    heteroscedastic = False
    if sky_counts is not None:
        ray_counts = np.array([sky_counts[pids_sorted[p]][f]
                               for p, f in zip(pos_in, flat_in)], dtype=float)
        heteroscedastic = True

    Xc, yc, noise_c = _upper_envelope_cells(
        X, z, dz_in, gx, gy, index=index, q_hi=q_hi, min_count=min_count,
        ray_counts=ray_counts)
    if Xc.shape[0] < 3:
        raise ValueError("too few populated cells to fit a surface; lower min_count")

    # cells are already few (<= nx*ny); the max_points guard rarely triggers.
    if Xc.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(Xc.shape[0], max_points, replace=False)
        sel.sort(); Xc, yc, noise_c = Xc[sel], yc[sel], noise_c[sel]

    mean = float(np.average(yc))
    hypers = fit_hyperparams(Xc, yc, noise_c, mean, n_restarts=n_restarts)
    noise_var = noise_c + (hypers.noise_floor * hypers.signal_std) ** 2

    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Xstar = np.stack([GX.ravel(), GY.ravel()], axis=1)
    mstar, sstar = predict(Xc, yc, noise_var, mean, Xstar, hypers)

    covered = sstar < cov_tau * hypers.signal_std
    H = np.where(covered, mstar, np.nan).reshape(nx, ny)
    sigma = np.where(covered, sstar, np.nan).reshape(nx, ny)
    support = np.where(covered, 1.0 - sstar / hypers.signal_std, np.nan).reshape(nx, ny)

    pred_cell, _ = predict(Xc, yc, noise_var, mean, Xc, hypers)
    ve = variance_explained(yc, pred_cell)

    detectors = [{"id": pid, "x": pos_pose[pid].x, "y": pos_pose[pid].y,
                  "z": pos_pose[pid].z} for pid in pids_sorted]
    cx = float(np.mean([d["x"] for d in detectors]))
    cy = float(np.mean([d["y"] for d in detectors]))
    radii = np.sqrt((Xc[:, 0] - cx) ** 2 + (Xc[:, 1] - cy) ** 2)
    coverage_radius_m = float(np.percentile(radii, 95))
    coverage_frac = float(np.mean(covered))
    n_rays = int(inside.sum())

    note = (
        f"Height is on an ASSUMED inverse-density scale a={a:g} (opacity-units per "
        "metre) -- this campaign's 2.2 m baseline does NOT determine it; only the "
        f"fitted SHAPE at that scale is a genuine result. Estimator: per-cell "
        f"flux-weighted (cos^{index:g}) UPPER quantile q_hi={q_hi:g} of exit-z "
        "(tracks the maximum overburden per column; robust to the radial "
        "opacity-sorting artifact that otherwise dips the detector footprints). "
        f"GP: Matern-5/2, length-scale={hypers.length_scale:.2f} m (ML-II), "
        f"{'heteroscedastic (counts)' if heteroscedastic else 'geometry-only (no counts; pass --run)'}. "
        f"sigma is the posterior std. {coverage_frac:.0%} of the grid within "
        f"~{coverage_radius_m:.1f} m of the detector centroid is data-supported; the "
        f"rest is NaN. variance_explained={ve:.3f} over {Xc.shape[0]} cells "
        f"({n_rays} rays). Opacity gauge zero-point not corrected (deferred)."
    )
    return SurfaceResult(H=H, sigma=sigma, support=support, gx=gx, gy=gy, a=a,
                         variance_explained=ve, coverage_frac=coverage_frac,
                         coverage_radius_m=coverage_radius_m, n_rays=n_rays,
                         detectors=detectors, scale_assumed=True, note=note,
                         length_scale_m=hypers.length_scale,
                         signal_std=hypers.signal_std, noise_floor=hypers.noise_floor)
```
Update the `fit_surface` signature to add `q_hi: float = 0.85, min_count: int = 8`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_hillside_surface.py tests/test_hillside_gp.py -q`
Expected: PASS. If the dip gate's `H_det >= ring - 1.0` fails for one detector, FIRST measure `H_det` and `ring` (print), then per the spec ruling raise `q_hi` toward 0.9 AND re-confirm `test_synthetic_hill_recovery_gate` still gives corr > 0.9. Do NOT loosen the gate below `ring - 1.0` and do NOT weaken the synthetic corr threshold. If both cannot hold together, STOP and report the measured numbers (this is the spec's documented conflict case).

- [ ] **Step 5: Commit**

```bash
git add megido/hillside_surface.py tests/test_hillside_surface.py
git commit -m "feat(hillside): fit GP to flux-weighted upper envelope, fix detector dip

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6"
```

---

### Task 3: CLI knobs, meta, real regen, full suite

**Files:**
- Modify: `megido/cli.py` (hillside: `--surface-qhi`, `--surface-min-count`; pass to `fit_surface`; add `q_hi` to meta)
- Modify: `tests/test_cli_hillside_surface.py` (assert `q_hi` in meta; keep fast fit flags)

**Interfaces:**
- Consumes: `fit_surface(..., q_hi=, min_count=)`.

- [ ] **Step 1: Write the failing test**

In `tests/test_cli_hillside_surface.py`, extend the existing heteroscedastic-meta test (which already runs `hillside --run ... --surface-max-points 150 --surface-restarts 1`) with:
```python
    assert "q_hi" in meta
    assert 0.0 < meta["q_hi"] <= 1.0
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_cli_hillside_surface.py -k heteroscedastic -q`
Expected: FAIL (`q_hi` not in meta).

- [ ] **Step 3: Implement**

In `megido/cli.py` hillside argparser (next to `--surface-cell`):
```python
    h.add_argument("--surface-qhi", type=float, default=0.85, dest="surface_qhi",
                   help="upper-quantile level for the per-cell exit-z envelope "
                        "(higher tracks the crown harder; robust to spurious low rays)")
    h.add_argument("--surface-min-count", type=int, default=8, dest="surface_min_count",
                   help="minimum exit points per grid cell to fit a target there")
```
In `_cmd_hillside`, pass them:
```python
    result = fit_surface(sol, cfg, a=args.surface_a, cell_m=args.surface_cell,
                         sky_counts=sky_counts, max_points=args.surface_max_points,
                         n_restarts=args.surface_restarts,
                         q_hi=args.surface_qhi, min_count=args.surface_min_count)
```
Add to the meta dict (next to `heteroscedastic`): `"q_hi": args.surface_qhi,`.

- [ ] **Step 4: Run CLI tests**

Run: `uv run pytest tests/test_cli_hillside_surface.py -q`
Expected: PASS.

- [ ] **Step 5: Regenerate the real surface and record before/after**

Run: `uv run python -m megido.cli hillside --solve runs/solve --run runs/ingest --out runs/voxels`
Expected: exit 0; prints the upper-envelope note. Record VE, coverage, and (via a quick check) the detector-footprint heights vs their rings — confirm the dip is gone (footprint no longer ~2.4 m below ring).

- [ ] **Step 6: Full suite + dead-check**

Run: `grep -rn "dz_in\b" megido/hillside_surface.py` (confirm any leftover is intentional) and `uv run pytest -q` (all pass).

- [ ] **Step 7: Commit**

```bash
git add megido/cli.py tests/test_cli_hillside_surface.py
git commit -m "feat(hillside): CLI --surface-qhi/--surface-min-count + q_hi in meta

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 exit_points dz re-added → Task 2. ✓
- §3.2 per-cell flux-weighted upper-quantile reduction → Task 1 (`_upper_envelope_cells`) + Task 2 (wiring). ✓
- §3.3 GP fit on cells → Task 2. ✓
- §3.4 predict/coverage/honesty/note → Task 2. ✓
- §4.1 interfaces (q_hi, min_count) → Task 2; §4.2 CLI knobs + meta → Task 3. ✓
- §5 tests: weighted-quantile unit (T1.1), dip gate (T2.2), no-regression synthetic/calibration (T2, kept), flux applied (T1.3 upweights vertical), honesty (kept tests) → covered. ✓
- §6 rulings (q_hi tuning vs synthetic; VE secondary) → Task 2 Step 4 instruction. ✓
- §7 gauge deferred → note text (Task 2). ✓

**2. Placeholder scan:** No TBD/TODO. The one conditional (dip-gate threshold tuning) is bounded by an explicit ruling and a hard floor (`ring - 1.0`, corr `> 0.9`), not a placeholder.

**3. Type consistency:** `exit_points` 4-tuple `(pts, pos_index, dz, sky_flat)` consumed only in `fit_surface` and the heteroscedastic test (both updated to 4). `_upper_envelope_cells` returns `(X_cell, y_cell, noise_cell)` consumed in `fit_surface`. `_weighted_quantile(v,w,q)` used only inside `_upper_envelope_cells` + its unit test. `SurfaceResult` fields unchanged (trailing hypers set as before). `variance_explained(z, pred, w=None)` called with cell targets. CLI `surface_qhi`/`surface_min_count` dest names match `fit_surface` kwargs `q_hi`/`min_count`.
