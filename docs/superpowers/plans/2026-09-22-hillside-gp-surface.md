# Gaussian-Process Hillside Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the linear (bilinear + Laplacian) hillside surface fit with a Matérn-5/2 Gaussian process that learns its smoothness (ML-II), weights observations heteroscedastically, and reports honest posterior-std error bars — keeping the `SurfaceResult` interface so the CLI and viewer are unchanged.

**Architecture:** A new numerics module `megido/hillside_gp.py` holds the GP (Matérn-5/2 kernel, negative-log-marginal-likelihood, ML-II hyperparameter fit, posterior predict). `megido/hillside_surface.py`'s `fit_surface` is reworked to assemble exit points + heteroscedastic noise, call the GP, and return the same `SurfaceResult`. The `hillside` CLI gains `--run` to rebuild per-sky-bin counts (Poisson noise), mirroring `reconstruct --bootstrap`.

**Tech Stack:** Python, numpy, scipy (`scipy.linalg.cho_factor/cho_solve`, `scipy.optimize.minimize`). No new dependency. matplotlib stays dev-only for the PNG.

**Spec:** `docs/superpowers/specs/2026-09-22-hillside-gp-surface-design.md`

## Global Constraints

- No new runtime dependency; GP is hand-rolled on numpy + scipy only.
- Units: metres throughout the hillside path (Phase 3 convention). `Pose.x/y/z` are metres.
- `np.nan` means "not constrained" — never 0. Unconstrained grid nodes are NaN.
- Absolute height scale is ASSUMED via `a` (opacity-units per metre); only the fitted SHAPE is data-driven. `SurfaceResult.scale_assumed` stays `True`; the honesty `note` is preserved and updated.
- Feed the GP `BaselineSolution.normalized_opacity(pid)` (via `exit_points`), never raw `.opacity`.
- `topography.csv` is TEST-ONLY synthetic ground truth; it is never read by `megido/` or the CLI deliverable path. `calibration_open_sky.csv` is never read.
- `SurfaceResult` field names/types are unchanged so the viewer (`hill_surface.npy` / `_sigma.npy` / `_meta.json`) needs no change.
- TDD: red before green, every task. A gate must be able to fail for a broken implementation.
- Commit messages end with the two attribution lines (see each Commit step).

---

### Task 1: GP numerics core (`megido/hillside_gp.py`)

**Files:**
- Create: `megido/hillside_gp.py`
- Test: `tests/test_hillside_gp.py`

**Interfaces:**
- Produces:
  - `matern52(X1, X2, length_scale: float, signal_var: float) -> np.ndarray` — `(n1,n2)` kernel matrix; `X1`,`X2` are `(n,2)`.
  - `nll(theta, X, z, noise_base, mean) -> float` — negative log marginal likelihood; `theta = (log ℓ, log σ_f, log η)`; `noise_base` is a per-point `(n,)` array of the count/geometry variance; `η` is the noise-floor fraction, contributing `(η·σ_f)²`.
  - `fit_hyperparams(X, z, noise_base, mean, *, n_restarts=3) -> GPHypers`.
  - `predict(X, z, noise_var, mean, Xstar, hypers) -> tuple[np.ndarray, np.ndarray]` — returns `(mean_star, std_star)`; `noise_var` is the FULL per-point noise `(n,)` (`noise_base + (η σ_f)²`).
  - `GPHypers` frozen dataclass: `length_scale: float, signal_std: float, noise_floor: float, nll: float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hillside_gp.py
import numpy as np
from megido.hillside_gp import matern52, nll, fit_hyperparams, predict, GPHypers


def test_matern52_psd_and_peak():
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, (40, 2))
    K = matern52(X, X, length_scale=2.0, signal_var=1.5)
    assert np.allclose(K, K.T, atol=1e-12)              # symmetric
    assert np.allclose(np.diag(K), 1.5, atol=1e-9)      # k(0)=signal_var
    np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))       # PSD (raises if not)
    # decreasing with distance
    k_near = matern52(np.array([[0.0, 0.0]]), np.array([[0.5, 0.0]]), 2.0, 1.5)[0, 0]
    k_far = matern52(np.array([[0.0, 0.0]]), np.array([[4.0, 0.0]]), 2.0, 1.5)[0, 0]
    assert k_near > k_far


def test_ml2_recovers_length_scale():
    # Data sampled from a smooth function with a known length-scale; ML-II
    # should land near it, NOT return a fixed guess.
    rng = np.random.default_rng(1)
    X = rng.uniform(-10, 10, (200, 2))
    true_ls = 3.0
    # draw a smooth field from a Matern prior at true_ls
    K = matern52(X, X, true_ls, 1.0) + 1e-8 * np.eye(len(X))
    z = np.linalg.cholesky(K) @ rng.standard_normal(len(X))
    noise_base = np.full(len(X), 1e-4)
    h = fit_hyperparams(X, z, noise_base, mean=0.0, n_restarts=3)
    assert 0.4 * true_ls < h.length_scale < 2.5 * true_ls


def test_predict_std_grows_away_from_data():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z = np.array([0.0, 0.0, 0.0, 0.0])
    noise = np.full(4, 1e-6)
    h = GPHypers(length_scale=0.5, signal_std=1.0, noise_floor=0.01, nll=0.0)
    near, std_near = predict(X, z, noise, 0.0, np.array([[0.5, 0.5]]), h)
    far, std_far = predict(X, z, noise, 0.0, np.array([[20.0, 20.0]]), h)
    assert std_far[0] > std_near[0]
    assert abs(std_far[0] - 1.0) < 0.05          # -> prior std (signal_std) far away


def test_predict_interpolates_noise_free_data():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]])
    z = np.array([1.0, 2.0, 2.0, 3.0, 2.0])
    noise = np.full(len(X), 1e-8)
    h = GPHypers(length_scale=1.5, signal_std=2.0, noise_floor=1e-4, nll=0.0)
    m, _ = predict(X, z, noise, float(z.mean()), X, h)
    assert np.allclose(m, z, atol=1e-2)          # near-interpolation at training points
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hillside_gp.py -q`
Expected: FAIL (module `megido.hillside_gp` not found).

- [ ] **Step 3: Implement `megido/hillside_gp.py`**

```python
"""Gaussian-process surface core: Matern-5/2 kernel, ML-II hyperparameters,
heteroscedastic-noise posterior. Hand-rolled on numpy + scipy (no sklearn):
at N ~ 3k the exact GP is a single Cholesky, seconds per fit.

theta parametrisation for the optimiser is log-space:
theta = (log length_scale, log signal_std, log noise_floor_fraction).
Working in logs keeps every hyperparameter positive and scales the
optimisation well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


@dataclass(frozen=True)
class GPHypers:
    length_scale: float
    signal_std: float
    noise_floor: float   # fraction of signal_std added as a homogeneous nugget
    nll: float


def _sqdist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    a2 = np.sum(X1 ** 2, axis=1)[:, None]
    b2 = np.sum(X2 ** 2, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (X1 @ X2.T), 0.0)


def matern52(X1: np.ndarray, X2: np.ndarray, length_scale: float,
             signal_var: float) -> np.ndarray:
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))
    d = np.sqrt(_sqdist(X1, X2))
    s = np.sqrt(5.0) * d / length_scale
    return signal_var * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


def _assemble(theta):
    ls, sf, nf = np.exp(theta)
    return ls, sf, nf


def nll(theta, X, z, noise_base, mean) -> float:
    ls, sf, nf = _assemble(theta)
    n = len(z)
    K = matern52(X, X, ls, sf ** 2)
    K[np.diag_indices_from(K)] += np.asarray(noise_base) + (nf * sf) ** 2
    try:
        c, low = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e25
    r = np.asarray(z, float) - mean
    alpha = cho_solve((c, low), r)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    return float(0.5 * r @ alpha + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi))


def fit_hyperparams(X, z, noise_base, mean, *, n_restarts: int = 3) -> GPHypers:
    X = np.atleast_2d(np.asarray(X, float))
    z = np.asarray(z, float)
    noise_base = np.asarray(noise_base, float)

    span = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0)))
    # nearest-neighbour scale as a lower bound on length_scale
    lo_ls = max(span / len(X) ** 0.5 * 0.25, 1e-3)
    hi_ls = max(span, lo_ls * 10.0)
    zstd = float(np.std(z - mean)) + 1e-9
    bounds = [(np.log(lo_ls), np.log(hi_ls)),
              (np.log(zstd * 1e-2), np.log(zstd * 1e2)),
              (np.log(1e-3), np.log(1.0))]

    rng = np.random.default_rng(0)
    inits_ls = np.geomspace(lo_ls * 1.5, hi_ls * 0.5, n_restarts)
    best = None
    for i in range(n_restarts):
        theta0 = np.log([inits_ls[i], zstd, 0.1])
        res = minimize(nll, theta0, args=(X, z, noise_base, mean),
                       method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    ls, sf, nf = _assemble(best.x)
    return GPHypers(float(ls), float(sf), float(nf), float(best.fun))


def predict(X, z, noise_var, mean, Xstar, hypers: GPHypers):
    X = np.atleast_2d(np.asarray(X, float))
    Xstar = np.atleast_2d(np.asarray(Xstar, float))
    z = np.asarray(z, float)
    sv = hypers.signal_std ** 2
    K = matern52(X, X, hypers.length_scale, sv)
    K[np.diag_indices_from(K)] += np.asarray(noise_var, float)
    c, low = cho_factor(K, lower=True)
    alpha = cho_solve((c, low), z - mean)
    Ks = matern52(Xstar, X, hypers.length_scale, sv)      # (m, n)
    mean_star = mean + Ks @ alpha
    v = cho_solve((c, low), Ks.T)                          # (n, m)
    var_star = sv - np.sum(Ks * v.T, axis=1)
    std_star = np.sqrt(np.maximum(var_star, 0.0))
    return mean_star, std_star
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hillside_gp.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add megido/hillside_gp.py tests/test_hillside_gp.py
git commit -m "feat(hillside-gp): Matern-5/2 GP core with ML-II and heteroscedastic noise

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6"
```

---

### Task 2: Rework `fit_surface` onto the GP (`megido/hillside_surface.py`)

**Files:**
- Modify: `megido/hillside_surface.py` (rework `exit_points` return; rework `fit_surface`; remove `bilinear_matrix`, `laplacian_matrix`, `solve_surface`, `_solve_with_ridge`; keep `_footprint_grid`, `_positions`, `variance_explained`, `SurfaceResult`).
- Modify: `tests/test_hillside_surface.py` (remove linear-solver tests; adapt the synthetic-hill gate to the new signature; add calibration + posterior-std tests).

**Interfaces:**
- Consumes: `megido.hillside_gp.{fit_hyperparams, predict}`.
- Produces:
  - `exit_points(sol, cfg, a=1.0, transparent_quantile=0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning `(pts (N,3), pos_index (N,), dz (N,), sky_flat (N,))` — `dz` is each ray's vertical direction cosine `1/norm`; `sky_flat` is the ray's flat sky-bin index into `sol.sky`.
  - `fit_surface(sol, cfg, *, a=8.0, cell_m=1.0, sky_counts=None, cov_tau=0.9, n_restarts=3) -> SurfaceResult`. `sky_counts: dict[str,np.ndarray] | None` maps position id → per-sky-bin observed counts (Poisson weights); `None` → geometric-leverage fallback.
  - `SurfaceResult` unchanged fields; `support` now carries `1 - sigma/signal_std` (data-influence proxy).

- [ ] **Step 1: Write/adapt the failing tests**

Replace the linear-solver tests (`test_bilinear_matrix_partition_of_unity`, `test_solve_recovers_a_plane`, `test_laplacian_penalises_curvature_not_planes`, `test_bootstrap_sigma_nonzero_and_masked_with_coverage`, `test_solve_stability_with_unsupported_corner_does_not_raise`) with the GP-oriented tests below. KEEP the helpers `_true_hill`, `_hill_term`, `_path_length_to_hill`, `_fake_cfg`, `_FakeSol`, `_build_fake_sol_from_hill` (they generate synthetic ground truth through the real geometry). Update the import block to drop the removed names.

```python
# imports at top become:
from megido.hillside_surface import exit_points, fit_surface, variance_explained


def test_synthetic_hill_recovery_gate():
    """Load-bearing: a known hill sampled through the real two-position sky
    geometry must be recovered in SHAPE by the GP fit at a=1."""
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()
    result = fit_surface(sol, cfg, a=1.0, cell_m=0.5)
    GX, GY = np.meshgrid(result.gx, result.gy, indexing="ij")
    Htrue = _true_hill(GX, GY)
    covered = np.isfinite(result.H)
    assert covered.sum() > 20
    corr = np.corrcoef(result.H[covered], Htrue[covered])[0, 1]
    assert corr > 0.9, f"shape correlation too low: {corr}"


def test_gp_uncertainty_is_calibrated():
    """Honest error bars: ~60-95% of covered truth points fall within +-1 sigma
    of the GP mean. A fit with arbitrarily tiny sigma fails the lower bound."""
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()
    result = fit_surface(sol, cfg, a=1.0, cell_m=0.5)
    GX, GY = np.meshgrid(result.gx, result.gy, indexing="ij")
    Htrue = _true_hill(GX, GY)
    covered = np.isfinite(result.H) & np.isfinite(result.sigma) & (result.sigma > 0)
    within = np.abs(result.H[covered] - Htrue[covered]) <= result.sigma[covered]
    frac = within.mean()
    assert 0.6 <= frac <= 0.98, f"1-sigma coverage {frac:.2f} not calibrated"


def test_unconstrained_nodes_are_nan_not_zero():
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()
    result = fit_surface(sol, cfg, a=1.0, cell_m=0.5)
    # a grid always has corners far from the two-position footprint centre
    assert np.isnan(result.H).any()
    assert not np.any(result.H[np.isnan(result.H)] == 0.0)  # NaN, never 0


def test_shape_invariant_to_assumed_scale():
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()
    r1 = fit_surface(sol, cfg, a=1.0, cell_m=0.5)
    r2 = fit_surface(sol, cfg, a=2.0, cell_m=0.5)
    m = np.isfinite(r1.H) & np.isfinite(r2.H)
    # different assumed scale -> same normalised SHAPE
    c = np.corrcoef((r1.H[m] - r1.H[m].mean()), (r2.H[m] - r2.H[m].mean()))[0, 1]
    assert c > 0.99
    assert r1.scale_assumed is True and "ASSUMED" in r1.note


def test_heteroscedastic_downweights_low_count_bins():
    """With sky_counts, low-count (noisy) rays are trusted less than with a
    flat weighting: the fit at the low-count region moves toward its
    high-count neighbours."""
    sol = _build_fake_sol_from_hill()
    cfg = _fake_cfg()
    _, pos_index, _, sky_flat = exit_points(sol, cfg, a=1.0)
    # fabricate counts: one position's bins all high, other's all low
    counts = {}
    for pid in sorted(set(_positions_ids(cfg))):
        counts[pid] = np.full(sol.sky.flat_size, 500.0)
    # knock down a contiguous chunk of sky bins to few counts
    any_pid = sorted(counts)[0]
    counts[any_pid][:] = 500.0
    lo = np.unique(sky_flat)[: max(1, len(np.unique(sky_flat)) // 5)]
    counts[any_pid][lo] = 3.0
    r_flat = fit_surface(sol, cfg, a=1.0, cell_m=0.5, sky_counts=None)
    r_het = fit_surface(sol, cfg, a=1.0, cell_m=0.5, sky_counts=counts)
    # both produce a covered surface; het must not crash and must change sigma
    cov = np.isfinite(r_het.sigma) & np.isfinite(r_flat.sigma)
    assert cov.sum() > 10
    assert not np.allclose(r_het.sigma[cov], r_flat.sigma[cov])
```

Add a tiny local helper near the test helpers so the heteroscedastic test can list position ids without importing internals:

```python
def _positions_ids(cfg):
    from megido.baseline import position_ids
    return list(position_ids(cfg).values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hillside_surface.py -q`
Expected: FAIL — `exit_points` returns 2 values not 4; `fit_surface` rejects `sky_counts`/`cov_tau`; removed names still imported elsewhere.

- [ ] **Step 3: Rework the implementation**

In `megido/hillside_surface.py`:

1. Replace the scipy.sparse imports/usage. New top imports:
```python
from megido.hillside_gp import fit_hyperparams, predict
```
Remove `import scipy.sparse as sp` / `scipy.sparse.linalg` and the functions `bilinear_matrix`, `laplacian_matrix`, `solve_surface`, `_solve_with_ridge`. Keep `variance_explained` but reimplement it dependency-free (below). Keep `_footprint_grid`, `_positions`, `SurfaceResult`.

2. Extend `exit_points` to also return `dz` and `sky_flat`:
```python
def exit_points(sol, cfg, a: float = 1.0, transparent_quantile: float = 0.05):
    pos_pose = _positions(cfg)
    pids_sorted = sorted(pos_pose)
    centers = sol.sky.centers
    sx_grid, sy_grid = np.meshgrid(centers, centers, indexing="ij")
    sx_flat = sx_grid.ravel()
    sy_flat = sy_grid.ravel()

    pts_list, idx_list, dz_list, flat_list = [], [], [], []
    for k, pid in enumerate(pids_sorted):
        pose = pos_pose[pid]
        lam = sol.normalized_opacity(pid, transparent_quantile=transparent_quantile)
        mask = np.isfinite(lam) & (lam > 0)
        if not mask.any():
            continue
        sx, sy, l = sx_flat[mask], sy_flat[mask], lam[mask]
        norm = np.sqrt(sx ** 2 + sy ** 2 + 1.0)
        dx, dy, dz = sx / norm, sy / norm, 1.0 / norm
        x = pose.x + a * l * dx
        y = pose.y + a * l * dy
        z = pose.z + a * l * dz
        pts_list.append(np.stack([x, y, z], axis=1))
        idx_list.append(np.full(int(mask.sum()), k, dtype=int))
        dz_list.append(dz)
        flat_list.append(np.nonzero(mask)[0])

    if not pts_list:
        z0 = np.zeros((0,), dtype=int)
        return np.zeros((0, 3)), z0, np.zeros((0,)), z0
    return (np.concatenate(pts_list), np.concatenate(idx_list),
            np.concatenate(dz_list), np.concatenate(flat_list))
```

3. Dependency-free `variance_explained` on dense arrays:
```python
def variance_explained(z, pred, w=None) -> float:
    z = np.asarray(z, float); pred = np.asarray(pred, float)
    w = np.ones_like(z) if w is None else np.asarray(w, float)
    resid = pred - z
    num = np.sum(w * resid ** 2)
    zbar = np.sum(w * z) / np.sum(w)
    denom = np.sum(w * (z - zbar) ** 2)
    return float(1.0 - num / denom)
```

4. New `fit_surface`:
```python
def fit_surface(sol, cfg, *, a: float = 8.0, cell_m: float = 1.0,
                sky_counts: dict | None = None, cov_tau: float = 0.9,
                n_restarts: int = 3) -> SurfaceResult:
    pts, pos_index, dz, sky_flat = exit_points(sol, cfg, a=a)
    if pts.shape[0] == 0:
        raise ValueError("no finite, positive-opacity sky pixels to fit a surface from")

    gx, gy = _footprint_grid(pts[:, :2], cell_m)
    nx, ny = len(gx), len(gy)
    inside = ((pts[:, 0] >= gx[0]) & (pts[:, 0] <= gx[-1])
              & (pts[:, 1] >= gy[0]) & (pts[:, 1] <= gy[-1]))
    X = pts[inside, :2]
    z = pts[inside, 2]
    dz_in = dz[inside]
    flat_in = sky_flat[inside]
    pos_in = pos_index[inside]

    pos_pose = _positions(cfg)
    pids_sorted = sorted(pos_pose)

    # heteroscedastic base variance: (a * dz * sigma_lambda)^2
    # sigma_lambda ~ |z - pose.z| / sqrt(N_counts)  (Poisson, relative), with a
    # small relative floor; falls back to pure geometric leverage without counts.
    heights = np.abs(z - np.array([pos_pose[pids_sorted[p]].z for p in pos_in]))
    if sky_counts is not None:
        N = np.array([max(1.0, sky_counts[pids_sorted[p]][f])
                      for p, f in zip(pos_in, flat_in)])
        sigma_lam = heights / np.sqrt(N) + 0.02 * heights
        heteroscedastic = True
    else:
        sigma_lam = 0.05 * heights + 0.02 * np.median(heights)  # geometry only
        heteroscedastic = False
    noise_base = (dz_in * sigma_lam) ** 2 + 1e-6

    mean = float(np.average(z))
    hypers = fit_hyperparams(X, z, noise_base, mean, n_restarts=n_restarts)
    noise_var = noise_base + (hypers.noise_floor * hypers.signal_std) ** 2

    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Xstar = np.stack([GX.ravel(), GY.ravel()], axis=1)
    mstar, sstar = predict(X, z, noise_var, mean, Xstar, hypers)

    # coverage: a node whose posterior std is still ~ the prior std learned
    # nothing from data -> unconstrained -> NaN (never 0).
    covered = sstar < cov_tau * hypers.signal_std
    H = np.where(covered, mstar, np.nan).reshape(nx, ny)
    sigma = np.where(covered, sstar, np.nan).reshape(nx, ny)
    support = np.where(covered, 1.0 - sstar / hypers.signal_std, np.nan).reshape(nx, ny)

    pred_train, _ = predict(X, z, noise_var, mean, X, hypers)
    ve = variance_explained(z, pred_train)

    detectors = [{"id": pid, "x": pos_pose[pid].x, "y": pos_pose[pid].y,
                  "z": pos_pose[pid].z} for pid in pids_sorted]
    cx = float(np.mean([d["x"] for d in detectors]))
    cy = float(np.mean([d["y"] for d in detectors]))
    radii = np.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2)
    coverage_radius_m = float(np.percentile(radii, 95))
    coverage_frac = float(np.mean(covered))

    note = (
        f"Height is on an ASSUMED inverse-density scale a={a:g} (opacity-units "
        "per metre) -- this campaign's 2.2 m baseline does NOT determine it; only "
        f"the fitted SHAPE at that scale is a genuine result. GP: Matern-5/2, "
        f"length-scale={hypers.length_scale:.2f} m (ML-II), "
        f"{'heteroscedastic Poisson+leverage' if heteroscedastic else 'geometry-only (no counts; pass --run)'} "
        f"noise. sigma is the posterior std (a real per-cell error bar). "
        f"{coverage_frac:.0%} of the grid within ~{coverage_radius_m:.1f} m of the "
        f"detector centroid is data-supported; the rest is NaN (unconstrained). "
        f"variance_explained={ve:.3f} over {len(z)} rays."
    )
    return SurfaceResult(H=H, sigma=sigma, support=support, gx=gx, gy=gy, a=a,
                         variance_explained=ve, coverage_frac=coverage_frac,
                         coverage_radius_m=coverage_radius_m, n_rays=int(len(z)),
                         detectors=detectors, scale_assumed=True, note=note)
```

Note the `_FakeSol` test stub must expose `sol.sky.flat_size`; `make_sky_grid()` already provides `flat_size`. If `_FakeSol` wraps the real sky grid (it does, via `make_sky_grid`), no change is needed.

- [ ] **Step 4: Run the hillside tests**

Run: `uv run pytest tests/test_hillside_surface.py tests/test_hillside_gp.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add megido/hillside_surface.py tests/test_hillside_surface.py
git commit -m "feat(hillside): fit surface with the Matern-5/2 GP, retire linear solver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6"
```

---

### Task 3: Per-sky-bin counts helper + `hillside --run` wiring

**Files:**
- Modify: `megido/baseline.py` (extract a reusable `position_sky_counts(grid, cfg)` from `_build_terms`'s summation logic).
- Modify: `megido/cli.py` (`_cmd_hillside`: add `--run`; rebuild counts; pass `sky_counts`; add hypers + `heteroscedastic` to `hill_surface_meta.json`).
- Test: `tests/test_baseline.py` (add `position_sky_counts` test), `tests/test_cli_hillside_surface.py` (heteroscedastic meta + fallback note).

**Interfaces:**
- Consumes: `megido.hillside_surface.fit_surface(..., sky_counts=...)`; `megido.angular.load_analysis_grid`.
- Produces: `megido.baseline.position_sky_counts(grid, cfg) -> dict[str, np.ndarray]` — per-position `(sky.flat_size,)` summed live observed counts, using the same live mask + `MIN_SKY_COUNTS` threshold as `_build_terms`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baseline.py  (add)
def test_position_sky_counts_matches_build_terms():
    from megido.baseline import position_sky_counts, _build_terms, position_ids
    from megido.config import load_site_config
    from megido.detector import DetectorGeometry
    from megido.sky import make_sky_grid
    from megido.basis import make_smooth_basis
    from tests.simhelp import make_scene_grid   # existing synthetic grid helper
    cfg = load_site_config("configs/megido.yaml")
    grid = make_scene_grid(cfg)                  # AnalysisGrid with counts
    sky = make_sky_grid(); basis = make_smooth_basis()
    psc = position_sky_counts(grid, cfg)
    # totals per position equal the live-summed counts _build_terms would use
    terms = _build_terms(grid, cfg, DetectorGeometry.megiddo(), sky, basis)
    for pid in psc:
        assert psc[pid].shape[0] == sky.flat_size
        assert psc[pid].sum() > 0
```

If no `tests/simhelp.py`/`make_scene_grid` exists, build the grid inline in the test with `megido.simbaseline.make_synthetic_scene(cfg).grid` (that returns an `AnalysisGrid` with `counts`).

```python
# tests/test_cli_hillside_surface.py  (add)
def test_hillside_run_produces_heteroscedastic_meta(tmp_path):
    # after running solve into a dir and ingest available, `hillside --run`
    # writes meta with heteroscedastic: true and GP hypers.
    ...  # follow the existing fixture pattern in this file for solve/ingest dirs
```
Follow whatever fixture the file already uses to produce a `runs/solve` and ingest dir; assert `json.loads(meta)["heteroscedastic"] is True` and `"length_scale" in meta`. Add a companion assertion that omitting `--run` yields `heteroscedastic: false` and a note containing `--run`.

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_baseline.py -k position_sky_counts tests/test_cli_hillside_surface.py -q`
Expected: FAIL (`position_sky_counts` undefined; CLI has no `--run`).

- [ ] **Step 3: Implement**

In `megido/baseline.py`, factor the summation out of `_build_terms` and have `_build_terms` call it (DRY — do not duplicate the mask logic):
```python
def position_sky_counts(grid, cfg) -> dict:
    from megido.acceptance import geometric_acceptance
    from megido.detector import DetectorGeometry
    from megido.sky import make_sky_grid, detector_to_sky
    geom = DetectorGeometry.megiddo()
    sky = make_sky_grid()
    tx, ty = grid.tan_mesh()
    acc = geometric_acceptance(tx, ty, geom)
    positions = position_ids(cfg)
    out: dict[str, np.ndarray] = {}
    for eid, counts in grid.counts.items():
        exp = cfg.exposure(eid)
        sx, sy, on_sky = detector_to_sky(tx, ty, exp.pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acc > 0) & on_sky & in_grid
        acc_arr = out.setdefault(positions[eid], np.zeros(sky.flat_size))
        np.add.at(acc_arr, flat[live], counts[live].astype(np.float64))
    return out
```
(If `_build_terms` can be refactored to call this without changing its behaviour, do so; otherwise leave `_build_terms` as-is and keep this as the public helper — verify equality via the Step-1 test.)

In `megido/cli.py` `_cmd_hillside` (the surface-writing branch around lines 369-372) and the argparser (around 506-510):
```python
# argparser for the hillside subcommand:
h.add_argument("--run", default="runs/ingest",
               help="ingest dir; enables heteroscedastic Poisson weights")
h.add_argument("--rebin", type=int, default=10,
               help="angular rebin factor (match the solve that made the baseline)")
```
```python
# in _cmd_hillside, before fit_surface:
from megido.baseline import position_sky_counts
from megido.angular import load_analysis_grid
sky_counts = None
run_dir = Path(args.run)
ids = [e.id for e in cfg.exposures]
if run_dir.is_dir() and all((run_dir / f"counts_{e}.npz").exists() for e in ids):
    grid = load_analysis_grid(run_dir, ids, factor=args.rebin)
    sky_counts = position_sky_counts(grid, cfg)
else:
    print(f"no ingest counts under {run_dir}; heteroscedastic weights off "
          "(geometry-only). Pass --run <ingest dir> for Poisson weighting.")
result = fit_surface(sol, cfg, a=args.surface_a, cell_m=args.surface_cell,
                     sky_counts=sky_counts)
```
Extend the meta JSON dict (where `cell_m`/`a` are written) with:
```python
"heteroscedastic": sky_counts is not None,
"length_scale_m": None,   # filled below from note-free source
```
Since `SurfaceResult` does not carry the hypers as fields, expose them minimally: add three optional fields to `SurfaceResult` with defaults so nothing else breaks — `length_scale_m: float = float("nan")`, `signal_std: float = float("nan")`, `noise_floor: float = float("nan")` — set them in `fit_surface`, and write them into the meta. (Adding trailing dataclass fields with defaults is backward-compatible.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_baseline.py tests/test_cli_hillside_surface.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add megido/baseline.py megido/cli.py megido/hillside_surface.py tests/test_baseline.py tests/test_cli_hillside_surface.py
git commit -m "feat(hillside): --run rebuilds per-sky-bin counts for heteroscedastic GP noise

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6"
```

---

### Task 4: End-to-end real-data run, dead-code sweep, full suite

**Files:**
- Modify: `tests/test_hillside.py` (only if it imported any removed linear function — update or leave).
- Verify: no remaining import of `bilinear_matrix`/`laplacian_matrix`/`solve_surface`/`_solve_with_ridge` anywhere.

- [ ] **Step 1: Grep for dead references**

Run: `grep -rn "bilinear_matrix\|laplacian_matrix\|solve_surface\|_solve_with_ridge" megido/ tests/`
Expected: no hits in `megido/`; any test hit must be updated/removed. Fix any that remain.

- [ ] **Step 2: Regenerate the real hillside surface**

Run: `uv run python -m megido.cli hillside --solve runs/solve --run runs/ingest --out runs/voxels`
Expected: exits 0; prints the GP note with a learned length-scale and `heteroscedastic` weighting; writes `hill_surface.npy`, `hill_surface_sigma.npy`, `hill_surface_meta.json`, `hill_surface.png`. Record the printed `variance_explained` and `length_scale`.

- [ ] **Step 3: Real-data smoke assertion (controller ruling)**

Confirm VE is finite and the covered region is non-trivial. If VE ≥ ~0.40 (linear baseline), note the improvement. If VE is modestly below, record a ledger ruling: the calibration gate (Task 2) is the primary quality bar and better-calibrated uncertainty is the intended trade — decided against the real number, not assumed.

- [ ] **Step 4: Full suite**

Run: `uv run pytest -q`
Expected: all pass. Then the viewer suite is unaffected (no viewer files changed), but run it if the repo's runner is quick: `cd viewer && <the node test command>`; expected unchanged pass counts.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(hillside): regenerate GP surface for runs/voxels, remove dead linear refs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 observations → Task 2 `exit_points` (extended). ✓
- §3.2 Matérn-5/2 kernel → Task 1 `matern52`. ✓
- §3.3 heteroscedastic noise (Poisson + leverage, fallback) → Task 2 `fit_surface` noise assembly + Task 3 counts. ✓
- §3.4 ML-II → Task 1 `fit_hyperparams`. ✓
- §3.5 prediction (mean, std) → Task 1 `predict`. ✓
- §3.6 coverage via posterior std → Task 2 `cov_tau` masking. ✓
- §4.1 module interface → Task 1. ✓  §4.2 fit_surface interface → Task 2. ✓  §4.3 CLI `--run` + meta → Task 3. ✓  §4.4 viewer unchanged → Task 4 verify. ✓
- §5 counts rebuild → Task 3 `position_sky_counts`. ✓
- §6 tests 1-8 → Task 1 (1,4-partial), Task 2 (gate 7, calibration, coverage, invariants, hetero), Task 4 (real smoke 8). ✓
- §7 rulings → Task 4 Step 3. ✓

**2. Placeholder scan:** The Task-3 CLI test body is described-not-coded because it must follow the file's existing solve/ingest fixture, which the implementer reads in place; the assertions are stated exactly (`heteroscedastic is True`, `length_scale` present, fallback note contains `--run`). All algorithmic code is complete.

**3. Type consistency:** `exit_points` 4-tuple return is consumed only in Task 2 `fit_surface` and the Task-2 hetero test (which unpacks 4). `SurfaceResult` gains trailing default fields only (`length_scale_m`, `signal_std`, `noise_floor`) — backward-compatible; viewer reads npy/meta, not the dataclass. `variance_explained` signature changed to `(z, pred, w=None)` and is called only inside `fit_surface` — no external caller. GP names (`fit_hyperparams`, `predict`, `GPHypers`, `matern52`, `nll`) consistent across Tasks 1-2.
