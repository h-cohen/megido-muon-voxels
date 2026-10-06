# Voxel Averaging Kernels Implementation Plan

**STOPPED after Task 4 (user decision, 2026-10-06): Task 5's gates failed — see the spec §10. Tasks 5-8 were not executed. Task 4's gate-1 geometry was changed by controller ruling R3 (9 detectors at 3 m pitch; the 25-detector geometry put the probe in view of one detector), and delta_sigma is 10 (ruling R4).**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the point-spread function (PSF) of the delivered voxel solver at a lattice of probe voxels, so that every reported voxel carries four honest kernel metrics. These make the claim "x̂ ± σ estimates the kernel-blurred truth" testable, and a count-level coverage gate tests it.

**Architecture:** A new pure module, `megido/kernels.py`, perturbs the measured opacity by `δ·A·e_j` for batches of well-separated probe voxels. It re-solves with the production solver, stopped at the nominal solve's best iterate, and turns each probe's response into metrics that are interpolated to the voxel grid. The CLI writes them as four viewer layers. The bootstrap is fixed so its replicas fit the same σ-weighted estimator that is delivered. The viewer gains a depth-spread gate and kernel readouts.

**Tech Stack:** Python 3.12, numpy, scipy.sparse, pytest; viewer: vanilla ES modules + WebGL2 GLSL, `node --test`, Playwright (pytest-playwright).

**Spec:** `docs/superpowers/specs/2026-10-06-voxel-averaging-kernels-design.md`

## Global Constraints

- Units: Phase 3 is metres everywhere in this plan.
- NaN means "not characterised / not constrained", never 0.
- `INVERSION_VERSION` goes 2 → 3 (`megido/raycast.py`).
- Tunables live in the `kernels:` YAML block, never hardcoded: `spacing_m` 1.0, `sep_m` 3.0, `z_levels_m` (default: every `spacing_m` from `z_min + spacing_m/2`), `delta_sigma` 10.0.
- Gate thresholds are fixed now and are never loosened to get a pass: well-resolved mass ≥ 0.8, spread ≤ 2 voxels, |shift| ≤ 1 voxel; linearity relative L2 ≤ 0.2; coverage of `R·t` within **[0.58, 0.78]**. If a gate fails, find out which side is wrong and report it.
- `.npy` volumes are numpy C-order `(nx, ny, nz)`. Every new 3D viewer texture goes through `makeVolumeTexture` (which applies `reorderForTexture`).
- Never run two pytest sessions at once. Viewer browser tests rebuild `viewer/dist/index.html`.
- Commit identity is repo-local; end every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DZtbXLj4bU1djEj1oH6HNs
  ```

## Review Focus

1. **A probe whose rows all have weight 0** (holdout masks or an excluded direction). It must be skipped, not given δ = NaN. Pinned in Task 4 (`test_probe_with_no_weighted_rows_is_skipped`).
2. **Probes at the grid edge**, where the lateral cell runs off the grid. The cell must clip to the grid and metrics must still be finite. Pinned in Task 4 (`test_edge_probe_cell_is_clipped_not_wrapped`).
3. **A zero response** (best_iter 0, or a probe whose bump the solver ignores). Metrics are NaN, not divide-by-zero. Pinned in Task 4 (`test_zero_response_gives_nan_metrics_not_errors`).
4. **Stale cache after a config change.** Changing any `kernels:` value or `delta_sigma` must change the cache key. Pinned in Task 6 (`test_cache_key_changes_with_every_input`).
5. **A viewer run without kernel layers.** The depth gate stays inert and disabled, with no hidden voxels. Pinned in Task 7 (node test `depth gate inert without psf_dz`).

---

## File Structure

| File | Responsibility |
|---|---|
| `megido/kernels.py` (new) | probe lattice/batches, perturbation PSFs, metrics, interpolation, `KernelResult`, cache key |
| `megido/inversion.py` | `best_iter` in `info`; `stop_at` returns a given iterate |
| `megido/voxuncert.py` | `voxel_bootstrap(sigma=)` |
| `megido/config.py` + `configs/*.yaml` | `KernelConfig`, `kernels:` block |
| `megido/raycast.py` | `INVERSION_VERSION = 3` |
| `megido/cli.py` | `reconstruct --kernels`, pass σ to the bootstrap |
| `megido/volexport.py` | export the four `psf_*` layers |
| `viewer/src/{gates,model,layers,picker,app}.mjs`, `viewer/shell.html` | depth-spread gate, labels, hover readout, SNR relabel |
| `tests/test_kernels.py` (new) | gates 1–5 |
| `tests/test_kernels_calibration.py` (new) | gate 6 |
| `docs/…`, `CLAUDE.md` | Task 8 |

---

### Task 1: Solver iterate hook (`best_iter`, `stop_at`) and version bump

**Files:**
- Modify: `megido/inversion.py` (`sirt`, `sirt_tv`, `solve`)
- Modify: `megido/raycast.py:29` (`INVERSION_VERSION = 3`)
- Test: `tests/test_inversion.py`

**Interfaces:**
- Produces: `sirt(fwd, data, rc, *, fit_offsets=True, stop_at: int | None = None)`, the same for `sirt_tv` and `solve`. `info["best_iter"]: int` is the number of x-updates applied to the returned iterate. With `stop_at=k`, the solver returns the iterate after exactly `k` updates, ignoring `chi2_target` and best-χ² selection. `solve(..., stop_at=info["best_iter"])` reproduces the nominal x bit for bit.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_inversion.py`)

```python
@pytest.mark.parametrize("algorithm", ["sirt", "tv"])
def test_stop_at_best_iter_reproduces_the_nominal_solution_bit_for_bit(algorithm):
    fwd, truth, data = _toy()
    noisy = FitData(lam=data.lam + np.random.default_rng(3).normal(0, 0.05, data.lam.size),
                    w=np.full(data.lam.size, 1 / 0.05**2), rows=data.rows)
    rc = Reconstruction(algorithm=algorithm, n_iter=60, tv_alpha=0.02, chi2_target=1.0)
    x, info = solve(fwd, noisy, rc)
    assert 0 <= info["best_iter"] <= rc.n_iter
    x2, info2 = solve(fwd, noisy, rc, stop_at=info["best_iter"])
    np.testing.assert_array_equal(x2, x)
    assert info2["offsets"] == info["offsets"]
    assert info2["best_iter"] == info["best_iter"]


@pytest.mark.parametrize("algorithm", ["sirt", "tv"])
def test_stop_at_k_applies_exactly_k_updates(algorithm):
    fwd, truth, data = _toy()
    rc = Reconstruction(algorithm=algorithm, n_iter=50, tv_alpha=0.0, chi2_target=1e-12)
    x0, i0 = solve(fwd, data, rc, stop_at=0)
    assert i0["best_iter"] == 0
    assert np.all(x0 == 0.0)                       # no update applied
    x5, _ = solve(fwd, data, rc, stop_at=5)
    x6, _ = solve(fwd, data, rc, stop_at=6)
    assert np.linalg.norm(x6 - x5) > 0             # one more update moved it


def test_stop_at_ignores_the_discrepancy_target():
    fwd, truth, data = _toy()
    rc = Reconstruction(algorithm="sirt", n_iter=200, chi2_target=1e30)  # would stop at k=0
    x, info = solve(fwd, data, rc)
    assert info["best_iter"] == 0
    x7, info7 = solve(fwd, data, rc, stop_at=7)
    assert info7["best_iter"] == 7 and np.linalg.norm(x7) > 0
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_inversion.py -q -k "stop_at"`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'stop_at'`

- [ ] **Step 3: Implement** in `megido/inversion.py`

Replace `sirt` with:

```python
def sirt(fwd: ForwardModel, data: FitData, rc: Reconstruction, *,
         fit_offsets: bool = True, stop_at: int | None = None) -> tuple[np.ndarray, dict]:
    """Weighted SIRT with nonnegativity and per-position offset refinement.

    Stops at rc.chi2_target by the discrepancy principle: fitting past the noise
    floor is how SIRT turns counting statistics into structure.

    `stop_at=k` returns the iterate after exactly k updates and ignores the
    discrepancy stop. Kernel probes (megido.kernels) use it to re-solve
    perturbed data along the nominal solve's path: a perturbation must not
    change WHICH iterate is delivered, or the point-spread function would jump.
    `info["best_iter"]` is the number of updates applied to the returned x.
    """
    A, lam, w = fwd.A, data.lam, data.w
    n_pos = len(fwd.rows.position_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_pos)
    row_inv, col_inv = _scalings(A, w)
    damp = _coverage_damping(col_inv, rc.coverage_damping)
    n_used = max(int(np.count_nonzero(w)), 1)

    history: list[float] = []
    chi2 = float("inf")
    n_updates = 0
    for k in range(rc.n_iter):
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        if fit_offsets:
            c = c + _update_offsets(resid, w, data.rows.pos_of_row, n_pos)
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if k % 20 == 0:
            history.append(chi2)
        if stop_at is not None:
            if k == stop_at:
                break
        elif chi2 <= rc.chi2_target:
            break
        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
        if damp is not None:
            x /= damp
        n_updates = k + 1
    history.append(chi2)
    return x, {"offsets": _named(c, fwd.rows.position_ids),
               "chi2_history": history,
               "best_chi2": float(min(history)),
               "n_iter_used": n_updates,
               "best_iter": n_updates}
```

Replace `sirt_tv` with:

```python
def sirt_tv(fwd: ForwardModel, data: FitData, rc: Reconstruction, *,
            fit_offsets: bool = True, stop_at: int | None = None) -> tuple[np.ndarray, dict]:
    """SIRT with a per-iteration anisotropic-TV proximal (denoising) step.

    tv_alpha is a fraction of the reconstructed scale (x's p95), so it transfers
    across datasets instead of needing a retune per campaign. Runs the full
    budget and returns the best-chi2 iterate: the TV step keeps it from
    overfitting the way plain SIRT does, so there is no discrepancy stop.

    `info["best_iter"]` = k means the returned x is the iterate after k updates.
    `stop_at=k` returns exactly that iterate (with its offsets) instead of the
    best-chi2 one; see `sirt` for why the kernel probes need it.
    """
    A, lam, w = fwd.A, data.lam, data.w
    shape = fwd.grid.shape
    n_pos = len(fwd.rows.position_ids)
    x = np.zeros(A.shape[1])
    c = np.zeros(n_pos)
    row_inv, col_inv = _scalings(A, w)
    damp = _coverage_damping(col_inv, rc.coverage_damping)
    n_used = max(int(np.count_nonzero(w)), 1)
    dual = np.zeros((3,) + shape)

    history: list[float] = []
    best = (float("inf"), x.copy(), c.copy(), 0)
    for k in range(rc.n_iter):
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        if fit_offsets:
            c = c + _update_offsets(resid, w, data.rows.pos_of_row, n_pos)
        resid = lam - (A @ x + c[data.rows.pos_of_row])
        chi2 = float(np.sum(w * resid**2) / n_used)
        if stop_at is not None and k == stop_at:
            return x, {"offsets": _named(c, fwd.rows.position_ids),
                       "chi2_history": history + [chi2], "best_chi2": chi2,
                       "n_iter_used": k, "best_iter": k}
        if chi2 < best[0]:
            best = (chi2, x.copy(), c.copy(), k)
        if k % 20 == 0:
            history.append(chi2)

        x = x + col_inv * (A.T @ (w * resid * row_inv))
        if rc.nonneg:
            np.maximum(x, 0.0, out=x)
        if damp is not None:
            x /= damp
        scale = float(np.percentile(x[x > 0], 95)) if (x > 0).any() else 0.0
        gamma = rc.tv_alpha * max(scale, 1e-9)
        x = _prox_tv(x.reshape(shape), gamma, rc.tv_z_weight, dual).ravel()

    history.append(best[0])
    return best[1], {"offsets": _named(best[2], fwd.rows.position_ids),
                     "chi2_history": history,
                     "best_chi2": float(best[0]),
                     "n_iter_used": rc.n_iter,
                     "best_iter": int(best[3])}
```

Replace `solve` with:

```python
def solve(fwd: ForwardModel, data: FitData, rc: Reconstruction, *,
          fit_offsets: bool = True, stop_at: int | None = None) -> tuple[np.ndarray, dict]:
    """`fit_offsets=False` holds every c_p at 0: only for a MEASURED opacity
    gauge (BaselineSolution.absolute). Fitting c_p against a measured level
    re-opens the degeneracy that moves a flat overburden into the offset and
    its oblique excess into the outer shell of the volume.
    `stop_at`: see `sirt` (kernel probes only)."""
    if rc.algorithm not in SOLVERS:
        raise ValueError(f"unknown algorithm {rc.algorithm!r}; have {sorted(SOLVERS)}")
    return SOLVERS[rc.algorithm](fwd, data, rc, fit_offsets=fit_offsets, stop_at=stop_at)
```

In `megido/raycast.py` change line 29 to `INVERSION_VERSION = 3`, and add a one-line comment above it: `# 3: info.best_iter / stop_at (kernels); bootstrap replicas use the delivered sigma weights.`

- [ ] **Step 4: Run the inversion tests plus the whole suite (the version bump changes cache keys)**

Run: `uv run pytest tests/test_inversion.py -q` and then `uv run pytest -q --ignore=tests/viewer`
Expected: all PASS. If a test pins `INVERSION_VERSION == 2` or `n_iter_used` semantics, update its expectation **only** if it pins the version number itself. Any other failure is a defect to report.

- [ ] **Step 5: Commit**

```bash
git add megido/inversion.py megido/raycast.py tests/test_inversion.py
git commit -m "feat(inversion): best_iter in info and stop_at hook; INVERSION_VERSION 3"
```

---

### Task 2: Bootstrap replicas use the delivered σ weights

**Files:**
- Modify: `megido/voxuncert.py` (`voxel_bootstrap`)
- Modify: `megido/cli.py` (the `voxel_bootstrap(...)` call in `_cmd_reconstruct`)
- Test: `tests/test_voxuncert.py`

**Interfaces:**
- Produces: `voxel_bootstrap(grid, cfg, *, n_replicas=8, seed=0, cache_dir=..., solve_kwargs=None, solver=solve_baseline, sigma: dict[str, np.ndarray] | None = None) -> BootstrapResult`. `sigma` is forwarded to every replica's `solve_voxels` (and to the lattice-pinning nominal solve).

- [ ] **Step 1: Write the failing test** (append to `tests/test_voxuncert.py`)

```python
def test_replicas_fit_with_the_same_sigma_weights_as_the_delivered_volume(tmp_path, monkeypatch):
    """Gate 7: sigma must describe the DELIVERED (sigma-weighted) estimator."""
    import megido.voxuncert as vu
    cfg = _cfg(tmp_path)
    seen = []
    real = vu.solve_voxels

    def spy(sol, cfg_, **kw):
        seen.append(kw.get("sigma"))
        return real(sol, cfg_, **kw)

    monkeypatch.setattr(vu, "solve_voxels", spy)
    sigma = {"pos0": np.full(10_000, 0.05), "pos1": np.full(10_000, 0.05)}
    vu.voxel_bootstrap(_grid(), cfg, n_replicas=2, cache_dir=None,
                       solve_kwargs={"n_iter": 5}, sigma=sigma)
    assert len(seen) == 3                      # nominal + 2 replicas
    assert all(s is sigma for s in seen)
```

(Sky grids have `n_bins**2` = 10,000 flat bins by default (`make_sky_grid`, 100 bins). If `build_fit_data` indexes σ by sky bin and that size is wrong, read `sol.sky.flat_size` from a nominal `solve_baseline(_grid(), cfg)` and size σ from it. Don't change the assertion.)

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/test_voxuncert.py -q -k same_sigma`
Expected: FAIL with `TypeError: voxel_bootstrap() got an unexpected keyword argument 'sigma'`

- [ ] **Step 3: Implement**

In `megido/voxuncert.py`, add `sigma: dict[str, np.ndarray] | None = None` to the `voxel_bootstrap` signature, after `solver=solve_baseline`. Append this paragraph to the docstring:

```
    `sigma` is the per-sky-bin weighting the DELIVERED volume was fitted with.
    Every replica is fitted with the same weights, so the spread describes the
    estimator that is shown. Replicas fitted unweighted described a different,
    noisier estimator (~1.4x the weighted sigma on the cafeteria; spike
    spike/sigma-calibration). The weights are held fixed, not re-bootstrapped:
    they define the estimator, they are not part of the noise.
```

Change the two `solve_voxels(...)` calls to pass `sigma=sigma`:

```python
    vgrid: VoxelGrid = solve_voxels(nominal_sol, cfg, sigma=sigma, cache_dir=cache_dir,
                                    holdouts=False)["full"].grid
    ...
        fits = solve_voxels(sol, cfg, sigma=sigma, cache_dir=cache_dir, holdouts=False,
                            grid=vgrid)
```

In `megido/cli.py`, `_cmd_reconstruct`, pass the σ the delivered fit used:

```python
        boot = voxel_bootstrap(counts_grid, cfg, n_replicas=args.bootstrap,
                               cache_dir=args.cache,
                               solve_kwargs={"n_iter": args.iters},
                               solver=solver, sigma=sigma)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_voxuncert.py tests/test_cli.py tests/test_cli2.py tests/test_cli3.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add megido/voxuncert.py megido/cli.py tests/test_voxuncert.py
git commit -m "fix(voxuncert): bootstrap replicas use the delivered sigma weights"
```

---

### Task 3: `KernelConfig` and the `kernels:` YAML blocks

**Files:**
- Modify: `megido/config.py`
- Modify: `configs/megido.yaml`, `configs/cafeteria.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `KernelConfig(spacing_m: float = 1.0, sep_m: float = 3.0, z_levels_m: tuple[float, ...] | None = None, delta_sigma: float = 10.0)`, frozen. `SiteConfig.kernels: KernelConfig` (default `KernelConfig()`). The loader reads `raw["kernels"]` and turns `z_levels_m` into a tuple of floats.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_config.py`)

```python
def test_kernels_block_defaults_and_overrides(tmp_path):
    from megido.config import KernelConfig, load_site_config
    base = ("site: t\ndata_dir: /tmp\nexposures:\n  - id: P0\n    runs: DET1-DET2\n"
            "    pose: {x: 0, y: 0, z: 0, tilt_deg: 0, az_deg: 0}\n")
    p = tmp_path / "a.yaml"; p.write_text(base)
    assert load_site_config(p).kernels == KernelConfig()
    q = tmp_path / "b.yaml"
    q.write_text(base + "kernels: {spacing_m: 2.0, sep_m: 4.0, z_levels_m: [3, 5.5], delta_sigma: 0.5}\n")
    k = load_site_config(q).kernels
    assert k == KernelConfig(spacing_m=2.0, sep_m=4.0, z_levels_m=(3.0, 5.5), delta_sigma=0.5)


def test_both_site_configs_carry_a_kernels_block():
    from megido.config import load_site_config
    for path in ("configs/megido.yaml", "configs/cafeteria.yaml"):
        k = load_site_config(path).kernels
        assert k.sep_m >= 2 * k.spacing_m        # batches need >= 2x2 offsets per level
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_config.py -q -k kernels`
Expected: FAIL with `ImportError: cannot import name 'KernelConfig'`

- [ ] **Step 3: Implement**

In `megido/config.py`, after `Reconstruction`, add:

```python
@dataclass(frozen=True)
class KernelConfig:
    """Point-spread probes of the delivered solver (megido.kernels).

    spacing_m   probe lattice pitch, laterally and between z levels
    sep_m       lateral spacing of probes that share ONE perturbed solve; must
                exceed the PSF's lateral reach, or neighbouring probes' responses
                mix (the linearity gate in tests/test_kernels.py measures this)
    z_levels_m  probe heights; None -> every spacing_m from z_min + spacing_m/2
    delta_sigma probe amplitude in units of the median opacity sigma of the rows
                crossing the probe voxel: big enough to dominate round-off, small
                enough to stay in the solver's linear regime
    """
    spacing_m: float = 1.0
    sep_m: float = 3.0
    z_levels_m: tuple | None = None
    delta_sigma: float = 10.0
```

Add `kernels: KernelConfig = field(default_factory=KernelConfig)` to `SiteConfig`, after `detector`. In `load_site_config`, before the `return`:

```python
    ker_raw = dict(raw.get("kernels", {}))
    if ker_raw.get("z_levels_m") is not None:
        ker_raw["z_levels_m"] = tuple(float(v) for v in ker_raw["z_levels_m"])
    kernels = KernelConfig(**ker_raw)
```

and pass `kernels=kernels` to `SiteConfig(...)`.

Append to `configs/megido.yaml`:

```yaml
# Point-spread probes of the delivered solver (`reconstruct --kernels`,
# megido/kernels.py, spec 2026-10-06-voxel-averaging-kernels-design.md).
#   spacing_m 1.0: four voxels at 0.25 m; finer only multiplies solves (~1 h here).
#   sep_m 3.0: probes 3 m apart share a solve. The lateral PSF reach measured on
#     the cafeteria spike is ~1-2 m; the linearity gate catches it if not.
#   delta_sigma 10.0: bump = ten median noise-sigmas of opacity through one voxel; the TV solver is amplitude-dependent and a 1-sigma bump gives an erratic PSF.
kernels:
  spacing_m: 1.0
  sep_m: 3.0
  delta_sigma: 10.0
```

Append the same block to `configs/cafeteria.yaml`, with the comment's first bullet reading `spacing_m 1.0: five voxels at 0.20 m; ~50 solves (~12 min).`

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add megido/config.py configs/megido.yaml configs/cafeteria.yaml tests/test_config.py
git commit -m "feat(config): kernels block for point-spread probes"
```

---

### Task 4: `megido/kernels.py` core: lattice, PSFs, metrics (+ gate 1, gate 5)

**Files:**
- Create: `megido/kernels.py`
- Test: `tests/test_kernels.py`

**Interfaces:**
- Consumes: `solve(..., stop_at=)` and `info["best_iter"]` (Task 1); `KernelConfig` (Task 3); `FitData`, `ForwardModel`, `VoxelGrid`.
- Produces:
  - `probe_batches(grid: VoxelGrid, rays3: np.ndarray, kc: KernelConfig) -> list[np.ndarray]`: each element is an `[n, 3]` int array of voxel indices `(i, j, k)`.
  - `probe_deltas(fwd, data, probes: np.ndarray, kc) -> np.ndarray`: `[n]` amplitudes, NaN where no weighted row crosses the probe.
  - `@dataclass(frozen=True) class PSF: probe: tuple[int, int, int]; lo: tuple[int, int]; values: np.ndarray` (`values[a, b, :]` is the response at voxel `(lo[0]+a, lo[1]+b, :)` per unit probe density).
  - `point_spreads(fwd, data, rc, x_hat, best_iter, probes, deltas, *, fit_offsets, cell_half: int) -> list[PSF]`
  - `cell_half_width(grid, kc) -> int`
  - `psf_metrics(psf: PSF, grid: VoxelGrid) -> dict[str, float]` with keys `psf_mass`, `psf_dz`, `psf_shift`, `psf_lat` (NaN when Σ|PSF| = 0).
  - `METRICS = ("psf_mass", "psf_dz", "psf_shift", "psf_lat")`
  - Test helpers in `tests/test_kernels.py`: `_megiddo_problem(tmp_path, positions=2)` and `_generous_problem(tmp_path)`, each returning `(fwd, data, rc, x_hat, info, rays3)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_kernels.py`)

```python
"""Point-spread probes of the delivered solver (spec 2026-10-06, section 4)."""
import dataclasses

import numpy as np
import pytest

from megido.config import KernelConfig, Reconstruction, load_site_config
from megido.fitdata import FitData
from megido.forward import build_forward_model
from megido.inversion import solve
from megido.kernels import (PSF, cell_half_width, point_spreads, probe_batches,
                            probe_deltas, psf_metrics)
from megido.phantom import project, sky_rows
from megido.resolution import rays_per_voxel
from megido.voxels import VoxelGrid

SIGMA = 0.02


def _slab(grid, z0, z1, rho):
    zc = grid.axis_centers(2)
    t = np.zeros(grid.shape)
    t[:, :, (zc >= z0) & (zc < z1)] = rho
    return t


def _problem(cfg, rows, background):
    fwd = build_forward_model(rows, cfg, cache_dir=None)
    truth = background(fwd.grid)
    d = project(fwd, truth.ravel(), sigma=SIGMA, seed=11)
    data = FitData(lam=d.lam, w=d.w, rows=rows)
    x, info = solve(fwd, data, cfg.reconstruction)
    rays3 = rays_per_voxel(fwd).reshape(fwd.grid.shape)
    return fwd, data, cfg.reconstruction, x, info, rays3


def _megiddo_problem(tmp_path, positions=2):
    """The real campaign's geometry at 0.5 m: P0/T20 at the origin, P1 at 2.2 m."""
    exps = ("  - id: P0\n    runs: DET1-DET2\n"
            "    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}\n"
            "  - id: T20\n    runs: DET3-DET4\n"
            "    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}\n")
    if positions == 2:
        exps += ("  - id: P1\n    runs: DET5-DET6\n"
                 "    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}\n")
    p = tmp_path / "megiddo.yaml"
    p.write_text(
        "site: megido-kernels\ndata_dir: /tmp\n"
        "volume: {z_min_m: 1.0, z_max_m: 11.0, spacing_m: 0.5, n_aperture_sub: 2,\n"
        "         xy_m: [[-8.0, 10.0], [-8.0, 8.0]]}\n"
        "reconstruction: {algorithm: tv, n_iter: 300, tv_alpha: 0.005, tv_z_weight: 0.3}\n"
        "kernels: {spacing_m: 2.0, sep_m: 4.0, z_levels_m: [2.25, 4.25, 6.25, 8.25, 10.25]}\n"
        "exposures:\n" + exps)
    cfg = load_site_config(p)
    pids = ("pos0", "pos1") if positions == 2 else ("pos0",)
    rows = sky_rows(pids, t_max=1.0, n_bins=24)
    return cfg, _problem(cfg, rows, lambda g: _slab(g, 8.0, 9.0, 0.05))


def _generous_problem(tmp_path):
    """25 detectors over +-20 m: baselines 10-40 m against z ~ 4 m."""
    dets = [(x, y) for x in (-20, -10, 0, 10, 20) for y in (-20, -10, 0, 10, 20)]
    blk = "\n".join(
        f"  - id: D{i}\n    runs: DET{i}-DET{i}\n"
        f"    pose: {{x: {dx}, y: {dy}, z: 0, tilt_deg: 0, az_deg: 0}}"
        for i, (dx, dy) in enumerate(dets))
    p = tmp_path / "generous.yaml"
    p.write_text(
        "site: generous\ndata_dir: /tmp\n"
        "volume: {z_min_m: 0.0, z_max_m: 8.0, spacing_m: 1.0, n_aperture_sub: 2,\n"
        "         xy_m: [[-30, 30], [-30, 30]]}\n"
        "reconstruction: {algorithm: tv, n_iter: 400, tv_alpha: 0.002, tv_z_weight: 0.3}\n"
        "exposures:\n" + blk + "\n")
    cfg = load_site_config(p)
    rows = sky_rows(tuple(f"pos{i}" for i in range(25)), t_max=1.2, n_bins=25)
    return cfg, _problem(cfg, rows, lambda g: _slab(g, 6.0, 7.0, 0.05))


def test_probe_batches_cover_the_lattice_once_and_keep_probes_apart(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    kc = cfg.kernels
    batches = probe_batches(fwd.grid, rays3, kc)
    allp = np.concatenate(batches)
    assert len({tuple(p) for p in allp}) == len(allp)              # each probe once
    assert all(rays3[tuple(p)] >= 2 for p in allp)                  # covered only
    sep_vox = round(kc.sep_m / fwd.grid.spacing)
    for b in batches:
        assert len(set(b[:, 2])) == 1                               # one z level per batch
        d = np.abs(b[:, None, :2] - b[None, :, :2]).max(-1)
        np.fill_diagonal(d, 10**9)
        assert d.min() >= sep_vox                                   # laterally separated


def test_probe_with_no_weighted_rows_is_skipped(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    probes = np.concatenate(probe_batches(fwd.grid, rays3, cfg.kernels))
    dead = FitData(lam=data.lam, w=np.zeros_like(data.w), rows=data.rows)
    assert np.isnan(probe_deltas(fwd, dead, probes, cfg.kernels)).all()
    live = probe_deltas(fwd, data, probes, cfg.kernels)
    assert np.isfinite(live).all() and (live > 0).all()
    # delta = delta_sigma * median sigma / voxel spacing; sigma is SIGMA everywhere
    np.testing.assert_allclose(live, cfg.kernels.delta_sigma * SIGMA / fwd.grid.spacing)


def test_psf_metrics_of_hand_built_responses():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(5, 5, 10))
    v = np.zeros((5, 5, 10)); v[2, 2, 4] = 1.0                      # a perfect delta
    m = psf_metrics(PSF(probe=(2, 2, 4), lo=(0, 0), values=v), grid)
    assert m["psf_mass"] == pytest.approx(1.0)
    assert m["psf_dz"] == pytest.approx(0.68)                       # 16-84% of one voxel
    assert m["psf_shift"] == pytest.approx(0.0)
    assert m["psf_lat"] == pytest.approx(0.0)
    s = np.zeros((5, 5, 10)); s[2, 2, :] = 0.1                      # smeared up the column
    m = psf_metrics(PSF(probe=(2, 2, 4), lo=(0, 0), values=s), grid)
    assert m["psf_mass"] == pytest.approx(1.0)
    assert m["psf_dz"] == pytest.approx(6.8)                        # 16-84% of 10 m
    assert m["psf_shift"] == pytest.approx(5.0 - 4.5)               # centroid 5.0, probe 4.5


def test_zero_response_gives_nan_metrics_not_errors():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(3, 3, 4))
    m = psf_metrics(PSF(probe=(1, 1, 1), lo=(0, 0), values=np.zeros((3, 3, 4))), grid)
    assert all(np.isnan(m[k]) for k in ("psf_mass", "psf_dz", "psf_shift", "psf_lat"))


def test_edge_probe_cell_is_clipped_not_wrapped(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    nx, ny, _ = fwd.grid.shape
    h = cell_half_width(fwd.grid, cfg.kernels)
    covered = np.argwhere(rays3 >= 2)
    p = covered[np.argmin(covered[:, 0])]                           # lowest-i covered voxel
    probes = p[None, :]
    deltas = probe_deltas(fwd, data, probes, cfg.kernels)
    (psf,) = point_spreads(fwd, data, rc, x, info["best_iter"], probes, deltas,
                           fit_offsets=True, cell_half=h)
    assert psf.lo[0] >= 0 and psf.lo[0] + psf.values.shape[0] <= nx
    assert psf.lo[1] >= 0 and psf.lo[1] + psf.values.shape[1] <= ny
    assert np.isfinite(psf.values).all()


def test_psfs_are_deterministic(tmp_path):
    """Gate 5."""
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    b = probe_batches(fwd.grid, rays3, cfg.kernels)[3]
    d = probe_deltas(fwd, data, b, cfg.kernels)
    h = cell_half_width(fwd.grid, cfg.kernels)
    a1 = point_spreads(fwd, data, rc, x, info["best_iter"], b, d, fit_offsets=True, cell_half=h)
    a2 = point_spreads(fwd, data, rc, x, info["best_iter"], b, d, fit_offsets=True, cell_half=h)
    for p, q in zip(a1, a2):
        np.testing.assert_array_equal(p.values, q.values)


def test_well_resolved_geometry_gives_a_compact_psf(tmp_path):
    """Gate 1: 10-40 m baselines against z ~ 4 m (analytic dz ~ 0.03 m)."""
    cfg, (fwd, data, rc, x, info, rays3) = _generous_problem(tmp_path)
    kc = dataclasses.replace(cfg.kernels, spacing_m=1.0, sep_m=6.0)
    i, j = np.array(fwd.grid.shape[:2]) // 2
    k = int((3.5 - fwd.grid.origin[2]) / fwd.grid.spacing)          # probe at z = 3.5 m
    probes = np.array([[i, j, k]])
    assert rays3[i, j, k] >= 2
    deltas = probe_deltas(fwd, data, probes, kc)
    (psf,) = point_spreads(fwd, data, rc, x, info["best_iter"], probes, deltas,
                           fit_offsets=True, cell_half=cell_half_width(fwd.grid, kc))
    m = psf_metrics(psf, fwd.grid)
    print(f"\ngate 1 (generous): {m}")
    h = fwd.grid.spacing
    assert m["psf_mass"] >= 0.8
    assert m["psf_dz"] <= 2 * h
    assert abs(m["psf_shift"]) <= 1 * h
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_kernels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'megido.kernels'`

- [ ] **Step 3: Implement `megido/kernels.py` (core part)**

```python
"""Point-spread functions of the DELIVERED voxel solver.

Spec: docs/superpowers/specs/2026-10-06-voxel-averaging-kernels-design.md.

A delivered voxel x_hat +- sigma estimates the truth blurred by the solver's
local resolution operator R (TV + non-negativity + damping + c_p, as
configured), not the point truth: count-level phantoms showed the bootstrap
sigma is a correct NOISE sigma that covers the point truth in only 33-55% of
voxels, because the bias of the one-sided geometry dominates
(spike/sigma-calibration). This module measures columns of R by perturbation:

    PSF_j = (x_hat(lam + delta A e_j) - x_hat(lam)) / delta

re-solved along the nominal solve's path (`stop_at=best_iter`), for batches of
probes far enough apart laterally to share one solve. Columns suffice: the
gate's R t = sum_j t_j PSF_j, and rows would need the adjoint of a nonlinear
iterative solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.config import KernelConfig, Reconstruction
from megido.fitdata import FitData
from megido.forward import ForwardModel
from megido.inversion import solve
from megido.voxels import VoxelGrid

METRICS = ("psf_mass", "psf_dz", "psf_shift", "psf_lat")


@dataclass(frozen=True)
class PSF:
    """One probe's response inside its lateral cell, per unit probe density.

    values[a, b, :] is voxel (lo[0] + a, lo[1] + b, :) -- the full z column.
    """
    probe: tuple[int, int, int]
    lo: tuple[int, int]
    values: np.ndarray


def _lattice_step(grid: VoxelGrid, kc: KernelConfig) -> int:
    return max(1, int(round(kc.spacing_m / grid.spacing)))


def cell_half_width(grid: VoxelGrid, kc: KernelConfig) -> int:
    """Half-width (voxels) of a probe's cell. (sep - 1) // 2 keeps the cells of
    one batch disjoint: probes are >= sep voxels apart."""
    sep = max(1, int(round(kc.sep_m / grid.spacing)))
    return max(0, (sep - 1) // 2)


def z_level_indices(grid: VoxelGrid, kc: KernelConfig) -> np.ndarray:
    z0, h, nz = grid.origin[2], grid.spacing, grid.shape[2]
    levels = (kc.z_levels_m if kc.z_levels_m is not None
              else np.arange(z0 + kc.spacing_m / 2, z0 + nz * h, kc.spacing_m))
    k = np.floor((np.asarray(levels, dtype=float) - z0) / h).astype(int)
    return np.unique(k[(k >= 0) & (k < nz)])


def probe_batches(grid: VoxelGrid, rays3: np.ndarray, kc: KernelConfig) -> list[np.ndarray]:
    """Probe voxels grouped into solves: one z level per batch, probes >= sep_m
    apart laterally; the (sep/step)^2 offset batches of a level fill the
    spacing_m lattice. Only voxels crossed by >= 2 rays are probed: below that
    a voxel is its ray's private unknown (the noise shell)."""
    step = _lattice_step(grid, kc)
    sep = max(step, int(round(kc.sep_m / grid.spacing)))
    m = -(-sep // step)                                    # ceil: offsets per axis
    nx, ny, _ = grid.shape
    li = np.arange(step // 2, nx, step)
    lj = np.arange(step // 2, ny, step)
    batches: list[np.ndarray] = []
    for k in z_level_indices(grid, kc):
        for oa in range(m):
            for ob in range(m):
                ii, jj = np.meshgrid(li[oa::m], lj[ob::m], indexing="ij")
                p = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k)], -1)
                p = p[rays3[p[:, 0], p[:, 1], p[:, 2]] >= 2]
                if len(p):
                    batches.append(p.astype(np.int64))
    return batches


def probe_deltas(fwd: ForwardModel, data: FitData, probes: np.ndarray,
                 kc: KernelConfig) -> np.ndarray:
    """delta_j = delta_sigma * median(sigma of weighted rows crossing j) / spacing:
    the density whose vertical path through one voxel adds ~delta_sigma noise
    sigmas of opacity. NaN when no weighted row crosses j (probe skipped)."""
    A = fwd.A.tocsc()
    flat = np.ravel_multi_index(probes.T, fwd.grid.shape)
    out = np.full(len(flat), np.nan)
    for n, j in enumerate(flat):
        rows = A.indices[A.indptr[j]:A.indptr[j + 1]]
        w = data.w[rows]
        w = w[w > 0]
        if w.size:
            out[n] = kc.delta_sigma * float(np.median(1.0 / np.sqrt(w))) / fwd.grid.spacing
    return out


def point_spreads(fwd: ForwardModel, data: FitData, rc: Reconstruction,
                  x_hat: np.ndarray, best_iter: int, probes: np.ndarray,
                  deltas: np.ndarray, *, fit_offsets: bool, cell_half: int) -> list[PSF]:
    """One perturbed solve for a batch; each probe's PSF is the response in its
    own lateral cell (clipped to the grid), divided by its delta. Probes with a
    NaN delta are skipped."""
    shape = fwd.grid.shape
    ok = np.isfinite(deltas)
    probes, deltas = probes[ok], deltas[ok]
    if not len(probes):
        return []
    bump = np.zeros(fwd.A.shape[1])
    bump[np.ravel_multi_index(probes.T, shape)] = deltas
    lam2 = data.lam + fwd.A @ bump
    x2, _ = solve(fwd, FitData(lam=lam2, w=data.w, rows=data.rows), rc,
                  fit_offsets=fit_offsets, stop_at=best_iter)
    resp = (x2 - np.asarray(x_hat).ravel()).reshape(shape)
    out = []
    for (i, j, k), d in zip(probes, deltas):
        i0, j0 = max(0, i - cell_half), max(0, j - cell_half)
        i1, j1 = min(shape[0], i + cell_half + 1), min(shape[1], j + cell_half + 1)
        out.append(PSF(probe=(int(i), int(j), int(k)), lo=(i0, j0),
                       values=resp[i0:i1, j0:j1, :] / d))
    return out


def _edge_quantile(p: np.ndarray, edges: np.ndarray, q: float) -> float:
    """q-quantile of a histogram p over bins with the given edges (p sums to 1)."""
    cdf = np.concatenate([[0.0], np.cumsum(p)])
    n = int(np.searchsorted(cdf, q, side="left"))
    n = min(max(n, 1), len(p))
    frac = (q - cdf[n - 1]) / p[n - 1] if p[n - 1] > 0 else 0.0
    return float(edges[n - 1] + frac * (edges[n] - edges[n - 1]))


def psf_metrics(psf: PSF, grid: VoxelGrid) -> dict[str, float]:
    """mass recovery, depth spread (16-84% width of |PSF|'s z-marginal), depth
    shift (|PSF| z-centroid minus probe z), lateral width (2x the radius about
    the probe holding 68% of |PSF|'s xy-marginal). NaN if the response is zero."""
    v = psf.values
    w = np.abs(v)
    tot = float(w.sum())
    if not tot > 0:
        return {k: float("nan") for k in METRICS}
    h = grid.spacing
    z_edges = grid.origin[2] + h * np.arange(grid.shape[2] + 1)
    zc = grid.axis_centers(2)
    pz = w.sum((0, 1)) / tot
    dz = _edge_quantile(pz, z_edges, 0.84) - _edge_quantile(pz, z_edges, 0.16)
    shift = float((pz * zc).sum() - zc[psf.probe[2]])
    pxy = w.sum(2) / tot
    a = np.arange(v.shape[0]) + psf.lo[0] - psf.probe[0]
    b = np.arange(v.shape[1]) + psf.lo[1] - psf.probe[1]
    r = h * np.hypot(a[:, None], b[None, :]).ravel()
    order = np.argsort(r, kind="stable")
    cum = np.cumsum(pxy.ravel()[order])
    r68 = float(r[order][min(int(np.searchsorted(cum, 0.68 - 1e-12)), r.size - 1)])
    return {"psf_mass": float(v.sum()), "psf_dz": float(dz),
            "psf_shift": shift, "psf_lat": 2.0 * r68}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_kernels.py -q -s`
Expected: PASS, with the gate 1 metrics printed. If gate 1 fails, do **not** change its thresholds. First check `stop_at`/`best_iter` (did the probe solve run any updates? `info["best_iter"] > 0`). Then check that the bump lands where intended (`fwd.A @ bump` should be nonzero on rows crossing the probe). Report the measured metrics to the controller.

- [ ] **Step 5: Commit**

```bash
git add megido/kernels.py tests/test_kernels.py
git commit -m "feat(kernels): probe lattice, perturbation PSFs and metrics"
```

---

### Task 5: `compute_kernels`, interpolation, `KernelResult` (+ gates 2, 3, 4)

**Files:**
- Modify: `megido/kernels.py`
- Test: `tests/test_kernels.py`

**Interfaces:**
- Consumes: everything from Task 4.
- Produces:
  - `@dataclass(frozen=True) class KernelResult: probes: np.ndarray ([n,3] int); metrics: dict[str, np.ndarray] ([n] each); volumes: dict[str, np.ndarray] ([nx,ny,nz] float32, NaN = not characterised); grid: VoxelGrid; config: dict; version: int; psfs: list[PSF] | None = None` (not saved). Methods: `save(path)`, `load(path) -> KernelResult` (static), `summary() -> list[dict]` (one row per probed z level: `z_m, n, psf_mass, psf_dz, psf_shift, psf_lat`, medians over finite values).
  - `compute_kernels(fwd, data, rc, x_hat, info, rays3, kc, *, fit_offsets: bool, batches: list[np.ndarray] | None = None, keep_psfs: bool = False, progress=None) -> KernelResult`
  - `interpolate_to_grid(probes: np.ndarray, values: np.ndarray, grid: VoxelGrid, kc: KernelConfig) -> np.ndarray`
  - `superpose(psfs: list[PSF], amplitudes: dict[tuple[int, int, int], float], shape) -> np.ndarray`: `Σ_j t_j PSF_j` embedded in the full grid.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_kernels.py`)

```python
from megido.kernels import (KernelResult, compute_kernels, interpolate_to_grid,
                            superpose)
from megido.resolution import depth_resolution
from megido.cli import _SKY_SIGMA_T


def test_interpolation_is_exact_on_lattice_points_and_nan_far_away():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=0.5, shape=(12, 12, 8))
    kc = KernelConfig(spacing_m=1.0, sep_m=2.0, z_levels_m=(1.25, 2.25))
    probes = np.array([[1, 1, 2], [3, 1, 2], [1, 3, 2], [3, 3, 2],
                       [1, 1, 4], [3, 1, 4], [1, 3, 4], [3, 3, 4]])
    vals = probes[:, 0] + 10.0 * probes[:, 2]                        # linear in i and k
    vol = interpolate_to_grid(probes, vals.astype(float), grid, kc)
    for p, v in zip(probes, vals):
        assert vol[tuple(p)] == pytest.approx(v)
    assert vol[2, 2, 3] == pytest.approx(2 + 30.0)                   # trilinear midpoint
    assert np.isnan(vol[11, 11, 7])                                  # > one spacing from any probe


def test_kernel_result_round_trips_and_summarises(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    res = compute_kernels(fwd, data, rc, x, info, rays3, cfg.kernels, fit_offsets=True)
    res.save(tmp_path / "kernels.npz")
    back = KernelResult.load(tmp_path / "kernels.npz")
    np.testing.assert_array_equal(back.probes, res.probes)
    for k in res.volumes:
        np.testing.assert_array_equal(back.volumes[k], res.volumes[k])
    assert back.grid == res.grid and back.version == res.version
    rows = res.summary()
    assert [r["z_m"] for r in rows] == sorted(r["z_m"] for r in rows)
    assert all(r["n"] > 0 for r in rows)


def test_depth_spread_cannot_beat_parallax(tmp_path):
    """Gate 2: one 2.2 m baseline. Spread (a 68% width ~ 2 sigma) >= analytic dz
    (1 sigma) at every level, and larger at the top than at the bottom."""
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    rows = compute_kernels(fwd, data, rc, x, info, rays3, cfg.kernels, fit_offsets=True).summary()
    print("\ngate 2:", [(r["z_m"], round(r["psf_dz"], 2)) for r in rows])
    for r in rows:
        assert r["psf_dz"] >= depth_resolution(r["z_m"], 2.2, _SKY_SIGMA_T)
    assert rows[-1]["psf_dz"] > rows[0]["psf_dz"]


def test_one_position_leaves_depth_unconstrained(tmp_path):
    """Gate 3 (negative control): no parallax at all."""
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path, positions=1)
    rows = compute_kernels(fwd, data, rc, x, info, rays3, cfg.kernels, fit_offsets=True).summary()
    mid = min(rows, key=lambda r: abs(r["z_m"] - 6.25))
    print("\ngate 3:", mid)
    assert mid["psf_dz"] >= 0.5 * (11.0 - 1.0)


def test_psfs_superpose_and_are_amplitude_independent(tmp_path):
    """Gate 4: batched PSFs predict the response to a whole-lattice spike phantom,
    and PSFs at 0.5x and 2x delta agree."""
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    kc = cfg.kernels
    res = compute_kernels(fwd, data, rc, x, info, rays3, kc, fit_offsets=True, keep_psfs=True)
    t = 0.04
    amps = {p.probe: t for p in res.psfs}
    pred = superpose(res.psfs, amps, fwd.grid.shape)
    spikes = np.zeros(fwd.A.shape[1])
    spikes[np.ravel_multi_index(np.array(list(amps)).T, fwd.grid.shape)] = t
    lam2 = data.lam + fwd.A @ spikes
    x2, _ = solve(fwd, FitData(lam=lam2, w=data.w, rows=data.rows), rc,
                  fit_offsets=True, stop_at=info["best_iter"])
    actual = (x2 - x).reshape(fwd.grid.shape)
    rel = np.linalg.norm(pred - actual) / np.linalg.norm(actual)
    print(f"\ngate 4 superposition rel L2 = {rel:.3f}")
    assert rel <= 0.2

    b = probe_batches(fwd.grid, rays3, kc)[0]
    h = cell_half_width(fwd.grid, kc)
    d = probe_deltas(fwd, data, b, kc)
    lo = point_spreads(fwd, data, rc, x, info["best_iter"], b, 0.5 * d, fit_offsets=True, cell_half=h)
    hi = point_spreads(fwd, data, rc, x, info["best_iter"], b, 2.0 * d, fit_offsets=True, cell_half=h)
    a = np.concatenate([p.values.ravel() for p in lo])
    c = np.concatenate([p.values.ravel() for p in hi])
    rel_amp = np.linalg.norm(a - c) / np.linalg.norm(c)
    print(f"gate 4 amplitude rel L2 = {rel_amp:.3f}")
    assert rel_amp <= 0.2
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_kernels.py -q -k "interpolation or round_trips or parallax or one_position or superpose"`
Expected: FAIL with `ImportError: cannot import name 'KernelResult'`

- [ ] **Step 3: Implement** (append to `megido/kernels.py`; add `import json`, `from dataclasses import asdict, field`, `from pathlib import Path`, `from megido.raycast import INVERSION_VERSION`)

```python
def _axis_weights(n: int, lattice: np.ndarray, reach: float):
    """Per voxel index 0..n-1: two lattice slots (a0, a1), weight of a1, and
    whether any lattice point lies within `reach` voxels."""
    idx = np.arange(n)
    pos = np.searchsorted(lattice, idx, side="right") - 1
    a0 = np.clip(pos, 0, len(lattice) - 1)
    a1 = np.clip(pos + 1, 0, len(lattice) - 1)
    span = (lattice[a1] - lattice[a0]).astype(float)
    t = np.where(span > 0, (idx - lattice[a0]) / np.where(span > 0, span, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    near = np.min(np.abs(idx[:, None] - lattice[None, :]), axis=1) <= reach
    return a0, a1, t, near


def interpolate_to_grid(probes: np.ndarray, values: np.ndarray, grid: VoxelGrid,
                        kc: KernelConfig) -> np.ndarray:
    """Trilinear over the probe lattice, renormalised over the corners that hold
    a finite value. NaN ("not characterised") where no probe lies within one
    spacing_m on every axis, or no corner is finite."""
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(values)
    if not ok.any():
        return out
    probes, values = probes[ok], values[ok]
    L = [np.unique(probes[:, a]) for a in range(3)]
    M = np.full([len(l) for l in L], np.nan)
    M[tuple(np.searchsorted(L[a], probes[:, a]) for a in range(3))] = values
    reach = kc.spacing_m / grid.spacing
    W = [_axis_weights(grid.shape[a], L[a], reach) for a in range(3)]
    num = np.zeros(grid.shape)
    den = np.zeros(grid.shape)
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                ia = W[0][cx]; ja = W[1][cy]; ka = W[2][cz]
                wx = W[0][2] if cx else 1 - W[0][2]
                wy = W[1][2] if cy else 1 - W[1][2]
                wz = W[2][2] if cz else 1 - W[2][2]
                w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
                v = M[np.ix_(ia, ja, ka)]
                f = np.isfinite(v) & (w > 0)
                num += np.where(f, w * np.nan_to_num(v), 0.0)
                den += np.where(f, w, 0.0)
    near = W[0][3][:, None, None] & W[1][3][None, :, None] & W[2][3][None, None, :]
    good = near & (den > 0)
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def superpose(psfs: list[PSF], amplitudes: dict, shape) -> np.ndarray:
    """sum_j t_j PSF_j on the full grid: the linearised estimator applied to a
    truth made of probe-lattice spikes (the calibration gate's R t)."""
    out = np.zeros(shape)
    for p in psfs:
        t = amplitudes.get(p.probe, 0.0)
        if t:
            a, b, _ = p.values.shape
            out[p.lo[0]:p.lo[0] + a, p.lo[1]:p.lo[1] + b, :] += t * p.values
    return out


@dataclass(frozen=True)
class KernelResult:
    probes: np.ndarray
    metrics: dict
    volumes: dict
    grid: VoxelGrid
    config: dict
    version: int = INVERSION_VERSION
    psfs: list | None = field(default=None, repr=False, compare=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"metric_{k}": np.asarray(v) for k, v in self.metrics.items()}
        arrays.update({f"volume_{k}": np.asarray(v, dtype=np.float32)
                       for k, v in self.volumes.items()})
        np.savez_compressed(
            path, probes=self.probes.astype(np.int64),
            origin=np.asarray(self.grid.origin, dtype=np.float64),
            spacing=np.asarray(self.grid.spacing, dtype=np.float64),
            shape=np.asarray(self.grid.shape, dtype=np.int64),
            version=np.asarray(self.version, dtype=np.int64),
            config=np.array(json.dumps(self.config)), **arrays)

    @staticmethod
    def load(path: str | Path) -> "KernelResult":
        d = np.load(path, allow_pickle=False)
        grid = VoxelGrid(origin=tuple(float(v) for v in d["origin"]),
                         spacing=float(d["spacing"]),
                         shape=tuple(int(v) for v in d["shape"]))
        return KernelResult(
            probes=d["probes"],
            metrics={k: d[f"metric_{k}"] for k in METRICS},
            volumes={k: d[f"volume_{k}"] for k in METRICS},
            grid=grid, config=json.loads(str(d["config"])), version=int(d["version"]))

    def summary(self) -> list[dict]:
        zc = self.grid.axis_centers(2)
        rows = []
        for k in np.unique(self.probes[:, 2]):
            sel = self.probes[:, 2] == k
            row = {"z_m": float(zc[k]), "n": int(sel.sum())}
            for m in METRICS:
                v = self.metrics[m][sel]
                v = v[np.isfinite(v)]
                row[m] = float(np.median(v)) if v.size else float("nan")
            rows.append(row)
        return rows


def compute_kernels(fwd: ForwardModel, data: FitData, rc: Reconstruction,
                    x_hat: np.ndarray, info: dict, rays3: np.ndarray, kc: KernelConfig,
                    *, fit_offsets: bool, batches: list | None = None,
                    keep_psfs: bool = False, progress=None) -> KernelResult:
    """All probe batches -> per-probe metrics -> per-voxel layers."""
    grid = fwd.grid
    batches = batches if batches is not None else probe_batches(grid, rays3, kc)
    half = cell_half_width(grid, kc)
    probes, values, kept = [], {m: [] for m in METRICS}, []
    for n, b in enumerate(batches):
        deltas = probe_deltas(fwd, data, b, kc)
        for psf in point_spreads(fwd, data, rc, x_hat, int(info["best_iter"]), b, deltas,
                                 fit_offsets=fit_offsets, cell_half=half):
            probes.append(psf.probe)
            for m, v in psf_metrics(psf, grid).items():
                values[m].append(v)
            if keep_psfs:
                kept.append(psf)
        if progress:
            progress(n + 1, len(batches))
    P = np.array(probes, dtype=np.int64).reshape(-1, 3)
    metrics = {m: np.array(values[m], dtype=float) for m in METRICS}
    volumes = {m: interpolate_to_grid(P, metrics[m], grid, kc) for m in METRICS}
    config = {**asdict(kc), "fit_offsets": fit_offsets, "best_iter": int(info["best_iter"])}
    return KernelResult(probes=P, metrics=metrics, volumes=volumes, grid=grid,
                        config=config, psfs=kept if keep_psfs else None)
```

(`asdict(kc)` returns `z_levels_m` as a tuple or None, both JSON-serialisable.)

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_kernels.py -q -s`
Expected: PASS, with gates 2–4 printing their numbers. If gate 4's superposition fails, widen `sep_m` in the **test geometry's** `kernels:` block (4 m → 6 m) and rerun, then report both numbers. Never raise the 0.2 threshold. If gates 2 or 3 fail, report the measured spreads. They are physical claims, and a failure is a finding.

- [ ] **Step 5: Commit**

```bash
git add megido/kernels.py tests/test_kernels.py
git commit -m "feat(kernels): compute_kernels, lattice interpolation, KernelResult; gates 2-4"
```

---

### Task 6: Calibration gate, CLI `--kernels`, cache, export

**Files:**
- Modify: `megido/kernels.py` (`cache_key`)
- Modify: `megido/cli.py` (`--kernels` flag and block; SNR caveat in the bootstrap printout)
- Modify: `megido/volexport.py` (export the four layers)
- Test: `tests/test_kernels_calibration.py` (new), `tests/test_kernels.py` (cache key), `tests/test_volexport.py`, `tests/test_cli3.py` (or wherever `_cmd_reconstruct` is tested; check with `grep -ln "reconstruct" tests/test_cli*.py`)

**Interfaces:**
- Consumes: `compute_kernels`, `KernelResult`, `superpose`, `probe_batches` (Tasks 4–5); `voxel_bootstrap(sigma=)` (Task 2).
- Produces: `cache_key(fwd, data, rc, kc, fit_offsets: bool) -> str` (16 hex characters). CLI `reconstruct --kernels` writes `<out>/kernels.npz` and `<out>/psf_{mass,dz,shift,lat}.npy`, cached at `<cache>/kernels_<key>.npz`. `export_volume` lists the four names in `meta.layers` when present.

- [ ] **Step 1: Write the failing tests**

`tests/test_kernels_calibration.py`:

```python
"""Gate 6: x_hat +- sigma covers the KERNEL-BLURRED truth R t (spec section 4).

A small sky-reference campaign at count level: the cafeteria config's two
positions, a truth of probe-lattice spikes, one observed Poisson draw, sigma
from 16 replicas that Poisson-resample the OBSERVED counts (as production).
The band [0.58, 0.78] was fixed before building, from the spike's positive
control (0.66-0.74) plus the 8-vs-32 replica spread.
"""
from dataclasses import replace
from functools import partial

import numpy as np

from megido.acceptance import geometric_acceptance
from megido.angular import AnalysisGrid
from megido.baseline import position_ids
from megido.config import KernelConfig, load_site_config
from megido.detector import DetectorGeometry
from megido.fitdata import RowIndex, build_fit_data
from megido.forward import build_forward_model
from megido.kernels import compute_kernels, probe_batches, superpose
from megido.reconstruct import solve_voxels
from megido.resolution import rays_per_voxel
from megido.sky import detector_to_sky
from megido.skyref import skyref_sigma, solve_skyref
from megido.voxuncert import voxel_bootstrap

LIVE = {"pos0": 1000.0, "pos1": 1000.0, "SKY": 3000.0}
RHO_T = 0.2


def _cfg():
    cfg = load_site_config("configs/cafeteria.yaml")
    return replace(cfg,
                   volume=replace(cfg.volume, spacing_m=0.5, viewer_crop_xy_m=None),
                   reconstruction=replace(cfg.reconstruction, n_iter=60),
                   kernels=KernelConfig(spacing_m=2.0, sep_m=4.0,
                                        z_levels_m=(3.25, 5.25, 7.25)))


def _expected_counts(cfg, edges, lam_truth, sky):
    grid0 = AnalysisGrid(edges=edges, counts={})
    tx, ty = grid0.tan_mesh()
    acc = geometric_acceptance(tx, ty, DetectorGeometry.for_site(cfg))
    n_sky = 3000.0 * acc / acc.max() / (1 + tx**2 + ty**2)
    mu = {"SKY": n_sky}
    pos = position_ids(cfg)
    for e in cfg.exposures:
        sx, sy, on = detector_to_sky(tx, ty, e.pose)
        flat, ing = sky.bin_index(sx, sy)
        lam = np.zeros_like(tx)
        okk = on & ing
        lam[okk] = lam_truth[pos[e.id]][flat[okk]]
        mu[e.id] = LIVE[e.id] / LIVE["SKY"] * n_sky * np.exp(-lam)
    return mu


def _truth_lambda(cfg, sol, vgrid, truth3):
    sky = sol.sky
    c = sky.centers
    ii, jj = np.meshgrid(np.arange(sky.n_bins), np.arange(sky.n_bins), indexing="ij")
    sx, sy, flat = c[ii].ravel(), c[jj].ravel(), (ii * sky.n_bins + jj).ravel()
    keep = (np.abs(sx) <= 1.0) & (np.abs(sy) <= 1.0)
    pids = tuple(sorted(sol.opacity))
    rows = RowIndex(position_ids=pids, pos_of_row=np.repeat(np.arange(len(pids)), keep.sum()),
                    sx=np.tile(sx[keep], len(pids)), sy=np.tile(sy[keep], len(pids)),
                    sky_flat=np.tile(flat[keep], len(pids)))
    lam_rows = build_forward_model(rows, cfg, grid=vgrid, cache_dir=None).predict(truth3)
    out = {}
    for k, pid in enumerate(pids):
        lam = np.zeros(sky.flat_size)
        m = rows.pos_of_row == k
        lam[rows.sky_flat[m]] = lam_rows[m]
        out[pid] = lam
    return out


def _poisson(mu, rng):
    return {k: rng.poisson(np.clip(v, 0, None)).astype(np.int64) for k, v in mu.items()}


```

Then the gate itself, in the same file:

```python
def test_sigma_covers_the_kernel_blurred_truth_not_the_point_truth():
    cfg = _cfg()
    edges = np.linspace(-1.0, 1.0, 41)
    solver = partial(solve_skyref, live_time=LIVE)
    from megido.sky import make_sky_grid
    sky = make_sky_grid()
    zero = {"pos0": np.zeros(sky.flat_size), "pos1": np.zeros(sky.flat_size)}

    # 1. Lattice and voxel grid from the noise-free, truth-free campaign.
    clear = AnalysisGrid(edges=edges, counts={k: np.round(v).astype(np.int64) for k, v in
                                              _expected_counts(cfg, edges, zero, sky).items()})
    sol0 = solver(clear, cfg)
    vgrid = solve_voxels(sol0, cfg, cache_dir=None, holdouts=False)["full"].grid
    fwd0 = build_forward_model(build_fit_data(sol0, cfg).rows, cfg, grid=vgrid, cache_dir=None)
    rays0 = rays_per_voxel(fwd0).reshape(vgrid.shape)
    batches = probe_batches(vgrid, rays0, cfg.kernels)
    probes = np.concatenate(batches)
    truth = np.zeros(vgrid.shape)
    truth[tuple(probes.T)] = RHO_T

    # 2. Observe the spike truth once.
    rng = np.random.default_rng(0)
    obs = _poisson(_expected_counts(cfg, edges, _truth_lambda(cfg, sol0, vgrid, truth), sky), rng)
    og = AnalysisGrid(edges=edges, counts=obs)
    sol = solver(og, cfg)
    sigma = skyref_sigma(og, cfg)
    full = solve_voxels(sol, cfg, sigma=sigma, cache_dir=None, holdouts=False, grid=vgrid)["full"]
    data = build_fit_data(sol, cfg, sigma=sigma)
    fwd = build_forward_model(data.rows, cfg, grid=vgrid, cache_dir=None)
    rays3 = rays_per_voxel(fwd).reshape(vgrid.shape)

    # 3. R t from the delivered solver's PSFs; sigma from the delivered-weight bootstrap.
    kr = compute_kernels(fwd, data, cfg.reconstruction, full.rho, full.info, rays3,
                         cfg.kernels, fit_offsets=False, batches=batches, keep_psfs=True)
    Rt = superpose(kr.psfs, {p.probe: RHO_T for p in kr.psfs}, vgrid.shape)
    boot = voxel_bootstrap(og, cfg, n_replicas=16, cache_dir=None, solver=solver, sigma=sigma)
    assert boot.grid == vgrid
    xhat = full.rho3()
    sig = boot.sigma
    scored = (rays3 >= 2) & (sig > 0)

    cov_rt = float((np.abs(xhat - Rt) <= sig)[scored].mean())
    cov_half = float((np.abs(xhat - Rt) <= 0.5 * sig)[scored].mean())
    cov_point = float((np.abs(xhat - truth) <= sig)[scored].mean())
    print(f"\ngate 6: coverage R t {cov_rt:.3f}  (sigma/2: {cov_half:.3f})  point truth {cov_point:.3f}")
    assert 0.58 <= cov_rt <= 0.78          # calibrated for what it estimates
    assert cov_half < 0.58                 # teeth: an over-confident sigma fails
    assert cov_point < cov_rt              # the reason this feature exists
```


Append to `tests/test_kernels.py`:

```python
from megido.kernels import cache_key


def test_cache_key_changes_with_every_input(tmp_path):
    cfg, (fwd, data, rc, x, info, rays3) = _megiddo_problem(tmp_path)
    kc = cfg.kernels
    base = cache_key(fwd, data, rc, kc, True)
    assert cache_key(fwd, data, rc, kc, True) == base
    variants = [
        cache_key(fwd, data, rc, kc, False),
        cache_key(fwd, data, dataclasses.replace(rc, tv_alpha=0.01), kc, True),
        cache_key(fwd, data, rc, dataclasses.replace(kc, delta_sigma=0.5), True),
        cache_key(fwd, data, rc, dataclasses.replace(kc, sep_m=6.0), True),
        cache_key(fwd, data, rc, dataclasses.replace(kc, z_levels_m=(3.0,)), True),
        cache_key(fwd, FitData(lam=data.lam + 1e-9, w=data.w, rows=data.rows), rc, kc, True),
        cache_key(fwd, FitData(lam=data.lam, w=data.w * 2, rows=data.rows), rc, kc, True),
    ]
    assert len(set(variants)) == len(variants) and base not in variants
```

Append to `tests/test_volexport.py` (reuse the module's existing fixture that writes a run directory with `volume_full.npz`; find it with `grep -n "def _\|def test" tests/test_volexport.py`):

```python
def test_kernel_layers_are_exported_when_present(tmp_path):
    run, cfg = _run_dir(tmp_path)          # adapt to the module's existing helper name
    shape = tuple(np.load(run / "volume_full.npz")["shape"])
    for name in ("psf_mass", "psf_dz", "psf_shift", "psf_lat"):
        np.save(run / f"{name}.npy", np.full(shape, 1.0, dtype=np.float32))
    export_volume(run, cfg)
    meta = json.loads((run / "meta.json").read_text())
    for name in ("psf_mass", "psf_dz", "psf_shift", "psf_lat"):
        assert name in meta["layers"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_kernels_calibration.py tests/test_kernels.py tests/test_volexport.py -q -k "covers or cache_key or kernel_layers"`
Expected: FAIL. The calibration test may already pass if Tasks 1–5 are in, which is acceptable since it gates behaviour already built. `cache_key` fails with `ImportError` and the layers test fails on `assert name in meta["layers"]`.

- [ ] **Step 3: Implement**

Append to `megido/kernels.py` (add `import hashlib`):

```python
def cache_key(fwd: ForwardModel, data: FitData, rc: Reconstruction, kc: KernelConfig,
              fit_offsets: bool) -> str:
    """Content address of a kernel computation: the system matrix, the data and
    weights, the solver settings, the probe settings and the code version."""
    h = hashlib.sha256()
    h.update(fwd.rows.key().encode())
    h.update(fwd.grid.key().encode())
    A = fwd.A.tocsr()
    for arr in (A.data, A.indices, A.indptr, data.lam, data.w):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(repr(rc).encode())
    h.update(repr(kc).encode())
    h.update(f"{fit_offsets}|{INVERSION_VERSION}".encode())
    return h.hexdigest()[:16]
```

In `megido/cli.py`:
- Add the import `from megido.kernels import KernelResult, cache_key, compute_kernels`.
- Add the reconstruct argument after `--backproject-z`:
  ```python
  r.add_argument("--kernels", action="store_true",
                 help="measure point-spread functions of the delivered solver "
                      "(kernels.npz + psf_*.npy; ~50+ solves)")
  ```
- In `_cmd_reconstruct`, after `np.save(out / "rays.npy", rays_per_voxel(fwd))`, add:

```python
    if args.kernels:
        fit_c = not sol.absolute
        key = cache_key(fwd, data, cfg.reconstruction, cfg.kernels, fit_c)
        cached = Path(args.cache) / f"kernels_{key}.npz" if args.cache else None
        if cached is not None and cached.exists():
            kres = KernelResult.load(cached)
            print(f"kernels         cached ({cached.name})")
        else:
            rays3 = rays_per_voxel(fwd).reshape(fwd.grid.shape)
            kres = compute_kernels(
                fwd, data, cfg.reconstruction, full.rho, full.info, rays3, cfg.kernels,
                fit_offsets=fit_c,
                progress=lambda i, n: print(f"  kernel batch {i}/{n}", flush=True))
            if cached is not None:
                kres.save(cached)
        kres.save(out / "kernels.npz")
        for name, vol in kres.volumes.items():
            np.save(out / f"{name}.npy", np.asarray(vol, dtype=np.float32))
        print("kernels         x_hat +- sigma estimates the truth blurred by these PSFs")
        print("  z [m]   n   mass   depth spread [m]   depth shift [m]   lateral [m]")
        for row in kres.summary():
            print(f"  {row['z_m']:5.2f} {row['n']:4d}  {row['psf_mass']:5.2f}"
                  f"   {row['psf_dz']:16.2f}   {row['psf_shift']:+15.2f}   {row['psf_lat']:11.2f}")
```

- In the bootstrap printout, change both `above SNR 3` strings to `above noise-SNR 3 (noise only, not detections)`.

In `megido/volexport.py`, extend the optional-layer loop tuple:

```python
    for name in ("views", "rays", "systematic", "backprojection",
                 "psf_mass", "psf_dz", "psf_shift", "psf_lat"):
```

Add a CLI test where the module already exercises `_cmd_reconstruct` on a small config. Run `reconstruct --kernels` with `--no-holdouts --no-systematic` and a tiny `kernels:` block (`spacing_m` = 2× the voxel spacing, `sep_m` = 4× it), then assert:

```python
    for name in ("kernels.npz", "psf_mass.npy", "psf_dz.npy", "psf_shift.npy", "psf_lat.npy"):
        assert (out / name).exists()
    assert np.load(out / "psf_dz.npy").shape == tuple(np.load(out / "volume_full.npz")["shape"])
    # second run hits the cache
    assert "kernels         cached" in capsys.readouterr().out   # after re-running main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_kernels_calibration.py tests/test_kernels.py tests/test_volexport.py tests/test_cli*.py -q -s`
Expected: PASS, with gate 6 printing its three coverages. If `cov_rt` falls outside [0.58, 0.78], report all three numbers and do **not** move the band. Check `boot.grid == vgrid` first, then the PSF amplitude test (gate 4).

- [ ] **Step 5: Commit**

```bash
git add megido/kernels.py megido/cli.py megido/volexport.py tests/
git commit -m "feat(kernels): calibration gate, reconstruct --kernels with cache, export layers"
```

---

### Task 7: Viewer: kernel layers, depth-spread gate, hover readout, SNR relabel

**Files:**
- Modify: `viewer/src/layers.mjs`, `viewer/src/model.mjs`, `viewer/src/gates.mjs`, `viewer/src/picker.mjs`, `viewer/src/app.mjs`, `viewer/shell.html`
- Test: `viewer/test/gates.test.mjs`, `viewer/test/picker.test.mjs`, `viewer/test/model.test.mjs` (only if it pins the EFFECTS keys), `tests/viewer/test_gate_parity.py`

**Interfaces:**
- Produces:
  - State `depthGateEnabled: false`, `maxDepthSpread: 3`, `hasPsfDz: false`.
  - EFFECTS `depthGateEnabled` and `maxDepthSpread`: `['window', 'render']`.
  - `voxelGates(settings, layers)` reads `layers.psf_dz`; uniforms gain `depthEnabled`, `maxDepthSpread`.
  - GLSL uniforms `uDzTex` (unit 7), `uDepthGateEnabled`, `uMaxDepthSpread`.
  - `hoverText(hit, layerData, shape) -> string` exported from `picker.mjs`.

- [ ] **Step 1: Write the failing node tests**

Append to `viewer/test/gates.test.mjs`:

```js
test('depth gate hides spread above max and NaN (not characterised)', () => {
  const psf_dz = Float32Array.from([1, 5, NaN, 1, 1, 1]);
  const g = voxelGates({ ...OFF, depthGateEnabled: true, maxDepthSpread: 3 }, { ...L, psf_dz });
  assert.deepEqual([0, 1, 2].map(g.keep), [true, false, false]);
  assert.equal(g.keepForWindow(1), false);
  assert.equal(g.windowActive, true);
  assert.equal(g.uniforms.depthEnabled, true);
  assert.equal(g.uniforms.maxDepthSpread, 3);
});

test('depth gate inert without psf_dz', () => {
  const g = voxelGates({ ...OFF, depthGateEnabled: true, maxDepthSpread: 3 }, L);
  for (let n = 0; n < 6; n++) assert.equal(g.keep(n), true);
  assert.equal(g.uniforms.depthEnabled, false);
});

test('key changes with the depth gate inputs', () => {
  const psf_dz = Float32Array.from([1, 5, NaN, 1, 1, 1]);
  const on = { ...ALL, depthGateEnabled: true, maxDepthSpread: 3 };
  const base = voxelGates(on, { ...L, psf_dz }).key;
  assert.notEqual(voxelGates({ ...on, maxDepthSpread: 4 }, { ...L, psf_dz }).key, base);
  assert.notEqual(voxelGates({ ...on, depthGateEnabled: false }, { ...L, psf_dz }).key, base);
  assert.notEqual(voxelGates(on, L).key, base);
});
```

Update the existing `uniforms` deepEqual assertion in the "a gate whose layer is absent" test to include `depthEnabled: false, maxDepthSpread: undefined`. To avoid that `undefined`, add `depthGateEnabled: false, maxDepthSpread: 3` to the `OFF` constant and expect `maxDepthSpread: 3`.

Append to `viewer/test/picker.test.mjs`:

```js
import { hoverText } from '../src/picker.mjs';

test('hoverText appends kernel metrics when the layers are loaded', () => {
  const shape = [2, 2, 2];
  const at = (v) => { const a = new Float32Array(8).fill(NaN); a[1 * 4 + 0 * 2 + 1] = v; return a; };
  const layers = new Map([['psf_mass', at(0.42)], ['psf_dz', at(2.5)],
                          ['psf_shift', at(-0.75)], ['psf_lat', at(1.2)]]);
  const s = hoverText({ i: 1, j: 0, k: 1, value: 0.0123 }, layers, shape);
  assert.match(s, /voxel \(1, 0, 1\)\s+value 0\.0123/);
  assert.match(s, /mass 0\.42/);
  assert.match(s, /depth spread 2\.50 m/);
  assert.match(s, /shift -0\.75 m/);
  assert.match(s, /lateral 1\.20 m/);
  assert.equal(hoverText({ i: 0, j: 0, k: 0, value: 1 }, new Map(), shape),
               'voxel (0, 0, 0)  value 1.0000');
  assert.match(hoverText({ i: 0, j: 0, k: 0, value: 1 }, layers, shape), /kernel not characterised/);
});
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `node --test viewer/test/`
Expected: FAIL (the depth-gate tests fail on `keep`; `hoverText` is not exported).

- [ ] **Step 3: Implement the pure modules**

`viewer/src/layers.mjs`, added to `KNOWN_LAYERS`:

```js
  psf_mass: { label: 'Kernel: mass recovered (1 = all)', kind: 'scalar' },
  psf_dz: { label: 'Kernel: depth spread (m)', kind: 'scalar' },
  psf_shift: { label: 'Kernel: depth shift (m)', kind: 'signed' },
  psf_lat: { label: 'Kernel: lateral width (m)', kind: 'scalar' },
```

`viewer/src/model.mjs`: after `hasSnr: false,`:

```js
    // Depth-spread gate (display-only): hide voxels whose point-spread function
    // spreads over more than maxDepthSpread metres in z (psf_dz.npy from
    // `reconstruct --kernels`). NaN = not characterised = hidden while on.
    // Off by default: it is a reading aid, not a data-quality verdict.
    depthGateEnabled: false,
    maxDepthSpread: 3,
    hasPsfDz: false,
```

and in `EFFECTS`, after the SNR line: `depthGateEnabled: ['window', 'render'], maxDepthSpread: ['window', 'render'],`

`viewer/src/gates.mjs`: update the NaN-semantics comment with the line `//   psf_dz NaN -> hidden (the kernel was not characterised there)`, then:

```js
  const dz = settings.depthGateEnabled && layers.psf_dz ? layers.psf_dz : null;
  const maxDz = settings.maxDepthSpread;
  const keepForWindow = (n) => (!rays || !(rays[n] < minRays)) && (!snr || snr[n] >= minSnr)
    && (!dz || dz[n] <= maxDz);
  ...
    windowActive: !!(rays || snr || dz),
    uniforms: {
      sigmaEnabled: !!sigma, sigmaValue,
      coverageEnabled: !!rays, minRays,
      snrEnabled: !!snr, minSnr,
      depthEnabled: !!dz, maxDepthSpread: maxDz,
    },
    key: [!!sigma, sigmaValue, !!rays, minRays, !!snr, minSnr, !!dz, maxDz].join('|'),
```

`viewer/src/picker.mjs`: add

```js
// Hover text: the picked value, plus the kernel metrics when the run ships
// them. Layer arrays are numpy C-order (nx, ny, nz): index i*ny*nz + j*nz + k.
export function hoverText(hit, layerData, shape) {
  const base = `voxel (${hit.i}, ${hit.j}, ${hit.k})  value ${hit.value.toFixed(4)}`;
  if (!layerData.has('psf_dz')) return base;
  const [, ny, nz] = shape;
  const n = hit.i * ny * nz + hit.j * nz + hit.k;
  const v = (name) => (layerData.has(name) ? layerData.get(name)[n] : NaN);
  const dz = v('psf_dz');
  if (!Number.isFinite(dz)) return `${base}\nkernel not characterised`;
  return `${base}\nkernel: mass ${v('psf_mass').toFixed(2)}  depth spread ${dz.toFixed(2)} m`
    + `  shift ${v('psf_shift').toFixed(2)} m  lateral ${v('psf_lat').toFixed(2)} m`;
}
```

- [ ] **Step 4: Run the node tests**

Run: `node --test viewer/test/`
Expected: PASS

- [ ] **Step 5: Add the parity case (failing browser test)**

In `tests/viewer/test_gate_parity.py`:
- In `_run`, add `psf_dz = np.full(SHAPE, 1.0, dtype=np.float32); psf_dz[COLS["A"]] = 9.0`. Add `("psf_dz", psf_dz)` to the save list and `"psf_dz"` to `meta["layers"]`.
- In `_set_only`'s JS, add `s.depthGateEnabled = gate === 'depth'; s.maxDepthSpread = 3;`.
- Add `("depth", "A")` to the `gate, hidden` parametrize list (column A is free once only one gate is on at a time).

Run: `uv run pytest tests/viewer/test_gate_parity.py -q -k depth`
Expected: FAIL. The CPU side already hides column A, but the GPU footprint stays lit, so the `lit_in_hidden` assertion fails.

- [ ] **Step 6: Implement the GL/DOM side** (`viewer/src/app.mjs`, `viewer/shell.html`)

- Uniform declarations next to `uniform float uMinSnr;`:
  ```glsl
  uniform sampler3D uDzTex;
  uniform bool uDepthGateEnabled;
  uniform float uMaxDepthSpread;
  ```
- `voxelGated` (the `!uCubeBaked` block), after the SNR line:
  ```glsl
      if (uDepthGateEnabled && !clipped) clipped = !(texture(uDzTex, tex).r <= uMaxDepthSpread);
  ```
- The fog march copy, after the SNR block:
  ```glsl
      if (uDepthGateEnabled && !clipped) {
        // !(<=) so a NaN spread (not characterised) is hidden, matching gates.mjs.
        clipped = !(texture(uDzTex, tex).r <= uMaxDepthSpread);
      }
  ```
- The uniform-name list: add `'uDzTex', 'uDepthGateEnabled', 'uMaxDepthSpread',`.
- The initial state: add `dzTex: dummyVolume,` next to `snrTex: dummyVolume`.
- `stateVoxelGates()`: add `psf_dz: state.hasPsfDz ? state.layerData.get('psf_dz') : null,`.
- The filter list in `applyVolumeFilter`: add `[state.dzTex, gl.NEAREST]` (a per-voxel gate, so NEAREST for the same reason as `rays` and `snr`).
- `render()` uniform upload, after `uMinSnr`:
  ```js
      gl.uniform1i(uniforms.uDepthGateEnabled, vg.depthEnabled ? 1 : 0);
      gl.uniform1f(uniforms.uMaxDepthSpread, vg.maxDepthSpread);
  ```
- Texture binding, after the TEXTURE5 (SNR) block:
  ```js
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_3D, state.dzTex);
      gl.uniform1i(uniforms.uDzTex, 7);
  ```
- `loadRun`, after the SNR texture block:
  ```js
      state.hasPsfDz = state.layerData.has('psf_dz');
      state.dzTex = state.hasPsfDz
        ? makeVolumeTexture(gl, meta.shape, state.layerData.get('psf_dz'))
        : dummyVolume;
      for (const id of ['#depth-gate-enabled', '#depth-gate-value']) {
        const el = root.querySelector(id);
        if (el) el.disabled = !state.hasPsfDz;
      }
  ```
- Listeners, after the SNR listeners:
  ```js
    root.querySelector('#depth-gate-enabled').addEventListener('change', (ev) => {
      commit({ depthGateEnabled: ev.target.checked });
    });
    root.querySelector('#depth-gate-value').addEventListener('input', (ev) => {
      const n = parseFloat(ev.target.value);
      if (!Number.isFinite(n)) return;
      commit({ maxDepthSpread: n }, { interactive: true });
    });
  ```
- Hover: import `hoverText` from `./picker.mjs`, and replace the `el.textContent = hit ? ... : ''` expression with `el.textContent = hit ? hoverText(hit, state.layerData, state.meta.shape) : '';`. Also add `el.style.whiteSpace = 'pre';` once at the top of the handler so the second line renders.
- `viewer/shell.html`, Uncertainty section. Replace the SNR label and its value row with:
  ```html
          <label title="Bootstrap NOISE signal-to-noise per voxel. It filters streaks the noisiest directions leave; it is not a detection test. On cafeteria phantoms 92% of SNR >= 3 voxels were empty in truth, because the bias of the one-sided geometry is not noise (see the kernel layers). Display-only."><input id="snr-gate-enabled" type="checkbox" style="width:auto" checked disabled> hide voxels with noise SNR below</label>
          <div class="row"><input id="snr-gate-value" type="number" min="0" max="100" step="0.5" value="3" style="width:4.5em" disabled> <span>(noise only &mdash; not a detection)</span></div>
          <label title="Depth spread of the solver's point-spread function at this voxel (reconstruct --kernels): a feature here would appear smeared over this many metres in z. NaN = not characterised = hidden while on. Display-only."><input id="depth-gate-enabled" type="checkbox" style="width:auto" disabled> hide voxels with depth spread above</label>
          <div class="row"><input id="depth-gate-value" type="number" min="0" max="50" step="0.25" value="3" style="width:4.5em" disabled> <span>m (kernel)</span></div>
  ```

- [ ] **Step 7: Run the viewer suites**

Run: `node --test viewer/test/` and then `uv run pytest tests/viewer -q`
Expected: PASS, including `test_shader_gate_matches_cpu_gate[depth-A-fog]` and `[depth-A-cubes]`.

- [ ] **Step 8: Commit**

```bash
git add viewer/ tests/viewer/test_gate_parity.py
git commit -m "feat(viewer): kernel layers, depth-spread gate, kernel hover readout; SNR is noise-only"
```

---

### Task 8: Full-scale runs and docs

**Files:**
- Modify: `CLAUDE.md`, `docs/cafeteria-run.md`, `docs/phase3-reconstruction-report.md`
- Run artefacts: `runs/cafeteria/voxels`, `runs/voxels` (git-ignored)

**Interfaces:** Consumes the CLI from Task 6. Produces only docs.

- [ ] **Step 1: Cafeteria at full scale** (in the foreground; about 12 min of kernels plus the bootstrap)

```bash
C=configs/cafeteria.yaml; R=runs/cafeteria
uv run python -m megido.cli reconstruct --config $C --solve $R/solve --run $R/ingest \
    --out $R/voxels --cache $R/.cache --bootstrap 8 --backproject-z 7.0 --kernels \
    | tee $R/reconstruct_kernels.log
uv run python -m megido.cli export --config $C --run $R/voxels
```

Expected: the kernel table prints. Record it verbatim. The bootstrap σ will be smaller than before (Task 2), so record the new SNR counts too.

- [ ] **Step 2: Megiddo at full scale** (in the foreground; about 1 h)

```bash
uv run python -m megido.cli reconstruct --bootstrap 8 --run runs/ingest --kernels \
    | tee runs/reconstruct_kernels.log
uv run python -m megido.cli export
```

- [ ] **Step 3: Viewer check.** Run `uv run python -m megido.cli view`, load `runs/cafeteria/voxels`, and confirm the four kernel layers appear in the layer switch, hover shows the kernel line, and the depth gate hides voxels as N falls. Take a screenshot for the report.

- [ ] **Step 4: Write the docs from the recorded numbers**

- `docs/cafeteria-run.md`: a new section, "Averaging kernels (what a voxel means)". Include the per-height table from Step 1; the spike's coverage numbers (point truth 0.33–0.40; `R·t` per gate 6); the reading rule "x̂ ± σ estimates the truth blurred by the PSF at that voxel; SNR is noise-only"; and the new SNR counts after the bootstrap fix.
- `docs/phase3-reconstruction-report.md`: a "Megiddo averaging kernels" section with the Step 2 table and the same reading rule.
- `CLAUDE.md`, "Two things that are true", first bullet: after "recovers **lateral** (column-integrated) structure", add the clause: ", in *pattern* (correlation), not amplitude: on the cafeteria phantom column integrals came out at a median 0.43× truth and mass is pulled toward the detector, because bias dominates noise there (`spike/sigma-calibration`; per-height kernel numbers in `docs/cafeteria-run.md`)".
- `CLAUDE.md`, the Phase 4 viewer paragraph: change the SNR gate description to "a noise filter, not a detection test (92% of SNR ≥ 3 voxels were truth-empty on cafeteria phantoms)". Add one sentence on the kernel layers and the depth-spread gate (off by default; NaN = not characterised = hidden). Add `megido/kernels.py` to the S4 description: "`reconstruct --kernels`: point-spread functions of the delivered solver; x̂ ± σ estimates the kernel-blurred truth".

- [ ] **Step 5: Run the full suite once**

Run: `uv run pytest -q` (one session only)
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs/cafeteria-run.md docs/phase3-reconstruction-report.md
git commit -m "docs: averaging kernels at full scale; SNR is noise-only; lateral claim is pattern, not amplitude"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §2 claim, §3.1 PSFs, sign, amplitude, determinism, offsets | 4 (PSF/δ), 1 (`stop_at`), 5/6 (`fit_offsets` passthrough) |
| §3.2 batching | 4 |
| §3.3 metrics + interpolation | 4, 5 |
| §4 gates 1–7 | 4 (1, 5), 5 (2, 3, 4), 6 (6), 2 (7) |
| §5 modules | 1–7 |
| §7 docs | 8 |
| §8 out of scope | none (intentionally) |

**2. Placeholder scan.** A broken draft of the gate-6 test was removed from Task 6 Step 1; only the complete body remains. The other steps carry full code.

**3. Type consistency.** `compute_kernels(..., batches=, keep_psfs=)`, `superpose(psfs, amplitudes, shape)`, `PSF.probe` as a tuple key, and `info["best_iter"]` are used identically in Tasks 4–6. The `hoverText(hit, layerData, shape)` signature matches its test and its app call.

**4. Review Focus.** Each of the five lines has a pinned test: Task 4 (three), Task 6 (cache key), Task 7 (inert gate).
