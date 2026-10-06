# Voxel averaging kernels — honest per-voxel uncertainty

Status: design approved in brainstorming, 2026-10-06. Binding authority for the
physics remains `2026-09-16-megido-muon-voxels-design.md`; this spec adds one S4
product and changes no physics.

## 1. Why

A spike (branch `spike/sigma-calibration`, `spikes/calib/RESULTS.md`) ran
count-level phantoms through the production chain and asked whether the
delivered voxel σ is calibrated: does the truth fall within ±1σ in 68% of the
voxels?

| scene | coverage of point truth |
|---|---|
| cafeteria beams (ceiling 7.0–7.4 m, beams 6.6–7.0 m) | 0.33–0.39; structure voxels 0.00 |
| cafeteria, same scene 2 m lower | 0.33–0.40 |
| Megiddo, production volume used as truth (best case) | 0.42–0.55 |
| Megiddo, same field flipped in z | 0.32–0.39; 1–4 m band 0.05 |

The positive control (one replica scored against the replica mean) gives
0.66–0.74, so the σ is a correct **noise** σ. What it misses is **bias**. On the
cafeteria beams, the median |E[x̂] − truth| is 1.7–1.8σ, while |x̂ − E[x̂]| is
0.4–0.6σ. The fit recovers 49% of the true mass, and only 22% of that sits in
the true 6.4–7.6 m band (truth: 99%). Vertical-column integrals come out at a
median of 0.43× truth. 92% of the voxels the viewer shows at SNR ≥ 3 are
empty in the truth. Earlier "lateral recovery" claims rest on correlation,
which cannot see this scale bias. That is the same blind spot that hid the
Phase 2 gauge bug.

Also found: `voxel_bootstrap` fits its replicas **without** the σ weights the
delivered volume uses, so σ describes a different estimator. For the weighted
estimator it is ~1.4× too large.

A neural-field representation was tried first (branch `spike/nerf-field`) and
rejected: see CLAUDE.md, "Honest limits".

## 2. The claim this spec makes honest

A delivered voxel value `x̂ ± σ` estimates the **kernel-blurred truth** `R·t`,
not the point truth `t`. `R` is the local resolution operator of the solver
that is actually delivered (TV + non-negativity + coverage damping + `c_p`
handling as configured), linearised around `x̂`. σ stays the counting-noise σ,
which the spike shows is calibrated for exactly that quantity. The four kernel
metrics (§3.3) tell the reader what the blur is at each location.

No prior is added, and no posterior σ for point truth is produced. In the null
space, a prior would set σ, the way a depth prior set depth (CLAUDE.md,
"Honest limits").

## 3. Method

### 3.1 Point-spread functions by perturbation

For a probe voxel `j`, add a bump to the measurement and re-solve:

    λ' = λ̂_meas + δ · A e_j
    PSF_j = (x̂(λ') − x̂(λ̂_meas)) / δ

`PSF_j` is column `j` of `R`: how a feature at `j` shows up in the delivered
volume. Columns are all that is needed. For the gate (§4),
`R·t = Σ_j t_j PSF_j`. True averaging kernels (rows of `R`) would need the
adjoint of a nonlinear iterative solver, which does not exist. Readers use the
PSFs ("a feature here would appear smeared over …").

- **Sign.** δ > 0 only: the response to *added* mass, which is the physically
  relevant direction. Under non-negativity, voxels at `x̂ = 0` respond one-sidedly.
  That is a property of the delivered estimator and is recorded, not
  corrected.
- **Amplitude.** `δ = delta_sigma · median(σ_rows crossing j) / spacing_m`.
  This is the density whose vertical path through one voxel adds about
  `delta_sigma` noise-σ of opacity. Linearity is checked at 0.5δ and 2δ (§4).
- **Determinism.** Production `sirt_tv` returns the best-χ² iterate. For
  perturbed data that can be a different iteration, which would make the PSF
  jump. Probe solves therefore run to the nominal solve's `best_iter` and
  return that iterate (`stop_at=` hook, §5). Plain `sirt` gets the same hook.
- **Offsets.** Probe solves reuse the nominal `fit_offsets` setting. At
  Megiddo, `c_p` is fitted, so part of a probe's mass may go into `c_p`. That is
  part of the delivered estimator, and it shows up as reduced mass recovery.

### 3.2 Batching

Probes at one height share a solve when they are far enough apart laterally.
A batch is the set of probes at one `z_level` on a lateral lattice spaced
`sep_m`. `(sep_m / spacing_m)²` offset batches per level fill the `spacing_m`
lattice. Only voxels crossed by ≥ 2 rays are probed (below that, the voxel is
those rays' private unknown; CLAUDE.md, viewer coverage gate). Each probe's
PSF is the batch response inside its own lateral cell (the `sep_m` square
centred on the probe, all z).

Cost at the defaults: cafeteria ~50 solves (~12 min); Megiddo ~1 h.
Interference between probes in a batch is not assumed away. The linearity
gate (§4, test 4) measures it.

### 3.3 Metrics per probe

Over the probe's cell, with `w = |PSF_j|`:

1. **Mass recovery** `Σ PSF_j` (dimensionless; 1 = all of the probe's mass
   recovered somewhere in its cell).
2. **Depth spread**: the width of the 16th–84th percentile interval of the
   z-marginal of `w`.
3. **Depth shift**: the z-centroid of `w` minus the probe's z.
4. **Lateral width**: twice the radius about the probe's (x, y) that holds
   68% of the xy-marginal of `w`.

Metrics are interpolated to every voxel trilinearly over the probe lattice,
renormalised over the available corners. A voxel with no probe within one
`spacing_m` is NaN ("not characterised", never 0).

## 4. Gates and tests

Each gate must be able to fail for a broken implementation. Tests use small
synthetic campaigns and run in the normal suite.

1. **Well-resolved control.** In the generous many-detector geometry of
   `tests/test_phantom.py`, a probe's PSF is compact: mass recovery ≥ 0.8,
   depth spread ≤ 2 voxels, |depth shift| ≤ 1 voxel. This catches broken probe,
   solve or metric plumbing.
2. **Cannot beat parallax.** In the two-position Megiddo geometry (one 2.2 m
   baseline), depth spread at height z is ≥ the analytic
   `dz ≈ √2·σ_t·z²/b` (`megido.resolution.depth_resolution`, with the same
   `σ_t` the CLI uses, `cli._SKY_SIGMA_T`) at every probed level, and it
   increases with z. The spread is a 68% interval width (≈ 2σ), while `dz` is
   a 1σ error, so the bound is conservative: failing it means the result is
   impossible, not just optimistic. This catches a solver or metric that
   claims depth localisation the data cannot support.
3. **One position (negative control).** With one position, depth spread at
   mid-height is ≥ half the grid's z extent.
4. **Linearity / superposition.** For a sparse spike phantom on the probe
   lattice, `Σ t_j PSF_j` predicts `x̂(λ̂ + A t) − x̂(λ̂)` with a relative L2
   error ≤ 0.2. PSFs measured at 0.5δ and 2δ agree to relative L2 ≤ 0.2.
   This catches batch interference, a broken iterate hook, or strong
   nonlinearity. If it fails on real geometry, the response is to widen
   `sep_m` or record the nonlinearity, never to loosen the threshold.
5. **Determinism.** The same inputs give bit-identical PSFs.
6. **Calibration (count level).** A small synthetic sky-reference campaign
   (the cafeteria config's two positions, 1.9 m apart; 0.5 m voxels; 40×40
   detector bins over |t| ≤ 1.0; live times 1000/1000/3000 s) with a spike
   phantom on the probe lattice, observed once, and σ from 16 replicas that
   Poisson-resample the observed counts:
   - `P(|x̂ − R·t| ≤ σ)` lies in **[0.58, 0.78]**. The band was fixed before
     building, from the spike's positive control (0.66–0.74) plus the 8-vs-32
     replica spread (~0.06).
   - Teeth: the same check with σ halved falls below 0.58.
   - Point-truth coverage `P(|x̂ − t| ≤ σ)` is reported and asserted lower
     than the `R·t` coverage. This contrast is the reason the feature exists.
7. **Bootstrap weights.** `voxel_bootstrap` replicas are fitted with the same
   σ weights as the delivered volume. This fails on today's code.

**Full-scale runs** (acceptance, numbers recorded, no thresholds tuned to them):
`reconstruct --kernels` on the cafeteria and on Megiddo, with the per-height
summary tables written to the docs (§7).

## 5. Architecture

**New module `megido/kernels.py`** (pure logic; I/O only for its own result):

- `probe_batches(grid, rays, kc) -> list[np.ndarray]`: probe voxel indices
  per batch.
- `point_spreads(fwd, data, rc, x_hat, best_iter, batch, delta, *, fit_offsets)
  -> dict[int, np.ndarray]`: one perturbed solve, split into per-probe PSFs
  over their cells.
- `psf_metrics(psf, probe, grid) -> dict[str, float]`.
- `KernelResult` (frozen dataclass): probe lattice, per-probe metrics,
  per-voxel interpolated metrics `[nx, ny, nz]` ×4, config and version;
  `save` / `load` (`kernels.npz`).
- `compute_kernels(fwd, data, rc, x_hat, info, kc, *, fit_offsets)
  -> KernelResult`: the orchestrator.

**Changes to existing modules:**

- `megido/inversion.py`: `sirt` and `sirt_tv` report `info["best_iter"]` and
  accept `stop_at: int | None`, which runs to iterate `k` and returns it.
- `megido/voxuncert.py`: `voxel_bootstrap(..., sigma=None)` passes `sigma` to
  each replica's `solve_voxels`; the CLI passes the same σ the delivered volume
  uses.
- `megido/raycast.py`: `INVERSION_VERSION` 2 → 3 (bootstrap output changes;
  `info` gains `best_iter`).
- `megido/config.py` plus both site YAMLs: a `kernels:` block with
  `spacing_m` (default 1.0), `sep_m` (3.0), `z_levels_m` (default: every
  `spacing_m` from `z_min + spacing/2`), and `delta_sigma` (1.0), each
  justified inline in the YAML. Tunables live in config, not code.
- `megido/cli.py`: `reconstruct --kernels` (opt-in; ~50+ solves). It writes
  `kernels.npz` plus four layers, `psf_mass.npy`, `psf_dz.npy`,
  `psf_shift.npy`, `psf_lat.npy` (numpy C-order `(nx, ny, nz)`, NaN where not
  characterised), and prints the per-height summary (median mass recovery,
  depth spread and depth shift per `spacing_m` band). The result is cached
  under a content-addressed key: A key + data hash + rc + `INVERSION_VERSION`
  + kernel config.
- `megido/volexport.py`: exports the four layers when present (the same path
  as `snr`).

**Viewer** (following the CLAUDE.md recipe):

- The four layers load through `runload.mjs` and appear in the layer switch;
  hover shows all four values at the picked voxel.
- New data gate: "hide voxels with depth spread > N m" (NaN → hidden while
  on; off by default). It goes in `gates.mjs`, both GLSL copies, the texture
  in `loadRun`, and a `test_gate_parity.py` case. New 3D textures go through
  `reorderForTexture`.
- The SNR gate label becomes "noise SNR ≥ N (not a detection)". Its behaviour
  is unchanged.

## 6. Data flow

```
baseline.npz + σ ──► solve_voxels ──► x̂, info.best_iter
                                  │
             fwd.A, data, rc ─────┴─► compute_kernels ──► kernels.npz
                                                   └─► psf_*.npy ──► export ──► viewer
```

Downstream consumers read `kernels.npz` / `psf_*.npy` only. Nothing upstream
imports `kernels.py`.

## 7. Docs and reporting

Written from the full-scale numbers:

- CLAUDE.md, "Two things that are true": the lateral claim gets its amplitude
  caveat. The lateral *pattern* is recovered (correlation), but column
  *amplitude* is biased (cafeteria phantom median 0.43×) and mass is pulled
  toward the detector. Quote the kernel numbers.
- CLAUDE.md viewer paragraph: the SNR gate is a noise filter, not a detector
  (the 92% figure); point to the depth-spread gate and the kernel layers.
- `docs/cafeteria-run.md`, and a Megiddo section in
  `docs/phase3-reconstruction-report.md`: the per-height kernel table, the
  coverage of `R·t` against point truth, and how to read a voxel
  ("x̂ ± σ estimates the truth blurred by the PSF at that location").

## 8. Out of scope (follow-ons)

1. **More-positions study.** Hypothetical translated positions run through
   the kernel and wrong-depth gates, TV against the multiresolution field.
   This answers "does a neural field win with more data" and "where should
   the next Megiddo position go". It is the natural next spec and reuses
   `kernels.py`.
2. **Count-level Poisson joint fit** (S2+S4 in one likelihood). Deferred
   until kernels show whether the likelihood, rather than the geometry, is
   what limits the result.
3. **Posterior σ for point truth.** Declined (§2).
4. **Hillside-surface kernels.** S6 already has a GP posterior and an
   out-of-sample check.

## 9. Decision record

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| What σ claims | Kernel-blurred truth `R·t` | Point truth via a posterior | A prior would set σ in the null space |
| How `R` is measured | Perturbation PSFs through the delivered solver | Linear-surrogate resolution matrix | A surrogate describes a different estimator (the same trap as the unweighted bootstrap) |
| Columns or rows of `R` | Columns (PSFs) | Rows (averaging kernels) | Rows need the adjoint of a nonlinear iterative solver; the gate needs only columns |
| Probe iterate | The nominal `best_iter` | Each solve's own best-χ² iterate | Keeps PSFs continuous and deterministic |
| Bootstrap weights | Same σ as the delivered volume | Unweighted replicas (today) | σ must describe the delivered estimator |
| Coverage band | [0.58, 0.78], fixed before building | Tuned after running | CLAUDE.md: never tune a gate to the output |
