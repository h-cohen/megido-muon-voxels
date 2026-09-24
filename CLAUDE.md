# Megiddo muon voxels — working guide

Cosmic-ray muon tomography at the Megiddo cavern. The detector sits at several
positions and tilts inside a cavern; from the muon rate in each angular bin we
reconstruct a 3D opacity (rock-density) field of the surrounding rock, and from
the flux edge of the small hill overhead we fit the **hillside surface** itself.
The deliverables are a voxel field, that fitted surface, and one HTML viewer that
shows both — with **honest uncertainty as a first-class output, not an
appendix**.

The defining constraint: **there is no open-sky calibration run.** The detector's
own angular response and the rock's absorption must be separated from the cavern
data alone. That single fact shapes the whole design.

## The one physics idea the whole project rests on

The detector response is fixed in the **detector frame**; the rock opacity is
fixed in the **sky frame**. Tilting the detector (20° in this campaign) moves
every detector bin to a different patch of sky while the response stays put. That
displacement is what lets Phase 2 separate response from absorption with no
open-sky run. Read `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md`
before touching the physics — it is the binding authority and records the
reasoning, the dead ends, and every decision.

## Two things that are true and easy to forget

- **This is a one-sided geometry.** Every reconstructed ray points along
  `normalize(sx, sy, 1)` — upward — because muons arrive from above. The inverse
  problem is therefore limited-angle: the code reproduces the measurements almost
  perfectly and recovers **lateral** (column-integrated) structure, but **depth
  lives in the null space** and is not localized. A single-layer truth
  reconstructs smeared across z. This is physics, not a bug. Depth resolution
  comes only from horizontal parallax between positions
  (`dz ≈ √2·σ_t·z²/b`), and the real campaign has one baseline (2.2 m), giving
  ~1.4 m depth error at mid-range — depth is *not* resolved. Never write a gate
  or a claim that assumes depth is recovered. `megido/resolution.py` computes the
  honest numbers; report them. The productive response to this is a *different
  observable*, not despair: the detector sits under a small hill looking up, so
  the hill's silhouette is a sharp flux edge (thick overburden at the crown,
  near-open sky off the flanks) — a lateral feature this geometry resolves well,
  whose parallax between positions can extract the hillside surface where the
  voxel depth cannot. See spec §11 "Data-driven hillside extraction from the
  flux edge."
- **The absolute level of opacity is not measured.** Phase 2 pins each position's
  opacity gauge independently, so only differences are meaningful. Feed the solver
  `BaselineSolution.normalized_opacity(pid)`, never raw `.opacity[pid]` (which is
  gauge-pinned to median zero and half negative). The per-position additive
  offset `c_p` in the voxel solve is there to reabsorb this, not as a fudge.

## Honest limits — the ceiling, and what was tried against it

The pipeline is at the honest limit of what one 2.2 m baseline of muon-only data
supports. Three limits are **physical, not algorithmic** — do not try to code
around them, and do not accept a change that claims to beat them without new data:

- **Depth is null-space.** One-sided geometry; resolvable only by more baseline
  (a *translated* second position), never by a prior or a regulariser.
- **Absolute opacity level is gauge-degenerate.** `counts ∝ norm·exp(−λ)`, so
  `λ → λ+c` is exactly cancelled by `norm → norm·eᶜ`. Only differences (voxels)
  and shape (surface) are meaningful. Fixable only by an external reference
  (a surveyed overburden or known-density anchor), never from the data.
- **Absolute hillside height is that same degeneracy** in the surface: `H ∝ 1/ρ`
  via the assumed `a`. Shape is measured; scale is assumed and labelled so.

Ideas that were tried at full scale and **rejected with evidence** (don't
re-propose without new information — the findings are in the specs/plans):

- **Flux-weighting the voxel rows (cos²θ):** *hurts* lateral recovery. Unlike the
  surface, the voxel solve is a line-integral inversion where every oblique ray
  buys lateral coverage; down-weighting them (by flux or by 1/σ²) trades coverage
  for nothing, because this geometry is coverage-limited, not noise-limited.
- **Constraining the voxels to zero above the fitted surface:** rejected
  before building. `H` rides on the assumed `a` (at `a = 8` its median sits at
  the grid top; `a = 4` would halve the grid; `a = 16` does nothing), so the
  voxels' vertical structure would be set by an assumption — a prior shaping
  depth. Delivered instead as a display-only viewer clip.
- **"Fixing" the opacity gauge zero-point:** no free lunch. The clip-to-zero
  convention touches only ~5% of directions (the transparent quantile), picks
  physically sensible grazing directions, and cannot change the meaningful
  relative/shape outputs — because the absolute level is degenerate anyway.

The two real wins this revision landed: the volume **render axis-order fix** and
the **upper-envelope GP hillside surface** (the detector-spot dip was a genuine
defect — radial opacity-sorting — confirmed against ground truth and fixed).

## Architecture

Six content-addressed stages; downstream stages consume artifacts, never upstream
code:

```
S0-det  detector calibration   raw .data  -> channel map, bar geometry
S0-exp  exposure calibration   raw .data  -> pedestals, gains, live-time
S1      event reconstruction   raw .data  -> tracks -> per-exposure angular histograms
S2      baseline solve         histograms -> detector response B(d), per-position opacity
S3      forward model          poses      -> sparse path-length matrix A
S4      inversion              opacity, A -> voxel field + uncertainty
S5      viewer                 volume     -> interactive HTML
S6      hillside surface       opacity    -> fitted H(x,y) + posterior sigma
```

Built in phases, each with its own spec-referenced plan under
`docs/superpowers/plans/` and an exit gate:

- **Phase 1** (done): S0-det, S0-exp, S1 — raw `.data` to angular histograms.
- **Phase 2** (done): S2 — the baseline solve. This is the project core.
- **Phase 3** (done): S3, S4 — tilt-aware forward model, voxel inversion,
  uncertainty.
- **Phase 4** (done): S5 — the viewer. `viewer/src/*.mjs` (WebGL2
  raymarching, zero runtime dependencies) built by `megido/viewerbuild.py`
  into one self-contained `viewer/dist/index.html`. Loads any run directory
  at runtime via a local file picker; never bakes data into the shipped
  page. Rays march only inside the volume box (slab intersection,
  >= half-voxel steps, opacity-corrected to the legacy per-sample length so
  the transfer function keeps its look — exactly at the image centre; the old
  per-pixel step was longer toward the edges, so edges are now slightly
  denser). Display-only options: trilinear
  sampling (hardware `LINEAR`, manual 8-tap fallback when
  `OES_texture_float_linear` is missing — pixel-tested against each other),
  gradient shading, surface coloured by posterior σ, and "clip volume above
  surface" (off by default; it hides density the surface model calls air,
  it measures nothing).
- **Phase 5** (done): S6 — the hillside surface, the campaign's most resolvable
  observable. `megido/silhouette.py` extracts the flux-edge ridgeline;
  `megido/hillside_surface.py` fits a smooth height field `H(x,y)` by a
  **Gaussian process** (`megido/hillside_gp.py`, Matérn-5/2, ML-II
  hyperparameters, honest posterior-std error bars). The GP fits a **per-cell,
  flux-weighted (cos^index) upper quantile** of ray exit-z, NOT the raw exit
  points: the exit-point placement `o + a·λ·d̂` sorts rays radially by opacity,
  so a mean-like fit inverts the crown above each detector into a spurious dip;
  the upper envelope tracks the true maximum overburden per column and the flux
  weight restores the sparse high-flux near-vertical rays. Validated against
  `topography.csv` ground truth (test-only). Absolute height rides on an ASSUMED
  inverse-density scale `a` — the shape is measured, the scale is not (see
  Honest limits). `megido/hillside_check.py` is the surface's ray-space
  goodness-of-fit: it ray-traces every measured direction through the surface
  as a uniform solid (density `1/a`) and compares predicted with measured
  opacity, **gauge-invariant per position** (each position's unmeasured
  opacity level is fitted as an additive offset, reported, and removed — the
  role `c_p` plays in the voxel solve). Real run (heteroscedastic, `a = 8`,
  `q_hi = 0.85`, `cell_m = 1.0`): 32% VE per ray (pos0 r = 0.64, pos1
  r = 0.50; fitted offsets −0.18 / −0.78 = unmeasured level plus any constant
  model bias), against 81% per cell. The residual has coherent centre-vs-rim
  structure, but that is **not attributable to geology**: the offset removes
  only a constant, and a wrong `a` / density scale leaves a residual
  proportional to path length, which is itself a centre/rim pattern. The ray
  VE moves strongly with `a` — never quote it without `a` and the fit settings.

## Repo conventions — non-negotiable

- **Units.** Phase 1 and `megido/detector.py` are **centimetres**. Phase 3 is
  **metres**. `Pose.x/y/z` are metres. Convert at the boundary; never mix.
- **Poses live in data, not code.** All exposures, geometry, and reconstruction
  tunables are in `configs/megido.yaml`. Adding a datapoint means dropping files
  in the data dir and appending one exposure block — no code change. Never
  hardcode a pose, a baseline, or the aperture (it is
  `DetectorGeometry.megiddo().aperture_m = 0.384 m`; derive it, never retype it).
- **Cache versioning is a hard constraint.** Any change to reconstruction logic
  must bump `RECONSTRUCTION_VERSION` (Phase 1) or `INVERSION_VERSION`
  (`megido/raycast.py`, Phase 3). Both are part of the artifact cache key. In
  Phase 1 a cache that ignored the code version silently served stale artifacts
  after a fix; do not reopen that trap.
- **Reader memory.** The full ASCII ingest is ~6 GB; `dtype=str` chunked reads
  must stay at a bounded chunk size (50,000 rows measured safe; 200,000 exhausted
  the machine).
- **`np.nan` means "not constrained."** An unconstrained sky bin or an unviewed
  voxel is NaN, never 0 — "no ray went there" and "measured as zero" are
  different statements and nothing may conflate them.
- **Supplied detector constants are falsification-tested, never silently
  refitted.** If a constant fails its test against data, escalate to the
  engineers; do not quietly replace it.
- **Volume texture axis order.** `.npy` volumes are numpy C-order `(nx,ny,nz)`
  (z fastest); WebGL `texImage3D` reads x fastest. The viewer transposes at
  upload (`reorderForTexture` in `viewer/src/grid.mjs`) — the `.npy` files stay
  numpy-natural for every other consumer. A silent mismatch here rendered the
  volume as diagonal stripes and was misread as physics for a while; keep the
  transpose, and if a new 3D layer is added, route it through the same upload.

## Testing philosophy — learned the hard way

Every serious defect in every phase surfaced from **running real data at
full scale**, not from a green suite or a passing review. The sharpest cases: a
Phase 2 gauge degeneracy that 159 tests and an approving reviewer missed because
correlation and standard deviation are both invariant to the degenerate
direction; a Phase 3 phantom gate that revealed depth is unrecoverable only
when real geometry ran through it; a Phase 4 volume that rendered as diagonal
stripes from an axis-order upload bug and was briefly rationalised as physics;
and a Phase 5 hillside that inverted the crown above each detector into a dip —
caught only by checking the fit against `topography.csv` ground truth. So:

- **TDD**, red before green, every task.
- Prefer tests that assert **physical relationships** (a vertical ray deposits
  exactly the slab thickness; `dz` scales as `z²`) over tests that restate the
  implementation.
- A gate must be able to **fail for a broken implementation**. A gate that passes
  by construction tests nothing.
- When a hand-computed expected value and the code disagree, find out which side
  is wrong — do not tune the expectation to match the output, and never loosen a
  gate threshold to force green.
- Phantom gates are honest about what the geometry can do: data-space
  self-consistency and lateral recovery are asserted; depth fidelity is not,
  because it is physically impossible here.
- **Headless WebGL is software (SwiftShader).** The raymarch costs ~9.5 µs per
  pixel there, linear in pixel count; viewer browser tests therefore run at a
  760×520 viewport (`tests/viewer/conftest.py`). Never run two pytest
  sessions at once: each rebuilds `viewer/dist/index.html` and a half-written
  page makes the browser time out. Run test commands in the foreground and
  wait — a backgrounded run whose notification is lost looks like a stall.

## Development workflow

Work is driven by the `superpowers` skills: **brainstorming → writing-plans →
subagent-driven-development → finishing-a-development-branch**. Plans decompose
into TDD tasks; each task gets a fresh implementer subagent and a separate
reviewer, tracked in a git-ignored ledger under `.superpowers/sdd/`. Rulings
(decisions taken to keep a plan moving) are recorded in the ledger with what each
costs if wrong, and surfaced to the user at the end.

## Git

- Repo-local identity: `Hadar Cohen <hal.nls@gmail.com>`.
- Committing directly to `main` is authorized for this project.
- **Pushing is done by the user.** The permission classifier blocks the assistant
  from pushing to the public remote; suggest `! git push origin main` and let the
  user run it.

## Model selection

Match the model to the cognitive load of the task, not to the file count. Use the
least capable model that will reliably succeed — it is faster and cheaper — but
never under-resource physics, numerics, or the review of a load-bearing
algorithm.

**Always use /caveman skill in new agents**

| Task | Model |
|---|---|
| Overall architecture | Opus 4.8 |
| Physics / mathematical design | Opus 4.8 |
| Difficult numerical reasoning | Opus 4.8 |
| Algorithm choice | Opus 4.8 |
| Major debugging | Opus 4.8 |
| Code review of important algorithms | Opus 4.8 |
| Normal implementation | Sonnet 5 |
| Refactoring | Sonnet 5 |
| Tests | Sonnet 5 |
| Visualization | Sonnet 5 |
| Documentation | Sonnet 5 |
| Search codebase | Haiku |
| Find references / usages | Haiku |
| Simple data transformations | Haiku |
| Mechanical cleanup | Haiku |

How this maps onto the subagent workflow: the **controller** (planning, rulings,
adjudicating reviews, physics decisions) runs on Opus 4.8. **Implementer** subagents
run on Sonnet when the plan carries the full code, escalating to Opus 4.8 for a task
that turns out to need real numerical or algorithmic judgment. **Reviewers** scale
to the diff's risk — a mechanical diff takes Sonnet, but the review of a solver,
a forward model, or any load-bearing algorithm takes Opus. Broad whole-branch
reviews take Opus. Pure search and mechanical cleanup take Haiku. Always name the
model explicitly when dispatching; an omitted model silently inherits the
controller's, which is the most expensive one.

## Where things are

- Real data: `/home/hadar/Cloud/Work/Postdoc/01_data/processed/megido`
- Phantom / synthetic ground truth: `/home/hadar/Cloud/Work/Postdoc/01_data/raw`
  (`topography.csv`, `detector_config.csv`). `calibration_open_sky.csv` sits
  there too and is **deliberately never used** — this campaign has no open-sky
  run, and a phantom that used one would validate a pipeline we do not have.
- Reference project (read-only, never import): `../cafeteria_3d_modeling` — an
  earlier campaign with an open-sky run and different geometry. Port ideas and
  maths, never the dependency.
- Specs: `docs/superpowers/specs/`. Plans: `docs/superpowers/plans/`. Phase
  reports: `docs/phase*-report.md`.
- Code: `megido/`. Tests: `tests/` (one module per source module).

## Running it

```bash
uv run pytest -q                                   # the suite
uv run python -m megido.cli validate --exposure P0 # S0-det falsification checks
uv run python -m megido.cli ingest                 # S0-exp + S1 for every exposure
uv run python -m megido.cli solve                  # S2 baseline + opacity
uv run python -m megido.cli reconstruct --bootstrap --run runs/ingest  # S3/S4 voxels + uncertainty
uv run python -m megido.cli export                 # volume.npy + meta.json for the viewer
uv run python -m megido.cli hillside --run runs/ingest  # S6 silhouette + GP surface + ray check (hill_residual.png)
uv run python -m megido.cli view                   # Phase 4 viewer: build + open
# also: validate / compare subcommands
```

The hillside surface exposes `--surface-qhi` (upper-quantile level, default 0.85)
and `--surface-min-count`; `--run` enables heteroscedastic Poisson weights and is
required for `reconstruct --bootstrap` (counts are rebuilt from the ingest grid,
not carried in `baseline.npz`).
