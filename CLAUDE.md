# Megiddo muon voxels — working guide

Cosmic-ray muon tomography at the Megiddo cavern. The detector sits at several
positions and tilts inside a cavern; from the muon rate in each angular bin we
reconstruct a 3D opacity (rock-density) field of the surrounding rock. The
deliverable is a voxel field plus an HTML viewer, with **honest uncertainty as a
first-class output, not an appendix**.

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
```

Built in four phases, each with its own spec-referenced plan under
`docs/superpowers/plans/` and an exit gate:

- **Phase 1** (done): S0-det, S0-exp, S1 — raw `.data` to angular histograms.
- **Phase 2** (done): S2 — the baseline solve. This is the project core.
- **Phase 3** (in progress): S3, S4 — tilt-aware forward model, voxel inversion,
  uncertainty.
- **Phase 4** (done): S5 — the viewer. `viewer/src/*.mjs` (WebGL2
  raymarching, zero runtime dependencies) built by `megido/viewerbuild.py`
  into one self-contained `viewer/dist/index.html`. Loads any run directory
  at runtime via a local file picker; never bakes data into the shipped
  page.

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

## Testing philosophy — learned the hard way

Every serious defect in all three phases surfaced from **running real data at
full scale**, not from a green suite or a passing review. The sharpest cases: a
Phase 2 gauge degeneracy that 159 tests and an approving reviewer missed because
correlation and standard deviation are both invariant to the degenerate
direction; and a Phase 3 phantom gate that revealed depth is unrecoverable only
when real geometry ran through it. So:

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
uv run python -m megido.cli view                  # Phase 4 viewer: build + open
# Phase 3 (in progress): reconstruct / export / compare subcommands
```
