# Viewer deepening: one voxel gate set, and initViewer split into pure modules

Date: 2026-09-28. Source: architecture review candidates 1 and 2. The user
answered the design questions with the recommended options and said "go ahead,
keep it clean".

## Why

`viewer/src/app.mjs` is the most-changed file in the repo (in 49 of the last
150 commits). Most of it is one closure, `initViewer`: 1775 lines, 36 inner
functions, ~60 listeners and 104 state writes. Two costs follow:

1. **The voxel gates are written out six times.** The copies are the fog
   march (GLSL), `voxelGated` (GLSL), `castHoverRay`, `castCubeRay`, the
   `keep()` in `ensureCubeGrid`, and `gatedForWindow`. The cube-grid cache key
   and the uniform upload each list the gate inputs once more. Each copy
   re-decides gate order, NaN handling and the "is this layer loaded" guards,
   and nothing but comments keeps them in step. Every recent feature (coverage
   gate, SNR gate, cubes, surface clip) paid this tax. One copy has already
   diverged: the auto-window leaves out the σ gate, and nothing says why.
2. **Viewer logic can only be tested in a browser.** Hover picking, the cube
   merge, artifact routing on load, the auto-window and each handler's
   follow-ups (re-window? rebuild mesh? preview?) all live inside the closure.
   The only way to test them is headless Chromium on SwiftShader, at about
   9.5 µs per pixel.

## Decisions (settled with the user)

- **Order:** the gate set comes first as its own change. The split follows in
  slices: picker, then run loader, then viewer model. Each slice lands with the
  whole suite green.
- **σ and the auto-window:** σ stays excluded, now on purpose and documented.
  The σ gate is a continuous slider; re-windowing on every drag would shift the
  colours under the user's hand. Coverage and SNR are number boxes.
- **GLSL:** stays hand-written. It reads uniforms that the gate set produces.
  One browser parity test checks shader against JS gate by gate.
- **Pure refactor:** nothing the user sees changes. No features or looks ride
  along.
- **Test facade:** `window.__viewerState` keeps its field names and hook
  functions (`pick`, `render`, `idleNow`, `beginInteraction`, `ensureCubeGrid`,
  `setActiveLayer`, `applyWindowForLayer`, `applyVolumeFilter`,
  `updateHillColourLegend`, …), so the existing browser tests stay unchanged.
  New node tests cover the new modules.
- **Process:** spec → plan → subagent-driven development, with an Opus
  reviewer on the gate-set and picker tasks.

## Rulings made while writing this spec (reviewable)

- **R1 — NaN ray counts follow the renderer.** Today `gatedForWindow` hides a
  voxel whose `rays` value is NaN (`rays >= min` is false). The shader keeps it
  (`rays < min` is false). The gate set uses the shader's rule everywhere.
  `rays.npy` is written by `rays_per_voxel` as counts and is never NaN, so no
  real run changes. *If wrong:* one line in the gate set.
- **R2 — "Gate inert" means the layer is absent or the gate is off.** Today
  the shader's σ gate reads the dummy 1×1 texture (value 0) when no σ layer is
  loaded, and `0 > value` is false for every reachable slider value (≥ 0). The
  gate set reports `sigma` as enabled only when the layer is loaded. This is
  identical in effect and explicit instead of accidental.
- **R3 — The model is a data table, not a framework.** It is a table from
  state field to effects, plus one `commit` in app.mjs that runs those effects
  in a fixed order. There is no event emitter, no reactive store, and no
  change to state identity (tests hold the object).
- **R4 — GL and DOM stay in app.mjs.** A fake `gl` would cost more than it
  proves. The adapters stay thin and browser-tested.

## Design

Four new modules in `viewer/src/`. None imports DOM or GL, so each runs under
`node --test`. Every one is added to `_MODULE_ORDER` in
`megido/viewerbuild.py` after `surfacemesh.mjs` and before `app.mjs`, in the
order below (each may import only from modules earlier in the list).

### 1. `gates.mjs` — the voxel gate set

It has one interface, built from the gate settings and the loaded layers:

- `voxelGates(settings, layers)`. `settings` holds `sigmaGateEnabled`,
  `sigmaGateValue`, `coverageGateEnabled`, `minRays`, `snrGateEnabled` and
  `minSnr`. `layers` holds `sigma`, `rays` and `snr` (a flat numpy-order array
  or absent). It returns an object with:
  - `keep(n)`: the data gates for the voxel at flat index `n`, in the fixed
    order σ, coverage, SNR. NaN semantics are defined once:
    - σ NaN is kept;
    - rays NaN is kept (R1);
    - SNR NaN is hidden ("the bootstrap said nothing").
  - `keepForWindow(n)`: coverage and SNR only. σ is excluded on purpose, and
    the reason is written in the source.
  - `uniforms`: `{sigmaEnabled, sigmaValue, coverageEnabled, minRays,
    snrEnabled, minSnr}` with the presence guards applied (R2). `render()`
    uploads exactly these.
  - `key`: a string that changes exactly when `keep` could change. The cube
    grid caches on it.
- `spatialGates(settings, surface)`. `settings` holds `clipMin`, `clipMax`,
  `clipPlaneEnabled`, `clipPlaneNormal`, `clipPlaneD` and `surfClip`.
  `surface` is `{H, gx, gy}` of the displayed height field, or null. It returns
  `keep(tex, world)`, applying clip box, then clip plane, then surface clip,
  matching the shader.

The shader keeps its two copies (`voxelGated` and the fog march), but they now
read only `uniforms` from the gate set. A comment at each copy points to
`gates.mjs` as the reference.

### 2. `picker.mjs` — hover picking as a pure function

- `pickVoxel(ndc, invViewProj, scene)` returns `{i, j, k, value}` or null.
  `scene` carries:
  - `meta`, the active layer `data` and `window`;
  - `renderMode`, `raySteps` and `minStepVoxels`;
  - the two gate sets;
  - the cube grid and `cubeThreshold`.
  It holds today's fog march (rayBox, the step rule, first sample above
  `window[0]`) and cube march (voxelMarch, gates at voxel centres, first voxel
  ≥ threshold), unchanged.
- `grid.mjs` gains two functions:
  - `voxelIndex(shape, i, j, k)`: the flat index, or −1 outside the grid,
    with the floor convention. `sampleNearest` is rewritten on top of it.
  - `cubeGrid(data, meta, block, keep)` → `{shape, spacing, values, baked}`:
    the pure half of `ensureCubeGrid`, placed next to `blockAverage`. The
    texture upload and the cache stay in app.mjs.

`castHoverRay` in app.mjs shrinks to "client pixel → NDC → `pickVoxel`".

### 3. `runload.mjs` — reading a run directory

- `readRun(files)`: takes File-like objects (`name`, `text()`, `arrayBuffer()`)
  and returns a dataset record:
  - `meta` (it throws when `meta.json` is missing, as today);
  - `silhouette` or null;
  - `hillSurface` or null: `{H, gx, gy, meta, sigma, residual, residualLim}`.
    It is null unless both the surface and its meta are present. σ and
    residual are dropped on a length mismatch. `residualLim` falls back to
    `symmetricLimit`.
  - `layers`: a Map holding only arrays whose length equals nx·ny·nz;
  - `sigmaMax` (a plain loop, never a spread);
  - `cubeThreshold` (`suggested_iso[0]`, else the mid value range).
- `readCompareVolume(files, primaryMeta)`: the compare-run read and grid
  check. It returns null when there is no `meta.json`, `{mismatch: true}` when
  shape, spacing or origin differ, and `{volume}` otherwise.

`loadRun` in app.mjs becomes: `readRun` → apply the record to state → DOM
enable/disable → textures → window → render.

### 4. `model.mjs` — state and its follow-ups

- `createState(defaults)`: today's initial `state` literal, minus the GL
  handles, which app.mjs attaches.
- `EFFECTS`: a table from state field to the set of follow-ups it needs, drawn
  from `'filter'`, `'window'`, `'surfaceMesh'`, `'legend'`, `'histogram'`,
  `'lut'` and `'render'`. Examples: `minSnr` → window, render; `smoothSampling`
  → filter, render; `hillColourMode` → legend, surfaceMesh, render.
- `effectsFor(patch)`: the union for the fields in a patch. It throws on a
  field that is not in the table, so a new field cannot be forgotten.

app.mjs gains one `commit(patch, {interactive})`. It assigns the patch, runs
the effects in a fixed order (filter → window → surfaceMesh → legend →
histogram → lut → render), and calls `beginInteraction()` when `interactive`.
Each DOM handler reduces to reading its input and calling `commit`. Where a
handler has DOM-only side effects (a readout's text), those stay in the
handler.

## Testing

- **Node, new:**
  - `viewer/test/gates.test.mjs`: a truth table for `keep` over every
    NaN/threshold case, presence guards, `keepForWindow` excluding σ, `key`
    changing with each input, and the spatial gate order.
  - `picker.test.mjs`: fog hit is the first sample above the window; each gate
    hides; cube hit is the first cube; baked grids skip the data gates; misses
    return null.
  - `runload.test.mjs`: fake files for a missing meta, the layer length
    filter, the optional silhouette and surface, the residualLim fallback, and
    a σ length mismatch.
  - `model.test.mjs`: sample effects, union, and unknown field throws.
- **Browser, new:** `tests/viewer/test_gate_parity.py`. A fixture where σ,
  coverage and SNR each hide a different voxel. For each gate alone, in fog
  and in cubes, the lit pixels agree with `pick` on a grid (the ≤ 2% mismatch
  rule of `test_voxel_cubes`). This is the gate that catches the GLSL copies
  drifting from `gates.mjs`.
- **Browser, existing:** unchanged, green after every task. A browser test
  that needs editing means behaviour moved, and the task stops for a ruling.

## Constraints

- No user-visible change. The only semantic ruling is R1, a no-op on every
  real run.
- The new modules import no DOM or GL. They import only earlier modules in
  `_MODULE_ORDER`.
- `window.__viewerState` field names and hook functions are unchanged.
- No new runtime dependencies. The viewer ships as one self-contained file.
- The build concatenates the modules into one scope and strips only
  single-line `import { … } from '…';` statements. Imports therefore stay on
  one line, and every top-level name must be unique across
  `viewer/src/*.mjs`.
- Test commands run in the foreground, one pytest session at a time
  (`viewer/dist` is rebuilt per session).
- CLAUDE.md's viewer paragraph is updated in the last task to name the four
  modules and to state that `gates.mjs` is the gate reference.

## Out of scope

Deleting `megido/hillside.py` (candidate 4), the CLI and campaign candidates
(3, 5, 6, 7), pruning browser tests, and any new viewer feature.
