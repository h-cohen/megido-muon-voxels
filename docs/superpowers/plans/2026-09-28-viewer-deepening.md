# Viewer Deepening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the six hand-synced voxel-gate copies in the viewer with one pure gate set, and move hover picking, run loading and the state→follow-up rules out of `initViewer` into pure, node-tested modules. Nothing the user sees changes.

**Architecture:** Four new ES modules go in `viewer/src/`: `gates.mjs`, `picker.mjs`, `runload.mjs` and `model.mjs`. None touches DOM or GL. `app.mjs` keeps the GL and DOM adapters and calls into them. The bundler (`megido/viewerbuild.py`) concatenates modules in `_MODULE_ORDER`; each new module is added there.

**Tech Stack:** Plain ES modules, WebGL2 GLSL, `node --test`, pytest + Playwright (headless Chromium/SwiftShader).

**Spec:** `docs/superpowers/specs/2026-09-28-viewer-deepening-design.md`

## Global Constraints

- Pure refactor: no user-visible change. The only semantic ruling is R1 (NaN ray count kept, as the shader does); `rays.npy` is never NaN.
- The new modules import no DOM or GL, and only import modules earlier in `_MODULE_ORDER`.
- Imports stay on ONE line: `import { a, b } from './x.mjs';`. The bundler strips only that form.
- Every top-level name must be unique across all `viewer/src/*.mjs`, because the bundle shares one scope.
- `window.__viewerState` keeps its field names and hook functions (`pick`, `render`, `idleNow`, `beginInteraction`, `ensureCubeGrid`, `setActiveLayer`, `applyWindowForLayer`, `applyVolumeFilter`, `updateHillColourLegend`, `drawHistogram`, `drawXferEditor`, `drawLegend`). Existing browser tests stay unmodified. A test that needs editing means behaviour moved: stop and report.
- Gate sets are rebuilt from `state` at each use (they are cheap). Tests set `state.*` fields directly and then call `state.render()`, so nothing may cache a gate set across calls.
- No new dependencies.
- Run tests in the foreground and wait. Never run two pytest sessions at once, because each rebuilds `viewer/dist/index.html`.
- Commands: `node --test viewer/test/*.test.mjs` (node) and `uv run pytest -q tests/viewer` (browser). Before each commit, run the full suite with `uv run pytest -q`.
- Commit to `main` (authorized). End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016AiXeo91wS1Myr2DX76ypZ
  ```

---

### Task 1: `gates.mjs`, the single voxel gate set, wired everywhere

**Files:**
- Create: `viewer/src/gates.mjs`
- Create: `viewer/test/gates.test.mjs`
- Create: `tests/viewer/test_gate_parity.py`
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: add `"gates.mjs"` after `"surfacemesh.mjs"`)
- Modify: `viewer/src/app.mjs`. Change `render()` uniform upload (lines ~649-654), `gatedForWindow` (~1367-1377), `ensureCubeGrid` (~1818-1843), `castHoverRay` (~1937-1971), `castCubeRay` (~1992-2012), and the two GLSL gate comments (~111-113 and above the fog-march gates ~227).

**Interfaces:**
- Consumes: `insideClipBox(tex, min, max)` and `insideClipPlane(tex, normal, d)` from `clip.mjs`; `surfaceHeightAt(H, gx, gy, x, y)` from `surfacemesh.mjs`.
- Produces:
  - `voxelGates(settings, layers)` → `{ keep(n), keepForWindow(n), windowActive, uniforms: {sigmaEnabled, sigmaValue, coverageEnabled, minRays, snrEnabled, minSnr}, key }`.
    - `settings` has `sigmaGateEnabled, sigmaGateValue, coverageGateEnabled, minRays, snrGateEnabled, minSnr`.
    - `layers` has `sigma, rays, snr`, each a flat numpy-order array or null/undefined.
  - `spatialGates(settings, surface)` → `{ keep(tex, world) }`.
    - `settings` has `clipMin, clipMax, clipPlaneEnabled, clipPlaneNormal, clipPlaneD, surfClip`.
    - `surface` is `{H, gx, gy}` or null.
  - In app.mjs, two private helpers: `stateVoxelGates()` and `stateSpatialGates()`. They build the gate sets from `state`.

- [ ] **Step 1: Write the failing node test** `viewer/test/gates.test.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { voxelGates, spatialGates } from '../src/gates.mjs';

const OFF = { sigmaGateEnabled: false, sigmaGateValue: 1, coverageGateEnabled: false, minRays: 2, snrGateEnabled: false, minSnr: 3 };
const ALL = { ...OFF, sigmaGateEnabled: true, coverageGateEnabled: true, snrGateEnabled: true };
// index:                0    1    2    3    4    5
const sigma = Float32Array.from([0.5, 2.0, NaN, 0.5, 0.5, 0.5]);
const rays  = Float32Array.from([5,   5,   5,   1,   NaN, 5]);
const snr   = Float32Array.from([4,   4,   4,   4,   4,   NaN]);
const L = { sigma, rays, snr };

test('all gates off keeps every voxel', () => {
  const g = voxelGates(OFF, L);
  for (let n = 0; n < 6; n++) assert.equal(g.keep(n), true);
  assert.equal(g.windowActive, false);
});

test('keep truth table with every gate on (NaN semantics)', () => {
  const g = voxelGates(ALL, L);
  assert.deepEqual([0, 1, 2, 3, 4, 5].map(g.keep), [
    true,   // passes all
    false,  // sigma 2 > 1
    true,   // sigma NaN is kept
    false,  // rays 1 < 2
    true,   // rays NaN is kept (R1: shader rule)
    false,  // snr NaN is hidden
  ]);
});

test('keepForWindow ignores the sigma gate on purpose', () => {
  const g = voxelGates(ALL, L);
  assert.equal(g.keepForWindow(1), true);    // hidden by sigma, still windows
  assert.equal(g.keepForWindow(3), false);   // coverage still applies
  assert.equal(g.keepForWindow(5), false);   // snr still applies
  assert.equal(g.windowActive, true);
});

test('a gate whose layer is absent is inert and reports disabled', () => {
  const g = voxelGates(ALL, {});
  for (let n = 0; n < 6; n++) assert.equal(g.keep(n), true);
  assert.deepEqual(g.uniforms, { sigmaEnabled: false, sigmaValue: 1, coverageEnabled: false, minRays: 2, snrEnabled: false, minSnr: 3 });
  assert.equal(g.windowActive, false);
});

test('uniforms report enabled only when gate on AND layer present', () => {
  const g = voxelGates({ ...OFF, snrGateEnabled: true }, L);
  assert.equal(g.uniforms.snrEnabled, true);
  assert.equal(g.uniforms.sigmaEnabled, false);
  assert.equal(g.uniforms.coverageEnabled, false);
});

test('key changes with every input that can change keep()', () => {
  const base = voxelGates(ALL, L).key;
  for (const patch of [{ sigmaGateEnabled: false }, { sigmaGateValue: 1.5 }, { coverageGateEnabled: false },
    { minRays: 3 }, { snrGateEnabled: false }, { minSnr: 2 }]) {
    assert.notEqual(voxelGates({ ...ALL, ...patch }, L).key, base, JSON.stringify(patch));
  }
  assert.notEqual(voxelGates(ALL, { rays, snr }).key, base);   // sigma layer gone
  assert.equal(voxelGates({ ...ALL }, L).key, base);
});

const CLIP = { clipMin: [0, 0, 0], clipMax: [1, 1, 1], clipPlaneEnabled: false, clipPlaneNormal: [0, 0, 1], clipPlaneD: 0, surfClip: false };

test('spatial: clip box, clip plane and surface clip each hide', () => {
  const mid = [0.5, 0.5, 0.5], w = [0, 0, 5];
  assert.equal(spatialGates(CLIP, null).keep(mid, w), true);
  assert.equal(spatialGates({ ...CLIP, clipMax: [1, 1, 0.4] }, null).keep(mid, w), false);
  assert.equal(spatialGates({ ...CLIP, clipPlaneEnabled: true, clipPlaneD: 0.2 }, null).keep(mid, w), false);
  const surface = { H: Float32Array.from([4, 4, 4, 4]), gx: [-1, 1], gy: [-1, 1] };
  assert.equal(spatialGates({ ...CLIP, surfClip: true }, surface).keep(mid, w), false);   // z 5 above H 4
  assert.equal(spatialGates({ ...CLIP, surfClip: true }, surface).keep(mid, [0, 0, 3]), true);
  assert.equal(spatialGates({ ...CLIP, surfClip: false }, surface).keep(mid, w), true);
  assert.equal(spatialGates({ ...CLIP, surfClip: true }, null).keep(mid, w), true);       // no surface: inert
});
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `node --test viewer/test/gates.test.mjs`
Expected: FAIL (cannot find module `../src/gates.mjs`).

- [ ] **Step 3: Write `viewer/src/gates.mjs`**

Check `insideClipPlane`'s sign convention in `clip.mjs` first. The shader clips when `dot(tex-0.5, n) - d < 0`, and `insideClipPlane` must return false exactly then. The test above assumes plane normal +z and d = 0.2 hides the centre (0.5 - 0.5 - 0.2 < 0). If `clip.mjs` disagrees, the test's expectation is wrong for this repo: fix the test, not `clip.mjs`.

```js
import { insideClipBox, insideClipPlane } from './clip.mjs';
import { surfaceHeightAt } from './surfacemesh.mjs';

// The voxel gate set: the ONE reference for which voxels the viewer hides.
// Every CPU consumer (hover picking, the cube-size merge, the auto window)
// asks this module, and render() uploads `uniforms` for the two GLSL copies
// (voxelGated and the fog march in app.mjs), which must follow the same rules.
// tests/viewer/test_gate_parity.py checks shader against this, gate by gate.
//
// NaN semantics, defined once:
//   sigma NaN -> kept   (no bootstrap sigma is not "too uncertain")
//   rays  NaN -> kept   (the shader's rule; rays.npy is counts, never NaN)
//   snr   NaN -> hidden (the bootstrap said nothing: not significant)
// A gate is inert when it is off OR its layer is not loaded.
export function voxelGates(settings, layers) {
  const sigma = settings.sigmaGateEnabled && layers.sigma ? layers.sigma : null;
  const rays = settings.coverageGateEnabled && layers.rays ? layers.rays : null;
  const snr = settings.snrGateEnabled && layers.snr ? layers.snr : null;
  const sigmaValue = settings.sigmaGateValue, minRays = settings.minRays, minSnr = settings.minSnr;
  // The auto window deliberately ignores the sigma gate: sigma is a
  // continuous slider, and re-windowing on every drag would shift the colours
  // under the user's hand. Coverage and SNR hide the noise shell whose
  // inflated values would otherwise set the colour scale.
  const keepForWindow = (n) => (!rays || !(rays[n] < minRays)) && (!snr || snr[n] >= minSnr);
  const keep = (n) => (!sigma || !(sigma[n] > sigmaValue)) && keepForWindow(n);
  return {
    keep,
    keepForWindow,
    windowActive: !!(rays || snr),
    uniforms: {
      sigmaEnabled: !!sigma, sigmaValue,
      coverageEnabled: !!rays, minRays,
      snrEnabled: !!snr, minSnr,
    },
    key: [!!sigma, sigmaValue, !!rays, minRays, !!snr, minSnr].join('|'),
  };
}

// Geometric gates at a sample: clip box, clip plane, then surface clip, the
// shader's order. `tex` is the [0,1]^3 volume coordinate, `world` metres.
// `surface` is the DISPLAYED (possibly smoothed) height field {H, gx, gy}.
export function spatialGates(settings, surface) {
  const surf = settings.surfClip && surface && surface.H ? surface : null;
  return {
    keep(tex, world) {
      if (!insideClipBox(tex, settings.clipMin, settings.clipMax)) return false;
      if (settings.clipPlaneEnabled && !insideClipPlane(tex, settings.clipPlaneNormal, settings.clipPlaneD)) return false;
      if (surf) {
        const h = surfaceHeightAt(surf.H, surf.gx, surf.gy, world[0], world[1]);
        if (Number.isFinite(h) && world[2] > h) return false;
      }
      return true;
    },
  };
}
```

- [ ] **Step 4: Run the node test and confirm it passes**

Run: `node --test viewer/test/gates.test.mjs`
Expected: all 7 tests PASS.

- [ ] **Step 5: Write the failing browser parity test** `tests/viewer/test_gate_parity.py`

It uses the same helpers pattern as `tests/viewer/test_voxel_cubes.py`. There are three columns of bright voxels: σ hides the first, coverage the second, SNR the third. For each gate alone and each render mode, hover and pixels must agree, and the gated column must vanish from both.

```python
"""Shader gates agree with gates.mjs, gate by gate, in fog and cube mode.

Three bright columns: sigma hides column A, coverage hides B, SNR hides C.
With one gate on, `pick` (CPU, gates.mjs) and the lit pixels (GPU, GLSL
copies) must agree, and the gated column must be gone from both.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

SHAPE = (9, 3, 4)
COLS = {"A": 1, "B": 4, "C": 7}          # x index of each bright column
_GRID = [(fx / 40, fy / 40) for fx in range(2, 39) for fy in range(2, 39)]


def _run(run_fixture):
    run = run_fixture(shape=SHAPE)
    vol = np.zeros(SHAPE, dtype=np.float32)
    for x in COLS.values():
        vol[x, 1, :] = 1.0
    sigma = np.full(SHAPE, 0.1, dtype=np.float32)
    sigma[COLS["A"]] = 5.0
    rays = np.full(SHAPE, 10.0, dtype=np.float32)
    rays[COLS["B"]] = 1.0
    snr = np.full(SHAPE, 10.0, dtype=np.float32)
    snr[COLS["C"]] = 0.5
    for name, arr in [("volume", vol), ("sigma", sigma), ("rays", rays), ("snr", snr)]:
        np.save(run / f"{name}.npy", arr)
    meta = json.loads((run / "meta.json").read_text())
    meta["layers"] = ["volume", "sigma", "rays", "snr"]
    meta["value_range"] = [0.0, 1.0]
    meta["suggested_iso"] = [0.5, 0.8]
    (run / "meta.json").write_text(json.dumps(meta))
    return run


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-top").click()


def _set_only(page, gate):
    """Enable exactly one of sigma / coverage / snr via state, then idle-render."""
    page.evaluate("""(gate) => {
        const s = window.__viewerState;
        s.sigmaGateEnabled = gate === 'sigma';
        s.sigmaGateValue = 1.0;
        s.coverageGateEnabled = gate === 'coverage';
        s.minRays = 2;
        s.snrGateEnabled = gate === 'snr';
        s.minSnr = 3;
        s.window = [0.1, 1.0];
        s.idleNow();
    }""", gate)


def _picks_and_lit(page):
    return page.evaluate("""async (pts) => {
        const s = window.__viewerState;
        s.idleNow();
        const cv = document.querySelector('#gl-canvas');
        const r = cv.getBoundingClientRect();
        const img = new Image(); img.src = cv.toDataURL(); await img.decode();
        const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
        const g = c.getContext('2d'); g.drawImage(img, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data;
        const bg = [d[0], d[1], d[2]];
        return pts.map(([fx, fy]) => {
            const p = s.pick(r.left + fx * r.width, r.top + fy * r.height);
            const px = Math.min(c.width - 1, Math.floor(fx * c.width));
            const py = Math.min(c.height - 1, Math.floor(fy * c.height));
            const o = (py * c.width + px) * 4;
            const lit = Math.abs(d[o] - bg[0]) + Math.abs(d[o + 1] - bg[1]) + Math.abs(d[o + 2] - bg[2]) > 12;
            return [p ? p.i : null, lit];
        });
    }""", _GRID)


@pytest.mark.parametrize("mode", ["fog", "cubes"])
@pytest.mark.parametrize("gate, hidden", [("sigma", "A"), ("coverage", "B"), ("snr", "C")])
def test_shader_gate_matches_cpu_gate(page, dist_path, run_fixture, mode, gate, hidden):
    _load(page, dist_path, _run(run_fixture))
    if mode == "cubes":
        page.locator("#render-mode").select_option("cubes")
    _set_only(page, gate)
    rows = _picks_and_lit(page)
    picked_cols = {i for i, _ in rows if i is not None}
    assert COLS[hidden] not in picked_cols                      # CPU hides the gated column
    assert {COLS[k] for k in COLS if k != hidden} <= picked_cols  # and keeps the others
    mismatch = sum((i is not None) != lit for i, lit in rows)
    assert sum(lit for _, lit in rows) > 0
    assert mismatch <= 0.03 * len(rows)                          # GPU agrees with CPU
```

- [ ] **Step 6: Run the parity test on the current code (baseline)**

Run: `uv run pytest -q tests/viewer/test_gate_parity.py`
Expected: PASS on the current code. It is a characterization gate: the GLSL and today's JS copies agree now, and the test guards the refactor.

The 3% tolerance allows for edge pixels; the fog render is soft. If fog fails only on the mismatch fraction, raise `s.window[0]` (the fog lit threshold) until lit pixels are where samples exceed the window. Do not loosen the tolerance above 3%. Report either change in the task report.

If it fails for a real mismatch between the GPU and CPU gates, STOP and report: that is a pre-existing drift the controller must rule on.

- [ ] **Step 7: Wire `gates.mjs` into app.mjs (behaviour-identical)**

Add to the single-line import block at the top of app.mjs:

```js
import { voxelGates, spatialGates } from './gates.mjs';
```

Add `"gates.mjs"` to `_MODULE_ORDER` in `megido/viewerbuild.py`, right after `"surfacemesh.mjs"`.

Inside `initViewer`, just after `worldBounds()`, add:

```js
  // Gate sets are rebuilt from state at each use (cheap, and tests mutate
  // state directly before calling render()/pick()).
  function stateVoxelGates() {
    return voxelGates(state, {
      sigma: state.layerData.get('sigma'),
      rays: state.hasRays ? state.layerData.get('rays') : null,
      snr: state.hasSnr ? state.layerData.get('snr') : null,
    });
  }
  function stateSpatialGates() {
    const surface = state.hillSurface && state.hillDisplayH
      ? { H: state.hillDisplayH, gx: state.hillSurface.gx, gy: state.hillSurface.gy } : null;
    return spatialGates(state, surface);
  }
```

In `render()`, replace the six gate uniform lines (`uSigmaGateEnabled` through `uMinSnr`) with:

```js
    const vg = stateVoxelGates().uniforms;
    gl.uniform1i(uniforms.uSigmaGateEnabled, vg.sigmaEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uSigmaGateValue, vg.sigmaValue);
    gl.uniform1i(uniforms.uCoverageGateEnabled, vg.coverageEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uMinRays, vg.minRays);
    gl.uniform1i(uniforms.uSnrGateEnabled, vg.snrEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uMinSnr, vg.minSnr);
```

Replace the body of `gatedForWindow(data)`, and its comment, with:

```js
    // The auto window comes from the voxels the window gates KEEP (see
    // keepForWindow in gates.mjs for why sigma is not one of them).
    function gatedForWindow(data) {
      const g = stateVoxelGates();
      if (!g.windowActive) return data;
      const out = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) out[i] = g.keepForWindow(i) ? data[i] : NaN;
      return out;
    }
```

In `ensureCubeGrid()`, replace the key and the `sigma/rays/snr/keep` block:

```js
    const vgates = stateVoxelGates();
    const key = [b, state.activeLayer, state.loadSeq, vgates.key].join('|');
    ...
      const merged = blockAverage(data, meta.shape, b, vgates.keep);
```

Keep the rest of `ensureCubeGrid` as is. Task 2 moves its pure half.

In `castHoverRay`, delete the `sigmaData/sigmaGateActive/raysData/coverageGateActive/snrData/snrGateActive` block. Before the loop, add `const vgates = stateVoxelGates(), sgates = stateSpatialGates();`. Replace the loop body's gate section so it reads:

```js
      const [vi, vj, vk] = worldToVoxel(world, state.meta);
      const n = voxelFlatIndex(state.meta.shape, vi, vj, vk);
      if (n < 0 || !sgates.keep(tex, world) || !vgates.keep(n)) continue;
      const value = data[n];
      if (!Number.isNaN(value) && value > (state.window ? state.window[0] : 0)) {
        return { i: Math.floor(vi), j: Math.floor(vj), k: Math.floor(vk), value };
      }
```

`voxelFlatIndex` is a local helper added in this task inside `initViewer`: `const voxelFlatIndex = (shape, i, j, k) => { const [nx, ny, nz] = shape; const a = Math.floor(i), b2 = Math.floor(j), c = Math.floor(k); return (a < 0 || a >= nx || b2 < 0 || b2 >= ny || c < 0 || c >= nz) ? -1 : a * ny * nz + b2 * nz + c; };`. Task 2 moves it to `grid.mjs` as `voxelIndex`.

`n < 0` → skip is behaviour-identical: out of the grid, the old `sampleNearest` returned NaN and could never hit.

In `castCubeRay`, delete `sigmaData/raysData/snrData`, add `const vgates = stateVoxelGates(), sgates = stateSpatialGates();`, and make the visitor body:

```js
      if (!sgates.keep(tex, world)) return false;
      if (!grid.baked && !vgates.keep(i * ny * nz + j * nz + k)) return false;
      const v = grid.values[i * cy * cz + j * cz + k];
      if (v >= state.cubeThreshold) { value = v; return true; }
      return false;
```

Delete the now-unused `at` helper.

In `FRAGMENT_SRC`, change the comment above `voxelGated` to:

```
// Cube mode: every gate evaluated once per voxel, at its centre. GLSL twin of
// gates.mjs (the reference): uniforms come from voxelGates().uniforms;
// tests/viewer/test_gate_parity.py checks the two agree.
```

Add one line above the fog march's `if (uSigmaGateEnabled) {` block:

```
    // Gates: GLSL twin of gates.mjs (voxelGates + spatialGates); keep in step.
```

- [ ] **Step 8: Run all viewer tests**

Run: `node --test viewer/test/*.test.mjs` then `uv run pytest -q tests/viewer`
Expected: all PASS, with no existing test modified.

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q`
Expected: exit 0.

- [ ] **Step 10: Commit**

```bash
git add viewer/src/gates.mjs viewer/test/gates.test.mjs tests/viewer/test_gate_parity.py megido/viewerbuild.py viewer/src/app.mjs
git commit -m "refactor(viewer): one voxel gate set (gates.mjs) replaces six hand-synced copies" -m "<attribution lines>"
```

---

### Task 2: `picker.mjs`, hover picking as a pure function, and `cubeGrid` in grid.mjs

**Files:**
- Create: `viewer/src/picker.mjs`
- Create: `viewer/test/picker.test.mjs`
- Modify: `viewer/src/grid.mjs` (add `voxelIndex`, `cubeGrid`; rewrite `sampleNearest` on `voxelIndex`)
- Modify: `viewer/test/grid.test.mjs` (add tests for `voxelIndex`, `cubeGrid`)
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: `"picker.mjs"` after `"gates.mjs"`)
- Modify: `viewer/src/app.mjs` (`castHoverRay`, `castCubeRay`, `ensureCubeGrid`; delete `voxelFlatIndex`)

**Interfaces:**
- Consumes (Task 1): gate-set objects with `.keep(n)` and `.keep(tex, world)`; app's `stateVoxelGates()` / `stateSpatialGates()`.
- Produces:
  - `voxelIndex(shape, i, j, k)` → integer flat index (floor), or −1 outside.
  - `cubeGrid(data, meta, block, keep)` → `{shape, spacing, values, baked}`.
  - `pickVoxel(ndc, invViewProj, scene)` → `{i, j, k, value}` or null.
    - `ndc` is `[x, y]`; `invViewProj` is a column-major 16-array (mat4.mjs convention).
    - `scene` = `{ meta, data, window, renderMode, raySteps, minStepVoxels, voxel, spatial, cubeGrid, cubeThreshold }`. `voxel` and `spatial` are the gate sets; `cubeGrid` is `{shape, spacing, values, baked}`.

- [ ] **Step 1: Write the failing tests.** Append to `viewer/test/grid.test.mjs` (extend its import line with `voxelIndex, cubeGrid`):

```js
test('voxelIndex floors and returns -1 outside the grid', () => {
  const shape = [4, 3, 2];
  assert.equal(voxelIndex(shape, 0, 0, 0), 0);
  assert.equal(voxelIndex(shape, 1.99, 2.5, 1.2), 1 * 3 * 2 + 2 * 2 + 1);
  assert.equal(voxelIndex(shape, -0.01, 0, 0), -1);
  assert.equal(voxelIndex(shape, 4, 0, 0), -1);
  assert.equal(voxelIndex(shape, 0, 0, 2), -1);
});

test('cubeGrid at block 1 is the layer itself, unbaked', () => {
  const meta = { shape: [2, 2, 2], spacing_m: 0.5, origin_m: [0, 0, 0] };
  const data = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
  const g = cubeGrid(data, meta, 1, () => true);
  assert.equal(g.values, data);
  assert.deepEqual(g.shape, [2, 2, 2]);
  assert.equal(g.spacing, 0.5);
  assert.equal(g.baked, false);
});

test('cubeGrid above block 1 is the kept-voxel block mean, baked', () => {
  const meta = { shape: [2, 2, 2], spacing_m: 0.5, origin_m: [0, 0, 0] };
  const data = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 100]);
  const g = cubeGrid(data, meta, 2, (n) => n !== 7);
  assert.deepEqual(g.shape, [1, 1, 1]);
  assert.equal(g.spacing, 1.0);
  assert.equal(g.baked, true);
  assert.ok(Math.abs(g.values[0] - 4) < 1e-6);   // mean of 1..7
});
```

Create `viewer/test/picker.test.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickVoxel } from '../src/picker.mjs';
import { voxelGates, spatialGates } from '../src/gates.mjs';
import { lookAt, perspective, multiply, invert } from '../src/mat4.mjs';

// 4x4x4 grid of 1 m voxels at the origin; camera straight above the column
// (1,1,*) looking down -z.
const META = { shape: [4, 4, 4], spacing_m: 1, origin_m: [0, 0, 0] };
const idx = (i, j, k) => i * 16 + j * 4 + k;
const OFF = { sigmaGateEnabled: false, sigmaGateValue: 1, coverageGateEnabled: false, minRays: 2, snrGateEnabled: false, minSnr: 3 };
const CLIP = { clipMin: [0, 0, 0], clipMax: [1, 1, 1], clipPlaneEnabled: false, clipPlaneNormal: [0, 0, 1], clipPlaneD: 0, surfClip: false };

function camera() {
  const eye = [1.5, 1.5, 20], target = [1.5, 1.5, 0];
  const view = lookAt(eye, target, [0, 1, 0]);
  const proj = perspective(Math.PI / 8, 1, 0.05, 100);
  return invert(multiply(proj, view));
}

function scene(data, extra = {}) {
  return {
    meta: META, data, window: [0.1, 1], renderMode: 'fog', raySteps: 200, minStepVoxels: 0.5,
    voxel: voxelGates(OFF, {}), spatial: spatialGates(CLIP, null),
    cubeGrid: { shape: META.shape, spacing: 1, values: data, baked: false }, cubeThreshold: 0.5,
    ...extra,
  };
}

test('fog: returns the first voxel from the camera above the window', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.9;   // top
  data[idx(1, 1, 0)] = 0.9;   // bottom, occluded
  const hit = pickVoxel([0, 0], camera(), scene(data));
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 3]);
  assert.ok(Math.abs(hit.value - 0.9) < 1e-6);
});

test('fog: values at or below window[0] are not hits', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.05;
  data[idx(1, 1, 1)] = 0.5;
  const hit = pickVoxel([0, 0], camera(), scene(data));
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 1]);
});

test('fog: a voxel gate hides the top voxel, the pick falls through', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.9;
  data[idx(1, 1, 1)] = 0.9;
  const snr = new Float32Array(64).fill(10);
  snr[idx(1, 1, 3)] = 0.5;
  const s = scene(data, { voxel: voxelGates({ ...OFF, snrGateEnabled: true }, { snr }) });
  const hit = pickVoxel([0, 0], camera(), s);
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 1]);
});

test('fog: the clip box hides the top half', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.9;
  data[idx(1, 1, 0)] = 0.9;
  const s = scene(data, { spatial: spatialGates({ ...CLIP, clipMax: [1, 1, 0.5] }, null) });
  const hit = pickVoxel([0, 0], camera(), s);
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 0]);
});

test('a ray that misses the box returns null', () => {
  const data = new Float32Array(64).fill(1);
  assert.equal(pickVoxel([0.99, 0.99], camera(), scene(data)), null);
});

test('cubes: first voxel at or above threshold; data gates skipped when baked', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.6;
  data[idx(1, 1, 1)] = 0.9;
  const sig = new Float32Array(64);
  sig[idx(1, 1, 3)] = 9;
  const gated = voxelGates({ ...OFF, sigmaGateEnabled: true }, { sigma: sig });
  const unbaked = pickVoxel([0, 0], camera(), scene(data, { renderMode: 'cubes', voxel: gated }));
  assert.deepEqual([unbaked.i, unbaked.j, unbaked.k], [1, 1, 1]);   // sigma hides top
  const baked = pickVoxel([0, 0], camera(), scene(data, {
    renderMode: 'cubes', voxel: gated,
    cubeGrid: { shape: META.shape, spacing: 1, values: data, baked: true },
  }));
  assert.deepEqual([baked.i, baked.j, baked.k], [1, 1, 3]);          // baked: gates already applied
});
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `node --test viewer/test/grid.test.mjs viewer/test/picker.test.mjs`
Expected: FAIL (missing exports and module).

If `lookAt`, `perspective`, `multiply` or `invert` have different names or argument orders in `mat4.mjs`, adapt the test's `camera()` to match `cameraMatrices()` in app.mjs. That function is the reference usage.

- [ ] **Step 3: Implement.** In `grid.mjs`, replace `sampleNearest`'s body and add the two functions. Keep the existing comment above `sampleNearest`.

```js
// Flat numpy-order index of the voxel containing fractional voxel-space
// coordinate (i, j, k) (floor convention, see sampleNearest), or -1 outside.
export function voxelIndex(shape, i, j, k) {
  const [nx, ny, nz] = shape;
  const ii = Math.floor(i), jj = Math.floor(j), kk = Math.floor(k);
  if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) return -1;
  return ii * ny * nz + jj * nz + kk;
}

export function sampleNearest(data, shape, i, j, k) {
  const n = voxelIndex(shape, i, j, k);
  return n < 0 ? NaN : data[n];
}
```

After `blockAverage`, add:

```js
// The cube-mode grid for the "Cube size" control: at block 1 the layer itself
// (the shader applies the gates per voxel); above 1 the b^3 block means of the
// voxels keep(n) accepts, with the gates then "baked" into the values.
export function cubeGrid(data, meta, block, keep) {
  const b = Math.max(1, block | 0);
  if (b === 1 || !data) return { shape: meta.shape, spacing: meta.spacing_m, values: data, baked: false };
  const merged = blockAverage(data, meta.shape, b, keep);
  return { shape: merged.shape, spacing: meta.spacing_m * b, values: merged.values, baked: true };
}
```

Create `viewer/src/picker.mjs`:

```js
import { rayBox, voxelMarch, voxelIndex, worldToVoxel } from './grid.mjs';

// Hover picking, CPU twin of the renderer (FRAGMENT_SRC in app.mjs): the same
// camera unprojection, box march and gates, so the voxel under the cursor is
// the voxel that was drawn. Pure: all inputs come in `scene`.
//   scene = { meta, data, window, renderMode, raySteps, minStepVoxels,
//             voxel, spatial,            // gates.mjs gate sets
//             cubeGrid, cubeThreshold }  // cube mode
// Fog: steps tEnter + (i + 0.5) * stepLen, stepLen = max(minStepVoxels *
// voxel, span / raySteps) (always the FULL step count, never the preview),
// first sample above window[0]. Cubes: voxel-to-voxel DDA (voxelMarch), gates
// at voxel centres, first voxel >= cubeThreshold.
export function pickVoxel(ndc, invViewProj, scene) {
  const { meta, data } = scene;
  if (!meta || !data || !invViewProj) return null;
  const m = invViewProj;
  const unprojectAt = (z) => {
    const c = [ndc[0], ndc[1], z, 1];
    const w = m[3] * c[0] + m[7] * c[1] + m[11] * c[2] + m[15] * c[3];
    return [
      (m[0] * c[0] + m[4] * c[1] + m[8] * c[2] + m[12] * c[3]) / w,
      (m[1] * c[0] + m[5] * c[1] + m[9] * c[2] + m[13] * c[3]) / w,
      (m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14] * c[3]) / w,
    ];
  };
  const nearP = unprojectAt(-1), farP = unprojectAt(1);
  const d = [farP[0] - nearP[0], farP[1] - nearP[1], farP[2] - nearP[2]];
  const len = Math.hypot(d[0], d[1], d[2]);
  const dir = [d[0] / len, d[1] / len, d[2] / len];
  return scene.renderMode === 'cubes' ? pickCubeAlong(nearP, dir, scene) : pickFogAlong(nearP, dir, scene);
}

function pickFogAlong(nearP, dir, scene) {
  const { meta, data } = scene;
  const min = meta.origin_m;
  const extent = meta.shape.map((n) => n * meta.spacing_m);
  const hit = rayBox(nearP, dir, min, [min[0] + extent[0], min[1] + extent[1], min[2] + extent[2]]);
  if (!hit) return null;
  const [tEnter, tExit] = hit;
  const voxel = extent[0] / meta.shape[0];   // cubic voxels (single spacing)
  const minStep = scene.minStepVoxels != null ? scene.minStepVoxels : 0.5;
  const stepLen = Math.max(minStep * voxel, (tExit - tEnter) / scene.raySteps);
  const lo = scene.window ? scene.window[0] : 0;
  for (let i = 0; i < scene.raySteps; i++) {
    const tt = tEnter + (i + 0.5) * stepLen;
    if (tt > tExit) break;
    const world = [nearP[0] + dir[0] * tt, nearP[1] + dir[1] * tt, nearP[2] + dir[2] * tt];
    const tex = [0, 1, 2].map((a) => (world[a] - min[a]) / extent[a]);
    const [vi, vj, vk] = worldToVoxel(world, meta);
    const n = voxelIndex(meta.shape, vi, vj, vk);
    if (n < 0 || !scene.spatial.keep(tex, world) || !scene.voxel.keep(n)) continue;
    const value = data[n];
    if (!Number.isNaN(value) && value > lo) {
      return { i: Math.floor(vi), j: Math.floor(vj), k: Math.floor(vk), value };
    }
  }
  return null;
}

function pickCubeAlong(nearP, dir, scene) {
  const { meta, cubeGrid: grid } = scene;
  if (!grid || !grid.values) return null;
  const [, ny, nz] = meta.shape;
  const [, cy, cz] = grid.shape;
  const s = grid.spacing, o = meta.origin_m;
  const ext = meta.shape.map((n) => n * meta.spacing_m);
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  let value = NaN;
  const hit = voxelMarch(nearP, dir, { shape: grid.shape, origin_m: o, spacing_m: s }, (i, j, k) => {
    const world = [o[0] + (i + 0.5) * s, o[1] + (j + 0.5) * s, o[2] + (k + 0.5) * s];
    const tex = [0, 1, 2].map((a) => clamp01((world[a] - o[a]) / ext[a]));
    if (!scene.spatial.keep(tex, world)) return false;
    if (!grid.baked && !scene.voxel.keep(i * ny * nz + j * nz + k)) return false;
    const v = grid.values[i * cy * cz + j * cz + k];
    if (v >= scene.cubeThreshold) { value = v; return true; }
    return false;
  });
  return hit ? { i: hit[0], j: hit[1], k: hit[2], value } : null;
}
```

Before relying on it, check that `worldToVoxel` is exported from grid.mjs (it is, at line ~10) and that the names `pickVoxel`, `pickFogAlong` and `pickCubeAlong` are unused elsewhere in `viewer/src/` (`grep -rn "pickFogAlong\|pickCubeAlong\|pickVoxel" viewer/src`).

- [ ] **Step 4: Run node tests, confirm pass**

Run: `node --test viewer/test/*.test.mjs`
Expected: all PASS.

- [ ] **Step 5: Wire into app.mjs**

- Add `"picker.mjs"` to `_MODULE_ORDER` after `"gates.mjs"`.
- Import line: add `cubeGrid` to the grid.mjs import. Add `import { pickVoxel } from './picker.mjs';`. Remove imports that become unused (`insideClipBox`, `insideClipPlane`, `voxelMarch`, `sampleNearest`, `worldToVoxel` if unused; grep each).
- `ensureCubeGrid()` becomes:

```js
  function ensureCubeGrid() {
    const meta = state.meta;
    if (!meta) return null;
    const b = Math.max(1, state.cubeBlock | 0);
    const vgates = stateVoxelGates();
    const key = [b, state.activeLayer, state.loadSeq, vgates.key].join('|');
    if (key === state.cubeGridKey && state.cubeGrid) return state.cubeGrid;
    if (state.cubeGrid && state.cubeGrid.tex) gl.deleteTexture(state.cubeGrid.tex);
    const grid = cubeGrid(state.layerData.get(state.activeLayer), meta, b, vgates.keep);
    grid.tex = grid.baked ? makeVolumeTexture(gl, grid.shape, grid.values) : null;
    state.cubeGrid = grid;
    state.cubeGridKey = key;
    return state.cubeGrid;
  }
```

- Replace `castHoverRay` and `castCubeRay` entirely with:

```js
  // Hover picking: client pixel -> NDC -> pickVoxel (picker.mjs), sharing
  // cameraMatrices() with render() so it always agrees with what was drawn.
  function castHoverRay(clientX, clientY) {
    if (!state.meta) return null;
    const rect = canvas.getBoundingClientRect();
    const ndc = [((clientX - rect.left) / rect.width) * 2 - 1, -(((clientY - rect.top) / rect.height) * 2 - 1)];
    const { invViewProj } = cameraMatrices();
    return pickVoxel(ndc, invViewProj, {
      meta: state.meta,
      data: state.layerData.get(state.activeLayer),
      window: state.window,
      renderMode: state.renderMode,
      raySteps: RAY_STEPS,
      minStepVoxels: state.minStepVoxels,
      voxel: stateVoxelGates(),
      spatial: stateSpatialGates(),
      cubeGrid: state.renderMode === 'cubes' ? ensureCubeGrid() : null,
      cubeThreshold: state.cubeThreshold,
    });
  }
```

- Delete `voxelFlatIndex`.

- [ ] **Step 6: Run the viewer tests, then the full suite**

Run: `node --test viewer/test/*.test.mjs`, then `uv run pytest -q tests/viewer`, then `uv run pytest -q`
Expected: all PASS, no existing test modified.

- [ ] **Step 7: Commit**

```bash
git add viewer/src/picker.mjs viewer/test/picker.test.mjs viewer/src/grid.mjs viewer/test/grid.test.mjs megido/viewerbuild.py viewer/src/app.mjs
git commit -m "refactor(viewer): hover picking and cube grid as pure modules" -m "<attribution lines>"
```

---

### Task 3: `runload.mjs`, reading a run directory

**Files:**
- Create: `viewer/src/runload.mjs`
- Create: `viewer/test/runload.test.mjs`
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: `"runload.mjs"` after `"picker.mjs"`)
- Modify: `viewer/src/app.mjs` (`loadRun` ~1078-1411, the `#load-second-run-input` handler ~1421-1467, remove `readFile`)

**Interfaces:**
- Consumes: `parseNpy(arrayBuffer)` → `{data, shape}` from `npy.mjs`; `symmetricLimit(arr)` from `surfacemesh.mjs`.
- Produces:
  - `readRun(files)` → `Promise<{ meta, silhouette, hillSurface, layers: Map, sigmaMax, cubeThreshold }>`. It throws `Error('selected directory has no meta.json')`.
  - `readCompareVolume(files, primaryMeta)` → `Promise<null | {mismatch: true} | {volume}>`. It throws `Error('compare run has no volume.npy')`.

- [ ] **Step 1: Write the failing test** `viewer/test/runload.test.mjs`. Build fake files with a tiny `.npy` encoder. Check `viewer/test/npy.test.mjs` for an existing encoder helper and reuse its approach; the code below is self-contained.

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readRun, readCompareVolume } from '../src/runload.mjs';

function npyBuffer(values, shape) {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  let h = header;
  while ((10 + h.length + 1) % 64 !== 0) h += ' ';
  h += '\n';
  const buf = new ArrayBuffer(10 + h.length + values.length * 4);
  const u8 = new Uint8Array(buf);
  u8.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(buf).setUint16(8, h.length, true);
  for (let i = 0; i < h.length; i++) u8[10 + i] = h.charCodeAt(i);
  new Float32Array(buf, 10 + h.length).set(values);
  return buf;
}
const jsonFile = (name, obj) => ({ name, text: async () => JSON.stringify(obj), arrayBuffer: async () => { throw new Error('json'); } });
const npyFile = (name, values, shape) => ({ name, text: async () => { throw new Error('npy'); }, arrayBuffer: async () => npyBuffer(values, shape) });

const META = { shape: [2, 2, 2], spacing_m: 0.5, origin_m: [0, 0, 0], layers: ['volume', 'sigma', 'backprojection'], suggested_iso: [0.3, 0.6], value_range: [0, 1] };
const vol = [0, 1, 2, 3, 4, 5, 6, 7];

test('missing meta.json throws the same error as before', async () => {
  await assert.rejects(readRun([npyFile('volume.npy', vol, [2, 2, 2])]), /no meta\.json/);
});

test('only full-grid layers load; sigmaMax and cubeThreshold derived', async () => {
  const r = await readRun([
    jsonFile('meta.json', META),
    npyFile('volume.npy', vol, [2, 2, 2]),
    npyFile('sigma.npy', [1, 9, 1, 1, 1, 1, 1, 1], [2, 2, 2]),
    npyFile('backprojection.npy', [1, 2, 3, 4], [2, 2]),       // 2D: skipped
  ]);
  assert.deepEqual([...r.layers.keys()], ['volume', 'sigma']);
  assert.equal(r.sigmaMax, 9);
  assert.equal(r.cubeThreshold, 0.3);
  assert.equal(r.silhouette, null);
  assert.equal(r.hillSurface, null);
});

test('cubeThreshold falls back to mid value_range; sigmaMax 1 without sigma', async () => {
  const r = await readRun([jsonFile('meta.json', { ...META, layers: ['volume'], suggested_iso: null, value_range: [0, 2] }), npyFile('volume.npy', vol, [2, 2, 2])]);
  assert.equal(r.cubeThreshold, 1);
  assert.equal(r.sigmaMax, 1);
});

test('hill surface needs both files; mismatched sigma dropped; residualLim fallback', async () => {
  const hm = { gx: [0, 1], gy: [0, 1], residual_grid_lim: null };
  const base = [jsonFile('meta.json', META), npyFile('volume.npy', vol, [2, 2, 2])];
  const noMeta = await readRun([...base, npyFile('hill_surface.npy', [1, 2, 3, 4], [2, 2])]);
  assert.equal(noMeta.hillSurface, null);
  const r = await readRun([...base,
    npyFile('hill_surface.npy', [1, 2, 3, 4], [2, 2]),
    jsonFile('hill_surface_meta.json', hm),
    npyFile('hill_surface_sigma.npy', [1, 2, 3], [3]),                 // wrong length: dropped
    npyFile('hill_residual_grid.npy', [0.1, -0.2, 0.3, -0.4], [2, 2]),
  ]);
  assert.equal(r.hillSurface.sigma, null);
  assert.equal(r.hillSurface.residual.length, 4);
  assert.ok(Number.isFinite(r.hillSurface.residualLim) && r.hillSurface.residualLim > 0);
  assert.deepEqual(r.hillSurface.gx, [0, 1]);
});

test('silhouette json is read when present', async () => {
  const r = await readRun([jsonFile('meta.json', META), npyFile('volume.npy', vol, [2, 2, 2]), jsonFile('hill_silhouette.json', { fans: [] })]);
  assert.deepEqual(r.silhouette, { fans: [] });
});

test('readCompareVolume: no meta, grid mismatch, match', async () => {
  assert.equal(await readCompareVolume([npyFile('volume.npy', vol, [2, 2, 2])], META), null);
  const moved = await readCompareVolume([jsonFile('meta.json', { ...META, origin_m: [1, 0, 0] }), npyFile('volume.npy', vol, [2, 2, 2])], META);
  assert.deepEqual(moved, { mismatch: true });
  const ok = await readCompareVolume([jsonFile('meta.json', META), npyFile('volume.npy', vol, [2, 2, 2])], META);
  assert.equal(ok.volume.length, 8);
  await assert.rejects(readCompareVolume([jsonFile('meta.json', META)], META), /no volume\.npy/);
});
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `node --test viewer/test/runload.test.mjs`
Expected: FAIL (module missing). If the npy encoder above does not parse with `parseNpy`, copy the encoder from `viewer/test/npy.test.mjs`. The test must exercise the real `parseNpy`.

- [ ] **Step 3: Write `viewer/src/runload.mjs`**

```js
import { parseNpy } from './npy.mjs';
import { symmetricLimit } from './surfacemesh.mjs';

// Reads a run directory (File-like objects: name, text(), arrayBuffer()) into
// one dataset record. Pure apart from the file reads; app.mjs applies the
// record to GL, DOM and state. Every optional artifact is null when absent,
// never an error: older and synthetic runs lack them.
export async function readRun(files) {
  const byName = new Map();
  for (const f of files) byName.set(f.name, f);
  const metaFile = byName.get('meta.json');
  if (!metaFile) throw new Error('selected directory has no meta.json');
  const meta = JSON.parse(await metaFile.text());
  const npyData = async (f) => parseNpy(await f.arrayBuffer()).data;

  const silhouetteFile = byName.get('hill_silhouette.json');
  const silhouette = silhouetteFile ? JSON.parse(await silhouetteFile.text()) : null;

  // The fitted hillside surface needs BOTH its heights and its meta. Sigma
  // and residual grids are dropped unless they match the surface's size.
  let hillSurface = null;
  const surfFile = byName.get('hill_surface.npy');
  const surfMetaFile = byName.get('hill_surface_meta.json');
  if (surfFile && surfMetaFile) {
    const H = await npyData(surfFile);
    const hillMeta = JSON.parse(await surfMetaFile.text());
    const sameSize = async (name) => {
      const f = byName.get(name);
      if (!f) return null;
      const d = await npyData(f);
      return d.length === H.length ? d : null;
    };
    const sigma = await sameSize('hill_surface_sigma.npy');
    const residual = await sameSize('hill_residual_grid.npy');
    // The CLI records residual_grid_lim; when it is null (no finite residual
    // at write time, or an older run) derive the same 98th-percentile |r|
    // scale from the data rather than a meaningless default.
    const residualLim = Number.isFinite(hillMeta.residual_grid_lim)
      ? hillMeta.residual_grid_lim
      : (residual ? symmetricLimit(residual) : NaN);
    hillSurface = { H, gx: hillMeta.gx, gy: hillMeta.gy, meta: hillMeta, sigma, residual, residualLim };
  }

  // Only layers with one value per voxel are raymarch-able; lower-dimensional
  // diagnostics (e.g. a 2D backprojection plane) share meta.layers and are
  // skipped here, correctly, not reported as errors.
  const voxelCount = meta.shape[0] * meta.shape[1] * meta.shape[2];
  const layers = new Map();
  for (const name of meta.layers) {
    const f = byName.get(`${name}.npy`);
    if (!f) continue;
    const data = await npyData(f);
    if (data.length === voxelCount) layers.set(name, data);
  }

  // Plain loop, not Math.max(...sig): a spread blows the call stack on the
  // real campaign's 675,840-element sigma array.
  let sigmaMax = 1;
  if (layers.has('sigma')) {
    const sig = layers.get('sigma');
    sigmaMax = 0;
    for (let i = 0; i < sig.length; i++) if (sig[i] > sigmaMax) sigmaMax = sig[i];
  }

  const iso = Array.isArray(meta.suggested_iso) ? meta.suggested_iso[0] : null;
  const vr = Array.isArray(meta.value_range) ? meta.value_range : [0, 1];
  const cubeThreshold = Number.isFinite(iso) ? iso : 0.5 * (vr[0] + vr[1]);

  return { meta, silhouette, hillSurface, layers, sigmaMax, cubeThreshold };
}

// "Load compare": null when the directory has no meta.json; {mismatch: true}
// when the grid differs (shape + spacing + origin, as compare_volumes in
// megido/volexport.py keys it); otherwise {volume}.
export async function readCompareVolume(files, primaryMeta) {
  const byName = new Map(files.map((f) => [f.name, f]));
  const metaFile = byName.get('meta.json');
  if (!metaFile) return null;
  const second = JSON.parse(await metaFile.text());
  const mismatch =
    JSON.stringify(second.shape) !== JSON.stringify(primaryMeta.shape) ||
    second.spacing_m !== primaryMeta.spacing_m ||
    JSON.stringify(second.origin_m) !== JSON.stringify(primaryMeta.origin_m);
  if (mismatch) return { mismatch: true };
  const volFile = byName.get('volume.npy');
  if (!volFile) throw new Error('compare run has no volume.npy');
  return { volume: parseNpy(await volFile.arrayBuffer()).data };
}
```

- [ ] **Step 4: Run node tests, confirm pass**

Run: `node --test viewer/test/*.test.mjs`
Expected: all PASS.

- [ ] **Step 5: Wire into app.mjs**

- Add `"runload.mjs"` to `_MODULE_ORDER` after `"picker.mjs"`. Import: `import { readRun, readCompareVolume } from './runload.mjs';`.
- Change the start of `loadRun(files)`. After `state.ready = false` and hiding onboarding, do `const run = await readRun(files); const meta = run.meta;`. Then:
  - `state.meta = meta; state.detectors = meta.detectors || []`, as now.
  - `state.silhouette = run.silhouette`, replacing the file read.
  - The hill-surface branch becomes `if (run.hillSurface) { const surf = run.hillSurface; state.hillSurface = surf; state.hillColourMode = surf.sigma ? 'sigma' : 'flat'; if (surf.sigma) state.hillSigmaRange = robustRange(surf.sigma); ...rest unchanged }`. Delete the file lookups (`hillSurfaceFile`, `hillSurfaceMetaFile`, `hillSurfaceSigmaFile`, `hillResidualFile`) and the parse code. `hillMeta` inside the branch becomes `surf.meta`.
  - The layer loop becomes: `state.layerData.clear(); for (const [name, data] of run.layers) state.layerData.set(name, data);`.
  - The cube-threshold block becomes `state.cubeThreshold = run.cubeThreshold;` plus the same `#cube-threshold` element update.
  - The sigma branch becomes: `if (state.layerData.has('sigma')) state.sigmaTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('sigma')); state.sigmaMax = run.sigmaMax;`.
  - Everything else in `loadRun` (DOM enable/disable, banner, delta reset, frameAll, crop, textures, window, layer panel, render, ready) stays in the same order.
- `#load-second-run-input` handler: keep the `if (!state.meta)` message. Then `const res = await readCompareVolume(files, state.meta); if (!res) return; if (res.mismatch) { ...'grid mismatch: cannot diff'; return; } const secondVolume = res.volume;`. The rest (delta, scale, verdict, radio) stays unchanged.
  - The original order checked for a missing meta before the primary-run check. Keep that: first `if (!files.some((f) => f.name === 'meta.json')) return;`, then the primary check, then `readCompareVolume`.
- Delete `readFile` if it is now unused (grep).

- [ ] **Step 6: Run the viewer tests, then the full suite**

Run: `node --test viewer/test/*.test.mjs`, then `uv run pytest -q tests/viewer`, then `uv run pytest -q`
Expected: all PASS, no existing test modified.

- [ ] **Step 7: Commit**

```bash
git add viewer/src/runload.mjs viewer/test/runload.test.mjs megido/viewerbuild.py viewer/src/app.mjs
git commit -m "refactor(viewer): run-directory reading as a pure module (runload.mjs)" -m "<attribution lines>"
```

---

### Task 4: `model.mjs`, state defaults and the field → follow-up table; one `commit()`

**Files:**
- Create: `viewer/src/model.mjs`
- Create: `viewer/test/model.test.mjs`
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: `"model.mjs"` after `"runload.mjs"`)
- Modify: `viewer/src/app.mjs` (the `state` literal ~463-538; the DOM handlers listed in Step 5; `rebuildLut`)

**Interfaces:**
- Consumes: `CAMERA_PRESETS` (camera.mjs), `colormapStops` (colormap.mjs).
- Produces:
  - `createState()` → plain object holding today's non-GL state fields and defaults, plus `window: null`.
  - `EFFECTS`: an object mapping each committable field to an array of effect names.
  - `EFFECT_ORDER = ['filter', 'surfaceMesh', 'legend', 'window', 'histogram', 'lut', 'render']`.
  - `effectsFor(patch)` → effect names in `EFFECT_ORDER` order. It throws `Error('no effects declared for state field: X')` for a field not in `EFFECTS`.

- [ ] **Step 1: Write the failing test** `viewer/test/model.test.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createState, EFFECTS, EFFECT_ORDER, effectsFor } from '../src/model.mjs';

test('every committable field exists in the default state', () => {
  const s = createState();
  for (const field of Object.keys(EFFECTS)) assert.ok(field in s, field);
});

test('every effect named in the table is a known effect', () => {
  for (const [field, effects] of Object.entries(EFFECTS)) {
    for (const e of effects) assert.ok(EFFECT_ORDER.includes(e), `${field}: ${e}`);
  }
});

test('gate thresholds re-window; the sigma gate does not (on purpose)', () => {
  assert.deepEqual(effectsFor({ minSnr: 2 }), ['window', 'render']);
  assert.deepEqual(effectsFor({ coverageGateEnabled: false }), ['window', 'render']);
  assert.deepEqual(effectsFor({ sigmaGateValue: 0.3 }), ['render']);
});

test('union across a patch, in the fixed order', () => {
  assert.deepEqual(effectsFor({ hillColourMode: 'sigma', smoothSampling: false }),
    ['filter', 'surfaceMesh', 'legend', 'render']);
});

test('an undeclared field throws', () => {
  assert.throws(() => effectsFor({ notAField: 1 }), /no effects declared for state field: notAField/);
});

test('defaults match the viewer today', () => {
  const s = createState();
  assert.equal(s.renderMode, 'fog');
  assert.equal(s.opacity, 1);
  assert.equal(s.coverageGateEnabled, true);
  assert.equal(s.minRays, 2);
  assert.equal(s.snrGateEnabled, true);
  assert.equal(s.minSnr, 3);
  assert.equal(s.sigmaGateEnabled, false);
  assert.equal(s.hillColourMode, 'flat');
  assert.deepEqual(s.clipMax, [1, 1, 1]);
  assert.equal(s.window, null);
});
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `node --test viewer/test/model.test.mjs`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `viewer/src/model.mjs`**

Move every non-GL field of the `state` literal in app.mjs (lines ~463-538) into `createState()`, verbatim, with its comments. GL-owned fields stay in app.mjs and are assigned after `createState()`: `gl, program, uniforms, raysTex, snrTex, volumeTex, sigmaTex, lutTex, floatLinear`. Add `window: null`.

```js
import { CAMERA_PRESETS } from './camera.mjs';
import { colormapStops } from './colormap.mjs';

// The viewer's display state: defaults, and which follow-ups each field needs
// when it changes. app.mjs owns the GL/DOM side; commit() there runs the
// effects this table names, in EFFECT_ORDER.
export const EFFECT_ORDER = ['filter', 'surfaceMesh', 'legend', 'window', 'histogram', 'lut', 'render'];

export function createState() {
  return {
    // ... every non-GL field of today's app.mjs state literal, verbatim ...
    window: null,
  };
}

const R = ['render'];
export const EFFECTS = {
  camera: R,
  clipMin: R, clipMax: R, clipPlaneEnabled: R, clipPlaneD: R,
  // The sigma gate is a continuous slider: it never re-windows (gates.mjs).
  sigmaGateEnabled: R, sigmaGateValue: R,
  coverageGateEnabled: ['window', 'render'], minRays: ['window', 'render'],
  snrGateEnabled: ['window', 'render'], minSnr: ['window', 'render'],
  renderMode: R, cubeThreshold: R, cubeBlock: R, opacity: R,
  smoothSampling: ['filter', 'render'],
  shading: R, adaptiveQuality: R,
  showDetectors: R, showSilhouette: R, showHillSurface: R, surfClip: R,
  hillColourMode: ['legend', 'surfaceMesh', 'render'],
  hillSurfaceSmooth: ['surfaceMesh', 'render'],
  window: ['histogram', 'render'],
  transferStops: ['lut', 'render'],
};

export function effectsFor(patch) {
  const wanted = new Set();
  for (const field of Object.keys(patch)) {
    const effects = EFFECTS[field];
    if (!effects) throw new Error(`no effects declared for state field: ${field}`);
    for (const e of effects) wanted.add(e);
  }
  return EFFECT_ORDER.filter((e) => wanted.has(e));
}
```

The `// ...` in `createState` is filled with the moved fields. This is a transcription of existing code, not new content: copy it verbatim. Check that `createState` references no GL object.

- [ ] **Step 4: Run node tests, confirm pass**

Run: `node --test viewer/test/*.test.mjs`
Expected: all PASS.

- [ ] **Step 5: Wire into app.mjs**

- Add `"model.mjs"` to `_MODULE_ORDER` after `"runload.mjs"`. Import: `import { createState, effectsFor } from './model.mjs';`.
- Replace the `state` literal with:

```js
  const dummyVolume = makeVolumeTexture(gl, [1, 1, 1], new Float32Array([0]));
  const state = Object.assign(createState(), {
    gl, program, uniforms, floatLinear,
    raysTex: dummyVolume, snrTex: dummyVolume, volumeTex: dummyVolume, sigmaTex: dummyVolume,
    lutTex: makeLutTexture(gl, buildTransferLUT(colormapStops('viridis'))),
  });
```

- Add `commit` next to `beginInteraction`:

```js
  // The one way a UI control changes display state: assign the patch, run the
  // follow-ups model.mjs declares for those fields (fixed order), and render
  // a fast preview when the change is part of a continuous drag.
  function commit(patch, { interactive = false } = {}) {
    const effects = effectsFor(patch);
    Object.assign(state, patch);
    for (const e of effects) {
      if (e === 'filter') applyVolumeFilter();
      else if (e === 'surfaceMesh') rebuildHillSurfaceBuffer();
      else if (e === 'legend') { if (state.updateHillColourLegend) state.updateHillColourLegend(); }
      else if (e === 'window') { if (state.applyWindowForLayer) state.applyWindowForLayer(state.activeLayer); }
      else if (e === 'histogram') drawHistogram();
      else if (e === 'lut') {
        gl.deleteTexture(state.lutTex);
        state.lutTex = makeLutTexture(gl, buildTransferLUT(state.transferStops));
        drawXferEditor();
      } else if (e === 'render') {
        if (interactive) beginInteraction();
        render();
      }
    }
  }
```

- Rewrite these handlers to read their input, keep any DOM-only side effect (readout text, localStorage) in the handler, and call `commit`:
  - `#toggle-detectors`, `#toggle-smooth`, `#toggle-shading`, `#toggle-adaptive`, `#toggle-silhouette`, `#toggle-hill-surface`, `#toggle-surf-clip`, `#clip-plane-enabled`, `#sigma-gate-enabled`, `#coverage-gate-enabled`, `#snr-gate-enabled`, `#render-mode`, `#cube-size` (`commit({ field: value })`).
  - `#hill-colour-mode` → `commit({ hillColourMode: ev.target.value })`.
  - `#hill-surface-smooth`: keep the `if (!state.hillSurface) return;` guard and the readout, then `commit({ hillSurfaceSmooth: Number(ev.target.value) || 0 }, { interactive: true })`.
  - Interactive: `#clip-plane-d`, `#sigma-gate-value` (`commit({ sigmaGateValue: frac * max }, …)`), `#coverage-gate-value`, `#snr-gate-value`, `#opacity`, `#cube-threshold`.
  - Clip axes: `commit({ clipMin: next }, { interactive: true })`, where `next = [...state.clipMin]; next[axes[axis]] = v`. Do the same for `clipMax`.
  - `updateSlice`: build the two arrays, then `commit({ clipMin, clipMax }, { interactive })`. Pass `interactive: true` from the `#slice-pos` handler and false from `#slice-axis`. Remove its separate `beginInteraction()` call.
  - Camera drag: `commit({ camera: { ...state.camera, yaw, pitch } }, { interactive: true })`. Wheel: likewise for `distance`. Preset buttons: `commit({ camera: { ...state.camera, yaw, pitch } })`.
  - Histogram window drag: `const w = [...state.window]; …set lo or hi…; commit({ window: w }, { interactive: true })`.
  - Transfer-stop drag: this one mutates `draggingStop.t` in place (a stop object inside `state.transferStops`). Keep the mutation and call `commit({ transferStops: state.transferStops }, { interactive: true })`. The field is the same array, but the `lut` effect still rebuilds from it.
  - `applyColormap(name)`: set via `commit({ transferStops: colormapStops(name) })` after the localStorage write.
  - Delete `rebuildLut` (now unused).
- Leave `setActiveLayer`, `loadRun`, `frameAll`, saved-view apply, export PNG, theme and shortcuts as they are. They are not single-field changes.
- Behaviour check while editing:
  - Each rewritten handler must end with the same effects as before, in an equivalent order.
  - Two cases gain one extra redundant render versus today (colormap, transfer drag): acceptable, since the frames are identical.
  - Any other difference in effects or order: stop and report.

- [ ] **Step 6: Run the viewer tests, then the full suite**

Run: `node --test viewer/test/*.test.mjs`, then `uv run pytest -q tests/viewer`, then `uv run pytest -q`
Expected: all PASS, no existing test modified.

- [ ] **Step 7: Commit**

```bash
git add viewer/src/model.mjs viewer/test/model.test.mjs megido/viewerbuild.py viewer/src/app.mjs
git commit -m "refactor(viewer): state defaults and follow-ups table (model.mjs); one commit()" -m "<attribution lines>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md` (Phase 4 viewer paragraph)

- [ ] **Step 1: Update the Phase 4 paragraph.** After the sentence that introduces `viewer/src/*.mjs`, add:

"The viewer's logic lives in pure modules, all node-tested without a browser:
- `gates.mjs` is the ONE reference for which voxels are hidden: the σ, coverage and SNR gates plus the spatial clips, with NaN semantics defined there. The two GLSL gate copies read its uniforms, and `tests/viewer/test_gate_parity.py` checks shader against it, gate by gate.
- `picker.mjs` does hover picking.
- `runload.mjs` reads a run directory.
- `model.mjs` holds the state defaults and the field → follow-up table that `commit()` in app.mjs runs.

app.mjs is the GL and DOM adapter. Adding a display setting means adding its field to `createState` and its effects to `EFFECTS`, which throws if you forget. Adding a gate means editing `gates.mjs`, the two GLSL copies and a parity-test case."

Also replace "Hover picking mirrors the shader's march and clip order exactly (`rayBox` in `grid.mjs`)" with "Hover picking (`picker.mjs`) mirrors the shader's march exactly (`rayBox` in `grid.mjs`)".

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md names the viewer's pure modules" -m "<attribution lines>"
```

---

## Self-review notes

- **Spec coverage:** gates (T1), parity test (T1), picker + voxelIndex + cubeGrid (T2), runload + compare (T3), model + commit (T4), docs (T5).
- **Rulings R1-R4:** R1 and R2 are in T1's code, R3 is T4's table, R4 is followed throughout (GL/DOM stay in app).
- **Type consistency:** gate sets `{keep, keepForWindow, windowActive, uniforms, key}` / `{keep}`; `cubeGrid` returns `{shape, spacing, values, baked}` and app adds `.tex`; `pickVoxel(ndc, invViewProj, scene)`.
