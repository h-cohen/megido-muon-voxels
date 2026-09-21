# Phase 4b — "Instrument" viewer redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reskin and reorganize the S5 voxel viewer into a polished dark "scientific instrument" UI — a left collapsible control dock, canvas edge-to-edge, plus colormaps, saved views, keyboard shortcuts, an onboarding card, an axis gizmo and a colorbar legend — without changing the renderer, the data contract, or the honesty guardrail.

**Architecture:** The WebGL2 raymarcher, the `state` model, `render()`, and every logic module (`npy/mat4/grid/transfer/histogram/clip/camera/layers/delta`) are unchanged. The redesign is (a) a new inline-CSS design system + DOM structure in `viewer/shell.html`, (b) reorganized DOM wiring in `viewer/src/app.mjs` (same behavior, new layout, new features), and (c) new pure modules `colormap.mjs`, `views.mjs`, `shortcuts.mjs`, `dock.mjs`. Everything ships in one self-contained `viewer/dist/index.html` via `megido/viewerbuild.py`.

**Tech Stack:** WebGL2 (untouched), vanilla ES modules, inline CSS (CSS-variable design tokens), inline SVG icons, small hardcoded colormap LUTs, `localStorage` for per-viewer conveniences, Node `node:test` for pure logic, Playwright (`pytest-playwright`, headless Chromium, `file://`) for DOM/interaction.

**Spec:** the approved in-chat design (this plan is its argument). No separate spec file. The prior phase spec `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md` §8 still binds the viewer's data contract and honesty requirement.

## Global Constraints

- **Single self-contained file:** the deliverable is one `viewer/dist/index.html` built by `megido/viewerbuild.py` from `viewer/src/*.mjs` + `viewer/shell.html`. **Zero runtime dependencies, zero network requests, works from `file://`.** Colormaps are small inline LUTs; icons are inline SVG; **no webfonts** — system font stack only (`system-ui`/`ui-monospace`).
- **Module style:** every `viewer/src/*.mjs` uses only `export function`/`export const` at top level and `import { x } from './y.mjs'` for cross-module use — **no default export, no dynamic `import()`** — so each file loads both as a `node:test` import and as build-concatenation input. New modules must be added to `_MODULE_ORDER` in `megido/viewerbuild.py`, each **before** `app.mjs` and after any module it imports.
- **Do NOT touch** the fragment shader `FRAGMENT_SRC` or vertex shader in `app.mjs`, and do NOT remove `preserveDrawingBuffer: true` from the `getContext('webgl2', ...)` call. `initViewer(root)` stays the sole bundle entry point. `window.__viewerState` and `state.ready` must remain (tests depend on them; `state.ready` goes false at `loadRun` start and true only after the first `render()`).
- **Data contract unchanged:** reads `volume.npy` + `meta.json` + per-layer `.npy` exactly as `megido/volexport.py` writes them. Only layers whose element count equals `nx*ny*nz` are raymarch-able (non-grid layers like the 2D backprojection are silently skipped — keep that behavior).
- **Honesty guardrail (non-negotiable):** the depth-resolution verdict must remain unmissable. The `#resolution-banner` element keeps its id, always shows `meta.resolution.verdict` text after load, and reads **red** when `meta.resolution.depth_resolved` is false. It may be restyled as a pill, but it may never be hidden, empty (when a verdict exists), or green-on-not-resolved.
- **localStorage** is for per-viewer conveniences only (section open/closed, saved views, onboarding-dismissed, last colormap). Wrap **every** read and write in `try/catch`; the viewer must render correctly when storage throws or returns nothing. Never store run data there.
- **Playwright tests** load a run via `page.locator("#load-run-input").set_input_files(str(run_dir))` (directory-path form; a file **list** fails on Playwright 1.63) and wait on `window.__viewerState.ready`. Fixtures `dist_path` and `run_fixture(layers=(...))` already exist in `tests/viewer/conftest.py`.
- **The real-data exit gate `tests/viewer/test_smoke.py` must stay green** end to end (loads `runs/voxels`, canvas non-blank, `#resolution-banner` contains "depth NOT resolved", **zero console errors**), updated for any DOM id it references that this plan changes.
- **JS unit tests** run with an explicit file path, e.g. `node --test viewer/test/colormap.test.mjs` (a bare-directory arg fails on the installed Node v24.16.0). Build test: `uv run pytest tests/test_viewerbuild.py`.
- **Preserve existing control element ids** wherever the control still exists, so the existing Playwright suite keeps passing: `#gl-canvas`, `#load-run-input`, `#load-second-run-input`, `#delta-verdict`, `#resolution-banner`, `#layer-panel` (+ dynamic `#layer-<key>` radios), `#camera-preset-top|front|side|iso`, `#clip-x-min|x-max|y-min|y-max|z-min|z-max`, `#slice-axis`, `#slice-pos`, `#clip-plane-enabled`, `#clip-plane-d`, `#sigma-gate-enabled`, `#sigma-gate-value`, `#xfer-canvas`, `#histogram-canvas`, `#hover-readout`, `#export-png-btn`. A control that is genuinely replaced (the two `#window-lo`/`#window-hi` sliders → a histogram band, Task 3) updates its owning tests in the same task.
- Commit directly to `main` (authorized). Each commit message ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6`.

---

## File structure

- `viewer/shell.html` — restructured: `<style>` design-system tokens + all component CSS; `<body>` = left `#dock` (sections) + `#canvas-wrap` (canvas + overlays). Modified by Tasks 1,2,3,4,5,6,7.
- `viewer/src/app.mjs` — orchestrator; DOM wiring moves to new structure; new feature wiring added. Modified by Tasks 1,2,3,4,5,6,7,8.
- `viewer/src/dock.mjs` — **new** (Task 1): accordion section collapse/expand + persistence helpers (pure + a small DOM-init that's driven from app).
- `viewer/src/colormap.mjs` — **new** (Task 2): named preset colormaps → transfer stops.
- `viewer/src/views.mjs` — **new** (Task 6): saved-view serialize/deserialize + localStorage load/save.
- `viewer/src/shortcuts.mjs` — **new** (Task 7): pure keymap (event → action name).
- `viewer/test/*.test.mjs` — node tests for each new pure module.
- `tests/viewer/*.py` — Playwright tests, updated per task as ids change; `test_smoke.py` is the exit gate.
- `megido/viewerbuild.py` — modify `_MODULE_ORDER` to include the new modules.

---

### Task 1: Design system, left dock shell, banner pill, collapsible sections

**Files:**
- Modify: `viewer/shell.html` (full restructure)
- Create: `viewer/src/dock.mjs`
- Create: `viewer/test/dock.test.mjs`
- Modify: `viewer/src/app.mjs` (init the dock; move nothing's behavior, only container structure)
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: add `dock.mjs` before `app.mjs`)
- Modify: `tests/viewer/conftest.py` only if a selector helper is needed (likely not)
- Create: `tests/viewer/test_dock.py`

**Interfaces:**
- Produces `viewer/src/dock.mjs`: `sectionStorageKey(id) -> string`; `loadSectionState(id, fallbackOpen=true) -> boolean` (reads localStorage, try/catch, returns fallback on any failure); `saveSectionState(id, isOpen) -> void` (try/catch, silent on failure); `initDock(root) -> void` — finds every `[data-section]` element in `root`, wires its header click to toggle a `data-open` attribute + persist, and applies the stored state on init.
- Consumes: nothing from other new modules.
- The DOM contract every later task relies on: a left `<div id="dock">` containing, top to bottom, `<header id="dock-header">` and one `<section class="dock-section" data-section="<id>">` per group with ids `sec-layers`, `sec-transfer`, `sec-clip`, `sec-uncertainty`, `sec-camera`; a right `<div id="canvas-wrap">` containing `#gl-canvas` and overlay containers `#overlay-banner`, `#overlay-gizmo`, `#overlay-legend`, `#overlay-hover`, `#overlay-onboarding`, `#overlay-shortcuts`. **All existing control ids from Global Constraints are preserved, relocated into the correct sections.** `#resolution-banner` lives inside `#overlay-banner`.

- [ ] **Step 1: Write the failing dock-state test**

`viewer/test/dock.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sectionStorageKey, loadSectionState, saveSectionState } from '../src/dock.mjs';

// Minimal localStorage shim for Node.
function withStorage(fn) {
  const store = new Map();
  globalThis.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  };
  try { fn(); } finally { delete globalThis.localStorage; }
}

test('sectionStorageKey namespaces the id', () => {
  assert.match(sectionStorageKey('sec-layers'), /sec-layers/);
});

test('loadSectionState returns the fallback when nothing stored', () => {
  withStorage(() => {
    assert.equal(loadSectionState('sec-x', true), true);
    assert.equal(loadSectionState('sec-x', false), false);
  });
});

test('saveSectionState round-trips', () => {
  withStorage(() => {
    saveSectionState('sec-y', false);
    assert.equal(loadSectionState('sec-y', true), false);
    saveSectionState('sec-y', true);
    assert.equal(loadSectionState('sec-y', true), true);
  });
});

test('loadSectionState survives a throwing localStorage', () => {
  const prev = globalThis.localStorage;
  globalThis.localStorage = { getItem() { throw new Error('blocked'); } };
  try {
    assert.equal(loadSectionState('sec-z', true), true); // fallback, no throw
  } finally {
    if (prev === undefined) delete globalThis.localStorage; else globalThis.localStorage = prev;
  }
});
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `node --test viewer/test/dock.test.mjs`
Expected: FAIL — `dock.mjs` does not exist.

- [ ] **Step 3: Implement `viewer/src/dock.mjs`**

```js
const NS = 'megido-viewer:section:';

export function sectionStorageKey(id) {
  return NS + id;
}

export function loadSectionState(id, fallbackOpen = true) {
  try {
    const raw = localStorage.getItem(sectionStorageKey(id));
    if (raw === null) return fallbackOpen;
    return raw === '1';
  } catch {
    return fallbackOpen;
  }
}

export function saveSectionState(id, isOpen) {
  try {
    localStorage.setItem(sectionStorageKey(id), isOpen ? '1' : '0');
  } catch {
    /* per-viewer convenience only; ignore storage failures */
  }
}

export function initDock(root) {
  const sections = root.querySelectorAll('[data-section]');
  for (const sec of sections) {
    const id = sec.getAttribute('data-section');
    const header = sec.querySelector('.section-header');
    const open = loadSectionState(id, sec.getAttribute('data-default-open') !== 'false');
    sec.setAttribute('data-open', open ? 'true' : 'false');
    if (header) {
      header.addEventListener('click', () => {
        const next = sec.getAttribute('data-open') !== 'true';
        sec.setAttribute('data-open', next ? 'true' : 'false');
        saveSectionState(id, next);
      });
    }
  }
}
```

- [ ] **Step 4: Run it, confirm it passes**

Run: `node --test viewer/test/dock.test.mjs`
Expected: PASS (4 tests).

- [ ] **Step 5: Restructure `viewer/shell.html`**

Replace the whole file. Requirements (the implementer applies taste within these — this is the visual identity task):
- `<head><style>` defines the design-system tokens on `:root` and all component CSS. Tokens (exact values — the agreed palette):
  ```css
  :root{
    --bg:#0b0d10; --panel:#14181d; --panel-2:#1a1f26; --line:#232a31;
    --text:#e6edf3; --text-dim:#9fb0bd; --accent:#2dd4bf; --accent-dim:#1c8b7f;
    --warn:#e5484d; --warn-bg:#2a1113; --radius:8px; --gap:10px;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  ```
- `html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font:13px/1.45 var(--sans);}`
- Layout: `#app-root{display:flex;height:100%}` → `#dock{width:300px;flex:0 0 300px;background:var(--panel);border-right:1px solid var(--line);overflow-y:auto;display:flex;flex-direction:column}` and `#canvas-wrap{position:relative;flex:1 1 auto;min-width:0}`. `#gl-canvas{display:block;width:100%;height:100%}`.
- `.dock-section` collapse: header row (`.section-header`, clickable, with a caret) + body (`.section-body`). CSS: `.dock-section[data-open="false"] .section-body{display:none}`; caret rotates via `[data-open]`. Section headers use uppercase small-caps letter-spaced labels in `--text-dim`.
- Numerics (`--mono`, `font-variant-numeric:tabular-nums`), hairline borders, `--radius` corners, accent only on active/focus (`:focus-visible{outline:2px solid var(--accent)}`), 120ms transitions guarded by `@media (prefers-reduced-motion:reduce){*{transition:none!important}}`.
- Body structure:
  ```html
  <div id="app-root">
    <aside id="dock">
      <header id="dock-header">
        <div id="app-title">Megiddo · muon voxels</div>
        <div id="run-name" class="mono dim"></div>
        <div class="row">
          <label class="btn">Load run<input id="load-run-input" type="file" webkitdirectory multiple hidden></label>
          <label class="btn">Load compare<input id="load-second-run-input" type="file" webkitdirectory multiple hidden></label>
        </div>
        <div id="delta-verdict" class="mono dim"></div>
      </header>
      <section class="dock-section" data-section="sec-layers"><div class="section-header">Layers</div><div class="section-body"><div id="layer-panel"></div></div></section>
      <section class="dock-section" data-section="sec-transfer"><div class="section-header">Transfer</div><div class="section-body">
        <!-- Task 2 colormap picker, Task 3 histogram band live here; keep xfer/histogram canvases + window sliders for now -->
        <canvas id="xfer-canvas" width="512" height="40"></canvas>
        <canvas id="histogram-canvas" width="512" height="60"></canvas>
        <label>Window lo <input id="window-lo" type="range" min="0" max="1" step="0.001" value="0"></label>
        <label>Window hi <input id="window-hi" type="range" min="0" max="1" step="0.001" value="1"></label>
      </div></section>
      <section class="dock-section" data-section="sec-clip"><div class="section-header">Clip &amp; slice</div><div class="section-body">
        <!-- the six clip-*, slice-axis, slice-pos, clip-plane-* controls, ids preserved -->
      </div></section>
      <section class="dock-section" data-section="sec-uncertainty"><div class="section-header">Uncertainty</div><div class="section-body">
        <!-- sigma-gate-enabled, sigma-gate-value, ids preserved -->
      </div></section>
      <section class="dock-section" data-section="sec-camera"><div class="section-header">Camera &amp; view</div><div class="section-body">
        <div id="camera-presets" class="row">
          <button id="camera-preset-top">Top</button><button id="camera-preset-front">Front</button>
          <button id="camera-preset-side">Side</button><button id="camera-preset-iso">Iso</button>
        </div>
        <button id="export-png-btn">Export PNG</button>
      </div></section>
    </aside>
    <div id="canvas-wrap">
      <canvas id="gl-canvas"></canvas>
      <div id="overlay-banner"><div id="resolution-banner"></div></div>
      <div id="overlay-gizmo"></div>
      <div id="overlay-legend"></div>
      <div id="overlay-hover"><div id="hover-readout" class="mono"></div></div>
      <div id="overlay-onboarding"></div>
      <div id="overlay-shortcuts" hidden></div>
    </div>
  </div>
  <script>/* __VIEWER_BUNDLE__ */</script>
  ```
  Keep ALL existing control ids; only their container/placement changes. The `hidden` file inputs are triggered by their wrapping `.btn` labels.

- [ ] **Step 6: Restyle the banner as a pill and call `initDock`**

Modify `viewer/src/app.mjs`:
- Add `import { initDock } from './dock.mjs';` at the top.
- At the end of `initViewer` (before `window.__viewerState = state;`), add `initDock(root);`.
- In `loadRun`, where the banner is set (currently sets `background`/`color`), also set `#run-name` textContent to `meta.run` and keep the banner semantics but as a pill: set `banner.textContent = res.verdict || ''`, toggle a class `banner.classList.toggle('not-resolved', !res.depth_resolved)`. Add CSS in shell.html: `#resolution-banner{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;max-width:60ch}` and `#resolution-banner.not-resolved{color:var(--warn);border:1px solid var(--warn);background:var(--warn-bg)}`, `#overlay-banner{position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:5;pointer-events:none}`. **Keep `banner.textContent` = the full verdict string** (test_smoke asserts it contains "depth NOT resolved"); if you also implement hover-to-expand, the full text must still be present in the DOM (e.g. as the element's textContent or a `title`/child), never truncated out of the DOM.

- [ ] **Step 7: Update `megido/viewerbuild.py` `_MODULE_ORDER`**

Add `'dock.mjs',` to `_MODULE_ORDER` immediately before `'app.mjs'`.

- [ ] **Step 8: Write the dock Playwright test**

`tests/viewer/test_dock.py`:
```python
def test_dock_sections_collapse_and_persist(page, dist_path, run_fixture):
    run = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#dock")
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    sec = page.locator('[data-section="sec-clip"]')
    assert sec.get_attribute("data-open") == "true"
    sec.locator(".section-header").click()
    assert sec.get_attribute("data-open") == "false"

    # persisted across reload
    page.reload()
    page.wait_for_selector('[data-section="sec-clip"]')
    assert page.locator('[data-section="sec-clip"]').get_attribute("data-open") == "false"
```

- [ ] **Step 9: Run the full viewer suite; every existing test must still pass**

Run: `node --test viewer/test/*.mjs && uv run pytest tests/test_viewerbuild.py tests/viewer -v --browser chromium`
Expected: PASS — existing tests keep passing because every control id is preserved; new dock test passes. If a pre-existing test fails, an id was dropped or moved out of reach — fix the structure, not the test.

- [ ] **Step 10: Visual check + commit**

Run the observation script (`uv run python /tmp/.../scratchpad/observe.py` if present, else any quick screenshot) OR at minimum confirm via bounding boxes that `#dock` occupies the left 300px and `#gl-canvas` fills the rest. Then:
```bash
git add viewer/shell.html viewer/src/dock.mjs viewer/test/dock.test.mjs viewer/src/app.mjs megido/viewerbuild.py tests/viewer/test_dock.py
git commit -m "feat(viewer): dark instrument design system, left dock, collapsible sections, banner pill"
```

---

### Task 2: Colormap presets

**Files:**
- Create: `viewer/src/colormap.mjs`
- Create: `viewer/test/colormap.test.mjs`
- Modify: `viewer/shell.html` (add `#colormap-select` in the Transfer section)
- Modify: `viewer/src/app.mjs` (apply selected colormap; remember last in localStorage)
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: add `colormap.mjs` before `app.mjs`)
- Create: `tests/viewer/test_colormap.py`

**Interfaces:**
- Produces `viewer/src/colormap.mjs`: `COLORMAP_NAMES` (array, e.g. `['viridis','inferno','magma','grayscale']`); `colormapStops(name) -> Array<{t,r,g,b,a}>` returning transfer stops (same shape `defaultStops()` produces, consumable by `buildTransferLUT`). Alpha ramps 0→255 across t so low density stays transparent. Unknown name falls back to `'viridis'`.
- Consumes: nothing.
- app.mjs consumes `colormapStops`/`COLORMAP_NAMES`; sets `state.transferStops = colormapStops(name)` then rebuilds the LUT and redraws the editor.

- [ ] **Step 1: Write the failing test**

`viewer/test/colormap.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { COLORMAP_NAMES, colormapStops } from '../src/colormap.mjs';

test('exposes the four presets', () => {
  for (const n of ['viridis', 'inferno', 'magma', 'grayscale']) {
    assert.ok(COLORMAP_NAMES.includes(n), n);
  }
});

test('stops span t=0..1 with a rising alpha ramp', () => {
  const s = colormapStops('viridis');
  assert.equal(s[0].t, 0);
  assert.equal(s[s.length - 1].t, 1);
  assert.ok(s[0].a <= s[s.length - 1].a);          // alpha rises with density
  for (const st of s) {
    for (const c of ['r', 'g', 'b', 'a']) {
      assert.ok(st[c] >= 0 && st[c] <= 255, `${c}=${st[c]}`);
    }
  }
});

test('grayscale is neutral (r==g==b at every stop)', () => {
  for (const st of colormapStops('grayscale')) {
    assert.equal(st.r, st.g);
    assert.equal(st.g, st.b);
  }
});

test('unknown name falls back to viridis', () => {
  assert.deepEqual(colormapStops('nope'), colormapStops('viridis'));
});
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `node --test viewer/test/colormap.test.mjs` — FAIL, module missing.

- [ ] **Step 3: Implement `viewer/src/colormap.mjs`**

Use 5-stop approximations of the matplotlib colormaps (values below are accurate enough for a viewer; alpha ramps `0,60,140,200,255`). Grayscale is linear.
```js
const RAMP_A = [0, 60, 140, 200, 255];

const RGB = {
  viridis:  [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
  inferno:  [[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,255,164]],
  magma:    [[0,0,4],[81,18,124],[183,55,121],[252,137,97],[252,253,191]],
  grayscale:[[0,0,0],[64,64,64],[128,128,128],[192,192,192],[255,255,255]],
};

export const COLORMAP_NAMES = ['viridis', 'inferno', 'magma', 'grayscale'];

export function colormapStops(name) {
  const rgb = RGB[name] || RGB.viridis;
  return rgb.map((c, i) => ({
    t: i / (rgb.length - 1),
    r: c[0], g: c[1], b: c[2], a: RAMP_A[i],
  }));
}
```

- [ ] **Step 4: Run it, confirm it passes**

Run: `node --test viewer/test/colormap.test.mjs` — PASS (4 tests).

- [ ] **Step 5: Add the picker to the Transfer section**

In `viewer/shell.html`, at the top of the `sec-transfer` body, add:
```html
<label class="field">Colormap
  <select id="colormap-select"></select>
</label>
```

- [ ] **Step 6: Wire it in `app.mjs`**

- Add `import { COLORMAP_NAMES, colormapStops } from './colormap.mjs';`.
- After `initDock(root)` (or in a small setup block), populate the select and wire change:
  ```js
  const cmapSel = root.querySelector('#colormap-select');
  for (const name of COLORMAP_NAMES) {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    cmapSel.appendChild(opt);
  }
  let lastCmap = 'viridis';
  try { lastCmap = localStorage.getItem('megido-viewer:colormap') || 'viridis'; } catch { /* ignore */ }
  cmapSel.value = COLORMAP_NAMES.includes(lastCmap) ? lastCmap : 'viridis';
  function applyColormap(name) {
    state.transferStops = colormapStops(name);
    try { localStorage.setItem('megido-viewer:colormap', name); } catch { /* ignore */ }
    if (state.drawXferEditor) state.drawXferEditor();
    // rebuild the GL LUT + render
    gl.deleteTexture(state.lutTex);
    state.lutTex = makeLutTexture(gl, buildTransferLUT(state.transferStops));
    render();
  }
  cmapSel.addEventListener('change', (ev) => applyColormap(ev.target.value));
  ```
- On initial load, apply the remembered colormap so the default is not the old hardcoded `defaultStops()`: change the `state.transferStops` initializer to `colormapStops('viridis')`, and after wiring call `applyColormap(cmapSel.value)` once a run is present (or immediately — it only touches the LUT).

- [ ] **Step 7: Add `colormap.mjs` to `_MODULE_ORDER`** (before `app.mjs`).

- [ ] **Step 8: Playwright test**

`tests/viewer/test_colormap.py`:
```python
def test_colormap_switch_changes_render(page, dist_path, run_fixture):
    run = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#colormap-select").select_option("inferno")
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after
    assert page.evaluate("() => window.__viewerState.transferStops[4].r") is not None
```

- [ ] **Step 9: Run + commit**

Run: `node --test viewer/test/*.mjs && uv run pytest tests/viewer -v --browser chromium`
```bash
git add viewer/src/colormap.mjs viewer/test/colormap.test.mjs viewer/shell.html viewer/src/app.mjs megido/viewerbuild.py tests/viewer/test_colormap.py
git commit -m "feat(viewer): colormap presets (viridis/inferno/magma/grayscale)"
```

---

### Task 3: Histogram window band (replace the two window sliders)

**Files:**
- Modify: `viewer/shell.html` (remove `#window-lo`/`#window-hi`; the histogram canvas becomes the window control; add `#window-readout`)
- Modify: `viewer/src/histogram.mjs` (add pure band↔window mapping helpers)
- Modify: `viewer/test/histogram.test.mjs` (test the new helpers)
- Modify: `viewer/src/app.mjs` (draw the band over the histogram; drag its edges to set the window; live readout)
- Modify: `tests/viewer/test_transfer_ui.py` and `tests/viewer/test_smoke.py` (they drive `#window-lo`/`#window-hi` today)

**Interfaces:**
- Produces in `viewer/src/histogram.mjs`: `windowToBandPx(window, layerMax, widthPx) -> {loPx, hiPx}` and `bandPxToWindow(px, layerMax, widthPx) -> value` (maps a pixel x within the histogram canvas to a density in `[0, layerMax]`). Pure.
- Consumes: existing `computeHistogram`, `robustWindow`.
- Replaces the `#window-lo`/`#window-hi` slider inputs; `state.window` and `state.layerMax` semantics unchanged.

- [ ] **Step 1: Add failing helper tests**

Append to `viewer/test/histogram.test.mjs`:
```js
import { windowToBandPx, bandPxToWindow } from '../src/histogram.mjs';

test('windowToBandPx maps window onto pixel span', () => {
  const { loPx, hiPx } = windowToBandPx([0, 0.5], 1.0, 200);
  assert.equal(loPx, 0);
  assert.equal(hiPx, 100);
});

test('bandPxToWindow is the inverse', () => {
  assert.ok(Math.abs(bandPxToWindow(100, 1.0, 200) - 0.5) < 1e-9);
  assert.equal(bandPxToWindow(-5, 1.0, 200), 0);      // clamped low
  assert.equal(bandPxToWindow(999, 1.0, 200), 1.0);   // clamped to layerMax
});
```

- [ ] **Step 2: Confirm fail, then implement in `histogram.mjs`**

```js
export function windowToBandPx(window, layerMax, widthPx) {
  const max = layerMax > 0 ? layerMax : 1;
  const loPx = (window[0] / max) * widthPx;
  const hiPx = (window[1] / max) * widthPx;
  return { loPx, hiPx };
}

export function bandPxToWindow(px, layerMax, widthPx) {
  const max = layerMax > 0 ? layerMax : 1;
  const frac = Math.max(0, Math.min(1, px / widthPx));
  return frac * max;
}
```
Run: `node --test viewer/test/histogram.test.mjs` — PASS.

- [ ] **Step 3: Edit `viewer/shell.html`**

In `sec-transfer`, remove the two `<label>Window lo/hi ...</label>` lines and their inputs. Under the histogram canvas add:
```html
<div id="window-readout" class="mono dim"></div>
```
Give `#histogram-canvas` `style="width:100%;height:80px;display:block;cursor:ew-resize"` and `#xfer-canvas` `style="width:100%;height:24px;display:block"`.

- [ ] **Step 4: Wire the band in `app.mjs`**

- Add `windowToBandPx, bandPxToWindow` to the histogram import.
- Extend `drawHistogram()` to also draw the current window band as a translucent accent rectangle with two edge handles over the bars, using `windowToBandPx(state.window, state.layerMax, canvas.width)`. After drawing, update `#window-readout` textContent to `` `${state.window[0].toFixed(3)} – ${state.window[1].toFixed(3)} 1/m` ``.
- Replace the removed slider `input` handlers with pointer drag on `#histogram-canvas`: on `pointerdown`, pick whichever edge (lo/hi) is nearer the cursor x; on `pointermove` while dragging, set that edge via `bandPxToWindow(cursorXWithinCanvas, state.layerMax, canvas.width)` (convert client x to canvas pixel via `rect` and `canvas.width/rect.width`), keep `lo <= hi`, then `drawHistogram(); render();`. Use a drag flag local to this control so it never fights the camera or transfer-editor drags.
- In `syncWindowSliders()` (rename to `syncWindowUI()` or keep name) drop the slider writes; instead just call `drawHistogram()` so the band reflects the new window. Update the one call site.

- [ ] **Step 5: Update the two Playwright tests that used the sliders**

In `tests/viewer/test_transfer_ui.py` and `tests/viewer/test_smoke.py`, replace any `page.locator("#window-lo"/"#window-hi").fill(...).dispatch_event("input")` with a drag on the histogram band, e.g.:
```python
box = page.locator("#histogram-canvas").bounding_box()
page.mouse.move(box["x"] + box["width"] * 0.15, box["y"] + box["height"] / 2)
page.mouse.down()
page.mouse.move(box["x"] + box["width"] * 0.35, box["y"] + box["height"] / 2)
page.mouse.up()
```
and assert the render changed and `#window-readout` is non-empty. Keep every other assertion (especially, in test_smoke, the banner + non-blank + zero-console-errors). Do NOT reintroduce `#window-lo`/`#window-hi`.

- [ ] **Step 6: Run + commit**

Run: `node --test viewer/test/*.mjs && uv run pytest tests/viewer -v --browser chromium`
```bash
git add viewer/shell.html viewer/src/histogram.mjs viewer/test/histogram.test.mjs viewer/src/app.mjs tests/viewer/test_transfer_ui.py tests/viewer/test_smoke.py
git commit -m "feat(viewer): draggable window band over the histogram, live density readout"
```

---

### Task 4: Clip dual-handle rows, value labels, restyled slice & plane

**Files:**
- Modify: `viewer/shell.html` (restyle clip section; add value-label spans; keep all ids)
- Modify: `viewer/src/app.mjs` (update value labels live on input)
- Modify: `tests/viewer/test_clip.py` only if selectors need the label assertions (existing behavior unchanged)

**Interfaces:**
- No new module. All existing clip ids preserved (`#clip-x-min` … `#clip-z-max`, `#slice-axis`, `#slice-pos`, `#clip-plane-enabled`, `#clip-plane-d`). Add label spans with ids `#clip-x-min-val` … etc. and `#clip-plane-d-val`, updated on input.

- [ ] **Step 1: Restyle the clip section markup**

In `sec-clip` body, lay out each axis as one compact row with two range inputs (min/max) styled as a paired track, each followed by a small `<span class="mono val" id="clip-x-min-val">0.00</span>`. Keep the `<select id="slice-axis">`, `<input id="slice-pos">`, `<input id="clip-plane-enabled" type="checkbox">`, `<input id="clip-plane-d">` with a `#clip-plane-d-val` span. Exact CSS/markup at the implementer's taste, but every listed id must exist and remain operable.

- [ ] **Step 2: Live labels in `app.mjs`**

In each existing clip `input` handler, after updating state, also set the matching `-val` span textContent (e.g. `root.querySelector('#clip-x-min-val').textContent = ev.target.value`). For `#clip-plane-d`, format to 2 decimals. Initialize all `-val` spans once at setup to their input's current value.

- [ ] **Step 3: Run the existing clip test (must stay green) + add a label assertion**

Add to `tests/viewer/test_clip.py` an assertion that after moving `#clip-x-max`, `#clip-x-max-val` textContent reflects the new value. Keep the existing before/after render assertions.

Run: `uv run pytest tests/viewer/test_clip.py -v --browser chromium` — PASS.

- [ ] **Step 4: Commit**

```bash
git add viewer/shell.html viewer/src/app.mjs tests/viewer/test_clip.py
git commit -m "feat(viewer): compact clip rows with live value labels"
```

---

### Task 5: Canvas overlays — axis gizmo, colorbar legend, hover tooltip

**Files:**
- Modify: `viewer/shell.html` (style `#overlay-gizmo`, `#overlay-legend`, `#overlay-hover`)
- Modify: `viewer/src/app.mjs` (draw gizmo from camera; draw legend from LUT + layer range; reposition hover as a cursor tooltip)
- Create: `tests/viewer/test_overlays.py`

**Interfaces:**
- No new module (the gizmo uses existing `orbitToEye`/`mat4`). `#overlay-gizmo` gets a small `<canvas id="gizmo-canvas" width="80" height="80">`; `#overlay-legend` gets `<canvas id="legend-canvas" width="200" height="44">` + tick labels; `#hover-readout` keeps its id but moves into `#overlay-hover` and follows the cursor.

- [ ] **Step 1: Overlay CSS + elements**

CSS: `#overlay-gizmo{position:absolute;left:12px;bottom:12px;z-index:5;pointer-events:none}`, `#overlay-legend{position:absolute;right:12px;bottom:12px;z-index:5;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:6px}`, `#overlay-hover{position:absolute;inset:0;z-index:6;pointer-events:none}`, `#hover-readout{position:absolute;background:var(--panel-2);border:1px solid var(--line);border-radius:6px;padding:2px 6px;transform:translate(12px,12px)}`. Put a `<canvas id="gizmo-canvas" width="80" height="80">` in `#overlay-gizmo` and a `<canvas id="legend-canvas" width="200" height="24">` + a `<div id="legend-ticks" class="mono dim">` in `#overlay-legend`.

- [ ] **Step 2: Draw the gizmo from the camera**

Add a `drawGizmo()` called at the end of `render()`. Project the three world axes into view space using the same `yaw/pitch` (reuse `orbitToEye`/`lookAt` basis, or derive the 3 rotated unit vectors from `yaw`/`pitch`), and draw three labeled lines (X red, Y green, Z blue) from the gizmo-canvas center. Respect the accent-restrained palette (axis colors are the exception, they encode orientation).

- [ ] **Step 3: Draw the colorbar legend**

Add `drawLegend()` called whenever the LUT or window changes (colormap change, window drag, layer switch). Fill `#legend-canvas` with the current `buildTransferLUT(state.transferStops)` gradient (left→right), and set `#legend-ticks` to three labels: `state.window[0]`, midpoint, `state.window[1]`, formatted with units `1/m` (for non-volume layers use their own units label — plain number is fine). Call it from the same places `drawXferEditor()`/`drawHistogram()` are called.

- [ ] **Step 4: Hover tooltip follows the cursor**

Change the existing `canvas.addEventListener('pointermove', ...)` hover handler: keep `castHoverRay`, but position `#hover-readout` at the cursor (`el.style.left = (ev.clientX - wrapRect.left) + 'px'; el.style.top = (ev.clientY - wrapRect.top) + 'px'`, where `wrapRect` is `#canvas-wrap`'s rect), and set `el.hidden = !hit`. Keep the same text format.

- [ ] **Step 5: Playwright test**

`tests/viewer/test_overlays.py`: load a run; assert `#gizmo-canvas` and `#legend-canvas` are present and non-blank (dataURL differs from a blank canvas of the same size); assert `#legend-ticks` textContent is non-empty; move the mouse over the canvas center and assert `#hover-readout` either updates text or stays hidden without throwing (console-error-free).

- [ ] **Step 6: Run + commit**

Run: `uv run pytest tests/viewer -v --browser chromium`
```bash
git add viewer/shell.html viewer/src/app.mjs tests/viewer/test_overlays.py
git commit -m "feat(viewer): axis gizmo, colorbar legend, cursor-following hover tooltip"
```

---

### Task 6: Saved views

**Files:**
- Create: `viewer/src/views.mjs`
- Create: `viewer/test/views.test.mjs`
- Modify: `viewer/shell.html` (saved-views UI + "Frame all" button in `sec-camera`)
- Modify: `viewer/src/app.mjs` (capture/apply a view; list persisted views)
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: add `views.mjs` before `app.mjs`)
- Create: `tests/viewer/test_saved_views.py`

**Interfaces:**
- Produces `viewer/src/views.mjs`: `captureView(state) -> viewObj` (plain JSON: `{name, camera:{yaw,pitch,distance,target}, window, activeLayer}`); `serializeViews(list) -> string`; `deserializeViews(str) -> list` (returns `[]` on malformed input); `loadViews() -> list` and `saveViews(list) -> void` (localStorage, try/catch). Pure except the load/save pair.
- Consumes: reads `state.camera`, `state.window`, `state.activeLayer`.
- app.mjs uses these to save the current view under a name, list saved views, apply one (set camera/window/layer then `render()`), and delete one.

- [ ] **Step 1: Failing tests**

`viewer/test/views.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { captureView, serializeViews, deserializeViews } from '../src/views.mjs';

test('captureView copies camera/window/layer, not references', () => {
  const state = { camera: { yaw: 1, pitch: 2, distance: 3, target: [1, 2, 3] }, window: [0.1, 0.4], activeLayer: 'volume' };
  const v = captureView(state, 'A');
  assert.equal(v.name, 'A');
  assert.deepEqual(v.window, [0.1, 0.4]);
  state.camera.yaw = 99; state.window[0] = 99;
  assert.equal(v.camera.yaw, 1);   // deep copy
  assert.equal(v.window[0], 0.1);
});

test('serialize/deserialize round-trips', () => {
  const list = [captureView({ camera: { yaw: 0, pitch: 0, distance: 1, target: [0,0,0] }, window: [0,1], activeLayer: 'sigma' }, 'v1')];
  assert.deepEqual(deserializeViews(serializeViews(list)), list);
});

test('deserializeViews returns [] on garbage', () => {
  assert.deepEqual(deserializeViews('not json'), []);
  assert.deepEqual(deserializeViews('{"x":1}'), []); // not an array
});
```

- [ ] **Step 2: Confirm fail, implement `views.mjs`**

```js
const KEY = 'megido-viewer:views';

export function captureView(state, name) {
  const c = state.camera;
  return {
    name: name || 'view',
    camera: { yaw: c.yaw, pitch: c.pitch, distance: c.distance, target: [...c.target] },
    window: [state.window[0], state.window[1]],
    activeLayer: state.activeLayer,
  };
}

export function serializeViews(list) {
  return JSON.stringify(list);
}

export function deserializeViews(str) {
  try {
    const v = JSON.parse(str);
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

export function loadViews() {
  try { return deserializeViews(localStorage.getItem(KEY) || '[]'); } catch { return []; }
}

export function saveViews(list) {
  try { localStorage.setItem(KEY, serializeViews(list)); } catch { /* ignore */ }
}
```
Run: `node --test viewer/test/views.test.mjs` — PASS.

- [ ] **Step 3: UI in `sec-camera`**

Add after the presets row:
```html
<button id="frame-all-btn">Frame all</button>
<div class="row"><input id="view-name" type="text" placeholder="view name"><button id="save-view-btn">Save view</button></div>
<ul id="saved-views"></ul>
```

- [ ] **Step 4: Wire in `app.mjs`**

- Add `import { captureView, loadViews, saveViews } from './views.mjs';`.
- `#frame-all-btn`: re-run the framing math from `loadRun` (target = grid center, distance = 1.3·diagonal) then `render()`. Factor that framing into a `frameAll()` helper so both `loadRun` and the button call it.
- `#save-view-btn`: `const list = loadViews(); list.push(captureView(state, nameInput.value || 'view ' + (list.length+1))); saveViews(list); renderViewList();`.
- `renderViewList()`: rebuild `#saved-views` with one `<li>` per saved view: an Apply button (sets `state.camera` from the view — deep copy — `state.window`, calls `state.setActiveLayer(view.activeLayer)` if present else `render()`) and a Delete button (splice + `saveViews` + re-render). Call `renderViewList()` at startup (views persist across sessions).

- [ ] **Step 5: Add `views.mjs` to `_MODULE_ORDER`** (before `app.mjs`).

- [ ] **Step 6: Playwright test**

`tests/viewer/test_saved_views.py`: load a run; click a camera preset; type a name; Save view; assert a `#saved-views li` exists. Rotate camera (drag), then click the saved view's Apply; assert `window.__viewerState.camera.yaw` returned to the saved value. Delete it; assert the list is empty.

- [ ] **Step 7: Run + commit**

```bash
git add viewer/src/views.mjs viewer/test/views.test.mjs viewer/shell.html viewer/src/app.mjs megido/viewerbuild.py tests/viewer/test_saved_views.py
git commit -m "feat(viewer): frame-all + saved views (persisted)"
```

---

### Task 7: Keyboard shortcuts, cheatsheet overlay, onboarding card

**Files:**
- Create: `viewer/src/shortcuts.mjs`
- Create: `viewer/test/shortcuts.test.mjs`
- Modify: `viewer/shell.html` (`#overlay-onboarding`, `#overlay-shortcuts` content + styles)
- Modify: `viewer/src/app.mjs` (keydown → action dispatch; show/hide overlays; onboarding gating)
- Modify: `megido/viewerbuild.py` (`_MODULE_ORDER`: add `shortcuts.mjs` before `app.mjs`)
- Create: `tests/viewer/test_shortcuts.py`

**Interfaces:**
- Produces `viewer/src/shortcuts.mjs`: `SHORTCUTS` (array of `{keys, action, label}` for the cheatsheet); `keyToAction(ev) -> string|null` mapping a KeyboardEvent-like `{key}` to an action name (`layer1..layer5`, `preset-top/front/side/iso`, `frame-all`, `toggle-help`, `close-overlay`). Pure — takes a plain object, no DOM.
- Consumes: nothing.
- app.mjs maps action names to the same handlers the buttons use.

- [ ] **Step 1: Failing test**

`viewer/test/shortcuts.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SHORTCUTS, keyToAction } from '../src/shortcuts.mjs';

test('digits map to layer actions', () => {
  assert.equal(keyToAction({ key: '1' }), 'layer1');
  assert.equal(keyToAction({ key: '5' }), 'layer5');
});
test('r frames all, ? toggles help, Escape closes', () => {
  assert.equal(keyToAction({ key: 'r' }), 'frame-all');
  assert.equal(keyToAction({ key: '?' }), 'toggle-help');
  assert.equal(keyToAction({ key: 'Escape' }), 'close-overlay');
});
test('unmapped key returns null', () => {
  assert.equal(keyToAction({ key: 'q' }), null);
});
test('SHORTCUTS is a non-empty list with labels', () => {
  assert.ok(SHORTCUTS.length > 0);
  for (const s of SHORTCUTS) assert.ok(s.label && s.action);
});
```

- [ ] **Step 2: Confirm fail, implement `shortcuts.mjs`**

```js
export const SHORTCUTS = [
  { keys: '1–5', action: 'layerN', label: 'Switch layer' },
  { keys: 't/f/s/i', action: 'preset', label: 'Camera preset (top/front/side/iso)' },
  { keys: 'r', action: 'frame-all', label: 'Frame all' },
  { keys: '?', action: 'toggle-help', label: 'Toggle this help' },
  { keys: 'Esc', action: 'close-overlay', label: 'Close overlay' },
];

export function keyToAction(ev) {
  const k = ev.key;
  if (k >= '1' && k <= '5') return 'layer' + k;
  if (k === 't') return 'preset-top';
  if (k === 'f') return 'preset-front';
  if (k === 's') return 'preset-side';
  if (k === 'i') return 'preset-iso';
  if (k === 'r') return 'frame-all';
  if (k === '?') return 'toggle-help';
  if (k === 'Escape') return 'close-overlay';
  return null;
}
```
Run: `node --test viewer/test/shortcuts.test.mjs` — PASS.

- [ ] **Step 3: Overlay content + CSS**

`#overlay-onboarding`: a centered card (`position:absolute;inset:0;display:grid;place-items:center;z-index:7`) with a panel: title "Load a run to begin", 3 bullets ("Load run → pick a run directory (e.g. runs/voxels)", "Drag to orbit, scroll to zoom", "Press ? for shortcuts"), and a "Got it" dismiss button. `#overlay-shortcuts`: a centered panel listing `SHORTCUTS`, `hidden` by default.

- [ ] **Step 4: Wire in `app.mjs`**

- Add `import { SHORTCUTS, keyToAction } from './shortcuts.mjs';`.
- Populate `#overlay-shortcuts` from `SHORTCUTS` once at setup.
- Onboarding: show `#overlay-onboarding` at startup unless dismissed (`localStorage 'megido-viewer:onboarded'`), and hide it once a run loads. "Got it" sets the flag (try/catch) and hides it.
- `window.addEventListener('keydown', ev => { const a = keyToAction(ev); if (!a) return; ... })`: `layer1..5` → select the Nth available layer radio (guard out of range); `preset-*` → click the matching preset; `frame-all` → `frameAll(); render()`; `toggle-help` → toggle `#overlay-shortcuts` `hidden`; `close-overlay` → hide both overlays. Ignore keydown when the target is an `<input>`/`<select>`/`<textarea>` (so typing a view name doesn't trigger shortcuts).

- [ ] **Step 5: Add `shortcuts.mjs` to `_MODULE_ORDER`** (before `app.mjs`).

- [ ] **Step 6: Playwright test**

`tests/viewer/test_shortcuts.py`: on first load (fresh context) assert `#overlay-onboarding` is visible; load a run; assert it hides. Press `?`; assert `#overlay-shortcuts` becomes visible; press `Escape`; assert hidden. Press `2`; assert `window.__viewerState.activeLayer` changed to the 2nd layer. Ensure zero console errors.

- [ ] **Step 7: Run + commit**

```bash
git add viewer/src/shortcuts.mjs viewer/test/shortcuts.test.mjs viewer/shell.html viewer/src/app.mjs megido/viewerbuild.py tests/viewer/test_shortcuts.py
git commit -m "feat(viewer): keyboard shortcuts, cheatsheet overlay, onboarding card"
```

---

### Task 8: Integration polish + exit gate on real data

**Files:**
- Modify: `viewer/shell.html` / `viewer/src/app.mjs` (only polish fixes found here)
- Modify: `tests/viewer/test_smoke.py` (exercise the full new UI on the real run)
- Modify: `viewer/README.md` (screenshot-free description of the new UI)

**Interfaces:** consumes everything from Tasks 1–7.

- [ ] **Step 1: Extend the real-data smoke test**

Update `tests/viewer/test_smoke.py` to, on `runs/voxels`, in addition to its existing assertions: switch colormap; drag the histogram window band; collapse/expand a dock section; open the shortcuts overlay with `?` and close with Escape; save a view and apply it; assert `#gizmo-canvas` and `#legend-canvas` are non-blank; and keep the hard assertions — canvas non-blank, `#resolution-banner` contains "depth NOT resolved", `window.__viewerError` is null, and **`console_errors == []`** through the whole interaction. Use `set_input_files(str(REAL_RUN))` and wait on `state.ready`.

- [ ] **Step 2: Run it against real data**

Run: `uv run pytest tests/viewer/test_smoke.py -v --browser chromium`
Expected: PASS. If any control throws on real data (e.g. a spread over the 675,840-element arrays, or a null deref), fix the owning code — do not weaken the test.

- [ ] **Step 3: Visual confirmation**

Rebuild (`uv run python -m megido.cli view --no-open`) and screenshot the real run (reuse `scratchpad/observe.py` or an equivalent). Confirm by eye: dock on the left with tidy sections, volume clearly rendered and framed, banner as a red pill top-center, gizmo bottom-left, legend bottom-right. Note findings in the report.

- [ ] **Step 4: Whole-suite + docs + commit**

Run: `uv run pytest -q && node --test viewer/test/*.mjs`
Update `viewer/README.md`'s "Using it" section to describe the dock, colormaps, window band, saved views, and shortcuts (no screenshots — text only).
```bash
git add tests/viewer/test_smoke.py viewer/README.md viewer/shell.html viewer/src/app.mjs
git commit -m "test(viewer): full-UI exit-gate smoke on real campaign; README refresh"
```
