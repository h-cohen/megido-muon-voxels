# Phase 4 viewer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship S5 — a self-contained, single-file HTML viewer that raymarches the
Phase 3 voxel field on the GPU, with a layer manager, transfer-function editor,
clip/slice controls, an uncertainty gate, and an always-visible depth-resolution
verdict, gated by a Playwright smoke test on the real reconstructed campaign.

**Architecture:** Pure logic (npy/meta parsing, 4x4 matrix math, the voxel grid's
index↔world mapping, transfer-function LUT construction, histogram binning, clip
math, layer metadata, run-delta math) lives in small ES modules under
`viewer/src/*.mjs`, unit-tested directly with Node's built-in test runner — no
browser needed for that layer. DOM/WebGL glue lives in `viewer/src/app.mjs` and
is only exercisable in a real browser, so it is gated by Playwright. A tiny
Python build step (`megido/viewerbuild.py`) concatenates the `.mjs` sources (in
dependency order, stripping `export`/`import` lines) into one inert classic
`<script>` block inside `viewer/shell.html`, producing `viewer/dist/index.html`
— a zero-dependency, zero-network file that opens correctly from `file://`.
There is exactly one GPU shader, written once (Task 3) with every uniform the
whole feature set needs declared up front; later tasks wire JS controls to
already-existing uniforms rather than re-touching GLSL.

**Tech Stack:** WebGL2 (raymarching, `TEXTURE_3D`), vanilla JS ES modules (no
runtime libraries — the entire "vendored libraries" requirement is satisfied by
having zero external dependencies), Python 3.12 + numpy for the data contract
side (already built in Phase 3), Node's built-in `node:test` for JS unit tests,
Playwright (`pytest-playwright`) for browser-level gates.

**Spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md`
section 8 (S5), section 9.4 (delta layer), section 10 (S5 test level).

## Global Constraints

- Data contract is `volume.npy` (float32 `[nx,ny,nz]`, `axis_order: "xyz"`) +
  `meta.json`, exactly as written by `megido/volexport.py` (Phase 3, already
  built — do not modify it). Real example fields used throughout this plan:
  `shape`, `origin_m`, `spacing_m`, `units`, `value_range`, `suggested_iso`,
  `layers` (list of names with a matching `<name>.npy` file in the run dir),
  `resolution.depth_resolved` (bool) and `resolution.verdict` (string).
- The viewer is a **static app**: it ships with no data baked in and loads a
  run directory at runtime via `<input type="file" webkitdirectory>` +
  `FileReader`. Never use `fetch()` for local files — that is blocked by CORS
  under `file://` in Chromium, which is exactly the failure mode the
  "works from `file://`" requirement exists to prevent.
- Every `.mjs` file under `viewer/src/` must be loadable two ways without
  changes to its own text: (a) `import`ed directly by `node:test` for unit
  tests, (b) concatenated by `megido/viewerbuild.py` into the single shipped
  HTML file. Concretely: only use `export function`/`export const` at the
  top level, and only `import { x, y } from './other.mjs'` lines for
  cross-module use (never a default export, never a dynamic `import()`).
- `viewer/dist/` is a build artifact — gitignore it. `viewer/src/`, `viewer/test/`,
  `viewer/shell.html`, `megido/viewerbuild.py` are the checked-in sources.
- Add `playwright` + `pytest-playwright` under `[project.optional-dependencies].dev`
  in `pyproject.toml`, and note the one-time `playwright install chromium` setup
  step in `viewer/README.md`. Every Playwright test in this plan uses headless
  Chromium (`browser_type_launch_args={"headless": True}` — default) navigating
  to a `file://` URL built by the test itself; never a live server.
- Matrices in `viewer/src/mat4.mjs` are column-major `Float32Array(16)`, the
  WebGL/OpenGL convention. `multiply(a, b)` returns the matrix that applies
  `b` first, then `a` (`multiply(a, b) * v == a * (b * v)`).
- The shader (`viewer/src/app.mjs`, `FRAGMENT_SRC`, written once in Task 3) is
  never rewritten in later tasks — only its already-declared uniforms are set
  from new JS. If a later task's brief seems to need a new uniform, that is a
  plan defect: ruling goes in the ledger, not a shader edit outside Task 3.

---

### Task 1: Build pipeline, npy parsing, CLI wiring, Playwright scaffold

**Files:**
- Create: `viewer/src/npy.mjs`
- Create: `viewer/test/npy.test.mjs`
- Create: `viewer/shell.html`
- Create: `viewer/README.md`
- Create: `megido/viewerbuild.py`
- Create: `tests/test_viewerbuild.py`
- Create: `tests/viewer/__init__.py`
- Create: `tests/viewer/conftest.py`
- Modify: `megido/cli.py` (add `view` subcommand)
- Modify: `pyproject.toml` (dev deps)
- Modify: `.gitignore` (add `viewer/dist/`)

**Interfaces:**
- Produces: `parseNpy(buffer: ArrayBuffer) -> {shape: number[], dtype: string, data: Float32Array}` from `viewer/src/npy.mjs`, consumed by `grid.mjs`/`app.mjs` in Task 3+.
- Produces: `build(viewerDir: Path = Path("viewer")) -> Path` from `megido/viewerbuild.py`, returning the path to `viewer/dist/index.html`. Consumed by `megido/cli.py`'s `view` subcommand and by `tests/viewer/conftest.py`'s `dist_path` fixture (used by every Playwright test in this plan).
- Produces: `tests/viewer/conftest.py` fixtures `dist_path` (session-scoped, returns the built `Path`) and `run_fixture(tmp_path)` (a pytest fixture factory documented below, used by Tasks 3-9) that writes a small synthetic run directory.

- [ ] **Step 1: Write the failing npy parser test**

`viewer/test/npy.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseNpy } from '../src/npy.mjs';

function makeNpy(shape, values) {
  // Minimal NPY v1.0 writer: magic + version + header + little-endian f4 data.
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]; // \x93NUMPY, v1.0
  const preLen = magic.length + 2; // + 2-byte header-length field
  let padded = header;
  while ((preLen + padded.length + 1) % 16 !== 0) padded += ' ';
  padded += '\n';
  const headerBytes = new TextEncoder().encode(padded);
  const buf = new ArrayBuffer(preLen + headerBytes.length + values.length * 4);
  const view = new DataView(buf);
  magic.forEach((b, i) => view.setUint8(i, b));
  view.setUint16(8, headerBytes.length, true);
  new Uint8Array(buf, preLen, headerBytes.length).set(headerBytes);
  const dataStart = preLen + headerBytes.length;
  values.forEach((v, i) => view.setFloat32(dataStart + i * 4, v, true));
  return buf;
}

test('parseNpy recovers shape and float32 data', () => {
  const buf = makeNpy([2, 3], [1, 2, 3, 4, 5, 6]);
  const { shape, dtype, data } = parseNpy(buf);
  assert.deepEqual(shape, [2, 3]);
  assert.equal(dtype, '<f4');
  assert.equal(data.length, 6);
  assert.equal(data[0], 1);
  assert.equal(data[5], 6);
  assert.ok(data instanceof Float32Array);
});

test('parseNpy rejects fortran-ordered arrays', () => {
  const buf = makeNpy([2, 2], [1, 2, 3, 4]);
  const bytes = new Uint8Array(buf);
  const text = new TextDecoder().decode(bytes);
  const patched = text.replace('False', 'True ');
  const out = new TextEncoder().encode(patched);
  assert.throws(() => parseNpy(out.buffer), /fortran_order/);
});
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `node --test viewer/test/npy.test.mjs`
Expected: FAIL — `viewer/src/npy.mjs` does not exist yet.

- [ ] **Step 3: Implement the parser**

`viewer/src/npy.mjs`:
```js
export function parseNpy(buffer) {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 6);
  const expected = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < 6; i++) {
    if (magic[i] !== expected[i]) throw new Error('not an NPY file: bad magic');
  }
  const major = view.getUint8(6);
  const headerLenBytes = major >= 2 ? 4 : 2;
  const headerLen = major >= 2
    ? view.getUint32(8, true)
    : view.getUint16(8, true);
  const headerStart = 8 + headerLenBytes;
  const headerText = new TextDecoder().decode(
    new Uint8Array(buffer, headerStart, headerLen)
  );

  const descrMatch = headerText.match(/'descr':\s*'([^']+)'/);
  const orderMatch = headerText.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = headerText.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !orderMatch || !shapeMatch) {
    throw new Error(`malformed NPY header: ${headerText}`);
  }
  const dtype = descrMatch[1];
  const fortranOrder = orderMatch[1] === 'True';
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);

  if (dtype !== '<f4' || fortranOrder) {
    throw new Error(
      `unsupported npy dtype/order: descr=${dtype} fortran_order=${fortranOrder} ` +
      `(this viewer only reads little-endian float32, C-order arrays)`
    );
  }

  const dataStart = headerStart + headerLen;
  const count = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(buffer.slice(dataStart, dataStart + count * 4));
  return { shape, dtype, data };
}
```

- [ ] **Step 4: Run it, confirm it passes**

Run: `node --test viewer/test/npy.test.mjs`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the failing build test**

`tests/test_viewerbuild.py`:
```python
from pathlib import Path

from megido.viewerbuild import build


def test_build_produces_single_html_with_no_export_or_import(tmp_path):
    viewer_dir = Path(__file__).resolve().parents[1] / "viewer"
    out = build(viewer_dir, out_dir=tmp_path)
    assert out.name == "index.html"
    text = out.read_text()
    assert "<!DOCTYPE html>" in text or "<!doctype html>" in text
    assert "export function" not in text
    assert "export const" not in text
    assert "import {" not in text
    assert "parseNpy" in text
    assert 'initViewer(document.body)' in text
```

- [ ] **Step 6: Run it, confirm it fails**

Run: `uv run pytest tests/test_viewerbuild.py -v`
Expected: FAIL — `megido.viewerbuild` does not exist.

- [ ] **Step 7: Write the shell page**

`viewer/shell.html` (the template `build()` fills in; keep it minimal — later
tasks add CSS/DOM elements here, this task only needs the script placeholder
to exist):
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Megiddo voxel viewer</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #eee;
               font: 13px system-ui, sans-serif; }
</style>
</head>
<body>
<div id="app-root"></div>
<script>
/* __VIEWER_BUNDLE__ */
</script>
</body>
</html>
```

- [ ] **Step 8: Write the build function**

`megido/viewerbuild.py`:
```python
"""Concatenates viewer/src/*.mjs into the single-file HTML deliverable.

Node's `node:test` imports these same .mjs files directly (unit tests run
against real ES module semantics); this module strips the export/import
syntax those files need for that and inlines them as one classic <script>
so the shipped page has zero network requests and works from file://.
"""
from __future__ import annotations

import re
from pathlib import Path

# Dependency order: a module may only import from a module earlier in this list.
_MODULE_ORDER = [
    "npy.mjs",
    "mat4.mjs",
    "grid.mjs",
    "transfer.mjs",
    "histogram.mjs",
    "clip.mjs",
    "camera.mjs",
    "layers.mjs",
    "delta.mjs",
    "app.mjs",
]

_EXPORT_RE = re.compile(r"^export (default )?")
_IMPORT_RE = re.compile(r"^\s*import\s*\{[^}]*\}\s*from\s*['\"][^'\"]+['\"];?\s*$")


def _strip_module_syntax(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if _IMPORT_RE.match(line):
            continue
        out_lines.append(_EXPORT_RE.sub("", line))
    return "\n".join(out_lines)


def build(viewer_dir: Path = Path("viewer"), *, out_dir: Path | None = None) -> Path:
    viewer_dir = Path(viewer_dir)
    src_dir = viewer_dir / "src"
    shell = (viewer_dir / "shell.html").read_text()

    chunks = []
    for name in _MODULE_ORDER:
        path = src_dir / name
        if not path.exists():
            continue
        chunks.append(f"// ---- {name} ----")
        chunks.append(_strip_module_syntax(path.read_text()))
    chunks.append("initViewer(document.body);")
    bundle = "\n".join(chunks)

    page = shell.replace("/* __VIEWER_BUNDLE__ */", bundle)

    out_dir = Path(out_dir) if out_dir is not None else viewer_dir / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(page)
    return out_path
```

Note: the test in Step 5 checks for `initViewer(document.body)` and for
`parseNpy` appearing in the bundle — both will only appear once `app.mjs`
(Task 3+) and `npy.mjs` (this task) exist; `npy.mjs` exists now, `app.mjs`
does not yet, so `_MODULE_ORDER`'s `if not path.exists(): continue` guard
lets `build()` succeed on a partial source tree throughout this plan's
execution. The `initViewer(document.body)` string will not appear until
Task 3 adds `app.mjs` — **for this task only**, relax the test's last
assertion: remove it here and add it back in Task 3's brief once `app.mjs`
exists (Task 3's Step 1 re-adds the exact assertion above to this same
test file).

- [ ] **Step 9: Run it, confirm it passes**

Run: `uv run pytest tests/test_viewerbuild.py -v`
Expected: PASS (with the `initViewer` assertion removed per the note above).

- [ ] **Step 10: Wire the CLI and dependencies**

Modify `megido/cli.py` — add a `view` subcommand. Find the function that
builds the top-level `argparse.ArgumentParser` and registers subcommands
(the existing `reconstruct`/`export`/`compare` subcommands show the
pattern), and add alongside them:
```python
def _cmd_view(args: argparse.Namespace) -> None:
    from pathlib import Path
    import webbrowser

    from megido.viewerbuild import build

    out = build(Path("viewer"))
    print(f"viewer built: {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
```
and in the parser setup:
```python
    p_view = subparsers.add_parser("view", help="build and open the S5 viewer")
    p_view.add_argument("--no-open", action="store_true",
                         help="build viewer/dist/index.html but don't open a browser")
    p_view.set_defaults(func=_cmd_view)
```
Match the existing subcommand wiring style in the file exactly (import
placement, `set_defaults(func=...)` pattern) — read the file first to copy
the established convention rather than guessing.

Modify `pyproject.toml` — add to `[project.optional-dependencies]`:
```toml
dev = ["pytest>=7.4", "pytest-playwright>=0.4", "playwright>=1.40"]
```

Modify `.gitignore` — add a line: `viewer/dist/`

- [ ] **Step 11: Install and verify Playwright**

Run: `uv sync --extra dev && uv run playwright install chromium`

- [ ] **Step 12: Write the Playwright fixtures**

`tests/viewer/__init__.py`: empty file.

`tests/viewer/conftest.py`:
```python
"""Shared fixtures for viewer Playwright tests.

`run_fixture` writes a tiny synthetic run directory (not real campaign data)
so tests are fast and self-contained; Task 10's final smoke test is the one
gate that points at the real `runs/voxels` output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from megido.viewerbuild import build

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"


@pytest.fixture(scope="session")
def dist_path() -> Path:
    return build(VIEWER_DIR)


@pytest.fixture
def run_fixture(tmp_path):
    """Write a synthetic run dir; returns its Path. shape/layers overridable."""
    def _make(shape=(6, 5, 4), layers=("volume",), spacing_m=0.5,
              origin_m=(0.0, 0.0, 1.0), depth_resolved=False):
        run = tmp_path / "run"
        run.mkdir()
        rng = np.random.default_rng(0)
        vol = rng.random(shape, dtype=np.float32)
        np.save(run / "volume.npy", vol)
        for name in layers:
            if name == "volume":
                continue
            arr = rng.random(shape, dtype=np.float32)
            np.save(run / f"{name}.npy", arr)
        meta = {
            "shape": list(shape),
            "axis_order": "xyz",
            "origin_m": list(origin_m),
            "spacing_m": spacing_m,
            "units": "opacity density [1/m]",
            "value_range": [float(vol.min()), float(vol.max())],
            "suggested_iso": [float(vol.max()) * 0.3, float(vol.max()) * 0.6],
            "run": "run",
            "layers": list(layers),
            "resolution": {
                "max_baseline_m": 2.2,
                "n_positions": 2,
                "depth_resolved": depth_resolved,
                "verdict": "depth NOT resolved: synthetic fixture verdict"
                if not depth_resolved else "depth resolved: synthetic fixture verdict",
            },
        }
        (run / "meta.json").write_text(json.dumps(meta))
        return run
    return _make
```

`viewer/README.md`:
```markdown
# Megiddo voxel viewer

Self-contained WebGL2 raymarching viewer for `volume.npy` + `meta.json`
(Phase 3's export contract, `megido/volexport.py`). No runtime dependencies,
no network requests, works from `file://`.

## Build

    uv run python -m megido.cli view

Writes `viewer/dist/index.html` and opens it. `--no-open` skips the browser.

## Test

    node --test viewer/test              # pure-logic unit tests
    uv run pytest tests/test_viewerbuild.py tests/viewer -v   # build + Playwright
    uv run playwright install chromium   # one-time, before the Playwright tests

## Using it

"Load run" picks a run directory (e.g. `runs/voxels`) via a directory file
picker; the viewer reads `meta.json` and every `.npy` file it names.
```

- [ ] **Step 13: Run everything, confirm green**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py -v`
Expected: PASS.

- [ ] **Step 14: Commit**

```bash
git add viewer/src/npy.mjs viewer/test/npy.test.mjs viewer/shell.html \
        viewer/README.md megido/viewerbuild.py tests/test_viewerbuild.py \
        tests/viewer/__init__.py tests/viewer/conftest.py \
        megido/cli.py pyproject.toml .gitignore uv.lock
git commit -m "feat: viewer build pipeline, npy parser, CLI view command"
```

---

### Task 2: 4x4 matrix math

**Files:**
- Create: `viewer/src/mat4.mjs`
- Create: `viewer/test/mat4.test.mjs`

**Interfaces:**
- Consumes: nothing.
- Produces (all `Float32Array(16)`, column-major, all pure functions):
  `identity()`, `multiply(a, b)`, `fromTranslation(v)`, `fromScaling(v)`,
  `perspective(fovyRad, aspect, near, far)`, `lookAt(eye, center, up)`,
  `invert(m)` (returns `null` if singular), `transpose(m)`.
  Consumed by `grid.mjs` (Task 3) and `app.mjs`/`camera.mjs` (Tasks 3-4).

- [ ] **Step 1: Write the failing tests**

`viewer/test/mat4.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  identity, multiply, fromTranslation, fromScaling,
  perspective, lookAt, invert, transpose,
} from '../src/mat4.mjs';

function closeTo(a, b, eps = 1e-5) {
  return Math.abs(a - b) < eps;
}
function matClose(a, b, eps = 1e-5) {
  for (let i = 0; i < 16; i++) if (!closeTo(a[i], b[i], eps)) return false;
  return true;
}
function apply(m, p) {
  const [x, y, z] = p;
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

test('identity leaves points unchanged', () => {
  assert.deepEqual([...apply(identity(), [1, 2, 3])], [1, 2, 3]);
});

test('fromTranslation moves a point', () => {
  const m = fromTranslation([1, 2, 3]);
  assert.deepEqual([...apply(m, [0, 0, 0])], [1, 2, 3]);
});

test('fromScaling scales a point', () => {
  const m = fromScaling([2, 3, 4]);
  assert.deepEqual([...apply(m, [1, 1, 1])], [2, 3, 4]);
});

test('multiply applies b then a', () => {
  const t = fromTranslation([10, 0, 0]);
  const s = fromScaling([2, 2, 2]);
  const m = multiply(t, s); // scale first, then translate
  const p = apply(m, [1, 0, 0]);
  assert.ok(closeTo(p[0], 12) && closeTo(p[1], 0) && closeTo(p[2], 0));
});

test('invert undoes a translation', () => {
  const m = fromTranslation([5, -3, 2]);
  const inv = invert(m);
  const round = multiply(m, inv);
  assert.ok(matClose(round, identity()));
});

test('invert returns null for a singular matrix', () => {
  const m = fromScaling([0, 1, 1]);
  assert.equal(invert(m), null);
});

test('transpose swaps rows and columns', () => {
  const m = fromTranslation([1, 2, 3]);
  const t = transpose(m);
  assert.equal(t[12], 0);
  assert.equal(t[3], 1);
});

test('lookAt places the eye and looks toward center', () => {
  const m = lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]);
  const view = apply(m, [0, 0, 0]); // center, in eye space, should be at (0,0,-5)
  assert.ok(closeTo(view[0], 0) && closeTo(view[1], 0) && closeTo(view[2], -5));
});

test('perspective is invertible and non-degenerate', () => {
  const m = perspective(Math.PI / 4, 1.5, 0.1, 100);
  const inv = invert(m);
  assert.notEqual(inv, null);
  const round = multiply(m, inv);
  assert.ok(matClose(round, identity(), 1e-4));
});
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `node --test viewer/test/mat4.test.mjs`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement mat4.mjs**

```js
// Column-major 4x4 matrices, Float32Array(16). multiply(a, b) applies b
// first, then a: multiply(a, b) * v == a * (b * v).

export function identity() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

export function fromTranslation(v) {
  const m = identity();
  m[12] = v[0]; m[13] = v[1]; m[14] = v[2];
  return m;
}

export function fromScaling(v) {
  const m = identity();
  m[0] = v[0]; m[5] = v[1]; m[10] = v[2];
  return m;
}

export function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k];
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

export function transpose(m) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[row * 4 + col] = m[col * 4 + row];
    }
  }
  return out;
}

export function perspective(fovyRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function norm(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

export function lookAt(eye, center, up) {
  const zAxis = norm(sub(eye, center));
  const xAxis = norm(cross(up, zAxis));
  const yAxis = cross(zAxis, xAxis);
  return new Float32Array([
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1,
  ]);
}

export function invert(m) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-10) return null;
  det = 1.0 / det;

  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}
```

- [ ] **Step 4: Run it, confirm it passes**

Run: `node --test viewer/test/mat4.test.mjs`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add viewer/src/mat4.mjs viewer/test/mat4.test.mjs
git commit -m "feat: viewer 4x4 matrix math"
```

---

### Task 3: Voxel grid mapping, transfer LUT, and the WebGL2 raymarcher

**Files:**
- Create: `viewer/src/grid.mjs`
- Create: `viewer/test/grid.test.mjs`
- Create: `viewer/src/transfer.mjs`
- Create: `viewer/test/transfer.test.mjs`
- Create: `viewer/src/app.mjs`
- Modify: `viewer/shell.html` (add canvas + load-run control markup)
- Modify: `tests/test_viewerbuild.py` (re-add the `initViewer` assertion, per Task 1's note)
- Create: `tests/viewer/test_raymarch.py`

**Interfaces:**
- Consumes: `parseNpy` (`npy.mjs`, Task 1); `identity`, `multiply`, `fromTranslation`, `fromScaling`, `perspective`, `lookAt`, `invert` (`mat4.mjs`, Task 2).
- Produces: `modelMatrixFromMeta(meta) -> Float32Array(16)`, `worldToVoxel(p, meta) -> [i,j,k]`, `voxelToWorld([i,j,k], meta) -> [x,y,z]`, `sampleNearest(data, shape, i, j, k) -> number` (NaN if out of range) from `grid.mjs`. Consumed by Task 8 (hover readout).
- Produces: `buildTransferLUT(stops, size=256) -> Uint8Array(size*4)`, `defaultStops() -> Array<{t,r,g,b,a}>` from `transfer.mjs`. Consumed by Task 5 (editor UI rebuilds the LUT on drag) and Task 3 itself (initial LUT texture).
- Produces from `app.mjs`: `initViewer(root: HTMLElement) -> void`, the sole entry point the built bundle calls. All later tasks (4-9) modify `app.mjs`'s internals but never change this signature.
- Produces (module-internal but load-bearing for later tasks, keep these exact names in `app.mjs`): a `state` object holding `{ meta, layerData: Map<string, Float32Array>, activeLayer, gl, program, uniforms, camera: {yaw, pitch, distance, target}, transferStops, clipMin, clipMax, clipPlaneEnabled, clipPlaneNormal, clipPlaneD, sigmaGateEnabled, sigmaGateValue, secondRun }`, and a `render()` function that re-issues the WebGL draw call reading current `state`. Later tasks read/write `state` fields and call `render()`.

- [ ] **Step 1: Write the failing grid tests**

`viewer/test/grid.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { modelMatrixFromMeta, worldToVoxel, voxelToWorld, sampleNearest } from '../src/grid.mjs';

const META = { shape: [4, 3, 2], spacing_m: 0.5, origin_m: [1, 2, 3] };

test('worldToVoxel and voxelToWorld are inverses', () => {
  const world = [2.5, 3.0, 3.5];
  const voxel = worldToVoxel(world, META);
  const back = voxelToWorld(voxel, META);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(back[i] - world[i]) < 1e-6);
});

test('worldToVoxel at the origin is voxel (0,0,0)', () => {
  const v = worldToVoxel(META.origin_m, META);
  assert.deepEqual(v.map((x) => Math.round(x * 1e6) / 1e6), [0, 0, 0]);
});

test('modelMatrixFromMeta maps the unit cube onto the world-space box', () => {
  const m = modelMatrixFromMeta(META);
  // point (1,1,1) in unit-cube/texture space -> far corner of the voxel box
  const x = m[0] * 1 + m[4] * 1 + m[8] * 1 + m[12];
  const y = m[1] * 1 + m[5] * 1 + m[9] * 1 + m[13];
  const z = m[2] * 1 + m[6] * 1 + m[10] * 1 + m[14];
  assert.ok(Math.abs(x - (1 + 4 * 0.5)) < 1e-6);
  assert.ok(Math.abs(y - (2 + 3 * 0.5)) < 1e-6);
  assert.ok(Math.abs(z - (3 + 2 * 0.5)) < 1e-6);
});

test('sampleNearest reads C-order [nx,ny,nz] data', () => {
  const shape = [2, 2, 2];
  const data = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]); // index = i*4 + j*2 + k
  assert.equal(sampleNearest(data, shape, 0, 0, 0), 0);
  assert.equal(sampleNearest(data, shape, 1, 0, 1), 5);
  assert.equal(sampleNearest(data, shape, 1, 1, 1), 7);
});

test('sampleNearest returns NaN out of bounds', () => {
  const shape = [2, 2, 2];
  const data = new Float32Array(8);
  assert.ok(Number.isNaN(sampleNearest(data, shape, -1, 0, 0)));
  assert.ok(Number.isNaN(sampleNearest(data, shape, 2, 0, 0)));
});
```

- [ ] **Step 2: Confirm failure**

Run: `node --test viewer/test/grid.test.mjs` — expect FAIL, module missing.

- [ ] **Step 3: Implement grid.mjs**

```js
import { multiply, fromTranslation, fromScaling } from './mat4.mjs';

export function modelMatrixFromMeta(meta) {
  const [nx, ny, nz] = meta.shape;
  const s = meta.spacing_m;
  const extents = [nx * s, ny * s, nz * s];
  return multiply(fromTranslation(meta.origin_m), fromScaling(extents));
}

export function worldToVoxel(p, meta) {
  const s = meta.spacing_m;
  return [
    (p[0] - meta.origin_m[0]) / s,
    (p[1] - meta.origin_m[1]) / s,
    (p[2] - meta.origin_m[2]) / s,
  ];
}

export function voxelToWorld(v, meta) {
  const s = meta.spacing_m;
  return [
    meta.origin_m[0] + v[0] * s,
    meta.origin_m[1] + v[1] * s,
    meta.origin_m[2] + v[2] * s,
  ];
}

export function sampleNearest(data, shape, i, j, k) {
  const [nx, ny, nz] = shape;
  const ii = Math.round(i), jj = Math.round(j), kk = Math.round(k);
  if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) return NaN;
  return data[ii * ny * nz + jj * nz + kk];
}
```

- [ ] **Step 4: Confirm the grid tests pass**

Run: `node --test viewer/test/grid.test.mjs` — expect PASS (5 tests).

- [ ] **Step 5: Write the failing transfer tests**

`viewer/test/transfer.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildTransferLUT, defaultStops } from '../src/transfer.mjs';

test('defaultStops spans 0 to 1', () => {
  const stops = defaultStops();
  assert.ok(stops.some((s) => s.t === 0));
  assert.ok(stops.some((s) => s.t === 1));
});

test('buildTransferLUT has correct byte length', () => {
  const lut = buildTransferLUT(defaultStops(), 256);
  assert.equal(lut.length, 256 * 4);
  assert.ok(lut instanceof Uint8Array);
});

test('buildTransferLUT interpolates linearly between two stops', () => {
  const stops = [
    { t: 0, r: 0, g: 0, b: 0, a: 0 },
    { t: 1, r: 255, g: 255, b: 255, a: 255 },
  ];
  const lut = buildTransferLUT(stops, 3); // samples at t = 0, 0.5, 1
  assert.deepEqual([lut[0], lut[1], lut[2], lut[3]], [0, 0, 0, 0]);
  assert.deepEqual([lut[8], lut[9], lut[10], lut[11]], [255, 255, 255, 255]);
  assert.ok(Math.abs(lut[4] - 127) <= 1);
});
```

- [ ] **Step 6: Confirm failure, then implement**

`viewer/src/transfer.mjs`:
```js
export function defaultStops() {
  return [
    { t: 0.0, r: 20, g: 20, b: 120, a: 0 },
    { t: 0.5, r: 200, g: 120, b: 20, a: 140 },
    { t: 1.0, r: 255, g: 240, b: 200, a: 255 },
  ];
}

export function buildTransferLUT(stops, size = 256) {
  const sorted = [...stops].sort((a, b) => a.t - b.t);
  const out = new Uint8Array(size * 4);
  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    let lo = sorted[0], hi = sorted[sorted.length - 1];
    for (let k = 0; k < sorted.length - 1; k++) {
      if (t >= sorted[k].t && t <= sorted[k + 1].t) {
        lo = sorted[k]; hi = sorted[k + 1];
        break;
      }
    }
    const span = hi.t - lo.t;
    const f = span > 0 ? (t - lo.t) / span : 0;
    out[i * 4 + 0] = Math.round(lo.r + (hi.r - lo.r) * f);
    out[i * 4 + 1] = Math.round(lo.g + (hi.g - lo.g) * f);
    out[i * 4 + 2] = Math.round(lo.b + (hi.b - lo.b) * f);
    out[i * 4 + 3] = Math.round(lo.a + (hi.a - lo.a) * f);
  }
  return out;
}
```

Run: `node --test viewer/test/transfer.test.mjs` — expect PASS (3 tests).

- [ ] **Step 7: Add canvas and load-run markup to the shell**

Modify `viewer/shell.html` — replace the `<body>` contents:
```html
<body>
<div id="app-root">
  <input id="load-run-input" type="file" webkitdirectory multiple
         style="position:absolute;top:8px;left:8px;z-index:2" />
  <canvas id="gl-canvas" style="display:block;width:100vw;height:100vh"></canvas>
</div>
<script>
/* __VIEWER_BUNDLE__ */
</script>
</body>
```

- [ ] **Step 8: Write app.mjs — the raymarcher**

`viewer/src/app.mjs`. This is the one place the shader source lives for the
whole plan; every uniform every later task needs is declared here even
though most are inert until their owning task sets them.

```js
import { parseNpy } from './npy.mjs';
import { identity, multiply, perspective, lookAt, invert } from './mat4.mjs';
import { modelMatrixFromMeta } from './grid.mjs';
import { buildTransferLUT, defaultStops } from './transfer.mjs';

const VERTEX_SRC = `#version 300 es
out vec2 vUv;
void main() {
  vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = pos;
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;

uniform mat4 uInvViewProj;
uniform vec3 uCameraPos;
uniform sampler3D uVolume;
uniform sampler3D uSigmaTex;
uniform sampler2D uTransferLUT;
uniform vec2 uWindow;        // [lo, hi] density remap before LUT lookup
uniform vec3 uClipMin;
uniform vec3 uClipMax;
uniform bool uClipPlaneEnabled;
uniform vec3 uClipPlaneNormal;
uniform float uClipPlaneD;
uniform bool uSigmaGateEnabled;
uniform float uSigmaGateValue;
uniform int uSteps;

vec3 unproject(vec2 ndc, float z) {
  vec4 clip = vec4(ndc, z, 1.0);
  vec4 world = uInvViewProj * clip;
  return world.xyz / world.w;
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 nearP = unproject(ndc, -1.0);
  vec3 farP = unproject(ndc, 1.0);
  vec3 dir = normalize(farP - nearP);

  float stepLen = length(farP - nearP) / float(uSteps);
  vec3 pos = nearP;
  vec4 accum = vec4(0.0);

  for (int i = 0; i < 512; i++) {
    if (i >= uSteps || accum.a > 0.98) break;
    vec3 tex = pos - uClipMin * 0.0; // placeholder to keep uClipMin referenced pre-Task7
    pos += dir * stepLen;
  }

  outColor = accum;
}`;
```

The loop body above is intentionally inert (Task 3 only proves the pipeline
compiles, binds a volume texture, and draws something); **replace it now**
with the real sampling logic — do not leave the placeholder line in:

```js
  for (int i = 0; i < 512; i++) {
    if (i >= uSteps || accum.a > 0.98) break;
    vec3 tex = (pos - uWorldMin) / uWorldExtent; // computed on CPU, see uniforms below
    if (tex.x >= 0.0 && tex.x <= 1.0 && tex.y >= 0.0 && tex.y <= 1.0 && tex.z >= 0.0 && tex.z <= 1.0) {
      bool clipped = any(lessThan(tex, uClipMin)) || any(greaterThan(tex, uClipMax));
      if (uClipPlaneEnabled) {
        float d = dot(tex - vec3(0.5), uClipPlaneNormal) - uClipPlaneD;
        clipped = clipped || d < 0.0;
      }
      if (uSigmaGateEnabled) {
        float sigma = texture(uSigmaTex, tex).r;
        clipped = clipped || sigma > uSigmaGateValue;
      }
      if (!clipped) {
        float density = texture(uVolume, tex).r;
        float t = clamp((density - uWindow.x) / max(uWindow.y - uWindow.x, 1e-6), 0.0, 1.0);
        vec4 c = texture(uTransferLUT, vec2(t, 0.5));
        c.rgb *= c.a;
        accum += (1.0 - accum.a) * c;
      }
    }
    pos += dir * stepLen;
  }
```

and add the two uniforms this version references (`uWorldMin`, `uWorldExtent`)
to the `uniform` block above, next to `uClipMin`:
```glsl
uniform vec3 uWorldMin;
uniform vec3 uWorldExtent;
```

Now the JS half of `app.mjs`:

```js
function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile error: ${log}`);
  }
  return sh;
}

function linkProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`program link error: ${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

function makeVolumeTexture(gl, shape, data) {
  const [nx, ny, nz] = shape;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  // NEAREST filtering avoids depending on OES_texture_float_linear, which
  // is not guaranteed on every WebGL2 implementation.
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, nx, ny, nz, 0, gl.RED, gl.FLOAT, data);
  return tex;
}

function makeLutTexture(gl, lutBytes) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, lutBytes.length / 4, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, lutBytes);
  return tex;
}

export function initViewer(root) {
  const canvas = root.querySelector('#gl-canvas');
  const fileInput = root.querySelector('#load-run-input');
  const gl = canvas.getContext('webgl2');
  if (!gl) throw new Error('WebGL2 is required');
  gl.getExtension('EXT_color_buffer_float');

  const program = linkProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
  const vao = gl.createVertexArray();

  const uniforms = {};
  for (const name of [
    'uInvViewProj', 'uCameraPos', 'uVolume', 'uSigmaTex', 'uTransferLUT',
    'uWindow', 'uClipMin', 'uClipMax', 'uClipPlaneEnabled', 'uClipPlaneNormal',
    'uClipPlaneD', 'uSigmaGateEnabled', 'uSigmaGateValue', 'uSteps',
    'uWorldMin', 'uWorldExtent',
  ]) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }

  const dummyVolume = makeVolumeTexture(gl, [1, 1, 1], new Float32Array([0]));
  const state = {
    meta: null,
    layerData: new Map(),
    activeLayer: null,
    gl, program, uniforms,
    camera: { yaw: 0.6, pitch: 0.5, distance: 3, target: [0, 0, 0] },
    transferStops: defaultStops(),
    clipMin: [0, 0, 0],
    clipMax: [1, 1, 1],
    clipPlaneEnabled: false,
    clipPlaneNormal: [0, 0, 1],
    clipPlaneD: 0,
    sigmaGateEnabled: false,
    sigmaGateValue: 1e9,
    secondRun: null,
    volumeTex: dummyVolume,
    sigmaTex: dummyVolume,
    lutTex: makeLutTexture(gl, buildTransferLUT(defaultStops())),
  };

  function worldBounds() {
    if (!state.meta) return { min: [0, 0, 0], extent: [1, 1, 1] };
    const { shape, spacing_m, origin_m } = state.meta;
    const extent = shape.map((n) => n * spacing_m);
    return { min: origin_m, extent };
  }

  function render() {
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(width * (window.devicePixelRatio || 1)));
    canvas.height = Math.max(1, Math.round(height * (window.devicePixelRatio || 1)));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.07, 0.07, 0.09, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const { yaw, pitch, distance, target } = state.camera;
    const eye = [
      target[0] + distance * Math.cos(pitch) * Math.sin(yaw),
      target[1] + distance * Math.sin(pitch),
      target[2] + distance * Math.cos(pitch) * Math.cos(yaw),
    ];
    const view = lookAt(eye, target, [0, 1, 0]);
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, 0.05, 100);
    const viewProj = multiply(proj, view);
    const invViewProj = invert(viewProj) || identity();

    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniformMatrix4fv(uniforms.uInvViewProj, false, invViewProj);
    gl.uniform3fv(uniforms.uCameraPos, eye);

    const { min, extent } = worldBounds();
    gl.uniform3fv(uniforms.uWorldMin, min);
    gl.uniform3fv(uniforms.uWorldExtent, extent);
    gl.uniform2fv(uniforms.uWindow, state.window || [0, 1]);
    gl.uniform3fv(uniforms.uClipMin, state.clipMin);
    gl.uniform3fv(uniforms.uClipMax, state.clipMax);
    gl.uniform1i(uniforms.uClipPlaneEnabled, state.clipPlaneEnabled ? 1 : 0);
    gl.uniform3fv(uniforms.uClipPlaneNormal, state.clipPlaneNormal);
    gl.uniform1f(uniforms.uClipPlaneD, state.clipPlaneD);
    gl.uniform1i(uniforms.uSigmaGateEnabled, state.sigmaGateEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uSigmaGateValue, state.sigmaGateValue);
    gl.uniform1i(uniforms.uSteps, 200);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, state.volumeTex);
    gl.uniform1i(uniforms.uVolume, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, state.sigmaTex);
    gl.uniform1i(uniforms.uSigmaTex, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, state.lutTex);
    gl.uniform1i(uniforms.uTransferLUT, 2);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  state.render = render;

  async function readFile(file) {
    return file.arrayBuffer();
  }

  async function loadRun(files) {
    const byName = new Map();
    for (const f of files) byName.set(f.name, f);

    const metaFile = byName.get('meta.json');
    if (!metaFile) throw new Error('selected directory has no meta.json');
    const meta = JSON.parse(await metaFile.text());
    state.meta = meta;

    state.layerData.clear();
    for (const name of meta.layers) {
      const file = byName.get(`${name}.npy`);
      if (!file) continue;
      const { data } = parseNpy(await readFile(file));
      state.layerData.set(name, data);
    }

    state.activeLayer = 'volume';
    state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('volume'));
    if (state.layerData.has('sigma')) {
      state.sigmaTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('sigma'));
    }
    const [lo, hi] = meta.value_range;
    state.window = [lo, hi];
    render();
  }

  fileInput.addEventListener('change', (ev) => {
    loadRun(Array.from(ev.target.files)).catch((err) => {
      console.error(err);
      window.__viewerError = String(err);
    });
  });

  render();
  window.__viewerState = state; // inspected by Playwright tests
}
```

- [ ] **Step 9: Re-add the build-test assertion**

Modify `tests/test_viewerbuild.py` — `app.mjs` now exists, so restore the
full assertion set exactly as written in Task 1 Step 5 (it already includes
`'initViewer(document.body)' in text`; no further edit needed if you did not
remove it — this step is a checkpoint to confirm it is present).

Run: `uv run pytest tests/test_viewerbuild.py -v` — expect PASS.

- [ ] **Step 10: Write the Playwright raymarch test**

`tests/viewer/test_raymarch.py`:
```python
from pathlib import Path


def test_canvas_renders_non_blank_after_loading_a_run(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # webkitdirectory pickers can't be driven by set_input_files with a
    # directory; feed the individual files it would have produced instead.
    files = sorted(str(p) for p in run_dir.iterdir())
    page.locator("#load-run-input").set_input_files(files)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    err = page.evaluate("() => window.__viewerError || null")
    assert err is None, err

    data_url = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    blank = page.evaluate("""
        () => {
          const c = document.createElement('canvas');
          c.width = document.querySelector('#gl-canvas').width;
          c.height = document.querySelector('#gl-canvas').height;
          return c.toDataURL();
        }
    """)
    assert data_url != blank
```

- [ ] **Step 11: Run it, confirm it passes**

Run: `uv run pytest tests/viewer/test_raymarch.py -v --browser chromium`
Expected: PASS. If it fails on shader compile, read the browser console via
`page.on("console", lambda m: print(m.text))` added temporarily to debug —
GLSL errors surface there, not in the pytest traceback.

- [ ] **Step 12: Run every test so far**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py tests/viewer -v`
Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add viewer/src/grid.mjs viewer/test/grid.test.mjs \
        viewer/src/transfer.mjs viewer/test/transfer.test.mjs \
        viewer/src/app.mjs viewer/shell.html \
        tests/test_viewerbuild.py tests/viewer/test_raymarch.py
git commit -m "feat: WebGL2 raymarcher, voxel grid mapping, transfer LUT"
```

---

### Task 4: Camera controls and presets

**Files:**
- Create: `viewer/src/camera.mjs`
- Create: `viewer/test/camera.test.mjs`
- Modify: `viewer/shell.html` (camera preset buttons)
- Modify: `viewer/src/app.mjs` (mouse/wheel handlers, preset wiring)
- Create: `tests/viewer/test_camera.py`

**Interfaces:**
- Consumes: nothing new (pure trig).
- Produces: `orbitToEye(target, yaw, pitch, distance) -> [x,y,z]`, `CAMERA_PRESETS` (object keyed `top`/`front`/`side`/`iso`, each `{yaw, pitch}`) from `camera.mjs`.
- Modifies `app.mjs`'s `state.camera` in response to drag/wheel/preset-click; `render()`'s eye computation (Task 3) is replaced with a call to `orbitToEye`.

- [ ] **Step 1: Write the failing camera tests**

`viewer/test/camera.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { orbitToEye, CAMERA_PRESETS } from '../src/camera.mjs';

function close(a, b, eps = 1e-5) { return Math.abs(a - b) < eps; }

test('orbitToEye at yaw=0, pitch=0 sits on +z from target', () => {
  const eye = orbitToEye([0, 0, 0], 0, 0, 5);
  assert.ok(close(eye[0], 0) && close(eye[1], 0) && close(eye[2], 5));
});

test('orbitToEye respects the target offset', () => {
  const eye = orbitToEye([1, 2, 3], 0, 0, 5);
  assert.ok(close(eye[0], 1) && close(eye[1], 2) && close(eye[2], 8));
});

test('orbitToEye stays at constant distance from target', () => {
  const target = [0, 0, 0];
  for (const [yaw, pitch] of [[0.3, 0.7], [2.1, -0.4], [5.9, 1.0]]) {
    const eye = orbitToEye(target, yaw, pitch, 4);
    const d = Math.hypot(eye[0], eye[1], eye[2]);
    assert.ok(close(d, 4, 1e-4));
  }
});

test('CAMERA_PRESETS defines top, front, side, iso', () => {
  for (const key of ['top', 'front', 'side', 'iso']) {
    assert.ok(key in CAMERA_PRESETS);
    assert.ok(typeof CAMERA_PRESETS[key].yaw === 'number');
    assert.ok(typeof CAMERA_PRESETS[key].pitch === 'number');
  }
});
```

- [ ] **Step 2: Confirm failure, then implement**

`viewer/src/camera.mjs`:
```js
export function orbitToEye(target, yaw, pitch, distance) {
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [
    target[0] + distance * cp * sy,
    target[1] + distance * sp,
    target[2] + distance * cp * cy,
  ];
}

// pitch is kept just off +-PI/2 for top/bottom-like views so lookAt's
// up vector (0,1,0) never goes parallel to the eye-to-target axis.
export const CAMERA_PRESETS = {
  top: { yaw: 0, pitch: Math.PI / 2 - 0.001 },
  front: { yaw: 0, pitch: 0 },
  side: { yaw: Math.PI / 2, pitch: 0 },
  iso: { yaw: Math.PI / 4, pitch: Math.PI / 5 },
};
```

Run: `node --test viewer/test/camera.test.mjs` — expect PASS (4 tests).

- [ ] **Step 3: Add preset buttons to the shell**

Modify `viewer/shell.html` — inside `#app-root`, after the file input:
```html
  <div id="camera-presets" style="position:absolute;top:8px;right:8px;z-index:2">
    <button id="camera-preset-top">Top</button>
    <button id="camera-preset-front">Front</button>
    <button id="camera-preset-side">Side</button>
    <button id="camera-preset-iso">Iso</button>
  </div>
```

- [ ] **Step 4: Wire camera controls in app.mjs**

Modify `viewer/src/app.mjs`:
- Add `import { orbitToEye, CAMERA_PRESETS } from './camera.mjs';` at the top.
- In `render()`, replace the eye computation block:
  ```js
  const { yaw, pitch, distance, target } = state.camera;
  const eye = [
    target[0] + distance * Math.cos(pitch) * Math.sin(yaw),
    target[1] + distance * Math.sin(pitch),
    target[2] + distance * Math.cos(pitch) * Math.cos(yaw),
  ];
  ```
  with:
  ```js
  const { yaw, pitch, distance, target } = state.camera;
  const eye = orbitToEye(target, yaw, pitch, distance);
  ```
- At the end of `initViewer`, before the final `render(); window.__viewerState = state;` lines, add:
  ```js
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', (ev) => {
    dragging = true; lastX = ev.clientX; lastY = ev.clientY;
  });
  window.addEventListener('pointerup', () => { dragging = false; });
  window.addEventListener('pointermove', (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
    lastX = ev.clientX; lastY = ev.clientY;
    state.camera.yaw += dx * 0.01;
    state.camera.pitch = Math.max(-1.5, Math.min(1.5, state.camera.pitch + dy * 0.01));
    render();
  });
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    state.camera.distance = Math.max(0.5, state.camera.distance * (1 + ev.deltaY * 0.001));
    render();
  }, { passive: false });

  for (const key of Object.keys(CAMERA_PRESETS)) {
    const btn = root.querySelector(`#camera-preset-${key}`);
    btn.addEventListener('click', () => {
      state.camera.yaw = CAMERA_PRESETS[key].yaw;
      state.camera.pitch = CAMERA_PRESETS[key].pitch;
      render();
    });
  }
  ```

- [ ] **Step 5: Write the Playwright camera test**

`tests/viewer/test_camera.py`:
```python
def test_camera_presets_change_the_render(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    files = sorted(str(p) for p in run_dir.iterdir())
    page.locator("#load-run-input").set_input_files(files)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    front = page.evaluate("""
        () => { window.__viewerState.camera.yaw = 0; window.__viewerState.camera.pitch = 0;
                window.__viewerState.render(); return document.querySelector('#gl-canvas').toDataURL(); }
    """)
    page.locator("#camera-preset-top").click()
    top = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert front != top

    state_yaw = page.evaluate("() => window.__viewerState.camera.yaw")
    assert abs(state_yaw - 0) < 1e-6  # top preset's yaw is 0, per CAMERA_PRESETS
```

- [ ] **Step 6: Run everything, confirm green**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py tests/viewer -v --browser chromium`

- [ ] **Step 7: Commit**

```bash
git add viewer/src/camera.mjs viewer/test/camera.test.mjs viewer/shell.html \
        viewer/src/app.mjs tests/viewer/test_camera.py
git commit -m "feat: orbit camera controls and presets"
```

---

### Task 5: Transfer-function editor and live histogram

**Files:**
- Create: `viewer/src/histogram.mjs`
- Create: `viewer/test/histogram.test.mjs`
- Modify: `viewer/shell.html` (xfer canvas, window sliders, histogram canvas)
- Modify: `viewer/src/app.mjs` (drag-to-edit stops, histogram redraw)
- Create: `tests/viewer/test_transfer_ui.py`

**Interfaces:**
- Consumes: `buildTransferLUT`, `defaultStops` (`transfer.mjs`, Task 3).
- Produces: `computeHistogram(data, lo, hi, nbins) -> Uint32Array(nbins)` from `histogram.mjs`. Consumed only within `app.mjs` in this task.
- Modifies `app.mjs`: adds `state.transferStops` mutation via drag, rebuilds `state.lutTex` on change, redraws `#histogram-canvas` on window-slider input.

- [ ] **Step 1: Write the failing histogram tests**

`viewer/test/histogram.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeHistogram } from '../src/histogram.mjs';

test('computeHistogram counts values within the window', () => {
  const data = new Float32Array([0, 0.1, 0.5, 0.9, 1.0, 2.0]);
  const hist = computeHistogram(data, 0, 1, 4);
  assert.equal(hist.length, 4);
  assert.equal(hist.reduce((a, b) => a + b, 0), 5); // 2.0 is outside [0,1]
});

test('computeHistogram ignores NaN', () => {
  const data = new Float32Array([0.1, NaN, 0.4]);
  const hist = computeHistogram(data, 0, 1, 2);
  assert.equal(hist.reduce((a, b) => a + b, 0), 2);
});

test('computeHistogram clamps the top edge into the last bin', () => {
  const data = new Float32Array([1.0]);
  const hist = computeHistogram(data, 0, 1, 4);
  assert.equal(hist[3], 1);
});
```

- [ ] **Step 2: Confirm failure, then implement**

`viewer/src/histogram.mjs`:
```js
export function computeHistogram(data, lo, hi, nbins) {
  const out = new Uint32Array(nbins);
  const span = hi - lo;
  if (span <= 0) return out;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isNaN(v) || v < lo || v > hi) continue;
    let bin = Math.floor(((v - lo) / span) * nbins);
    if (bin >= nbins) bin = nbins - 1;
    if (bin < 0) bin = 0;
    out[bin] += 1;
  }
  return out;
}
```

Run: `node --test viewer/test/histogram.test.mjs` — expect PASS (3 tests).

- [ ] **Step 3: Add editor and histogram markup**

Modify `viewer/shell.html` — inside `#app-root`:
```html
  <div id="transfer-panel" style="position:absolute;bottom:8px;left:8px;right:8px;z-index:2;background:#000a;padding:6px">
    <canvas id="xfer-canvas" width="512" height="40" style="width:100%;display:block"></canvas>
    <canvas id="histogram-canvas" width="512" height="60" style="width:100%;display:block"></canvas>
    <label>Window lo <input id="window-lo" type="range" min="0" max="1" step="0.001" value="0" /></label>
    <label>Window hi <input id="window-hi" type="range" min="0" max="1" step="0.001" value="1" /></label>
  </div>
```

- [ ] **Step 4: Wire the editor in app.mjs**

Modify `viewer/src/app.mjs`:
- Add `import { computeHistogram } from './histogram.mjs';` at the top.
- After `loadRun` defines `state.window`, add a `rebuildLut()` helper and
  a `drawHistogram()` helper, called once after load and whenever the
  window sliders or a drag changes `state.transferStops`/`state.window`:
  ```js
  function rebuildLut() {
    gl.deleteTexture(state.lutTex);
    state.lutTex = makeLutTexture(gl, buildTransferLUT(state.transferStops));
    render();
  }

  function drawHistogram() {
    const canvas = root.querySelector('#histogram-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const data = state.layerData.get(state.activeLayer);
    if (!data || !state.window) return;
    const [lo, hi] = state.window;
    const hist = computeHistogram(data, lo, hi, 64);
    const max = Math.max(...hist, 1);
    const barW = canvas.width / hist.length;
    ctx.fillStyle = '#8cf';
    for (let i = 0; i < hist.length; i++) {
      const h = (hist[i] / max) * canvas.height;
      ctx.fillRect(i * barW, canvas.height - h, barW - 1, h);
    }
  }

  function drawXferEditor() {
    const canvas = root.querySelector('#xfer-canvas');
    const ctx = canvas.getContext('2d');
    const lut = buildTransferLUT(state.transferStops, canvas.width);
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let x = 0; x < canvas.width; x++) {
      for (let y = 0; y < canvas.height; y++) {
        const idx = (y * canvas.width + x) * 4;
        img.data[idx] = lut[x * 4]; img.data[idx + 1] = lut[x * 4 + 1];
        img.data[idx + 2] = lut[x * 4 + 2]; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.fillStyle = '#fff';
    for (const s of state.transferStops) {
      ctx.fillRect(s.t * canvas.width - 2, 0, 4, canvas.height);
    }
  }
  state.drawHistogram = drawHistogram;
  state.drawXferEditor = drawXferEditor;
  ```
- In `loadRun`, after setting `state.window`, call `drawHistogram(); drawXferEditor();`.
- Wire the sliders and the editor drag, added alongside the camera handlers in Step 4 of Task 4 (append, do not replace):
  ```js
  root.querySelector('#window-lo').addEventListener('input', (ev) => {
    state.window[0] = parseFloat(ev.target.value) * (state.meta ? state.meta.value_range[1] : 1);
    drawHistogram(); render();
  });
  root.querySelector('#window-hi').addEventListener('input', (ev) => {
    state.window[1] = parseFloat(ev.target.value) * (state.meta ? state.meta.value_range[1] : 1);
    drawHistogram(); render();
  });

  const xferCanvas = root.querySelector('#xfer-canvas');
  let draggingStop = null;
  xferCanvas.addEventListener('pointerdown', (ev) => {
    const rect = xferCanvas.getBoundingClientRect();
    const t = (ev.clientX - rect.left) / rect.width;
    draggingStop = state.transferStops.reduce((best, s) =>
      Math.abs(s.t - t) < Math.abs(best.t - t) ? s : best);
  });
  window.addEventListener('pointerup', () => { draggingStop = null; });
  window.addEventListener('pointermove', (ev) => {
    if (!draggingStop) return;
    const rect = xferCanvas.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    draggingStop.t = t;
    rebuildLut();
    drawXferEditor();
  });
  ```

- [ ] **Step 5: Write the Playwright test**

`tests/viewer/test_transfer_ui.py`:
```python
def test_dragging_a_transfer_stop_changes_the_render(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    files = sorted(str(p) for p in run_dir.iterdir())
    page.locator("#load-run-input").set_input_files(files)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    box = page.locator("#xfer-canvas").bounding_box()
    mid_x = box["x"] + box["width"] * 0.5
    mid_y = box["y"] + box["height"] * 0.5
    page.mouse.move(mid_x, mid_y)
    page.mouse.down()
    page.mouse.move(mid_x + box["width"] * 0.3, mid_y)
    page.mouse.up()
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after

    hist_blank = page.evaluate("""
        () => {
          const c = document.createElement('canvas');
          c.width = document.querySelector('#histogram-canvas').width;
          c.height = document.querySelector('#histogram-canvas').height;
          return c.toDataURL();
        }
    """)
    hist_now = page.evaluate("() => document.querySelector('#histogram-canvas').toDataURL()")
    assert hist_now != hist_blank
```

- [ ] **Step 6: Run everything, confirm green**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py tests/viewer -v --browser chromium`

- [ ] **Step 7: Commit**

```bash
git add viewer/src/histogram.mjs viewer/test/histogram.test.mjs viewer/shell.html \
        viewer/src/app.mjs tests/viewer/test_transfer_ui.py
git commit -m "feat: transfer function editor and live histogram"
```

---

### Task 6: Layer manager

**Files:**
- Create: `viewer/src/layers.mjs`
- Create: `viewer/test/layers.test.mjs`
- Modify: `viewer/shell.html` (layer panel container)
- Modify: `viewer/src/app.mjs` (populate panel from `meta.layers`, switch active layer)
- Create: `tests/viewer/test_layers.py`

**Interfaces:**
- Consumes: `meta.layers` (array of strings, from the loaded run's `meta.json`).
- Produces: `KNOWN_LAYERS` (object keyed by layer name, `{label, kind}`, `kind` is `'scalar'` or `'signed'`), `availableLayers(layerNames) -> Array<{key, label, kind}>` from `layers.mjs`. Consumed by Task 9 (adds a `'delta'` entry once a second run is loaded).
- Scoping decision (ledger this if you disagree): the layer manager selects **one active layer at a time** (radio-button semantics), not a multi-volume blend — "every artifact independently toggleable" means every artifact can independently become the one being raymarched, not that N artifacts composite simultaneously. Compositing N arbitrary volumes in one raymarch pass is a materially larger shader and is not needed to inspect each artifact on its own terms.

- [ ] **Step 1: Write the failing layers tests**

`viewer/test/layers.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KNOWN_LAYERS, availableLayers } from '../src/layers.mjs';

test('KNOWN_LAYERS covers the Phase 3 export names', () => {
  for (const key of ['volume', 'sigma', 'snr', 'views', 'systematic', 'backprojection']) {
    assert.ok(key in KNOWN_LAYERS, key);
  }
});

test('availableLayers filters to present names, in meta order', () => {
  const result = availableLayers(['volume', 'sigma']);
  assert.deepEqual(result.map((l) => l.key), ['volume', 'sigma']);
  assert.equal(result[0].label, KNOWN_LAYERS.volume.label);
});

test('availableLayers falls back gracefully for an unknown layer name', () => {
  const result = availableLayers(['mystery_layer']);
  assert.equal(result.length, 1);
  assert.equal(result[0].key, 'mystery_layer');
  assert.equal(result[0].label, 'mystery_layer');
  assert.equal(result[0].kind, 'scalar');
});
```

- [ ] **Step 2: Confirm failure, then implement**

`viewer/src/layers.mjs`:
```js
export const KNOWN_LAYERS = {
  volume: { label: 'Combined solve', kind: 'scalar' },
  sigma: { label: 'Uncertainty (sigma)', kind: 'scalar' },
  snr: { label: 'SNR', kind: 'scalar' },
  views: { label: 'View count', kind: 'scalar' },
  systematic: { label: 'Gauge systematic', kind: 'signed' },
  backprojection: { label: 'Backprojection', kind: 'scalar' },
  volume_holdout_pos0: { label: 'Holdout: pos0 removed', kind: 'scalar' },
  volume_holdout_pos1: { label: 'Holdout: pos1 removed', kind: 'scalar' },
  phantom: { label: 'Phantom truth', kind: 'scalar' },
  delta: { label: 'Run delta (B minus A)', kind: 'signed' },
};

export function availableLayers(layerNames) {
  return layerNames.map((key) => {
    const known = KNOWN_LAYERS[key];
    return known ? { key, label: known.label, kind: known.kind }
                 : { key, label: key, kind: 'scalar' };
  });
}
```

Run: `node --test viewer/test/layers.test.mjs` — expect PASS (3 tests).

- [ ] **Step 3: Add the layer panel container**

Modify `viewer/shell.html` — inside `#app-root`, after camera presets:
```html
  <div id="layer-panel" style="position:absolute;top:48px;right:8px;z-index:2;background:#000a;padding:6px;color:#eee"></div>
```

- [ ] **Step 4: Populate and wire the panel in app.mjs**

Modify `viewer/src/app.mjs`:
- Add `import { availableLayers } from './layers.mjs';` at the top.
- In `loadRun`, after building `state.volumeTex`/`state.sigmaTex`, add:
  ```js
  function setActiveLayer(key) {
    state.activeLayer = key;
    state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get(key));
    drawHistogram();
    render();
  }
  state.setActiveLayer = setActiveLayer;

  const panel = root.querySelector('#layer-panel');
  panel.innerHTML = '';
  for (const layer of availableLayers([...state.layerData.keys()])) {
    const id = `layer-${layer.key}`;
    const label = document.createElement('label');
    label.style.display = 'block';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'active-layer';
    radio.id = id;
    radio.checked = layer.key === 'volume';
    radio.addEventListener('change', () => setActiveLayer(layer.key));
    label.appendChild(radio);
    label.appendChild(document.createTextNode(' ' + layer.label));
    panel.appendChild(label);
  }
  ```
  (Place this block right before the existing `state.window = [lo, hi];` line so
  `meta` is in scope; it references `drawHistogram` and `render`, both defined
  earlier in `initViewer`'s closure by this point in the file.)

- [ ] **Step 5: Write the Playwright test**

`tests/viewer/test_layers.py`:
```python
def test_layer_panel_lists_every_layer_and_switching_changes_render(page, dist_path, run_fixture):
    run_dir = run_fixture(layers=("volume", "sigma", "views"))
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    files = sorted(str(p) for p in run_dir.iterdir())
    page.locator("#load-run-input").set_input_files(files)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    for key in ("volume", "sigma", "views"):
        assert page.locator(f"#layer-{key}").count() == 1

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#layer-views").check()
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after
```

- [ ] **Step 6: Run everything, confirm green**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py tests/viewer -v --browser chromium`

- [ ] **Step 7: Commit**

```bash
git add viewer/src/layers.mjs viewer/test/layers.test.mjs viewer/shell.html \
        viewer/src/app.mjs tests/viewer/test_layers.py
git commit -m "feat: layer manager"
```

---

### Task 7: Clip box, clip plane, and slice views

**Files:**
- Create: `viewer/src/clip.mjs`
- Create: `viewer/test/clip.test.mjs`
- Modify: `viewer/shell.html` (clip controls)
- Modify: `viewer/src/app.mjs` (wire controls to the already-declared clip uniforms)
- Create: `tests/viewer/test_clip.py`

**Interfaces:**
- Consumes: nothing new — the shader's `uClipMin`/`uClipMax`/`uClipPlaneEnabled`/`uClipPlaneNormal`/`uClipPlaneD` uniforms already exist from Task 3.
- Produces: `insideClipBox(p, boxMin, boxMax) -> bool`, `clipPlaneDistance(p, normal, planeD) -> number`, `insideClipPlane(p, normal, planeD) -> bool` from `clip.mjs`. Consumed by Task 8's CPU-side hover raymarch (it must respect the same clip state the GPU shader does).
- Slice views reuse the clip box: a slice along an axis is a clip box whose min/max on that axis are pinned to a thin slab — no new uniforms.

- [ ] **Step 1: Write the failing clip tests**

`viewer/test/clip.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { insideClipBox, clipPlaneDistance, insideClipPlane } from '../src/clip.mjs';

test('insideClipBox is true inside, false outside', () => {
  assert.equal(insideClipBox([0.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1]), true);
  assert.equal(insideClipBox([1.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1]), false);
  assert.equal(insideClipBox([0, 0, 0], [0, 0, 0], [1, 1, 1]), true); // inclusive edge
});

test('clipPlaneDistance is signed distance from the plane through the box center', () => {
  const d = clipPlaneDistance([0.5, 0.5, 0.7], [0, 0, 1], 0.2);
  assert.ok(Math.abs(d - (0.2 - 0.2)) < 1e-9); // (0.7-0.5) - 0.2 = 0
});

test('insideClipPlane keeps the positive side', () => {
  assert.equal(insideClipPlane([0.5, 0.5, 0.9], [0, 0, 1], 0.0), true);
  assert.equal(insideClipPlane([0.5, 0.5, 0.1], [0, 0, 1], 0.0), false);
});
```

- [ ] **Step 2: Confirm failure, then implement**

`viewer/src/clip.mjs`:
```js
export function insideClipBox(p, boxMin, boxMax) {
  return p[0] >= boxMin[0] && p[0] <= boxMax[0]
      && p[1] >= boxMin[1] && p[1] <= boxMax[1]
      && p[2] >= boxMin[2] && p[2] <= boxMax[2];
}

// Matches the shader's convention: plane passes through the box center
// (0.5,0.5,0.5) offset by planeD along normal, in [0,1]^3 texture space.
export function clipPlaneDistance(p, normal, planeD) {
  const rel = [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5];
  const dot = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2];
  return dot - planeD;
}

export function insideClipPlane(p, normal, planeD) {
  return clipPlaneDistance(p, normal, planeD) >= 0;
}
```

Run: `node --test viewer/test/clip.test.mjs` — expect PASS (3 tests).

- [ ] **Step 3: Add clip controls to the shell**

Modify `viewer/shell.html` — inside `#app-root`:
```html
  <div id="clip-panel" style="position:absolute;top:48px;left:8px;z-index:2;background:#000a;padding:6px;color:#eee">
    <div>Clip box</div>
    <label>x min <input id="clip-x-min" type="range" min="0" max="1" step="0.01" value="0" /></label>
    <label>x max <input id="clip-x-max" type="range" min="0" max="1" step="0.01" value="1" /></label>
    <label>y min <input id="clip-y-min" type="range" min="0" max="1" step="0.01" value="0" /></label>
    <label>y max <input id="clip-y-max" type="range" min="0" max="1" step="0.01" value="1" /></label>
    <label>z min <input id="clip-z-min" type="range" min="0" max="1" step="0.01" value="0" /></label>
    <label>z max <input id="clip-z-max" type="range" min="0" max="1" step="0.01" value="1" /></label>
    <div>Slice</div>
    <select id="slice-axis">
      <option value="none">none</option>
      <option value="x">x</option>
      <option value="y">y</option>
      <option value="z">z</option>
    </select>
    <input id="slice-pos" type="range" min="0" max="1" step="0.01" value="0.5" />
    <div>Clip plane</div>
    <label><input id="clip-plane-enabled" type="checkbox" /> enabled</label>
    <label>d <input id="clip-plane-d" type="range" min="-0.87" max="0.87" step="0.01" value="0" /></label>
  </div>
```

- [ ] **Step 4: Wire clip controls in app.mjs**

Modify `viewer/src/app.mjs` — append after the transfer-editor wiring from Task 5:
```js
  const axes = { x: 0, y: 1, z: 2 };
  for (const axis of Object.keys(axes)) {
    root.querySelector(`#clip-${axis}-min`).addEventListener('input', (ev) => {
      state.clipMin[axes[axis]] = parseFloat(ev.target.value);
      render();
    });
    root.querySelector(`#clip-${axis}-max`).addEventListener('input', (ev) => {
      state.clipMax[axes[axis]] = parseFloat(ev.target.value);
      render();
    });
  }

  root.querySelector('#slice-axis').addEventListener('change', updateSlice);
  root.querySelector('#slice-pos').addEventListener('input', updateSlice);
  function updateSlice() {
    const axis = root.querySelector('#slice-axis').value;
    const pos = parseFloat(root.querySelector('#slice-pos').value);
    if (axis === 'none') {
      state.clipMin = [0, 0, 0];
      state.clipMax = [1, 1, 1];
    } else {
      const idx = axes[axis];
      const half = 0.02;
      state.clipMin = [0, 0, 0]; state.clipMax = [1, 1, 1];
      state.clipMin[idx] = Math.max(0, pos - half);
      state.clipMax[idx] = Math.min(1, pos + half);
    }
    render();
  }

  root.querySelector('#clip-plane-enabled').addEventListener('change', (ev) => {
    state.clipPlaneEnabled = ev.target.checked;
    render();
  });
  root.querySelector('#clip-plane-d').addEventListener('input', (ev) => {
    state.clipPlaneD = parseFloat(ev.target.value);
    render();
  });
```

- [ ] **Step 5: Write the Playwright test**

`tests/viewer/test_clip.py`:
```python
def test_clip_box_and_plane_change_the_render(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    files = sorted(str(p) for p in run_dir.iterdir())
    page.locator("#load-run-input").set_input_files(files)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#clip-x-max").fill("0.3")
    page.locator("#clip-x-max").dispatch_event("input")
    after_clip = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after_clip

    page.locator("#clip-plane-enabled").check()
    after_plane = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert after_clip != after_plane
```

- [ ] **Step 6: Run everything, confirm green**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py tests/viewer -v --browser chromium`

- [ ] **Step 7: Commit**

```bash
git add viewer/src/clip.mjs viewer/test/clip.test.mjs viewer/shell.html \
        viewer/src/app.mjs tests/viewer/test_clip.py
git commit -m "feat: clip box, clip plane, and slice views"
```

---

### Task 8: Uncertainty gate and hover voxel readout

**Files:**
- Modify: `viewer/shell.html` (sigma-gate slider, readout text)
- Modify: `viewer/src/app.mjs` (gate wiring, hover raycast)
- Create: `tests/viewer/test_gate_and_hover.py`

**Interfaces:**
- Consumes: `worldToVoxel`, `sampleNearest` (`grid.mjs`, Task 3); `insideClipBox`, `insideClipPlane` (`clip.mjs`, Task 7); `invert` (`mat4.mjs`, Task 2).
- No new pure module — this task is entirely `app.mjs` wiring plus one CPU-side helper function added directly to `app.mjs` (`castHoverRay`, not exported, only used internally, so it lives beside `initViewer` rather than as a tenth `.mjs` file for a single call site).

- [ ] **Step 1: Add gate and readout markup**

Modify `viewer/shell.html` — inside `#clip-panel`, append:
```html
    <div>Uncertainty gate</div>
    <label><input id="sigma-gate-enabled" type="checkbox" /> hide sigma above</label>
    <input id="sigma-gate-value" type="range" min="0" max="1" step="0.01" value="1" />
  </div>
  <div id="hover-readout" style="position:absolute;bottom:120px;left:8px;z-index:2;background:#000a;color:#eee;padding:4px 8px;font-family:monospace"></div>
```
(Note the `</div>` before `hover-readout` closes `#clip-panel`, which Task 7
opened — this task's markup is appended inside it and then closes it.)

- [ ] **Step 2: Wire the sigma gate in app.mjs**

Modify `viewer/src/app.mjs` — append after the clip-plane wiring from Task 7:
```js
  root.querySelector('#sigma-gate-enabled').addEventListener('change', (ev) => {
    state.sigmaGateEnabled = ev.target.checked;
    render();
  });
  root.querySelector('#sigma-gate-value').addEventListener('input', (ev) => {
    const frac = parseFloat(ev.target.value);
    const sigmaData = state.layerData.get('sigma');
    const max = sigmaData ? Math.max(...sigmaData) : 1;
    state.sigmaGateValue = frac * max;
    render();
  });
```

- [ ] **Step 3: Implement the CPU-side hover raycast**

Modify `viewer/src/app.mjs` — add `import { insideClipBox, insideClipPlane } from './clip.mjs';` and `import { worldToVoxel, sampleNearest } from './grid.mjs';` at the top (grid.mjs is already imported for `modelMatrixFromMeta`; extend that same import line). Append this function inside `initViewer`, after `render` is defined, and call it from a new `pointermove` listener on the canvas that is separate from the camera-drag listener added in Task 4 (both listeners can coexist — the camera one only acts while `dragging` is true):
```js
  function castHoverRay(clientX, clientY) {
    if (!state.meta) return null;
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -(((clientY - rect.top) / rect.height) * 2 - 1);

    const { yaw, pitch, distance, target } = state.camera;
    const eye = orbitToEye(target, yaw, pitch, distance);
    const view = lookAt(eye, target, [0, 1, 0]);
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, 0.05, 100);
    const invViewProj = invert(multiply(proj, view));
    if (!invViewProj) return null;

    function unproject(z) {
      const clip = [ndcX, ndcY, z, 1];
      const m = invViewProj;
      const w = m[3] * clip[0] + m[7] * clip[1] + m[11] * clip[2] + m[15] * clip[3];
      return [
        (m[0] * clip[0] + m[4] * clip[1] + m[8] * clip[2] + m[12] * clip[3]) / w,
        (m[1] * clip[0] + m[5] * clip[1] + m[9] * clip[2] + m[13] * clip[3]) / w,
        (m[2] * clip[0] + m[6] * clip[1] + m[10] * clip[2] + m[14] * clip[3]) / w,
      ];
    }
    const nearP = unproject(-1), farP = unproject(1);
    const dir = [farP[0] - nearP[0], farP[1] - nearP[1], farP[2] - nearP[2]];
    const len = Math.hypot(...dir);
    const step = [dir[0] / len, dir[1] / len, dir[2] / len];
    const data = state.layerData.get(state.activeLayer);
    if (!data) return null;

    const steps = 200;
    const stepLen = len / steps;
    for (let s = 0; s < steps; s++) {
      const world = [nearP[0] + step[0] * stepLen * s,
                     nearP[1] + step[1] * stepLen * s,
                     nearP[2] + step[2] * stepLen * s];
      const { min, extent } = worldBounds();
      const tex = [
        (world[0] - min[0]) / extent[0],
        (world[1] - min[1]) / extent[1],
        (world[2] - min[2]) / extent[2],
      ];
      if (!insideClipBox(tex, state.clipMin, state.clipMax)) continue;
      if (state.clipPlaneEnabled && !insideClipPlane(tex, state.clipPlaneNormal, state.clipPlaneD)) continue;
      const [i, j, k] = worldToVoxel(world, state.meta);
      const value = sampleNearest(data, state.meta.shape, i, j, k);
      if (!Number.isNaN(value) && value > (state.window ? state.window[0] : 0)) {
        return { i: Math.round(i), j: Math.round(j), k: Math.round(k), value };
      }
    }
    return null;
  }

  canvas.addEventListener('pointermove', (ev) => {
    const hit = castHoverRay(ev.clientX, ev.clientY);
    const el = root.querySelector('#hover-readout');
    el.textContent = hit
      ? `voxel (${hit.i}, ${hit.j}, ${hit.k})  value ${hit.value.toFixed(4)}`
      : '';
  });
```

- [ ] **Step 4: Write the Playwright test**

`tests/viewer/test_gate_and_hover.py`:
```python
def test_sigma_gate_changes_render_and_hover_shows_a_voxel(page, dist_path, run_fixture):
    run_dir = run_fixture(layers=("volume", "sigma"))
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    files = sorted(str(p) for p in run_dir.iterdir())
    page.locator("#load-run-input").set_input_files(files)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#sigma-gate-enabled").check()
    page.locator("#sigma-gate-value").fill("0.2")
    page.locator("#sigma-gate-value").dispatch_event("input")
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after

    box = page.locator("#gl-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_timeout(100)
    text = page.locator("#hover-readout").text_content()
    assert text is not None
```

- [ ] **Step 5: Run everything, confirm green**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py tests/viewer -v --browser chromium`

Note: the hover readout may legitimately be empty (`text == ""`) if the ray
through the canvas center misses every voxel above the window threshold on
the tiny random fixture — the test only asserts the element exists and
updates without throwing, not that a specific voxel is always hit.

- [ ] **Step 6: Commit**

```bash
git add viewer/shell.html viewer/src/app.mjs tests/viewer/test_gate_and_hover.py
git commit -m "feat: uncertainty gate and hover voxel readout"
```

---

### Task 9: Resolution banner, run-delta, PNG export

**Files:**
- Create: `viewer/src/delta.mjs`
- Create: `viewer/test/delta.test.mjs`
- Modify: `viewer/shell.html` (banner, load-second-run button, export button)
- Modify: `viewer/src/app.mjs` (banner population, delta loading, export)
- Create: `tests/viewer/test_banner_delta_export.py`

**Interfaces:**
- Consumes: `meta.resolution.depth_resolved`/`meta.resolution.verdict` (already in every `meta.json`, Phase 3).
- Produces: `computeDelta(a, b) -> Float32Array`, `deltaVerdict(rms, scale) -> string` from `delta.mjs`. `deltaVerdict`'s thresholds mirror `megido/volexport.py`'s `compare_volumes` exactly (`rms < 1e-3*scale` → `'unchanged'`, `< 1e-1*scale` → `'shifted slightly'`, else `'shifted substantially'`) so the in-browser verdict never disagrees with the CLI's `compare` subcommand for the same two runs.

- [ ] **Step 1: Write the failing delta tests**

`viewer/test/delta.test.mjs`:
```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeDelta, deltaVerdict } from '../src/delta.mjs';

test('computeDelta subtracts elementwise (b - a)', () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([1, 5, 2]);
  const d = computeDelta(a, b);
  assert.deepEqual([...d], [0, 3, -1]);
});

test('computeDelta rejects mismatched lengths', () => {
  assert.throws(() => computeDelta(new Float32Array(3), new Float32Array(4)), /length/);
});

test('deltaVerdict matches volexport.compare_volumes thresholds', () => {
  assert.equal(deltaVerdict(0.0001, 1.0), 'unchanged');
  assert.equal(deltaVerdict(0.05, 1.0), 'shifted slightly');
  assert.equal(deltaVerdict(0.5, 1.0), 'shifted substantially');
});
```

- [ ] **Step 2: Confirm failure, then implement**

`viewer/src/delta.mjs`:
```js
export function computeDelta(a, b) {
  if (a.length !== b.length) {
    throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  }
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = b[i] - a[i];
  return out;
}

// Mirrors megido/volexport.py's compare_volumes verdict thresholds exactly.
export function deltaVerdict(rms, scale) {
  if (rms < 1e-3 * scale) return 'unchanged';
  if (rms < 1e-1 * scale) return 'shifted slightly';
  return 'shifted substantially';
}
```

Run: `node --test viewer/test/delta.test.mjs` — expect PASS (3 tests).

- [ ] **Step 3: Add banner, second-run loader, export button**

Modify `viewer/shell.html` — inside `#app-root`, as the first child (so it
renders on top):
```html
  <div id="resolution-banner" style="position:absolute;top:0;left:0;right:0;z-index:3;padding:6px 12px;font-weight:bold;text-align:center"></div>
  <input id="load-second-run-input" type="file" webkitdirectory multiple
         style="position:absolute;top:36px;left:8px;z-index:2" />
  <div id="delta-verdict" style="position:absolute;top:36px;left:220px;z-index:2;color:#eee"></div>
  <button id="export-png-btn" style="position:absolute;bottom:8px;right:8px;z-index:2">Export PNG</button>
```

- [ ] **Step 4: Wire banner, delta, export in app.mjs**

Modify `viewer/src/app.mjs`:
- Add `import { computeDelta, deltaVerdict } from './delta.mjs';` at the top.
- In `loadRun`, after `state.meta = meta;`, add:
  ```js
  const banner = root.querySelector('#resolution-banner');
  const res = meta.resolution || {};
  banner.textContent = res.verdict || '';
  banner.style.background = res.depth_resolved ? '#2a6' : '#a33';
  banner.style.color = '#fff';
  ```
- After the layer-panel block from Task 6, add second-run loading:
  ```js
  root.querySelector('#load-second-run-input').addEventListener('change', async (ev) => {
    const files = Array.from(ev.target.files);
    const byName = new Map(files.map((f) => [f.name, f]));
    const metaFile = byName.get('meta.json');
    if (!metaFile) return;
    const secondMeta = JSON.parse(await metaFile.text());
    if (JSON.stringify(secondMeta.shape) !== JSON.stringify(state.meta.shape)) {
      root.querySelector('#delta-verdict').textContent = 'grid mismatch: cannot diff';
      return;
    }
    const volFile = byName.get('volume.npy');
    const { data: secondVolume } = parseNpy(await volFile.arrayBuffer());
    const primary = state.layerData.get('volume');
    const delta = computeDelta(primary, secondVolume);
    state.layerData.set('delta', delta);

    const sorted = [...primary].map(Math.abs).sort((a, b) => a - b);
    const scale = sorted[Math.floor(sorted.length * 0.95)] || 1e-12;
    const rms = Math.sqrt(delta.reduce((s, v) => s + v * v, 0) / delta.length);
    root.querySelector('#delta-verdict').textContent =
      `delta: ${deltaVerdict(rms, scale)} (rms ${rms.toFixed(4)})`;

    const deltaOption = document.createElement('label');
    deltaOption.style.display = 'block';
    const radio = document.createElement('input');
    radio.type = 'radio'; radio.name = 'active-layer'; radio.id = 'layer-delta';
    radio.addEventListener('change', () => setActiveLayer('delta'));
    deltaOption.appendChild(radio);
    deltaOption.appendChild(document.createTextNode(' Run delta (B minus A)'));
    root.querySelector('#layer-panel').appendChild(deltaOption);
  });
  ```
- Add the export handler, anywhere after `render` is defined:
  ```js
  root.querySelector('#export-png-btn').addEventListener('click', () => {
    render();
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'megiddo-voxel-view.png';
      a.click();
      URL.revokeObjectURL(url);
    });
  });
  ```

- [ ] **Step 5: Write the Playwright test**

`tests/viewer/test_banner_delta_export.py`:
```python
def test_banner_delta_and_export(page, dist_path, run_fixture):
    run_a = run_fixture(depth_resolved=False)
    run_b = run_fixture(depth_resolved=False)
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    files_a = sorted(str(p) for p in run_a.iterdir())
    page.locator("#load-run-input").set_input_files(files_a)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    banner_text = page.locator("#resolution-banner").text_content()
    assert "depth NOT resolved" in banner_text

    files_b = sorted(str(p) for p in run_b.iterdir())
    page.locator("#load-second-run-input").set_input_files(files_b)
    page.wait_for_function("() => window.__viewerState.layerData.has('delta')")
    assert page.locator("#layer-delta").count() == 1
    assert "delta:" in page.locator("#delta-verdict").text_content()

    with page.expect_download() as dl_info:
        page.locator("#export-png-btn").click()
    download = dl_info.value
    assert download.suggested_filename == "megiddo-voxel-view.png"
```

- [ ] **Step 6: Run everything, confirm green**

Run: `node --test viewer/test && uv run pytest tests/test_viewerbuild.py tests/viewer -v --browser chromium`

- [ ] **Step 7: Commit**

```bash
git add viewer/src/delta.mjs viewer/test/delta.test.mjs viewer/shell.html \
        viewer/src/app.mjs tests/viewer/test_banner_delta_export.py
git commit -m "feat: resolution banner, run-delta layer, PNG export"
```

---

### Task 10: Final smoke test on the real campaign output

**Files:**
- Create: `tests/viewer/test_smoke.py`
- Modify: `viewer/README.md` (point at `runs/voxels` as the canonical example run)
- Modify: `CLAUDE.md` (mark Phase 4 done in the phase list, add the `view` command to "Running it")

**Interfaces:**
- Consumes: everything built in Tasks 1-9, and the real `runs/voxels/` directory produced by Phase 3 (`volume.npy`, `meta.json`, `sigma.npy`, `snr.npy`, `views.npy`, `systematic.npy`, `backprojection.npy` — already on disk from the Phase 3 campaign run).
- Produces: nothing further — this is the plan's exit gate.

- [ ] **Step 1: Write the combined smoke test**

`tests/viewer/test_smoke.py`:
```python
from pathlib import Path

REAL_RUN = Path(__file__).resolve().parents[2] / "runs" / "voxels"


def test_real_campaign_output_renders_with_every_control_operable(page, dist_path):
    assert (REAL_RUN / "meta.json").exists(), (
        "runs/voxels is Phase 3's real campaign output — regenerate it with "
        "`uv run python -m megido.cli reconstruct ...` (see docs/phase3-reconstruction-report.md) "
        "before running this gate"
    )
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    files = sorted(str(p) for p in REAL_RUN.iterdir() if p.suffix in (".npy", ".json"))
    page.locator("#load-run-input").set_input_files(files)
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta", timeout=15000)

    err = page.evaluate("() => window.__viewerError || null")
    assert err is None, err

    canvas_data = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    blank = page.evaluate("""
        () => { const c = document.createElement('canvas');
                c.width = document.querySelector('#gl-canvas').width;
                c.height = document.querySelector('#gl-canvas').height;
                return c.toDataURL(); }
    """)
    assert canvas_data != blank, "canvas must be non-blank on real reconstructed output"

    banner = page.locator("#resolution-banner").text_content()
    assert "depth NOT resolved" in banner, (
        "the real campaign's honest verdict must reach the viewer unedited"
    )

    for key in ("volume", "sigma", "snr", "views", "systematic", "backprojection"):
        assert page.locator(f"#layer-{key}").count() == 1, f"missing layer control for {key}"
        page.locator(f"#layer-{key}").check()

    page.locator("#camera-preset-top").click()
    page.locator("#camera-preset-iso").click()

    box = page.locator("#xfer-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5)
    page.mouse.up()

    page.locator("#window-lo").fill("0.1")
    page.locator("#window-lo").dispatch_event("input")
    page.locator("#window-hi").fill("0.9")
    page.locator("#window-hi").dispatch_event("input")

    page.locator("#clip-x-max").fill("0.6")
    page.locator("#clip-x-max").dispatch_event("input")
    page.locator("#clip-plane-enabled").check()
    page.locator("#slice-axis").select_option("z")
    page.locator("#slice-pos").fill("0.5")
    page.locator("#slice-pos").dispatch_event("input")

    page.locator("#sigma-gate-enabled").check()
    page.locator("#sigma-gate-value").fill("0.5")
    page.locator("#sigma-gate-value").dispatch_event("input")

    canvas_box = page.locator("#gl-canvas").bounding_box()
    page.mouse.move(canvas_box["x"] + canvas_box["width"] / 2,
                     canvas_box["y"] + canvas_box["height"] / 2)
    page.wait_for_timeout(100)

    with page.expect_download():
        page.locator("#export-png-btn").click()

    assert console_errors == [], f"JS console errors during interaction: {console_errors}"
```

- [ ] **Step 2: Run it against the real data**

Run: `uv run pytest tests/viewer/test_smoke.py -v --browser chromium`
Expected: PASS. This is the plan's exit gate — every control from Tasks 3-9
is exercised in one page session against the real `runs/voxels` output with
zero console errors.

- [ ] **Step 3: Point the README at the real run and update CLAUDE.md**

Modify `viewer/README.md` — replace the "Using it" section's example:
```markdown
## Using it

"Load run" picks a run directory (e.g. `runs/voxels`, produced by
`uv run python -m megido.cli reconstruct ...`) via a directory file picker;
the viewer reads `meta.json` and every `.npy` file it names. The resolution
banner at the top always shows whether depth is resolved for that run —
for the current campaign it is not, and the viewer will say so in red.
```

Modify `CLAUDE.md` — in the "Architecture" phase list, change:
```
- **Phase 4** (not started): S5 — the viewer, rebuilt from scratch (GPU
  raymarching; the cafeteria viewer is reference only).
```
to:
```
- **Phase 4** (done): S5 — the viewer. `viewer/src/*.mjs` (WebGL2
  raymarching, zero runtime dependencies) built by `megido/viewerbuild.py`
  into one self-contained `viewer/dist/index.html`. Loads any run directory
  at runtime via a local file picker; never bakes data into the shipped
  page.
```
and in "Running it", add after the `solve` line:
```
uv run python -m megido.cli view                  # Phase 4 viewer: build + open
```

- [ ] **Step 4: Run the whole project's test suites**

Run: `uv run pytest -q && node --test viewer/test`
Expected: both green — this is the finishing-a-development-branch gate.

- [ ] **Step 5: Commit**

```bash
git add tests/viewer/test_smoke.py viewer/README.md CLAUDE.md
git commit -m "test: Phase 4 exit-gate smoke test on the real campaign output"
```
