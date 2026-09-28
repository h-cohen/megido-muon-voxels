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
        s.window = [0.5, 1.0];
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
