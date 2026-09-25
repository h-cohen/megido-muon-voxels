"""Opacity slider: one display-only multiplier on how opaque the voxels draw.

Fog mode scales every sample's transfer-function alpha; cube mode turns the
opaque blocks translucent, so cubes behind the first one show through. At the
default (1) both modes must render exactly as before.

Fixture (same as test_voxel_cubes): an empty 6x5x4 volume with a bright voxel
(2,2,3) = 1.0 stacked above a dimmer one (2,2,1) = 0.8 in the same column,
viewed from the top so the upper cube hides the lower one.
"""
from __future__ import annotations

import json

import numpy as np

SHAPE = (6, 5, 4)


def _run(run_fixture):
    run = run_fixture(shape=SHAPE)
    vol = np.zeros(SHAPE, dtype=np.float32)
    vol[2, 2, 3] = 1.0
    vol[2, 2, 1] = 0.8
    np.save(run / "volume.npy", vol)
    meta = json.loads((run / "meta.json").read_text())
    meta["value_range"] = [0.0, 1.0]
    meta["suggested_iso"] = [0.3, 0.6]
    (run / "meta.json").write_text(json.dumps(meta))
    return run


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-top").click()
    page.evaluate("() => window.__viewerState.idleNow()")


def _set(page, selector, value):
    page.locator(selector).fill(str(value))
    page.locator(selector).dispatch_event("input")


def _opacity(page, v):
    page.locator("#opacity").evaluate(
        "(el, v) => { el.value = String(v); el.dispatchEvent(new Event('input')); }", v)


def _frame(page):
    """RGBA of the current full-quality frame, (h, w, 4) float."""
    data, w, h = page.evaluate("""async () => {
        window.__viewerState.idleNow();
        const cv = document.querySelector('#gl-canvas');
        const img = new Image(); img.src = cv.toDataURL(); await img.decode();
        const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
        const g = c.getContext('2d'); g.drawImage(img, 0, 0);
        return [Array.from(g.getImageData(0, 0, c.width, c.height).data), c.width, c.height];
    }""")
    return np.array(data, dtype=float).reshape(h, w, 4)


def test_slider_defaults_to_fully_opaque(page, dist_path, run_fixture):
    _load(page, dist_path, _run(run_fixture))
    el = page.locator("#opacity")
    assert el.get_attribute("type") == "range"
    assert float(el.input_value()) == 1.0
    assert page.evaluate("() => window.__viewerState.opacity") == 1.0
    assert page.locator("#opacity-readout").inner_text().strip() == "100%"


def test_fog_opacity_scales_the_render_and_zero_hides_it(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, _run(run_fixture))
    full = _frame(page)
    _opacity(page, 0.3)
    assert page.locator("#opacity-readout").inner_text().strip() == "30%"
    faint = _frame(page)
    _opacity(page, 0.0)
    none = _frame(page)
    _opacity(page, 1.0)
    back = _frame(page)
    lit = full[..., :3].max(axis=2) > 0
    assert lit.sum() > 0
    # Weaker, not gone, and back to identical at 1.
    assert 0 < faint[..., 3][lit].sum() < full[..., 3][lit].sum()
    assert none[..., :3].max() == 0 and none[..., 3].max() == 0
    assert np.array_equal(back, full)
    assert not errors


def test_translucent_cubes_let_the_lower_cube_show_through(page, dist_path, run_fixture):
    """Opaque: the pixels covered by the upper cube look the same whether or
    not the lower cube exists (it is hidden). Translucent: they differ, because
    the lower cube now shows through. A march that still stops at the first
    cube fails the second half."""
    _load(page, dist_path, _run(run_fixture))
    page.locator("#render-mode").select_option("cubes")

    def pair(opacity):
        _opacity(page, opacity)
        _set(page, "#cube-threshold", 0.9)       # only the upper cube
        upper = _frame(page)
        _set(page, "#cube-threshold", 0.5)       # both cubes
        both = _frame(page)
        return upper, both

    upper1, both1 = pair(1.0)
    covered = upper1[..., 3] > 0
    assert covered.sum() > 0
    assert np.array_equal(upper1[covered], both1[covered])      # hidden when opaque

    upper_h, both_h = pair(0.5)
    assert upper_h[..., 3][covered].max() < upper1[..., 3][covered].max()   # translucent
    diff = np.abs(both_h[covered] - upper_h[covered]).max(axis=1)
    assert np.mean(diff > 2) > 0.5                              # lower cube shows through


def test_cube_hover_still_returns_the_first_cube_when_translucent(page, dist_path, run_fixture):
    _load(page, dist_path, _run(run_fixture))
    page.locator("#render-mode").select_option("cubes")
    _set(page, "#cube-threshold", 0.5)
    _opacity(page, 0.4)
    page.evaluate("() => window.__viewerState.idleNow()")
    p = page.evaluate("""() => {
        const r = document.querySelector('#gl-canvas').getBoundingClientRect();
        let best = null;
        for (let fx = 0.1; fx < 0.9; fx += 0.02) for (let fy = 0.1; fy < 0.9; fy += 0.02) {
            const q = window.__viewerState.pick(r.left + fx * r.width, r.top + fy * r.height);
            if (q && q.k === 3) best = q;
        }
        return best;
    }""")
    assert p is not None and (p["i"], p["j"], p["k"]) == (2, 2, 3)
