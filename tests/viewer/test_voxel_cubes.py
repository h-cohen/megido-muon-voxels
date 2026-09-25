"""'Voxel cubes' render mode: every voxel >= threshold drawn as an opaque cube.

The fog raymarch blends dozens of voxels per pixel, so voxel boundaries never
show. Cube mode walks the grid voxel-to-voxel (DDA, voxelMarch in grid.mjs)
and stops at the first voxel >= threshold that passes every gate; hover must
return exactly that voxel.

Fixture: an otherwise empty 6x5x4 volume with a bright voxel (2,2,3) = 1.0
stacked above a dimmer one (2,2,1) = 0.8 in the same column.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

SHAPE = (6, 5, 4)
_GRID = [(fx / 30, fy / 30) for fx in range(3, 28) for fy in range(3, 28) if fy != 15]


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
    return run, vol


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-top").click()
    page.evaluate("() => window.__viewerState.idleNow()")


def _cubes(page, threshold=None):
    page.locator("#render-mode").select_option("cubes")
    if threshold is not None:
        page.locator("#cube-threshold").fill(str(threshold))
        page.locator("#cube-threshold").dispatch_event("input")
    page.evaluate("() => window.__viewerState.idleNow()")


def _picks(page):
    return page.evaluate(
        """(pts) => {
            const r = document.querySelector('#gl-canvas').getBoundingClientRect();
            return pts.map(([fx, fy]) =>
                window.__viewerState.pick(r.left + fx * r.width, r.top + fy * r.height));
        }""", _GRID)


def _luminance(page):
    """Per-pixel max(r,g,b) of the current frame, 0..255."""
    return page.evaluate("""async () => {
        window.__viewerState.idleNow();
        const url = document.querySelector('#gl-canvas').toDataURL();
        const img = new Image(); img.src = url; await img.decode();
        const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
        const g = c.getContext('2d'); g.drawImage(img, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data;
        const out = [];
        for (let i = 0; i < d.length; i += 4) out.push(Math.max(d[i], d[i + 1], d[i + 2]));
        return out;
    }""")


def test_controls_default_to_fog_with_threshold_from_meta(page, dist_path, run_fixture):
    run, _ = _run(run_fixture)
    _load(page, dist_path, run)
    assert page.locator("#render-mode").input_value() == "fog"
    assert float(page.locator("#cube-threshold").input_value()) == 0.3


def test_hover_returns_the_first_cube_from_the_camera(page, dist_path, run_fixture):
    """Occlusion: wherever a ray hits the upper cube it must return it, never
    the lower cube behind it. (Oblique perspective rays that pass BESIDE the
    upper cube may legitimately reach the lower one.)"""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run, vol = _run(run_fixture)
    _load(page, dist_path, run)
    _cubes(page, 0.9)                        # only the upper (2,2,3) = 1.0 voxel is a cube
    upper_only = _picks(page)
    assert {(p["i"], p["j"], p["k"]) for p in upper_only if p} == {(2, 2, 3)}
    _cubes(page, 0.5)                        # both cubes exist now
    both = _picks(page)
    assert all(vol[p["i"], p["j"], p["k"]] >= 0.5 for p in both if p)
    for a, b in zip(upper_only, both):
        if a:                                # this ray intersects the upper cube ...
            assert b and (b["i"], b["j"], b["k"]) == (2, 2, 3)   # ... so it must stop there
    assert not errors


def test_threshold_above_every_voxel_draws_and_picks_nothing(page, dist_path, run_fixture):
    run, _ = _run(run_fixture)
    _load(page, dist_path, run)
    _cubes(page, 5.0)
    assert all(p is None for p in _picks(page))
    assert max(_luminance(page)) == 0


def test_cubes_have_hard_edges_where_fog_is_soft(page, dist_path, run_fixture):
    run, _ = _run(run_fixture)
    _load(page, dist_path, run)
    fog = np.array(_luminance(page))
    _cubes(page, 0.5)
    cube = np.array(_luminance(page))

    def dim_fraction(lum):
        lit = lum[lum > 0]
        assert lit.size > 0
        return float(np.mean(lit < 0.3 * lit.max()))

    assert dim_fraction(cube) < 0.05          # a solid block: lit pixels are all near full
    assert dim_fraction(fog) > 0.2            # the fog render fades out softly


@pytest.mark.parametrize("size, threshold", [("1", 0.5), ("2", 0.05)])
def test_hover_and_shader_agree_on_which_pixels_hold_a_cube(page, dist_path, run_fixture, size, threshold):
    run, _ = _run(run_fixture)
    _load(page, dist_path, run)
    _cubes(page, threshold)
    page.locator("#cube-size").select_option(size)
    page.evaluate("() => window.__viewerState.idleNow()")
    picks = _picks(page)
    lum = np.array(_luminance(page))
    w, h = page.evaluate("() => [document.querySelector('#gl-canvas').width, document.querySelector('#gl-canvas').height]")
    img = lum.reshape(h, w)
    lit = [img[min(h - 1, int(fy * h)), min(w - 1, int(fx * w))] > 0 for fx, fy in _GRID]
    mismatch = sum(bool(p) != l for p, l in zip(picks, lit))
    assert sum(lit) > 0
    assert mismatch <= 0.02 * len(_GRID)


def test_threshold_arrows_step_by_a_thousandth(page, dist_path, run_fixture):
    run, _ = _run(run_fixture)
    _load(page, dist_path, run)
    box = page.locator("#cube-threshold")
    assert box.get_attribute("step") == "0.001"
    box.focus()
    page.keyboard.press("ArrowUp")
    assert abs(float(box.input_value()) - 0.301) < 1e-9
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    assert abs(float(box.input_value()) - 0.299) < 1e-9
    assert abs(page.evaluate("() => window.__viewerState.cubeThreshold") - 0.299) < 1e-9


def test_cube_size_merges_blocks_by_their_mean(page, dist_path, run_fixture):
    """At 2x, the block holding the 1.0 voxel (coarse (1,1,1), 8 voxels) has
    mean 1/8; the block holding the 0.8 voxel (coarse (1,1,0)) has mean 0.1."""
    run, _ = _run(run_fixture)
    _load(page, dist_path, run)
    _cubes(page, 0.11)
    page.locator("#cube-size").select_option("2")
    page.evaluate("() => window.__viewerState.idleNow()")
    picked = [p for p in _picks(page) if p]
    assert {(p["i"], p["j"], p["k"]) for p in picked} == {(1, 1, 1)}
    assert abs(picked[0]["value"] - 0.125) < 1e-6
    _cubes(page, 0.13)                       # above the diluted mean: nothing
    assert all(p is None for p in _picks(page))
    assert max(_luminance(page)) == 0


def test_cube_size_merges_only_voxels_that_pass_the_gates(page, dist_path, run_fixture):
    run, vol = _run(run_fixture)
    rays = np.full(SHAPE, 10.0, dtype=np.float32)
    rays[2, 2, 2] = 1.0                      # gated (coverage gate default 2), inside coarse (1,1,1)
    np.save(run / "rays.npy", rays)
    vol[2, 2, 2] = 50.0                      # a huge value that must NOT enter the mean
    np.save(run / "volume.npy", vol)
    meta = json.loads((run / "meta.json").read_text())
    meta["layers"] = ["volume", "rays"]
    (run / "meta.json").write_text(json.dumps(meta))
    _load(page, dist_path, run)
    _cubes(page, 0.11)
    page.locator("#cube-size").select_option("2")
    page.evaluate("() => window.__viewerState.idleNow()")
    picked = [p for p in _picks(page) if p]
    assert picked and all(abs(p["value"] - 1.0 / 7.0) < 1e-6 for p in picked)   # 7 kept voxels
