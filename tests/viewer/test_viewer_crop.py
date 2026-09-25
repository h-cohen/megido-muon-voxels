"""meta.json `viewer_crop_xy_m` sets the initial clip box (display-only).

Cropping the SOLVE box pushes oblique rays' opacity into the box walls
(docs/cafeteria-run.md); cropping only the display does not. The fixture
volume spans x 0..3 m, y 0..2.5 m (origin 0,0; 6x5 voxels at 0.5 m), so the
crop x 0.75..2.25, y 0.5..2.0 is fractions x 0.25..0.75, y 0.2..0.8.
"""
from __future__ import annotations

import json

import pytest

_GRID = [(fx / 20, fy / 20) for fx in range(2, 19) for fy in range(2, 19) if fy != 10]


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-top").click()
    page.evaluate("() => window.__viewerState.idleNow()")


def _picks(page):
    return page.evaluate(
        """(pts) => {
            const r = document.querySelector('#gl-canvas').getBoundingClientRect();
            return pts.map(([fx, fy]) =>
                window.__viewerState.pick(r.left + fx * r.width, r.top + fy * r.height));
        }""", _GRID)


def test_crop_sets_the_initial_clip_box_and_hover_stays_inside(page, dist_path, run_fixture):
    run = run_fixture(shape=(6, 5, 4), origin_m=(0.0, 0.0, 1.0), spacing_m=0.5)
    meta = json.loads((run / "meta.json").read_text())
    meta["viewer_crop_xy_m"] = [[0.75, 2.25], [0.5, 2.0]]
    (run / "meta.json").write_text(json.dumps(meta))
    _load(page, dist_path, run)

    lo = page.evaluate("() => window.__viewerState.clipMin")
    hi = page.evaluate("() => window.__viewerState.clipMax")
    assert lo[:2] == pytest.approx([0.25, 0.2]) and hi[:2] == pytest.approx([0.75, 0.8])
    assert lo[2] == 0 and hi[2] == 1                       # z untouched
    assert float(page.locator("#clip-x-min").input_value()) == pytest.approx(0.25, abs=0.01)
    assert float(page.locator("#clip-y-max").input_value()) == pytest.approx(0.8, abs=0.01)
    picks = [p for p in _picks(page) if p]
    assert picks and all(1 <= p["i"] <= 4 and 1 <= p["j"] <= 3 for p in picks)


def test_no_crop_keeps_the_full_box(page, dist_path, run_fixture):
    run = run_fixture(shape=(6, 5, 4))
    _load(page, dist_path, run)
    assert page.evaluate("() => window.__viewerState.clipMin") == [0, 0, 0]
    assert page.evaluate("() => window.__viewerState.clipMax") == [1, 1, 1]
