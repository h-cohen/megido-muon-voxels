"""Display-only 'Clip volume above surface' toggle (Task 5).

Off by default; when checked, the raymarch must skip volume samples above the
displayed (possibly smoothed) hillside height field, so fewer pixels light up.
The fixture's flat surface at z=2.0 cuts the z in [1, 3] volume in half.

Two of these gates (mean-row-moves-down, slope-edge-follows-corr) exist
because the ORIGINAL `lit_on < lit_off` gate cannot fail for an inverted
`world.z < h` clip (removes the OTHER half, same pixel count) or for a
transposed `aboveSurface` fetch on a flat field (still same count) -- see
review finding M1. Both new gates fail for those broken implementations.
"""
from __future__ import annotations

import numpy as np

_COUNT_LIT = """() => {
  const src = document.querySelector('#gl-canvas');
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const bg = [d[0], d[1], d[2]];
  let n = 0;
  for (let k = 0; k < d.length; k += 4) {
    if (Math.abs(d[k]-bg[0]) + Math.abs(d[k+1]-bg[1]) + Math.abs(d[k+2]-bg[2]) > 24) n++;
  }
  return n;
}"""

# Mean pixel-row index (0 = top of canvas) over all "lit" (non-background)
# pixels. Returns null when nothing is lit.
_MEAN_LIT_ROW = """() => {
  const src = document.querySelector('#gl-canvas');
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const bg = [d[0], d[1], d[2]];
  const w = c.width, h = c.height;
  let sumRow = 0, n = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const k = (y * w + x) * 4;
      if (Math.abs(d[k]-bg[0]) + Math.abs(d[k+1]-bg[1]) + Math.abs(d[k+2]-bg[2]) > 24) { sumRow += y; n++; }
    }
  }
  return n > 0 ? sumRow / n : null;
}"""

# For each canvas column that has at least one lit pixel, the row index of
# the TOPMOST lit pixel in that column (the visible upper silhouette edge).
_TOP_LIT_ROW_PER_COLUMN = """() => {
  const src = document.querySelector('#gl-canvas');
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const bg = [d[0], d[1], d[2]];
  const w = c.width, h = c.height;
  const cols = [], tops = [];
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const k = (y * w + x) * 4;
      if (Math.abs(d[k]-bg[0]) + Math.abs(d[k+1]-bg[1]) + Math.abs(d[k+2]-bg[2]) > 24) {
        cols.push(x); tops.push(y); break;
      }
    }
  }
  return { cols, tops };
}"""


def test_clip_above_surface_hides_volume_and_is_off_by_default(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    assert page.locator("#toggle-surf-clip").is_enabled()
    assert page.locator("#toggle-surf-clip").is_checked() is False
    page.locator("#toggle-hill-surface").uncheck()      # count volume pixels only
    page.wait_for_timeout(150)
    lit_off = page.evaluate(_COUNT_LIT)
    page.locator("#toggle-surf-clip").check(); page.wait_for_timeout(150)
    lit_on = page.evaluate(_COUNT_LIT)
    assert lit_on < lit_off, (lit_off, lit_on)
    assert not errors, errors


def test_clip_above_surface_moves_mass_down_on_front_view(page, dist_path, run_fixture):
    """A gate that can fail: `lit_on < lit_off` alone also passes for an
    inverted `world.z < h` clip (which removes the OTHER half, same pixel
    count). Front view (camera.mjs: yaw=pi/2, pitch=0, z up) shows world z as
    canvas rows, so clipping the surface's ABOVE half must move the mean lit
    row DOWN (larger row index = lower on screen = lower z). An inverted
    clip would move it up instead."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill=True)
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-front").click()
    page.locator("#toggle-hill-surface").uncheck()      # count volume pixels only
    page.wait_for_timeout(150)
    mean_row_off = page.evaluate(_MEAN_LIT_ROW)
    page.locator("#toggle-surf-clip").check(); page.wait_for_timeout(150)
    mean_row_on = page.evaluate(_MEAN_LIT_ROW)
    print(f"\nmean lit row: off={mean_row_off:.3f}, on={mean_row_on:.3f}")
    assert mean_row_off is not None and mean_row_on is not None
    assert mean_row_on > mean_row_off, (mean_row_off, mean_row_on)
    assert not errors, errors


def test_clip_edge_follows_slope(page, dist_path, run_fixture):
    """A gate that can fail for a transposed `aboveSurface` fetch, which would
    still pass the flat-fixture `lit_on < lit_off` count check unchanged.
    Uses the `hill="slope"` fixture, where H varies along world x only. Front
    preset (camera.mjs: yaw=pi/2, pitch=0 -> eye on +y looking toward -y,
    z up) projects world x onto the canvas's horizontal axis -- confirmed
    empirically by running this test (a flat clip edge under Front would fail
    the correlation assertion below; it does not). With the surface clip on,
    the visible upper silhouette's row should vary linearly with column
    (the slope); with the clip off, the top edge is just the box's flat top
    face and should show much weaker correlation.
    """
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    run = run_fixture(hill="slope")
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")
    page.locator("#camera-preset-front").click()
    page.locator("#toggle-hill-surface").uncheck()      # count volume pixels only
    page.wait_for_timeout(150)

    edge_off = page.evaluate(_TOP_LIT_ROW_PER_COLUMN)
    page.locator("#toggle-surf-clip").check(); page.wait_for_timeout(150)
    edge_on = page.evaluate(_TOP_LIT_ROW_PER_COLUMN)

    def corr(edge):
        cols, tops = edge["cols"], edge["tops"]
        assert len(cols) > 10, "too few lit columns to measure a slope"
        if np.std(tops) == 0:
            return 0.0   # perfectly flat edge: zero variance, zero correlation
        return float(np.corrcoef(cols, tops)[0, 1])

    corr_on = corr(edge_on)
    corr_off = corr(edge_off)
    print(f"\nslope edge corr: clip on={corr_on:.3f}, clip off={corr_off:.3f}")
    assert abs(corr_on) > 0.8, (corr_on, edge_on)
    assert abs(corr_on) > abs(corr_off), (corr_on, corr_off)
    assert not errors, errors
