"""Display-only 'Clip volume above surface' toggle (Task 5).

Off by default; when checked, the raymarch must skip volume samples above the
displayed (possibly smoothed) hillside height field, so fewer pixels light up.
The fixture's flat surface at z=2.0 cuts the z in [1, 3] volume in half.
"""
from __future__ import annotations

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
