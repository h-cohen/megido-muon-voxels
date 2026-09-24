def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")


def _png(page):
    return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")


def test_smooth_and_shading_toggles_each_change_the_render(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    st = page.evaluate("() => ({s: window.__viewerState.smoothSampling, h: window.__viewerState.shading})")
    assert st == {"s": True, "h": True}
    base = _png(page)
    page.locator("#toggle-shading").uncheck(); page.wait_for_timeout(150)
    no_shade = _png(page)
    assert no_shade != base
    page.locator("#toggle-smooth").uncheck(); page.wait_for_timeout(150)
    nearest = _png(page)
    assert nearest != no_shade
    assert not errors, errors


_PIXELS = """() => {
  const src = document.querySelector('#gl-canvas');
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0);
  return Array.from(ctx.getImageData(0, 0, c.width, c.height).data);
}"""


def _mean_abs_diff(a, b):
    assert len(a) == len(b)
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def test_manual_trilinear_fallback_agrees_with_hardware_linear(page, dist_path, run_fixture):
    """The manual 8-tap texelFetch path only runs on GPUs without
    OES_texture_float_linear, so headless never exercises it by default. Force
    it and require it to render like hardware LINEAR (same maths) and unlike
    NEAREST -- a transposed or half-texel-shifted fallback would fail this."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    if not page.evaluate("() => window.__viewerState.floatLinear"):
        import pytest
        pytest.skip("no OES_texture_float_linear: hardware reference unavailable")
    page.locator("#toggle-shading").uncheck(); page.wait_for_timeout(150)
    hw = page.evaluate(_PIXELS)
    page.evaluate("""() => { const s = window.__viewerState;
        s.floatLinear = false; s.applyVolumeFilter(); s.render(); }""")
    page.wait_for_timeout(150)
    manual = page.evaluate(_PIXELS)
    page.evaluate("""() => { const s = window.__viewerState;
        s.smoothSampling = false; s.applyVolumeFilter(); s.render(); }""")
    page.wait_for_timeout(150)
    nearest = page.evaluate(_PIXELS)
    d_manual = _mean_abs_diff(hw, manual)
    d_nearest = _mean_abs_diff(hw, nearest)
    print(f"\nmean |Δ| per channel: manual vs hw={d_manual:.3f}, nearest vs hw={d_nearest:.3f}")
    assert d_nearest > 1.0, d_nearest                 # the comparison has teeth
    assert d_manual < 0.25 * d_nearest, (d_manual, d_nearest)
    assert not errors, errors
