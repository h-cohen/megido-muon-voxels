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
