"""Task 3: adaptive quality (fast preview) while interacting.

Camera drags/zooms and slider drags call state.beginInteraction(), which
should render at a reduced step count with shading off for that frame only;
after the 150 ms idle timer (or the state.idleNow() test hook), the next
render() must be pixel-identical to the full-quality frame that was showing
before the interaction started, at the same camera.
"""


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")


def _png(page):
    return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")


def test_adaptive_quality_defaults_on_and_not_interacting(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))
    st = page.evaluate(
        "() => ({a: window.__viewerState.adaptiveQuality, i: window.__viewerState.interacting})"
    )
    assert st == {"a": True, "i": False}


def test_begin_interaction_previews_then_idle_restores_full_quality(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))

    page.evaluate("() => window.__viewerState.render()")
    full = _png(page)

    page.evaluate("() => { window.__viewerState.beginInteraction(); window.__viewerState.render(); }")
    interacting = page.evaluate("() => window.__viewerState.interacting")
    assert interacting is True
    preview = _png(page)
    assert preview != full, "preview frame (fewer steps, no shading) must differ from full quality"

    page.evaluate("() => window.__viewerState.idleNow()")
    assert page.evaluate("() => window.__viewerState.interacting") is False
    settled = _png(page)
    assert settled == full, "idleNow() must restore the exact full-quality frame (same camera)"
    assert not errors, errors


def test_adaptive_quality_disabled_skips_the_preview(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _load(page, dist_path, run_fixture(shape=(12, 10, 8)))

    page.locator("#toggle-adaptive").uncheck()
    page.evaluate("() => window.__viewerState.render()")
    full = _png(page)

    page.evaluate("() => { window.__viewerState.beginInteraction(); window.__viewerState.render(); }")
    assert page.evaluate("() => window.__viewerState.interacting") is True
    still_full = _png(page)
    assert still_full == full, (
        "with #toggle-adaptive unchecked, beginInteraction()+render() must still be full quality"
    )
    assert not errors, errors
