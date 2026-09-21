def test_onboarding_shows_then_hides_on_run_load(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # Fresh context, never dismissed before: onboarding card is visible.
    assert page.locator("#overlay-onboarding").is_visible()

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    assert not page.locator("#overlay-onboarding").is_visible()
    assert errors == []


def test_help_overlay_toggles_with_question_mark_and_escape(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    assert not page.locator("#overlay-shortcuts").is_visible()

    page.keyboard.press("Shift+Slash")  # '?'
    assert page.locator("#overlay-shortcuts").is_visible()

    page.keyboard.press("Escape")
    assert not page.locator("#overlay-shortcuts").is_visible()


def test_digit_key_switches_to_nth_layer(page, dist_path, run_fixture):
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    run_dir = run_fixture(layers=("volume", "sigma"))
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    page.keyboard.press("2")
    active = page.evaluate("() => window.__viewerState.activeLayer")
    assert active == "sigma"
    assert errors == []
