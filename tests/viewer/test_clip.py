def test_clip_box_and_plane_change_the_render(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # This Playwright version (1.63) requires a directory path for a
    # webkitdirectory input; pass the run directory itself (see
    # tests/viewer/test_camera.py for the same pattern).
    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#clip-x-max").fill("0.3")
    page.locator("#clip-x-max").dispatch_event("input")
    after_clip = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after_clip
    assert page.locator("#clip-x-max-val").text_content() == "0.30"

    page.locator("#clip-plane-enabled").check()
    after_plane = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert after_clip != after_plane
