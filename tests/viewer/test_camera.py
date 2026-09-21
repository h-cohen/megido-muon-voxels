import math


def test_camera_presets_change_the_render(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # This Playwright version (1.63) requires a directory path for a
    # webkitdirectory input; pass the run directory itself (see
    # tests/viewer/test_raymarch.py for the same pattern).
    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    front = page.evaluate("""
        () => { window.__viewerState.camera.yaw = 0; window.__viewerState.camera.pitch = 0;
                window.__viewerState.render(); return document.querySelector('#gl-canvas').toDataURL(); }
    """)
    page.locator("#camera-preset-top").click()
    top = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert front != top

    state_yaw = page.evaluate("() => window.__viewerState.camera.yaw")
    assert abs(state_yaw - 0) < 1e-6  # top preset's yaw is 0, per CAMERA_PRESETS

    # yaw==0 alone doesn't distinguish top from front (both have yaw=0) —
    # pitch is the discriminating field for the top preset, so assert it too.
    state_pitch = page.evaluate("() => window.__viewerState.camera.pitch")
    assert abs(state_pitch - (math.pi / 2 - 0.001)) < 1e-6
