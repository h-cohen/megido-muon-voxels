"""First-load defaults: robust window (not the [min,max] value_range) and a
camera framed on the grid, not the old hardcoded target=[0,0,0]/distance=3.
"""


def test_default_window_and_camera_frame_the_loaded_volume(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    # state.layerMax must be set from the active layer's own data, and
    # state.window must be exactly what robustWindow(data) computes for it -
    # not meta.value_range. Recompute robustWindow in-page (it's part of the
    # bundle) so this doesn't hardcode the fixture's random values.
    result = page.evaluate(
        """
        () => {
          const s = window.__viewerState;
          const data = s.layerData.get('volume');
          let max = 0;
          for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
          const expectedWindow = robustWindow(data);
          return {
            layerMax: s.layerMax,
            window: s.window,
            expectedMax: max,
            expectedWindow,
            cameraTarget: s.camera.target,
            cameraDistance: s.camera.distance,
            meta: s.meta,
          };
        }
        """
    )

    assert result["layerMax"] == result["expectedMax"]
    assert result["window"][0] == result["expectedWindow"][0]
    assert abs(result["window"][1] - result["expectedWindow"][1]) < 1e-9

    meta = result["meta"]
    shape, spacing, origin = meta["shape"], meta["spacing_m"], meta["origin_m"]
    expected_target = [origin[i] + 0.5 * shape[i] * spacing for i in range(3)]
    for got, want in zip(result["cameraTarget"], expected_target):
        assert abs(got - want) < 1e-6

    # Old hardcoded default was distance=3, target=[0,0,0]; the framing
    # distance must scale with the volume's own diagonal, not stay fixed.
    assert result["cameraDistance"] > 3
