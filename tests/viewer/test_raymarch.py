from pathlib import Path


def test_canvas_renders_non_blank_after_loading_a_run(page, dist_path, run_fixture):
    run_dir = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # This Playwright version (1.63) requires a directory path for a
    # webkitdirectory input (older versions wanted the individual files);
    # pass the run directory itself.
    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    err = page.evaluate("() => window.__viewerError || null")
    assert err is None, err

    data_url = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    blank = page.evaluate("""
        () => {
          const c = document.createElement('canvas');
          c.width = document.querySelector('#gl-canvas').width;
          c.height = document.querySelector('#gl-canvas').height;
          return c.toDataURL();
        }
    """)
    assert data_url != blank
