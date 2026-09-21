"""Task 10: combined smoke test on the real campaign output.

This is Phase 4's exit gate: every control from Tasks 3-9 is exercised in one
page session against the real `runs/voxels` reconstruction (Phase 3's actual
output, not a synthetic fixture) with zero console errors.
"""
from pathlib import Path

REAL_RUN = Path(__file__).resolve().parents[2] / "runs" / "voxels"


def test_real_campaign_output_renders_with_every_control_operable(page, dist_path):
    assert (REAL_RUN / "meta.json").exists(), (
        "runs/voxels is Phase 3's real campaign output - regenerate it with "
        "`uv run python -m megido.cli reconstruct ...` (see docs/phase3-reconstruction-report.md) "
        "before running this gate"
    )
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # Directory-picker upload: pass the directory path, not a filtered file
    # list. The app reads files by basename from meta.layers, so the extra
    # .npz files in runs/voxels are harmless (ignored).
    page.locator("#load-run-input").set_input_files(str(REAL_RUN))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready", timeout=15000)

    err = page.evaluate("() => window.__viewerError || null")
    assert err is None, err

    canvas_data = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    blank = page.evaluate("""
        () => { const c = document.createElement('canvas');
                c.width = document.querySelector('#gl-canvas').width;
                c.height = document.querySelector('#gl-canvas').height;
                return c.toDataURL(); }
    """)
    assert canvas_data != blank, "canvas must be non-blank on real reconstructed output"

    banner = page.locator("#resolution-banner").text_content()
    assert "depth NOT resolved" in banner, (
        "the real campaign's honest verdict must reach the viewer unedited"
    )

    # backprojection.npy is a 2D (58, 50) diagnostic plane, not a volume grid
    # layer (nx*ny*nz elements); loadRun correctly skips it, so it is not
    # asserted here.
    for key in ("volume", "sigma", "snr", "views", "systematic"):
        assert page.locator(f"#layer-{key}").count() == 1, f"missing layer control for {key}"
        page.locator(f"#layer-{key}").check()

    page.locator("#camera-preset-top").click()
    page.locator("#camera-preset-iso").click()

    box = page.locator("#xfer-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5)
    page.mouse.up()

    hist_box = page.locator("#histogram-canvas").bounding_box()
    page.mouse.move(hist_box["x"] + hist_box["width"] * 0.15, hist_box["y"] + hist_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(hist_box["x"] + hist_box["width"] * 0.35, hist_box["y"] + hist_box["height"] / 2)
    page.mouse.up()
    readout = page.locator("#window-readout").text_content()
    assert readout and readout.strip(), "window readout must reflect the dragged window"

    page.locator("#clip-x-max").fill("0.6")
    page.locator("#clip-x-max").dispatch_event("input")
    page.locator("#clip-plane-enabled").check()
    page.locator("#slice-axis").select_option("z")
    page.locator("#slice-pos").fill("0.5")
    page.locator("#slice-pos").dispatch_event("input")

    page.locator("#sigma-gate-enabled").check()
    page.locator("#sigma-gate-value").fill("0.5")
    page.locator("#sigma-gate-value").dispatch_event("input")

    canvas_box = page.locator("#gl-canvas").bounding_box()
    page.mouse.move(canvas_box["x"] + canvas_box["width"] / 2,
                     canvas_box["y"] + canvas_box["height"] / 2)
    page.wait_for_timeout(100)

    with page.expect_download():
        page.locator("#export-png-btn").click()

    assert console_errors == [], f"JS console errors during interaction: {console_errors}"
