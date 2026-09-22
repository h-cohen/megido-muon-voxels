"""Task 5: detector position markers.

run_fixture's synthetic meta has no `detectors` key, so this points at the
real `runs/voxels` reconstruction (which has P0/T20a/T20b/P1), same REAL_RUN
pattern as test_smoke.py's exit gate.
"""
from pathlib import Path

REAL_RUN = Path(__file__).resolve().parents[2] / "runs" / "voxels"


def test_detector_toggle_changes_render_with_no_console_errors(page, dist_path):
    assert (REAL_RUN / "meta.json").exists(), (
        "runs/voxels is Phase 3's real campaign output - regenerate it with "
        "`uv run python -m megido.cli reconstruct ...` before running this test"
    )
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(REAL_RUN))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready", timeout=15000)

    err = page.evaluate("() => window.__viewerError || null")
    assert err is None, err

    def canvas_data():
        return page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")

    disabled_data = canvas_data()

    page.locator("#toggle-detectors").check()
    page.wait_for_timeout(50)
    enabled_data = canvas_data()

    assert enabled_data != disabled_data, (
        "enabling #toggle-detectors must change the rendered canvas (markers drawn)"
    )

    page.locator("#toggle-detectors").uncheck()
    page.wait_for_timeout(50)
    back_off_data = canvas_data()
    assert back_off_data == disabled_data, (
        "disabling #toggle-detectors must remove the markers again"
    )

    assert console_errors == [], f"JS console errors during interaction: {console_errors}"


def test_detector_labels_track_the_two_distinct_positions(page, dist_path):
    """runs/voxels has 4 exposures (P0/T20a/T20b/P1) at 2 distinct world
    positions (origin, and (2.2, 0, 0)); expect one HTML label per distinct
    position, gated by the same #toggle-detectors checkbox as the markers."""
    assert (REAL_RUN / "meta.json").exists(), (
        "runs/voxels is Phase 3's real campaign output - regenerate it with "
        "`uv run python -m megido.cli reconstruct ...` before running this test"
    )
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(REAL_RUN))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready", timeout=15000)

    def visible_label_texts():
        return page.evaluate(
            "() => Array.from(document.querySelectorAll('#detector-labels .detector-label'))"
            ".filter(el => !el.hidden).map(el => el.textContent)"
        )

    # Off by default: no visible labels.
    assert visible_label_texts() == []

    page.locator("#toggle-detectors").check()
    page.wait_for_timeout(50)
    texts = visible_label_texts()
    assert len(texts) == 2, f"expected 2 detector labels (2 distinct positions), got {texts}"
    joined = "/".join(texts)
    assert "P0" in joined
    assert "P1" in joined

    page.locator("#toggle-detectors").uncheck()
    page.wait_for_timeout(50)
    assert visible_label_texts() == [], "unchecking #toggle-detectors must hide all labels"

    assert console_errors == [], f"JS console errors during interaction: {console_errors}"
