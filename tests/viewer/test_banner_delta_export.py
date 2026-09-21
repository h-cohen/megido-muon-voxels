"""Task 9: resolution banner, run-delta layer, PNG export.

`run_fixture` (see conftest.py) always writes to `tmp_path / "run"`, so
calling it twice in one test collides on the same directory (the second
`run.mkdir()` raises `FileExistsError`). To get two independent run
directories on disk, this test calls `run_fixture` once for run A, then
copies that directory tree to a sibling `run_b` directory for run B rather
than calling the fixture factory again.
"""
import shutil

from megido.viewerbuild import build  # noqa: F401  (dist_path fixture uses this)


def test_banner_delta_and_export(page, dist_path, run_fixture, tmp_path):
    run_a = run_fixture(depth_resolved=False)
    run_b = tmp_path / "run_b"
    shutil.copytree(run_a, run_b)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    page.locator("#load-run-input").set_input_files(str(run_a))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.meta")

    banner_text = page.locator("#resolution-banner").text_content()
    assert "depth NOT resolved" in banner_text

    page.locator("#load-second-run-input").set_input_files(str(run_b))
    page.wait_for_function("() => window.__viewerState.layerData.has('delta')")
    assert page.locator("#layer-delta").count() == 1
    assert "delta:" in page.locator("#delta-verdict").text_content()

    with page.expect_download() as dl_info:
        page.locator("#export-png-btn").click()
    download = dl_info.value
    assert download.suggested_filename == "megiddo-voxel-view.png"
