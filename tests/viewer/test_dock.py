"""Task 1: left dock sections collapse and persist their open/closed state."""


def test_dock_sections_collapse_and_persist(page, dist_path, run_fixture):
    run = run_fixture()
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#dock")
    page.locator("#load-run-input").set_input_files(str(run))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    sec = page.locator('[data-section="sec-clip"]')
    assert sec.get_attribute("data-open") == "true"
    sec.locator(".section-header").click()
    assert sec.get_attribute("data-open") == "false"

    # persisted across reload
    page.reload()
    page.wait_for_selector('[data-section="sec-clip"]')
    assert page.locator('[data-section="sec-clip"]').get_attribute("data-open") == "false"
