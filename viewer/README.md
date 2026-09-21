# Megiddo voxel viewer

Self-contained WebGL2 raymarching viewer for `volume.npy` + `meta.json`
(Phase 3's export contract, `megido/volexport.py`). No runtime dependencies,
no network requests, works from `file://`.

## Build

    uv run python -m megido.cli view

Writes `viewer/dist/index.html` and opens it. `--no-open` skips the browser.

## Test

    node --test viewer/test/*.test.mjs   # pure-logic unit tests
    uv run pytest tests/test_viewerbuild.py tests/viewer -v   # build + Playwright
    uv run playwright install chromium   # one-time, before the Playwright tests

## Using it

"Load run" picks a run directory (e.g. `runs/voxels`, produced by
`uv run python -m megido.cli reconstruct ...`) via a directory file picker;
the viewer reads `meta.json` and every `.npy` file it names. The resolution
banner at the top always shows whether depth is resolved for that run —
for the current campaign it is not, and the viewer will say so in red.
