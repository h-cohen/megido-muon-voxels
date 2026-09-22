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

The left dock is the instrument panel. Each section (Layers, Transfer, Clip &
slice, Uncertainty, Camera & view) collapses independently by clicking its
header, and the open/closed state is remembered per browser. Layers picks
which volume is shown (combined solve, sigma, SNR, view count, gauge
systematic); Transfer sets the colormap and the density-to-color window — drag
inside the histogram to move the window band, or drag the transfer editor
handles directly. Clip & slice cuts a sub-box or a single-plane slice out of
the volume; Uncertainty can hide voxels above a sigma threshold. Camera & view
has orbit presets (top/front/side/iso), a "Frame all" button, and saved
views: name the current camera + window + active layer and "Save view" to add
it to the list, then "Apply" to jump back to it later (persisted across
reloads) or "Delete" to remove it.

Around the canvas: a small axis gizmo (bottom-left) shows the current
orientation, a colorbar legend (bottom-right) shows the active layer's units
and range, and hovering the volume shows the voxel index and value under the
cursor.

Keyboard shortcuts (ignored while a text field has focus): digits `1`-`5`
switch layers, `t`/`f`/`s`/`i` jump to the top/front/side/iso camera presets,
`r` frames all, `?` opens the shortcuts cheatsheet, and `Escape` closes any
open overlay. A one-time onboarding card explains the basics on first load
and does not reappear once dismissed.
