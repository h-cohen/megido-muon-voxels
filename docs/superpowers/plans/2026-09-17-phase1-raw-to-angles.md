# Phase 1 — Raw Data to Angular Histograms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn 70 raw CAEN `.data` files into validated per-exposure tan(θx), tan(θy) angular histograms, with every supplied detector constant falsification-tested against the data before use.

**Architecture:** A streaming pandas reader feeds a pure-numpy hit/track pipeline. Detector constants come from the engineers' files and are loaded as frozen dataclasses, never fitted. A synthetic event simulator generates data through the same geometry so the whole chain has a ground-truth gate. Outputs are two artifacts per exposure: a tracks Parquet (for the existing S-curve corrector) and a counts `.npz` (the ingest seam into the ported cafeteria solver).

**Tech Stack:** Python 3.12 (uv-managed venv), numpy, scipy, pandas + pyarrow, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md`

---

## Global Constraints

- **Package name** `megido`. In-tree execution via `python -m megido.<module>`; no console_scripts.
- **Detector constants are supplied, not fitted.** Values come from `docs/parameters.data` and `docs/detector_info.txt`. A constant that fails its validation test is escalated to the engineers, never silently replaced.
- **Bar index, never channel number.** Every position calculation uses the bar index from the §1.3 map. Any function taking a raw channel number must convert before use.
- **Preserve integer counts.** Angular histograms store raw un-normalized `int64` counts. Downstream Poisson bootstrap and MLEM both need them.
- **All physical lengths in this phase are centimetres** (matching `parameters.data`). Conversion to metres happens at the S2/S3 boundary, not here.
- **Every commit message ends with these two lines:**
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GncmtYkvUMu61VW6Ub1KP6
  ```
- **Run tests with** `uv run pytest`.

### Authoritative detector constants (spec §1.2)

| Constant | Value |
|---|---|
| `n_bars` | 23 |
| `base_cm` (bar width) | 3.2 |
| `height_cm` | 1.7 |
| `side_cm` | 2.3345 |
| `length_cm` | 40.0 |
| `layer_z_cm` (layer 0→3) | 0.0, 6.2, 31.5, 37.7 |
| `asic_to_layer` | (1, 3, 2, 0) |
| layer coordinate, bottom-up | X, Y, X, Y |
| campaign azimuth | 241° |

Derived and used throughout: `pitch_cm = base_cm / 2 = 1.6`, `active_width_cm = 22*1.6 + 3.2 = 38.4`, `Δz_x = Δz_y = 31.5`.

### Two corrections to the spec, applied in this plan

1. **There is no track χ².** Each coordinate is measured by exactly two layers, so the straight-line fit is exactly determined and χ² is identically zero. Spec §5 step 4–5 calls for a χ² cut; that cut does not exist. Quality selection comes from cluster topology and charge consistency instead. (`layers_fit_calibration/method.md` records the same degeneracy.)
2. **Angular range is ±1.25, not ±1.** Geometric acceptance reaches `|tan θ|max = active_width / Δz = 38.4 / 31.5 = 1.219`. The spec's ±1 crop, inherited from an assumed 80 cm layer separation, would discard real data. Binning is 500 × 500 over ±1.25 (0.005 tan units per bin).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `megido/__init__.py` | Version only |
| `megido/detector.py` | `BarGeometry`, `DetectorGeometry` — supplied constants, channel↔bar, layer↔ASIC, bar positions |
| `megido/config.py` | `Pose`, `PoseSigma`, `Exposure`, `Binning`, `SiteConfig` — YAML exposure registry |
| `megido/reader.py` | Streaming chunked reader for raw `.data`, header-repeat aware |
| `megido/calib.py` | S0-exp: per-channel pedestal and gain (MPV) |
| `megido/hits.py` | Threshold, cluster, channel→bar, charge-sharing position |
| `megido/tracks.py` | Two-point slope per coordinate → tan θx, tan θy |
| `megido/sim.py` | Synthetic event generator through the real geometry |
| `megido/validate.py` | S0-det falsification tests on supplied constants |
| `megido/anghist.py` | Angular histogramming, `counts_<exp>.npz` writer |
| `megido/trackfile.py` | `tracks_<exp>.parquet` writer, 7-column schema |
| `megido/pipeline.py` | Per-exposure orchestration + content-addressed artifact cache |
| `megido/cli.py` | `python -m megido.cli validate \| ingest` |
| `configs/megido.yaml` | The exposure registry |
| `tests/…` | One test module per source module |

---

## Task 1: Scaffold and detector geometry

**Files:**
- Create: `pyproject.toml`, `megido/__init__.py`, `megido/detector.py`, `tests/test_detector.py`, `.gitignore` additions

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  BarGeometry(n_bars:int, base_cm:float, height_cm:float, side_cm:float, length_cm:float)
      .pitch_cm -> float
      .active_width_cm -> float
      .bar_center_cm(bar:int) -> float
  BarGeometry.check_consistency() -> None     # raises if side != hypot(base/2, height)
  DetectorGeometry(asic_channels:dict[int,tuple[int,...]], asic_to_layer:tuple[int,...],
                   layer_z_cm:tuple[float,...], layer_coord:tuple[str,...], bar:BarGeometry)
      .channel_to_bar(asic:int, channel:int) -> int | None
      .bar_to_channel(asic:int, bar:int) -> int
      .unmapped_channels(asic:int) -> tuple[int,...]   # the 9 channels with no bar
      .layer_of_asic(asic:int) -> int
      .z_of_asic(asic:int) -> float
      .coord_of_asic(asic:int) -> str
      .asics_for_coord(coord:str) -> tuple[int,int]   # ordered by ascending z
      .dz_cm(coord:str) -> float
      .max_tan() -> float
  DetectorGeometry.megiddo() -> DetectorGeometry      # classmethod, the supplied constants
  ```

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "megido"
version = "0.1.0"
description = "Muon tomography 3D voxel reconstruction, Megiddo cavern"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "pyarrow>=14",
    "PyYAML>=6",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.setuptools.packages.find]
include = ["megido*"]
```

- [ ] **Step 2: Create the venv and install**

```bash
cd /home/hadar/Cloud/Work/Postdoc/03_projects/megido-muon-voxels
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Expected: succeeds, `.venv/` created. Append `.venv/`, `runs/`, `cache/` to `.gitignore` if not already covered.

- [ ] **Step 3: Write the failing test**

Create `tests/test_detector.py`:

```python
import pytest
from megido.detector import BarGeometry, DetectorGeometry


def test_bar_geometry_derived_values():
    b = BarGeometry()
    assert b.n_bars == 23
    assert b.base_cm == pytest.approx(3.2)
    assert b.pitch_cm == pytest.approx(1.6)
    assert b.active_width_cm == pytest.approx(38.4)
    # side must be consistent with base and height: sqrt((base/2)^2 + height^2)
    assert b.side_cm == pytest.approx(2.3345, abs=1e-3)


def test_bar_centers_are_evenly_spaced():
    b = BarGeometry()
    assert b.bar_center_cm(0) == pytest.approx(1.6)
    assert b.bar_center_cm(22) == pytest.approx(36.8)
    for i in range(22):
        assert b.bar_center_cm(i + 1) - b.bar_center_cm(i) == pytest.approx(1.6)


def test_each_asic_maps_23_distinct_channels():
    g = DetectorGeometry.megiddo()
    for asic in range(4):
        chans = g.asic_channels[asic]
        assert len(chans) == 23
        assert len(set(chans)) == 23
        assert all(0 <= c <= 31 for c in chans)


def test_channel_bar_roundtrip():
    g = DetectorGeometry.megiddo()
    for asic in range(4):
        for bar in range(23):
            ch = g.bar_to_channel(asic, bar)
            assert g.channel_to_bar(asic, ch) == bar


def test_unmapped_channels_return_none():
    g = DetectorGeometry.megiddo()
    # ASIC 0 uses 23 of 32; the other 9 must be unmapped
    unmapped = [c for c in range(32) if g.channel_to_bar(0, c) is None]
    assert len(unmapped) == 9
    assert sorted(unmapped) == [1, 2, 5, 6, 9, 10, 12, 13, 14]


def test_known_map_entries():
    g = DetectorGeometry.megiddo()
    # detector_info.txt: "channel 28 of ASIC_0 connected to bar number 1"
    assert g.bar_to_channel(0, 0) == 28
    assert g.bar_to_channel(0, 1) == 30
    assert g.bar_to_channel(3, 0) == 12


def test_layer_assignment_matches_detector_info():
    g = DetectorGeometry.megiddo()
    # asic_to_layer = [1, 3, 2, 0]; layers bottom-up are X, Y, X, Y
    assert (g.layer_of_asic(0), g.coord_of_asic(0), g.z_of_asic(0)) == (1, "y", 6.2)
    assert (g.layer_of_asic(1), g.coord_of_asic(1), g.z_of_asic(1)) == (3, "y", 37.7)
    assert (g.layer_of_asic(2), g.coord_of_asic(2), g.z_of_asic(2)) == (2, "x", 31.5)
    assert (g.layer_of_asic(3), g.coord_of_asic(3), g.z_of_asic(3)) == (0, "x", 0.0)


def test_coordinate_pairs_and_lever_arms():
    g = DetectorGeometry.megiddo()
    assert g.asics_for_coord("x") == (3, 2)   # ascending z: z=0 then z=31.5
    assert g.asics_for_coord("y") == (0, 1)   # ascending z: z=6.2 then z=37.7
    assert g.dz_cm("x") == pytest.approx(31.5)
    assert g.dz_cm("y") == pytest.approx(31.5)


def test_max_tan_is_geometric_acceptance_limit():
    g = DetectorGeometry.megiddo()
    # active_width / dz = 38.4 / 31.5
    assert g.max_tan() == pytest.approx(38.4 / 31.5, rel=1e-6)
    assert g.max_tan() > 1.0   # the spec's inherited +-1 crop would lose real data
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.detector'`

- [ ] **Step 5: Implement `megido/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 6: Implement `megido/detector.py`**

```python
"""Detector constants supplied by the engineers, 2026-09-17.

Source: docs/parameters.data, docs/detector_info.txt (filed under "# Detector 3";
the Megiddo campaign used that unit).

These values are NOT fitted. megido.validate falsification-tests each of them
against data; a failure is escalated to the engineers, never silently replaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

# Lookup tables: index is bar position (0-based), value is the DAQ channel.
# detector_info.txt: "channel 28 of ASIC_0 connected to bar number 1".
ASIC_CHANNELS: dict[int, tuple[int, ...]] = {
    0: (28, 30, 3, 31, 29, 0, 24, 26, 7, 27, 25, 4, 20, 22, 11, 23, 21, 8, 16, 18, 15, 19, 17),
    1: (3, 31, 2, 0, 28, 1, 7, 27, 6, 4, 24, 5, 11, 23, 10, 8, 20, 9, 15, 19, 14, 12, 16),
    2: (19, 15, 18, 16, 12, 17, 23, 11, 22, 20, 8, 21, 27, 7, 26, 24, 4, 25, 31, 3, 30, 28, 0),
    3: (12, 14, 19, 15, 13, 16, 8, 10, 23, 11, 9, 20, 4, 6, 27, 7, 5, 24, 0, 2, 31, 3, 1),
}

ASIC_TO_LAYER: tuple[int, ...] = (1, 3, 2, 0)
LAYER_Z_CM: tuple[float, ...] = (0.0, 6.2, 31.5, 37.7)
LAYER_COORD: tuple[str, ...] = ("x", "y", "x", "y")  # bottom-up


@dataclass(frozen=True)
class BarGeometry:
    """Triangular extruded scintillator bars, interleaved apex-up / apex-down.

    Bar `b` spans [b*pitch, b*pitch + base]; consecutive bars overlap by `pitch`.
    """

    n_bars: int = 23
    base_cm: float = 3.2
    height_cm: float = 1.7
    side_cm: float = 2.3345
    length_cm: float = 40.0

    @property
    def pitch_cm(self) -> float:
        """Centre-to-centre spacing. Interleaved triangles overlap by half a base."""
        return self.base_cm / 2.0

    @property
    def active_width_cm(self) -> float:
        return (self.n_bars - 1) * self.pitch_cm + self.base_cm

    def bar_center_cm(self, bar: int) -> float:
        if not 0 <= bar < self.n_bars:
            raise ValueError(f"bar {bar} out of range 0..{self.n_bars - 1}")
        return (bar + 1) * self.pitch_cm

    def check_consistency(self) -> None:
        """side == sqrt((base/2)^2 + height^2) for an isoceles triangle."""
        expected = hypot(self.base_cm / 2.0, self.height_cm)
        if abs(expected - self.side_cm) > 1e-3:
            raise ValueError(f"side_cm {self.side_cm} inconsistent with base/height ({expected:.4f})")


@dataclass(frozen=True)
class DetectorGeometry:
    asic_channels: dict[int, tuple[int, ...]]
    asic_to_layer: tuple[int, ...]
    layer_z_cm: tuple[float, ...]
    layer_coord: tuple[str, ...]
    bar: BarGeometry = field(default_factory=BarGeometry)

    @classmethod
    def megiddo(cls) -> "DetectorGeometry":
        bar = BarGeometry()
        bar.check_consistency()
        return cls(
            asic_channels=dict(ASIC_CHANNELS),
            asic_to_layer=ASIC_TO_LAYER,
            layer_z_cm=LAYER_Z_CM,
            layer_coord=LAYER_COORD,
            bar=bar,
        )

    def __post_init__(self) -> None:
        for asic, chans in self.asic_channels.items():
            if len(set(chans)) != self.bar.n_bars:
                raise ValueError(f"ASIC {asic}: expected {self.bar.n_bars} distinct channels")

    # --- channel <-> bar -------------------------------------------------

    def channel_to_bar(self, asic: int, channel: int) -> int | None:
        """Bar index for a DAQ channel, or None if the channel is unmapped."""
        try:
            return self.asic_channels[asic].index(channel)
        except ValueError:
            return None

    def bar_to_channel(self, asic: int, bar: int) -> int:
        return self.asic_channels[asic][bar]

    def unmapped_channels(self, asic: int) -> tuple[int, ...]:
        used = set(self.asic_channels[asic])
        return tuple(c for c in range(32) if c not in used)

    # --- layer geometry --------------------------------------------------

    def layer_of_asic(self, asic: int) -> int:
        return self.asic_to_layer[asic]

    def z_of_asic(self, asic: int) -> float:
        return self.layer_z_cm[self.layer_of_asic(asic)]

    def coord_of_asic(self, asic: int) -> str:
        return self.layer_coord[self.layer_of_asic(asic)]

    def asics_for_coord(self, coord: str) -> tuple[int, int]:
        """The two ASICs measuring `coord`, ordered by ascending z."""
        asics = [a for a in range(4) if self.coord_of_asic(a) == coord]
        if len(asics) != 2:
            raise ValueError(f"expected exactly 2 ASICs for coord {coord!r}, got {asics}")
        asics.sort(key=self.z_of_asic)
        return asics[0], asics[1]

    def dz_cm(self, coord: str) -> float:
        lo, hi = self.asics_for_coord(coord)
        return self.z_of_asic(hi) - self.z_of_asic(lo)

    def max_tan(self) -> float:
        """Geometric acceptance limit: a track must cross both layers of a coordinate."""
        return self.bar.active_width_cm / min(self.dz_cm("x"), self.dz_cm("y"))
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_detector.py -v`
Expected: 9 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml megido/ tests/test_detector.py .gitignore
git commit -m "feat: detector geometry from engineer-supplied constants"
```

---

## Task 2: Streaming raw-data reader

The files repeat their 424-column header roughly every 165 rows (338 times per file). Only `HIT_*` and `CHARGE_HG_*` are needed — 256 of 424 columns.

**Files:**
- Create: `megido/reader.py`, `tests/test_reader.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  HIT_COLUMNS: list[str]          # ["HIT_0_0", ... "HIT_3_31"], 128 names
  CHARGE_COLUMNS: list[str]       # ["CHARGE_HG_0_0", ... ], 128 names

  @dataclass(frozen=True)
  class EventChunk:
      hit: np.ndarray             # bool   [n_events, 4, 32]
      charge: np.ndarray          # int32  [n_events, 4, 32]
      n_events: int

  read_chunks(path: Path, chunksize: int = 200_000) -> Iterator[EventChunk]
  count_events(path: Path) -> int
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_reader.py`:

```python
import numpy as np
import pytest

from megido.reader import CHARGE_COLUMNS, HIT_COLUMNS, EventChunk, read_chunks


HEADER = ";".join(
    ["ID_CLUSTER", "CLUSTER_RUN_Timecode_ns", "CLUSTER_Timecode_ns", "NEventsInCluster"]
    + [f"{pre}_{a}" for a in range(4)
       for pre in ("ASIC", "EventCounter", "RUN_EventTimeCodeLSB", "RUN_EventTimecode_ns",
                   "T0_to_Event_Timecode", "T0_to_Event_Timecode_ns", "Trigger_ID",
                   "Validation_ID", "Flags")]
    + [f"HIT_{a}_{c}" for a in range(4) for c in range(32)]
    + [f"CHARGE_HG_{a}_{c}" for a in range(4) for c in range(32)]
    + [f"CHARGE_LG_{a}_{c}" for a in range(4) for c in range(32)]
)


def _row(hit_map, charge_map):
    """hit_map/charge_map: {(asic, channel): value}. Everything else zero."""
    vals = ["1", "1000", "1000", "4"] + ["0"] * 36
    vals += [str(int(hit_map.get((a, c), 0))) for a in range(4) for c in range(32)]
    vals += [str(int(charge_map.get((a, c), 2200))) for a in range(4) for c in range(32)]
    vals += ["0"] * 128
    return ";".join(vals)


@pytest.fixture
def fixture_file(tmp_path):
    """Two data rows, a REPEATED header between them, CRLF line endings."""
    p = tmp_path / "DET299999_BEAM_20260101_000000_filter.data"
    lines = [
        HEADER,
        _row({(0, 28): 1, (1, 3): 1}, {(0, 28): 5000, (1, 3): 6000}),
        HEADER,  # DAQ re-arm: header appears again mid-file
        _row({(2, 19): 1, (3, 12): 1}, {(2, 19): 7000, (3, 12): 8000}),
    ]
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    return p


def test_column_name_lists():
    assert len(HIT_COLUMNS) == 128
    assert len(CHARGE_COLUMNS) == 128
    assert HIT_COLUMNS[0] == "HIT_0_0"
    assert HIT_COLUMNS[-1] == "HIT_3_31"
    assert CHARGE_COLUMNS[0] == "CHARGE_HG_0_0"


def test_repeated_headers_are_dropped(fixture_file):
    chunks = list(read_chunks(fixture_file))
    total = sum(c.n_events for c in chunks)
    assert total == 2, "the repeated header row must not be parsed as an event"


def test_shapes_and_dtypes(fixture_file):
    c = next(iter(read_chunks(fixture_file)))
    assert c.hit.shape == (c.n_events, 4, 32)
    assert c.charge.shape == (c.n_events, 4, 32)
    assert c.hit.dtype == np.bool_
    assert c.charge.dtype == np.int32


def test_values_land_in_the_right_asic_and_channel(fixture_file):
    chunks = list(read_chunks(fixture_file))
    hit = np.concatenate([c.hit for c in chunks])
    charge = np.concatenate([c.charge for c in chunks])

    assert hit[0, 0, 28] and hit[0, 1, 3]
    assert hit[0].sum() == 2
    assert charge[0, 0, 28] == 5000
    assert charge[0, 1, 3] == 6000

    assert hit[1, 2, 19] and hit[1, 3, 12]
    assert charge[1, 3, 12] == 8000
    # unhit channels carry pedestal, not zero
    assert charge[1, 0, 0] == 2200


def test_chunking_splits_without_loss(fixture_file):
    chunks = list(read_chunks(fixture_file, chunksize=1))
    assert len(chunks) == 2
    assert all(c.n_events == 1 for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.reader'`

- [ ] **Step 3: Implement `megido/reader.py`**

```python
"""Streaming reader for CAEN DT5550W cluster dumps.

The files are semicolon-delimited ASCII with CRLF endings and 424 columns. The
header line repeats roughly every 165 rows (the DAQ re-arms), so those rows must
be dropped rather than parsed. Only HIT_* and CHARGE_HG_* are needed here.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_ASICS = 4
N_CHANNELS = 32

HIT_COLUMNS: list[str] = [f"HIT_{a}_{c}" for a in range(N_ASICS) for c in range(N_CHANNELS)]
CHARGE_COLUMNS: list[str] = [f"CHARGE_HG_{a}_{c}" for a in range(N_ASICS) for c in range(N_CHANNELS)]

_ID = "ID_CLUSTER"


@dataclass(frozen=True)
class EventChunk:
    hit: np.ndarray      # bool  [n_events, 4, 32]
    charge: np.ndarray   # int32 [n_events, 4, 32]

    @property
    def n_events(self) -> int:
        return int(self.hit.shape[0])


def read_chunks(path: Path, chunksize: int = 200_000) -> Iterator[EventChunk]:
    """Yield EventChunks. Repeated header rows are dropped."""
    usecols = [_ID] + HIT_COLUMNS + CHARGE_COLUMNS
    reader = pd.read_csv(
        path,
        sep=";",
        usecols=usecols,
        chunksize=chunksize,
        dtype=str,          # repeated header rows make numeric dtypes fail
        engine="c",
        low_memory=False,
    )
    for frame in reader:
        frame = frame[frame[_ID] != _ID]
        if frame.empty:
            continue
        hit = frame[HIT_COLUMNS].to_numpy(dtype=np.int8).reshape(-1, N_ASICS, N_CHANNELS)
        charge = frame[CHARGE_COLUMNS].to_numpy(dtype=np.int32).reshape(-1, N_ASICS, N_CHANNELS)
        yield EventChunk(hit=hit.astype(bool), charge=charge)


def count_events(path: Path) -> int:
    return sum(c.n_events for c in read_chunks(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reader.py -v`
Expected: 5 passed

- [ ] **Step 5: Smoke-test against a real file**

Run:
```bash
uv run python -c "
from pathlib import Path
from megido.reader import read_chunks
p = Path('/home/hadar/Cloud/Work/Postdoc/01_data/processed/megido/DET200084_BEAM_20260617_210539_filter.data')
n = 0
for c in read_chunks(p):
    n += c.n_events
print('events:', n)
"
```
Expected: `events: 56081`

- [ ] **Step 6: Commit**

```bash
git add megido/reader.py tests/test_reader.py
git commit -m "feat: streaming reader for raw CAEN cluster dumps"
```

---

## Task 3: Exposure registry

**Files:**
- Create: `megido/config.py`, `configs/megido.yaml`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  Pose(x:float, y:float, z:float, tilt_deg:float, az_deg:float)
      .rotation() -> np.ndarray          # 3x3, Rz(az_deg) @ Ry(tilt_deg)
  PoseSigma(xy:float=0.05, ang:float=1.0)
  Exposure(id:str, run_ids:tuple[int,...], pose:Pose, pose_sigma:PoseSigma,
           norm_group:str, note:str)
  Binning(t_max:float=1.25, n_bins:int=500)
      .edges() -> np.ndarray             # length n_bins+1
  SiteConfig(site:str, data_dir:Path, frame_origin:str, x_axis_bearing_deg:float,
             exposures:tuple[Exposure,...], binning:Binning)
      .exposure(eid:str) -> Exposure
      .files_for(eid:str) -> list[Path]  # existing files only, sorted
  load_site_config(path:Path) -> SiteConfig
  ```

- [ ] **Step 1: Create `configs/megido.yaml`**

```yaml
site: megido
data_dir: /home/hadar/Cloud/Work/Postdoc/01_data/processed/megido
frame:
  origin: P0
  x_axis_bearing_deg: 241

binning:
  t_max: 1.25
  n_bins: 500

exposures:
  - id: P0
    runs: DET200084-DET200104
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
    pose_sigma: {xy: 0.05, ang: 1.0}

  - id: T20a
    runs: DET200105-DET200116
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}
    norm_group: T20a

  - id: T20b
    runs: DET200119-DET200144
    pose: {x: 0.0, y: 0.0, z: 0.0, tilt_deg: 20, az_deg: 241}
    norm_group: T20b
    note: "-22% rate after the 20 Jul - 5 Aug gap, cause unknown"

  - id: P1
    runs: DET200145-DET200155
    pose: {x: 2.2, y: 0.0, z: 0.0, tilt_deg: 0, az_deg: 241}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_config.py`:

```python
import numpy as np
import pytest

from megido.config import Binning, Pose, load_site_config

CONFIG = "configs/megido.yaml"


def test_loads_four_exposures():
    cfg = load_site_config(CONFIG)
    assert [e.id for e in cfg.exposures] == ["P0", "T20a", "T20b", "P1"]
    assert cfg.site == "megido"
    assert cfg.x_axis_bearing_deg == 241


def test_run_ranges_are_inclusive():
    cfg = load_site_config(CONFIG)
    p0 = cfg.exposure("P0")
    assert p0.run_ids[0] == 200084
    assert p0.run_ids[-1] == 200104
    assert len(p0.run_ids) == 21


def test_norm_group_defaults_to_exposure_id():
    cfg = load_site_config(CONFIG)
    assert cfg.exposure("P0").norm_group == "P0"
    assert cfg.exposure("T20b").norm_group == "T20b"


def test_missing_run_ids_are_tolerated_not_errors():
    cfg = load_site_config(CONFIG)
    # T20b spans 200119-200144 with no gaps, but the loader must not require
    # every id in a range to exist on disk.
    files = cfg.files_for("T20b")
    assert len(files) == 26
    assert all(f.exists() for f in files)


def test_exposure_file_counts_sum_to_seventy():
    cfg = load_site_config(CONFIG)
    counts = {e.id: len(cfg.files_for(e.id)) for e in cfg.exposures}
    assert counts == {"P0": 21, "T20a": 12, "T20b": 26, "P1": 11}
    assert sum(counts.values()) == 70


def test_zero_tilt_rotation_is_pure_yaw():
    r = Pose(0, 0, 0, tilt_deg=0, az_deg=0).rotation()
    assert np.allclose(r, np.eye(3))


def test_rotation_is_orthonormal():
    r = Pose(0, 0, 0, tilt_deg=20, az_deg=241).rotation()
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_tilt_moves_the_normal_off_zenith_by_the_stated_angle():
    r = Pose(0, 0, 0, tilt_deg=20, az_deg=0).rotation()
    normal = r @ np.array([0.0, 0.0, 1.0])
    assert np.degrees(np.arccos(normal[2])) == pytest.approx(20.0)


def test_binning_edges():
    b = Binning(t_max=1.25, n_bins=500)
    e = b.edges()
    assert len(e) == 501
    assert e[0] == pytest.approx(-1.25)
    assert e[-1] == pytest.approx(1.25)
    assert (e[1] - e[0]) == pytest.approx(0.005)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.config'`

- [ ] **Step 4: Implement `megido/config.py`**

```python
"""Exposure registry. Poses live in data (configs/*.yaml), never in code.

Adding a datapoint: drop files in data_dir, append one exposure block, re-run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_RUN_RANGE = re.compile(r"^DET(\d+)\s*-\s*DET(\d+)$")


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    tilt_deg: float
    az_deg: float

    def rotation(self) -> np.ndarray:
        """R = Rz(az) @ Ry(tilt). Tilt takes the normal off zenith; az is the
        bearing of the detector's local +x (bar) axis."""
        t = np.radians(self.tilt_deg)
        a = np.radians(self.az_deg)
        ry = np.array([[np.cos(t), 0.0, np.sin(t)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(t), 0.0, np.cos(t)]])
        rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a), np.cos(a), 0.0],
                       [0.0, 0.0, 1.0]])
        return rz @ ry


@dataclass(frozen=True)
class PoseSigma:
    xy: float = 0.05     # metres
    ang: float = 1.0     # degrees


@dataclass(frozen=True)
class Exposure:
    id: str
    run_ids: tuple[int, ...]
    pose: Pose
    pose_sigma: PoseSigma = field(default_factory=PoseSigma)
    norm_group: str = ""
    note: str = ""


@dataclass(frozen=True)
class Binning:
    t_max: float = 1.25
    n_bins: int = 500

    def edges(self) -> np.ndarray:
        return np.linspace(-self.t_max, self.t_max, self.n_bins + 1)


@dataclass(frozen=True)
class SiteConfig:
    site: str
    data_dir: Path
    frame_origin: str
    x_axis_bearing_deg: float
    exposures: tuple[Exposure, ...]
    binning: Binning

    def exposure(self, eid: str) -> Exposure:
        for e in self.exposures:
            if e.id == eid:
                return e
        raise KeyError(f"no exposure {eid!r}; have {[e.id for e in self.exposures]}")

    def files_for(self, eid: str) -> list[Path]:
        """Existing data files for an exposure, sorted by run id.

        A run id in the configured range with no file on disk is skipped, not an
        error: the campaign has gaps (e.g. DET200117, DET200118).
        """
        out: list[Path] = []
        for rid in self.exposure(eid).run_ids:
            matches = sorted(self.data_dir.glob(f"DET{rid}_*.data"))
            out.extend(matches)
        return out


def _parse_runs(spec: str) -> tuple[int, ...]:
    m = _RUN_RANGE.match(str(spec).strip())
    if not m:
        raise ValueError(f"run spec {spec!r} must look like 'DET200084-DET200104'")
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi < lo:
        raise ValueError(f"run range {spec!r} is reversed")
    return tuple(range(lo, hi + 1))


def load_site_config(path: str | Path) -> SiteConfig:
    raw = yaml.safe_load(Path(path).read_text())
    frame = raw.get("frame", {})
    binning = Binning(**raw.get("binning", {}))

    exposures = []
    for block in raw["exposures"]:
        eid = block["id"]
        exposures.append(
            Exposure(
                id=eid,
                run_ids=_parse_runs(block["runs"]),
                pose=Pose(**block["pose"]),
                pose_sigma=PoseSigma(**block.get("pose_sigma", {})),
                norm_group=block.get("norm_group", eid),
                note=block.get("note", ""),
            )
        )
    return SiteConfig(
        site=raw["site"],
        data_dir=Path(raw["data_dir"]),
        frame_origin=frame.get("origin", ""),
        x_axis_bearing_deg=float(frame.get("x_axis_bearing_deg", 0.0)),
        exposures=tuple(exposures),
        binning=binning,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add megido/config.py configs/megido.yaml tests/test_config.py
git commit -m "feat: exposure registry with tilt-aware poses"
```

---

## Task 4: Per-exposure pedestal and gain calibration (S0-exp)

Pedestal comes from the `HIT=0` population; gain from the most-probable value of the `HIT=1` charge spectrum. The prior implementation measured pedestal but never subtracted it — we subtract.

**Files:**
- Create: `megido/calib.py`, `tests/test_calib.py`

**Interfaces:**
- Consumes: `megido.reader.EventChunk`
- Produces:
  ```python
  @dataclass(frozen=True)
  class ChannelCalibration:
      pedestal: np.ndarray      # float64 [4, 32]
      noise_sigma: np.ndarray   # float64 [4, 32]
      mpv: np.ndarray           # float64 [4, 32]  pedestal-subtracted
      gain: np.ndarray          # float64 [4, 32]  median(mpv) / mpv
      n_hits: np.ndarray        # int64   [4, 32]
      .threshold(n_sigma: float = 3.0) -> np.ndarray   # float64 [4,32], absolute ADC
      .save(path: Path) -> None
      .load(path: Path) -> ChannelCalibration          # staticmethod

  sigma_clipped_pedestal(values: np.ndarray, n_sigma: float = 3.0,
                         n_iter: int = 5) -> tuple[float, float]
  histogram_mpv(values: np.ndarray, bins: int = 200) -> float
  calibrate(chunks: Iterable[EventChunk]) -> ChannelCalibration
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_calib.py`:

```python
import numpy as np
import pytest

from megido.calib import (ChannelCalibration, calibrate, histogram_mpv,
                          sigma_clipped_pedestal)
from megido.reader import EventChunk


def test_sigma_clipped_pedestal_recovers_a_clean_gaussian():
    rng = np.random.default_rng(0)
    v = rng.normal(2200.0, 40.0, size=50_000)
    mean, sigma = sigma_clipped_pedestal(v)
    assert mean == pytest.approx(2200.0, abs=1.0)
    assert sigma == pytest.approx(40.0, rel=0.05)


def test_sigma_clipped_pedestal_rejects_a_signal_tail():
    """A high-side tail must not drag the pedestal up - that is why we clip."""
    rng = np.random.default_rng(1)
    core = rng.normal(2200.0, 40.0, size=50_000)
    tail = rng.normal(7000.0, 1500.0, size=5_000)
    mean, _ = sigma_clipped_pedestal(np.concatenate([core, tail]))
    assert mean == pytest.approx(2200.0, abs=5.0)
    assert np.mean(np.concatenate([core, tail])) > 2500  # the naive mean fails


def test_histogram_mpv_finds_the_landau_peak_not_the_mean():
    from scipy.stats import moyal
    rng = np.random.default_rng(2)
    v = moyal.rvs(loc=4000.0, scale=600.0, size=200_000, random_state=rng)
    mpv = histogram_mpv(v)
    assert mpv == pytest.approx(4000.0, rel=0.05)
    assert mpv < np.mean(v)   # the Landau tail pulls the mean above the MPV


def _synthetic_chunk(n=20_000, seed=0):
    """Every channel: pedestal 2200+-40; hit channels add a Moyal on top."""
    from scipy.stats import moyal
    rng = np.random.default_rng(seed)
    charge = rng.normal(2200.0, 40.0, size=(n, 4, 32))
    hit = rng.random((n, 4, 32)) < 0.05
    extra = moyal.rvs(loc=4000.0, scale=600.0, size=(n, 4, 32), random_state=rng)
    charge[hit] += extra[hit]
    return EventChunk(hit=hit, charge=charge.astype(np.int32))


def test_calibrate_recovers_pedestal_and_mpv():
    cal = calibrate([_synthetic_chunk()])
    assert cal.pedestal.shape == (4, 32)
    assert np.allclose(cal.pedestal, 2200.0, atol=5.0)
    assert np.allclose(cal.noise_sigma, 40.0, rtol=0.15)
    # mpv is pedestal-subtracted
    assert np.allclose(cal.mpv, 4000.0, rtol=0.10)


def test_gain_is_normalised_to_the_median_channel():
    cal = calibrate([_synthetic_chunk()])
    assert np.median(cal.gain) == pytest.approx(1.0, abs=0.02)
    assert np.all(cal.gain > 0)


def test_threshold_is_pedestal_plus_n_sigma():
    cal = calibrate([_synthetic_chunk()])
    thr = cal.threshold(n_sigma=3.0)
    assert np.allclose(thr, cal.pedestal + 3.0 * cal.noise_sigma)


def test_calibration_roundtrips_through_disk(tmp_path):
    cal = calibrate([_synthetic_chunk()])
    p = tmp_path / "cal.npz"
    cal.save(p)
    back = ChannelCalibration.load(p)
    assert np.allclose(back.pedestal, cal.pedestal)
    assert np.allclose(back.gain, cal.gain)
    assert np.array_equal(back.n_hits, cal.n_hits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_calib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.calib'`

- [ ] **Step 3: Implement `megido/calib.py`**

```python
"""S0-exp: per-channel pedestal and gain, computed per exposure.

Runs per exposure rather than once per detector so DAQ drift (such as the
unexplained 22% rate change on 6 Aug 2026) is caught rather than absorbed.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.reader import EventChunk

N_ASICS, N_CHANNELS = 4, 32


def sigma_clipped_pedestal(values: np.ndarray, n_sigma: float = 3.0,
                           n_iter: int = 5) -> tuple[float, float]:
    """Mean and sigma of the pedestal core, with the signal tail clipped away."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    keep = np.ones(v.shape, dtype=bool)
    mean = float(np.mean(v))
    sigma = float(np.std(v))
    for _ in range(n_iter):
        if sigma <= 0 or not np.isfinite(sigma):
            break
        new_keep = np.abs(v - mean) < n_sigma * sigma
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep
        mean = float(np.mean(v[keep]))
        sigma = float(np.std(v[keep]))
    return mean, sigma


def histogram_mpv(values: np.ndarray, bins: int = 200) -> float:
    """Most-probable value: the peak bin centre.

    Muon deposits follow a Landau with a long delta-ray tail, so the mean is
    both biased high and unstable between channels. The MPV is the stable
    estimator and is what the calibration normalises on.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 50:
        return float("nan")
    counts, edges = np.histogram(v, bins=bins)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


@dataclass(frozen=True)
class ChannelCalibration:
    pedestal: np.ndarray
    noise_sigma: np.ndarray
    mpv: np.ndarray
    gain: np.ndarray
    n_hits: np.ndarray

    def threshold(self, n_sigma: float = 3.0) -> np.ndarray:
        return self.pedestal + n_sigma * self.noise_sigma

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, pedestal=self.pedestal, noise_sigma=self.noise_sigma,
            mpv=self.mpv, gain=self.gain, n_hits=self.n_hits,
        )

    @staticmethod
    def load(path: Path) -> "ChannelCalibration":
        d = np.load(path)
        return ChannelCalibration(
            pedestal=d["pedestal"], noise_sigma=d["noise_sigma"],
            mpv=d["mpv"], gain=d["gain"], n_hits=d["n_hits"],
        )


def calibrate(chunks: Iterable[EventChunk], max_samples: int = 400_000) -> ChannelCalibration:
    """Accumulate per-channel HIT=0 and HIT=1 charge samples, then fit."""
    ped_samples: list[list[np.ndarray]] = [[[] for _ in range(N_CHANNELS)] for _ in range(N_ASICS)]
    sig_samples: list[list[np.ndarray]] = [[[] for _ in range(N_CHANNELS)] for _ in range(N_ASICS)]
    n_hits = np.zeros((N_ASICS, N_CHANNELS), dtype=np.int64)
    n_ped = np.zeros((N_ASICS, N_CHANNELS), dtype=np.int64)

    for chunk in chunks:
        for a in range(N_ASICS):
            for c in range(N_CHANNELS):
                h = chunk.hit[:, a, c]
                q = chunk.charge[:, a, c]
                n_hits[a, c] += int(h.sum())
                n_ped[a, c] += int((~h).sum())
                if n_ped[a, c] <= max_samples:
                    ped_samples[a][c].append(q[~h])
                if n_hits[a, c] <= max_samples:
                    sig_samples[a][c].append(q[h])

    pedestal = np.full((N_ASICS, N_CHANNELS), np.nan)
    noise = np.full((N_ASICS, N_CHANNELS), np.nan)
    mpv = np.full((N_ASICS, N_CHANNELS), np.nan)

    for a in range(N_ASICS):
        for c in range(N_CHANNELS):
            ped = np.concatenate(ped_samples[a][c]) if ped_samples[a][c] else np.empty(0)
            sig = np.concatenate(sig_samples[a][c]) if sig_samples[a][c] else np.empty(0)
            pedestal[a, c], noise[a, c] = sigma_clipped_pedestal(ped)
            peak = histogram_mpv(sig)
            mpv[a, c] = peak - pedestal[a, c] if np.isfinite(peak) else np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.nanmedian(mpv)
        gain = ref / mpv
    gain = np.where(np.isfinite(gain) & (gain > 0), gain, 1.0)

    return ChannelCalibration(pedestal=pedestal, noise_sigma=noise, mpv=mpv,
                              gain=gain, n_hits=n_hits)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_calib.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add megido/calib.py tests/test_calib.py
git commit -m "feat: per-exposure pedestal and gain calibration"
```

---

## Task 5: Hit finding and charge-sharing position

Position convention, fixed here and tested: bar `b` spans `[b*pitch, b*pitch + base]`, so bar centre is `(b+1)*pitch`. For a two-bar cluster on bars `(b, b+1)` with pedestal-subtracted gain-corrected charges `q_lo, q_hi`:

```
f    = q_hi / (q_lo + q_hi)
x_cm = (b + 1) * pitch_cm + pitch_cm * f
```

`f = 0` lands on the centre of bar `b`, `f = 1` on the centre of bar `b+1`, and the single-bar case `x_cm = (b+1)*pitch_cm` is the continuous limit. Residual S-curve bias is left to the existing `layers_fit_calibration` corrector.

**Files:**
- Create: `megido/hits.py`, `tests/test_hits.py`

**Interfaces:**
- Consumes: `megido.detector.DetectorGeometry`, `megido.calib.ChannelCalibration`, `megido.reader.EventChunk`
- Produces:
  ```python
  REJECT_OK = 0; REJECT_NO_HIT = 1; REJECT_UNMAPPED = 2
  REJECT_TOO_MANY = 3; REJECT_NON_ADJACENT = 4

  @dataclass(frozen=True)
  class LayerHits:
      position_cm: np.ndarray   # float64 [n_events, 4], NaN where rejected
      bar_lo: np.ndarray        # int16   [n_events, 4], -1 where rejected
      charge_lo: np.ndarray     # float64 [n_events, 4]
      charge_hi: np.ndarray     # float64 [n_events, 4], 0.0 for single-bar
      reject: np.ndarray        # uint8   [n_events, 4]

  find_hits(chunk, geom, cal, n_sigma=3.0) -> LayerHits
  hit_position_cm(bar_lo:int, q_lo:float, q_hi:float, bar:BarGeometry) -> float
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_hits.py`:

```python
import numpy as np
import pytest

from megido.calib import ChannelCalibration
from megido.detector import BarGeometry, DetectorGeometry
from megido.hits import (REJECT_NON_ADJACENT, REJECT_NO_HIT, REJECT_OK,
                         REJECT_TOO_MANY, REJECT_UNMAPPED, find_hits,
                         hit_position_cm)
from megido.reader import EventChunk


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


@pytest.fixture
def flat_cal():
    """Pedestal 2200, sigma 40, unit gain on every channel."""
    return ChannelCalibration(
        pedestal=np.full((4, 32), 2200.0),
        noise_sigma=np.full((4, 32), 40.0),
        mpv=np.full((4, 32), 4000.0),
        gain=np.ones((4, 32)),
        n_hits=np.zeros((4, 32), dtype=np.int64),
    )


# --- the position convention -------------------------------------------

def test_equal_charge_lands_midway_between_bar_centres():
    b = BarGeometry()
    x = hit_position_cm(5, 1000.0, 1000.0, b)
    assert x == pytest.approx(b.bar_center_cm(5) + b.pitch_cm / 2)


def test_all_charge_in_low_bar_lands_on_its_centre():
    b = BarGeometry()
    assert hit_position_cm(5, 1000.0, 0.0, b) == pytest.approx(b.bar_center_cm(5))


def test_all_charge_in_high_bar_lands_on_the_next_centre():
    b = BarGeometry()
    assert hit_position_cm(5, 0.0, 1000.0, b) == pytest.approx(b.bar_center_cm(6))


def test_position_is_monotonic_in_charge_fraction():
    b = BarGeometry()
    xs = [hit_position_cm(5, 1000.0 - q, q, b) for q in np.linspace(0, 1000, 11)]
    assert all(x2 > x1 for x1, x2 in zip(xs, xs[1:]))


# --- clustering ---------------------------------------------------------

def _chunk(spec, n_events=1):
    """spec: {(asic, channel): charge}. Hits are charges above 2320 ADC."""
    hit = np.zeros((n_events, 4, 32), dtype=bool)
    charge = np.full((n_events, 4, 32), 2200, dtype=np.int32)
    for (a, c), q in spec.items():
        hit[:, a, c] = True
        charge[:, a, c] = q
    return EventChunk(hit=hit, charge=charge)


def test_single_mapped_hit_is_accepted(geom, flat_cal):
    ch = geom.bar_to_channel(0, 7)
    hits = find_hits(_chunk({(0, ch): 6000}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_OK
    assert hits.bar_lo[0, 0] == 7
    assert hits.position_cm[0, 0] == pytest.approx(geom.bar.bar_center_cm(7))


def test_two_adjacent_bars_are_accepted_and_interpolated(geom, flat_cal):
    c7, c8 = geom.bar_to_channel(0, 7), geom.bar_to_channel(0, 8)
    hits = find_hits(_chunk({(0, c7): 5200, (0, c8): 5200}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_OK
    assert hits.bar_lo[0, 0] == 7
    expected = geom.bar.bar_center_cm(7) + geom.bar.pitch_cm / 2
    assert hits.position_cm[0, 0] == pytest.approx(expected)


def test_two_non_adjacent_bars_are_rejected(geom, flat_cal):
    c3, c15 = geom.bar_to_channel(0, 3), geom.bar_to_channel(0, 15)
    hits = find_hits(_chunk({(0, c3): 6000, (0, c15): 6000}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_NON_ADJACENT
    assert np.isnan(hits.position_cm[0, 0])


def test_three_hits_are_rejected(geom, flat_cal):
    chans = [geom.bar_to_channel(0, b) for b in (7, 8, 9)]
    hits = find_hits(_chunk({(0, c): 6000 for c in chans}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_TOO_MANY


def test_hit_on_an_unmapped_channel_is_rejected(geom, flat_cal):
    unmapped = geom.unmapped_channels(0)[0]
    hits = find_hits(_chunk({(0, unmapped): 6000}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_UNMAPPED


def test_no_hit_layer_is_rejected(geom, flat_cal):
    hits = find_hits(_chunk({}), geom, flat_cal)
    assert (hits.reject[0] == REJECT_NO_HIT).all()


def test_charge_below_threshold_does_not_count_as_a_hit(geom, flat_cal):
    """HIT flag set but charge inside pedestal+3sigma: not a real hit."""
    ch = geom.bar_to_channel(0, 7)
    hits = find_hits(_chunk({(0, ch): 2250}), geom, flat_cal)
    assert hits.reject[0, 0] == REJECT_NO_HIT


def test_gain_correction_is_applied(geom, flat_cal):
    """Halving one bar's gain must shift the interpolated position toward it."""
    c7, c8 = geom.bar_to_channel(0, 7), geom.bar_to_channel(0, 8)
    chunk = _chunk({(0, c7): 5200, (0, c8): 5200})
    gain = flat_cal.gain.copy()
    gain[0, c8] = 2.0     # bar 8 reads low, so its charge is scaled up
    cal2 = ChannelCalibration(flat_cal.pedestal, flat_cal.noise_sigma,
                              flat_cal.mpv, gain, flat_cal.n_hits)
    plain = find_hits(chunk, geom, flat_cal).position_cm[0, 0]
    boosted = find_hits(chunk, geom, cal2).position_cm[0, 0]
    assert boosted > plain
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.hits'`

- [ ] **Step 3: Implement `megido/hits.py`**

```python
"""Hit finding: threshold, cluster, map to bars, interpolate sub-bar position.

Everything here works in BAR index. The channel->bar map is a folded permutation
(spec 1.3); assuming channel number == bar number scores 0.2% adjacency against
44-47% for the real map, so the conversion is never optional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.calib import ChannelCalibration
from megido.detector import BarGeometry, DetectorGeometry
from megido.reader import EventChunk

REJECT_OK = 0
REJECT_NO_HIT = 1
REJECT_UNMAPPED = 2
REJECT_TOO_MANY = 3
REJECT_NON_ADJACENT = 4

REJECT_NAMES = {
    REJECT_OK: "ok",
    REJECT_NO_HIT: "no_hit",
    REJECT_UNMAPPED: "unmapped_channel",
    REJECT_TOO_MANY: "too_many_bars",
    REJECT_NON_ADJACENT: "non_adjacent_bars",
}


@dataclass(frozen=True)
class LayerHits:
    position_cm: np.ndarray
    bar_lo: np.ndarray
    charge_lo: np.ndarray
    charge_hi: np.ndarray
    reject: np.ndarray


def hit_position_cm(bar_lo: int, q_lo: float, q_hi: float, bar: BarGeometry) -> float:
    """Sub-bar position from charge sharing between bars (bar_lo, bar_lo+1).

    f = q_hi / (q_lo + q_hi) interpolates between the two bar centres, so
    f=0 -> centre of bar_lo and f=1 -> centre of bar_lo+1.
    """
    total = q_lo + q_hi
    if total <= 0:
        return float("nan")
    f = q_hi / total
    return bar.bar_center_cm(bar_lo) + bar.pitch_cm * f


def find_hits(chunk: EventChunk, geom: DetectorGeometry,
              cal: ChannelCalibration, n_sigma: float = 3.0) -> LayerHits:
    n = chunk.n_events
    position = np.full((n, 4), np.nan)
    bar_lo = np.full((n, 4), -1, dtype=np.int16)
    charge_lo = np.zeros((n, 4))
    charge_hi = np.zeros((n, 4))
    reject = np.full((n, 4), REJECT_NO_HIT, dtype=np.uint8)

    threshold = cal.threshold(n_sigma)
    # Pre-build channel -> bar lookup as an array for vectorised use.
    ch2bar = np.full((4, 32), -1, dtype=np.int16)
    for a in range(4):
        for b in range(geom.bar.n_bars):
            ch2bar[a, geom.bar_to_channel(a, b)] = b

    above = chunk.hit & (chunk.charge > threshold[None, :, :])
    corrected = (chunk.charge.astype(np.float64) - cal.pedestal[None, :, :]) * cal.gain[None, :, :]

    for a in range(4):
        fired = above[:, a, :]
        counts = fired.sum(axis=1)

        for ev in np.flatnonzero(counts > 0):
            chans = np.flatnonzero(fired[ev])
            bars = ch2bar[a, chans]
            if np.any(bars < 0):
                reject[ev, a] = REJECT_UNMAPPED
                continue
            if bars.size > 2:
                reject[ev, a] = REJECT_TOO_MANY
                continue

            order = np.argsort(bars)
            bars = bars[order]
            q = corrected[ev, a, chans[order]]

            if bars.size == 1:
                reject[ev, a] = REJECT_OK
                bar_lo[ev, a] = bars[0]
                charge_lo[ev, a] = q[0]
                position[ev, a] = geom.bar.bar_center_cm(int(bars[0]))
            elif bars[1] - bars[0] == 1:
                reject[ev, a] = REJECT_OK
                bar_lo[ev, a] = bars[0]
                charge_lo[ev, a] = q[0]
                charge_hi[ev, a] = q[1]
                position[ev, a] = hit_position_cm(int(bars[0]), q[0], q[1], geom.bar)
            else:
                reject[ev, a] = REJECT_NON_ADJACENT

    return LayerHits(position_cm=position, bar_lo=bar_lo, charge_lo=charge_lo,
                     charge_hi=charge_hi, reject=reject)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hits.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add megido/hits.py tests/test_hits.py
git commit -m "feat: hit finding with charge-sharing sub-bar interpolation"
```

---

## Task 6: Track angles

Each coordinate is measured by exactly two layers, so the slope is a two-point difference and no χ² exists. This is deliberate — see Global Constraints correction 1.

**Files:**
- Create: `megido/tracks.py`, `tests/test_tracks.py`

**Interfaces:**
- Consumes: `megido.hits.LayerHits`, `megido.detector.DetectorGeometry`
- Produces:
  ```python
  @dataclass(frozen=True)
  class Tracks:
      tan_x: np.ndarray    # float64 [n], NaN where the event failed
      tan_y: np.ndarray    # float64 [n]
      x_bot_cm: np.ndarray # float64 [n]  position in the lower X layer
      y_bot_cm: np.ndarray # float64 [n]  position in the lower Y layer
      valid: np.ndarray    # bool    [n]  all four layers accepted
      .theta_deg() -> np.ndarray
      .phi_deg() -> np.ndarray
      .n_valid -> int

  fit_tracks(hits: LayerHits, geom: DetectorGeometry) -> Tracks
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracks.py`:

```python
import numpy as np
import pytest

from megido.detector import DetectorGeometry
from megido.hits import REJECT_NO_HIT, REJECT_OK, LayerHits
from megido.tracks import Tracks, fit_tracks


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def _hits(positions, rejects=None):
    """positions: [n, 4] in ASIC order 0..3."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    rej = np.full((n, 4), REJECT_OK, dtype=np.uint8) if rejects is None else np.asarray(rejects, np.uint8)
    return LayerHits(position_cm=pos, bar_lo=np.zeros((n, 4), np.int16),
                     charge_lo=np.ones((n, 4)), charge_hi=np.zeros((n, 4)), reject=rej)


def test_vertical_track_has_zero_slope(geom):
    # ASIC order 0,1,2,3 -> coords y,y,x,x at z 6.2, 37.7, 31.5, 0.0
    hits = _hits([[19.2, 19.2, 19.2, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.valid[0]
    assert t.tan_x[0] == pytest.approx(0.0)
    assert t.tan_y[0] == pytest.approx(0.0)
    assert t.theta_deg()[0] == pytest.approx(0.0)


def test_known_x_slope_is_recovered(geom):
    """X is ASIC 3 (z=0) then ASIC 2 (z=31.5); dz = 31.5."""
    dx = 6.3
    hits = _hits([[19.2, 19.2, 19.2 + dx, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.tan_x[0] == pytest.approx(dx / 31.5)
    assert t.tan_y[0] == pytest.approx(0.0)


def test_known_y_slope_is_recovered(geom):
    """Y is ASIC 0 (z=6.2) then ASIC 1 (z=37.7); dz = 31.5."""
    dy = -3.15
    hits = _hits([[19.2, 19.2 + dy, 19.2, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.tan_y[0] == pytest.approx(dy / 31.5)
    assert t.tan_x[0] == pytest.approx(0.0)


def test_bottom_positions_come_from_the_lower_layers(geom):
    hits = _hits([[11.0, 19.2, 19.2, 5.0]])
    t = fit_tracks(hits, geom)
    assert t.x_bot_cm[0] == pytest.approx(5.0)    # ASIC 3, z=0
    assert t.y_bot_cm[0] == pytest.approx(11.0)   # ASIC 0, z=6.2


def test_event_with_a_rejected_layer_is_invalid(geom):
    rej = np.full((1, 4), REJECT_OK, np.uint8)
    rej[0, 2] = REJECT_NO_HIT
    hits = _hits([[19.2, 19.2, np.nan, 19.2]], rejects=rej)
    t = fit_tracks(hits, geom)
    assert not t.valid[0]
    assert np.isnan(t.tan_x[0])


def test_theta_and_phi_for_a_diagonal_track(geom):
    hits = _hits([[19.2, 19.2 + 31.5, 19.2 + 31.5, 19.2]])
    t = fit_tracks(hits, geom)
    assert t.tan_x[0] == pytest.approx(1.0)
    assert t.tan_y[0] == pytest.approx(1.0)
    assert t.theta_deg()[0] == pytest.approx(np.degrees(np.arctan(np.sqrt(2.0))))
    assert t.phi_deg()[0] == pytest.approx(45.0)


def test_n_valid_counts_only_complete_events(geom):
    rej = np.full((3, 4), REJECT_OK, np.uint8)
    rej[1, 0] = REJECT_NO_HIT
    hits = _hits([[19.2] * 4, [19.2] * 4, [19.2] * 4], rejects=rej)
    t = fit_tracks(hits, geom)
    assert t.n_valid == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.tracks'`

- [ ] **Step 3: Implement `megido/tracks.py`**

```python
"""Track angles from two-point slopes.

Each coordinate is measured by exactly TWO layers, so the straight-line fit is
exactly determined and chi2 is identically zero. There is no fit-quality cut to
be had here; selection comes from cluster topology in megido.hits instead.
(The same degeneracy is documented in layers_fit_calibration/method.md.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.detector import DetectorGeometry
from megido.hits import REJECT_OK, LayerHits


@dataclass(frozen=True)
class Tracks:
    tan_x: np.ndarray
    tan_y: np.ndarray
    x_bot_cm: np.ndarray
    y_bot_cm: np.ndarray
    valid: np.ndarray

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    def theta_deg(self) -> np.ndarray:
        return np.degrees(np.arctan(np.hypot(self.tan_x, self.tan_y)))

    def phi_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.tan_y, self.tan_x))


def fit_tracks(hits: LayerHits, geom: DetectorGeometry) -> Tracks:
    valid = (hits.reject == REJECT_OK).all(axis=1)

    x_lo_asic, x_hi_asic = geom.asics_for_coord("x")
    y_lo_asic, y_hi_asic = geom.asics_for_coord("y")
    dz_x = geom.dz_cm("x")
    dz_y = geom.dz_cm("y")

    x_bot = hits.position_cm[:, x_lo_asic]
    y_bot = hits.position_cm[:, y_lo_asic]

    tan_x = (hits.position_cm[:, x_hi_asic] - x_bot) / dz_x
    tan_y = (hits.position_cm[:, y_hi_asic] - y_bot) / dz_y

    tan_x = np.where(valid, tan_x, np.nan)
    tan_y = np.where(valid, tan_y, np.nan)

    return Tracks(tan_x=tan_x, tan_y=tan_y, x_bot_cm=np.where(valid, x_bot, np.nan),
                  y_bot_cm=np.where(valid, y_bot, np.nan), valid=valid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracks.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add megido/tracks.py tests/test_tracks.py
git commit -m "feat: two-point track angles, no chi2 by construction"
```

---

## Task 7: Synthetic event simulator

This is the ground-truth gate machinery. It throws muons through the real geometry and the real channel map, so the whole chain can be tested end-to-end against known angles.

**Files:**
- Create: `megido/sim.py`, `tests/test_sim.py`

**Interfaces:**
- Consumes: `megido.detector.DetectorGeometry`
- Produces:
  ```python
  @dataclass(frozen=True)
  class TruthEvents:
      chunk: EventChunk
      true_tan_x: np.ndarray   # float64 [n]
      true_tan_y: np.ndarray   # float64 [n]

  simulate(geom, n_events=20_000, seed=0, pedestal=2200.0, noise=40.0,
           mpv=4000.0, max_tan=None) -> TruthEvents
  write_raw_file(truth: TruthEvents, path: Path, header_every: int = 165) -> None
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_sim.py`:

```python
import numpy as np
import pytest

from megido.calib import calibrate
from megido.detector import DetectorGeometry
from megido.hits import find_hits
from megido.reader import read_chunks
from megido.sim import simulate, write_raw_file
from megido.tracks import fit_tracks


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def test_simulated_hits_land_only_on_mapped_channels(geom):
    truth = simulate(geom, n_events=2000, seed=0)
    for a in range(4):
        unmapped = list(geom.unmapped_channels(a))
        assert not truth.chunk.hit[:, a, unmapped].any()


def test_simulated_charges_sit_above_pedestal_where_hit(geom):
    truth = simulate(geom, n_events=2000, seed=1)
    hit = truth.chunk.hit
    assert truth.chunk.charge[hit].mean() > truth.chunk.charge[~hit].mean() + 1000


def test_pipeline_recovers_the_injected_angles(geom):
    """The end-to-end gate: raw arrays -> calib -> hits -> tracks -> truth."""
    truth = simulate(geom, n_events=20_000, seed=2)
    cal = calibrate([truth.chunk])
    tracks = fit_tracks(find_hits(truth.chunk, geom, cal), geom)

    v = tracks.valid
    assert v.sum() > 0.5 * truth.chunk.n_events, "simulator should mostly produce clean events"

    dx = tracks.tan_x[v] - truth.true_tan_x[v]
    dy = tracks.tan_y[v] - truth.true_tan_y[v]
    # 4 mm single-layer resolution over a 31.5 cm lever arm -> sigma_tan ~ 0.018
    assert np.abs(np.median(dx)) < 0.005, "no bias in tan_x"
    assert np.abs(np.median(dy)) < 0.005, "no bias in tan_y"
    assert np.std(dx) < 0.04
    assert np.std(dy) < 0.04


def test_adjacency_rate_matches_the_real_detector(geom):
    """The simulator must reproduce the 44-47% two-adjacent-bar rate."""
    truth = simulate(geom, n_events=20_000, seed=3)
    cal = calibrate([truth.chunk])
    hits = find_hits(truth.chunk, geom, cal)
    two_bar = (hits.charge_hi > 0).mean()
    assert 0.30 < two_bar < 0.70


def test_written_raw_file_roundtrips_through_the_reader(geom, tmp_path):
    truth = simulate(geom, n_events=500, seed=4)
    p = tmp_path / "DET299998_BEAM_20260101_000000_filter.data"
    write_raw_file(truth, p, header_every=50)

    back = list(read_chunks(p))
    n = sum(c.n_events for c in back)
    assert n == 500, "repeated headers must not be counted as events"
    hit = np.concatenate([c.hit for c in back])
    charge = np.concatenate([c.charge for c in back])
    assert np.array_equal(hit, truth.chunk.hit)
    assert np.array_equal(charge, truth.chunk.charge)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.sim'`

- [ ] **Step 3: Implement `megido/sim.py`**

```python
"""Synthetic events through the real geometry and the real channel map.

Used as the ground-truth gate: the full chain must recover injected angles
within resolution, and must reproduce the measured two-adjacent-bar rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import moyal

from megido.detector import DetectorGeometry
from megido.reader import CHARGE_COLUMNS, HIT_COLUMNS, EventChunk

_CLUSTER_COLS = ["ID_CLUSTER", "CLUSTER_RUN_Timecode_ns", "CLUSTER_Timecode_ns",
                 "NEventsInCluster"]
_ASIC_COLS = [f"{pre}_{a}" for a in range(4)
              for pre in ("ASIC", "EventCounter", "RUN_EventTimeCodeLSB",
                          "RUN_EventTimecode_ns", "T0_to_Event_Timecode",
                          "T0_to_Event_Timecode_ns", "Trigger_ID", "Validation_ID",
                          "Flags")]
_LG_COLS = [f"CHARGE_LG_{a}_{c}" for a in range(4) for c in range(32)]


@dataclass(frozen=True)
class TruthEvents:
    chunk: EventChunk
    true_tan_x: np.ndarray
    true_tan_y: np.ndarray


def _deposit(bar_geom, pos_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a crossing position into (bar_lo, fraction_in_high_bar).

    Inverse of megido.hits.hit_position_cm.
    """
    pitch = bar_geom.pitch_cm
    u = pos_cm / pitch - 1.0            # continuous bar coordinate
    bar_lo = np.floor(u).astype(int)
    frac = u - bar_lo
    return bar_lo, frac


def simulate(geom: DetectorGeometry, n_events: int = 20_000, seed: int = 0,
             pedestal: float = 2200.0, noise: float = 40.0, mpv: float = 4000.0,
             max_tan: float | None = None) -> TruthEvents:
    rng = np.random.default_rng(seed)
    bar = geom.bar
    if max_tan is None:
        max_tan = 0.6 * geom.max_tan()

    # Throw angles with a cos^2(theta) weighting, rejection-sampled inside max_tan.
    tan_x = np.empty(n_events)
    tan_y = np.empty(n_events)
    filled = 0
    while filled < n_events:
        need = n_events - filled
        tx = rng.uniform(-max_tan, max_tan, size=need * 3)
        ty = rng.uniform(-max_tan, max_tan, size=need * 3)
        cos_t = 1.0 / np.sqrt(1.0 + tx**2 + ty**2)
        keep = rng.random(tx.size) < cos_t**2
        tx, ty = tx[keep][:need], ty[keep][:need]
        tan_x[filled:filled + tx.size] = tx
        tan_y[filled:filled + ty.size] = ty
        filled += tx.size

    # Reference crossing point at z=0, kept well inside the active area so
    # every layer is crossed.
    margin = bar.active_width_cm * 0.25
    x0 = rng.uniform(margin, bar.active_width_cm - margin, size=n_events)
    y0 = rng.uniform(margin, bar.active_width_cm - margin, size=n_events)

    hit = np.zeros((n_events, 4, 32), dtype=bool)
    charge = rng.normal(pedestal, noise, size=(n_events, 4, 32))

    for asic in range(4):
        z = geom.z_of_asic(asic)
        coord = geom.coord_of_asic(asic)
        pos = (x0 + tan_x * z) if coord == "x" else (y0 + tan_y * z)

        bar_lo, frac = _deposit(bar, pos)
        inside = (bar_lo >= 0) & (bar_lo < bar.n_bars - 1)

        amplitude = moyal.rvs(loc=mpv, scale=mpv / 7.0, size=n_events, random_state=rng)
        amplitude = np.clip(amplitude, mpv * 0.3, mpv * 5.0)

        ev = np.flatnonzero(inside)
        for i in ev:
            b = int(bar_lo[i])
            f = float(frac[i])
            q_hi = amplitude[i] * f
            q_lo = amplitude[i] * (1.0 - f)
            # Only deposits that clear threshold register as hits.
            if q_lo > 4.0 * noise:
                c = geom.bar_to_channel(asic, b)
                hit[i, asic, c] = True
                charge[i, asic, c] += q_lo
            if q_hi > 4.0 * noise:
                c = geom.bar_to_channel(asic, b + 1)
                hit[i, asic, c] = True
                charge[i, asic, c] += q_hi

    chunk = EventChunk(hit=hit, charge=np.rint(charge).astype(np.int32))
    return TruthEvents(chunk=chunk, true_tan_x=tan_x, true_tan_y=tan_y)


def write_raw_file(truth: TruthEvents, path: Path, header_every: int = 165) -> None:
    """Write a file in the real raw format, including repeated header rows."""
    columns = _CLUSTER_COLS + _ASIC_COLS + HIT_COLUMNS + CHARGE_COLUMNS + _LG_COLS
    header = ";".join(columns)

    n = truth.chunk.n_events
    hit = truth.chunk.hit.reshape(n, -1).astype(np.int8)
    charge = truth.chunk.charge.reshape(n, -1)
    zeros_cluster = np.zeros((n, len(_CLUSTER_COLS) + len(_ASIC_COLS)), dtype=np.int64)
    zeros_cluster[:, 3] = 4          # NEventsInCluster
    zeros_lg = np.zeros((n, len(_LG_COLS)), dtype=np.int64)

    rows = np.concatenate([zeros_cluster, hit, charge, zeros_lg], axis=1)

    lines: list[str] = []
    for i in range(n):
        if i % header_every == 0:
            lines.append(header)
        lines.append(";".join(map(str, rows[i].tolist())))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(("\r\n".join(lines) + "\r\n").encode())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sim.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add megido/sim.py tests/test_sim.py
git commit -m "feat: synthetic event simulator as ground-truth gate"
```

---

## Task 8: S0-det validation suite

Falsification tests on the supplied constants. Each returns a pass/fail with the measured number, so a failure names what disagreed.

Note the two statistics deliberately **not** used, both recorded in spec §1.4: mutual information between ASIC pairs does not discriminate layer orientation on this detector, and track χ² does not exist. Position correlation is used instead.

**Files:**
- Create: `megido/validate.py`, `tests/test_validate.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  ```python
  @dataclass(frozen=True)
  class Check:
      name: str
      passed: bool
      measured: float
      expected: str
      detail: str

  adjacency_check(hits: LayerHits, min_rate: float = 0.40) -> list[Check]
  coordinate_pairing_check(hits, geom) -> Check
  active_width_check(hits, geom, tol: float = 0.10) -> Check
  acceptance_cutoff_check(tracks, geom, tol: float = 0.15) -> Check
  unmapped_loss_check(hits) -> Check
  run_all(chunk, geom, cal) -> list[Check]
  format_report(checks: list[Check]) -> str
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate.py`:

```python
import numpy as np
import pytest

from megido.calib import calibrate
from megido.detector import DetectorGeometry
from megido.hits import find_hits
from megido.sim import simulate
from megido.tracks import fit_tracks
from megido.validate import (Check, acceptance_cutoff_check, active_width_check,
                             adjacency_check, coordinate_pairing_check,
                             format_report, run_all, unmapped_loss_check)


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


@pytest.fixture
def simulated(geom):
    truth = simulate(geom, n_events=30_000, seed=7)
    cal = calibrate([truth.chunk])
    return truth, cal


def test_adjacency_passes_with_the_correct_map(geom, simulated):
    truth, cal = simulated
    checks = adjacency_check(find_hits(truth.chunk, geom, cal))
    assert len(checks) == 4
    assert all(c.passed for c in checks), [c.detail for c in checks]


def test_adjacency_fails_with_a_scrambled_map(geom, simulated):
    """The identity assumption scores ~0.2% on real data; any wrong map must fail."""
    truth, cal = simulated
    identity = {a: tuple(range(23)) for a in range(4)}
    wrong = DetectorGeometry(asic_channels=identity, asic_to_layer=geom.asic_to_layer,
                             layer_z_cm=geom.layer_z_cm, layer_coord=geom.layer_coord,
                             bar=geom.bar)
    checks = adjacency_check(find_hits(truth.chunk, wrong, cal))
    assert not all(c.passed for c in checks)


def test_coordinate_pairing_prefers_the_supplied_assignment(geom, simulated):
    truth, cal = simulated
    c = coordinate_pairing_check(find_hits(truth.chunk, geom, cal), geom)
    assert c.passed, c.detail


def test_active_width_matches_23_bars_at_the_derived_pitch(geom, simulated):
    """Spec 4.1 sub-task 0b: the occupied span must match 23 bars x 1.6 cm pitch."""
    truth, cal = simulated
    c = active_width_check(find_hits(truth.chunk, geom, cal), geom)
    assert c.measured > 0
    # the simulator confines tracks to the middle half, so the span is a lower bound
    assert c.measured <= geom.bar.active_width_cm * 1.10


def test_acceptance_cutoff_matches_geometry(geom, simulated):
    truth, cal = simulated
    tracks = fit_tracks(find_hits(truth.chunk, geom, cal), geom)
    c = acceptance_cutoff_check(tracks, geom)
    assert c.measured > 0
    # the simulator throws inside 0.6 * max_tan, so the cutoff must not EXCEED geometry
    assert c.measured <= geom.max_tan() * 1.05


def test_unmapped_loss_is_zero_in_simulation(geom, simulated):
    truth, cal = simulated
    c = unmapped_loss_check(find_hits(truth.chunk, geom, cal))
    assert c.measured == pytest.approx(0.0, abs=1e-6)


def test_run_all_returns_checks_and_formats(geom, simulated):
    truth, cal = simulated
    checks = run_all(truth.chunk, geom, cal)
    assert len(checks) >= 7
    assert all(isinstance(c, Check) for c in checks)
    report = format_report(checks)
    assert "PASS" in report or "FAIL" in report
    assert "adjacency" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.validate'`

- [ ] **Step 3: Implement `megido/validate.py`**

```python
"""S0-det: falsification tests on the engineer-supplied constants.

Each check is a test on a SUPPLIED number, not a fit. A constant that fails is
escalated to the engineers, never silently replaced.

Two statistics are deliberately not used here (spec 1.4):
  - mutual information between ASIC pairs does NOT discriminate layer
    orientation on this detector; it ranked the true same-coordinate pairs
    mid-table. Position correlation is used instead.
  - track chi2 does not exist: each coordinate has exactly two layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from megido.calib import ChannelCalibration
from megido.detector import DetectorGeometry
from megido.hits import REJECT_OK, REJECT_UNMAPPED, LayerHits, find_hits
from megido.reader import EventChunk
from megido.tracks import fit_tracks


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    measured: float
    expected: str
    detail: str


def adjacency_check(hits: LayerHits, min_rate: float = 0.40) -> list[Check]:
    """Two-hit clusters must land on ADJACENT bar indices.

    Chance level for two random distinct bars out of 23 is 2/23 = 8.7%. The
    supplied map measures 44-47% on real data; the identity assumption 0.2%.
    """
    out = []
    for asic in range(4):
        accepted = hits.reject[:, asic] == REJECT_OK
        n_acc = int(accepted.sum())
        two_bar = int(((hits.charge_hi[:, asic] > 0) & accepted).sum())
        rate = two_bar / n_acc if n_acc else 0.0
        out.append(Check(
            name=f"adjacency.asic{asic}",
            passed=rate >= min_rate,
            measured=rate,
            expected=f">= {min_rate:.0%} (chance 8.7%)",
            detail=f"ASIC {asic}: {two_bar}/{n_acc} accepted clusters span two adjacent bars",
        ))
    return out


def coordinate_pairing_check(hits: LayerHits, geom: DetectorGeometry) -> Check:
    """Same-coordinate layers must correlate in POSITION more than cross pairs."""
    pos = hits.position_cm
    ok = (hits.reject == REJECT_OK).all(axis=1)
    p = pos[ok]
    if p.shape[0] < 100:
        return Check("coordinate_pairing", False, 0.0, "n/a", "too few complete events")

    def r(a: int, b: int) -> float:
        return float(np.corrcoef(p[:, a], p[:, b])[0, 1])

    same = [geom.asics_for_coord("x"), geom.asics_for_coord("y")]
    same_r = [r(a, b) for a, b in same]
    cross = [(a, b) for a in range(4) for b in range(a + 1, 4)
             if geom.coord_of_asic(a) != geom.coord_of_asic(b)]
    cross_r = [r(a, b) for a, b in cross]

    margin = min(same_r) - max(cross_r)
    return Check(
        name="coordinate_pairing",
        passed=margin > 0.0,
        measured=margin,
        expected="min(same-coord r) > max(cross-coord r)",
        detail=(f"same-coordinate r={[round(v, 3) for v in same_r]} "
                f"{same}; cross r={[round(v, 3) for v in cross_r]} {cross}"),
    )


def active_width_check(hits: LayerHits, geom: DetectorGeometry,
                       tol: float = 0.10) -> Check:
    """Occupied hit-position span must match 23 bars at the derived pitch.

    Spec 4.1 sub-task 0b. A flat-topped occupancy with hard edges confirms both
    the bar count and the pitch; a short or long span means one of them is wrong.
    """
    ok = hits.reject == REJECT_OK
    pos = hits.position_cm[ok]
    if pos.size < 500:
        return Check("active_width", False, 0.0, "n/a", "too few accepted hits")

    lo, hi = np.quantile(pos, [0.001, 0.999])
    measured = float(hi - lo)
    expected = geom.bar.active_width_cm
    return Check(
        name="active_width",
        passed=measured <= expected * (1.0 + tol),
        measured=measured,
        expected=f"<= {expected:.1f} cm ({geom.bar.n_bars} bars x {geom.bar.pitch_cm} cm pitch)",
        detail=(f"hit positions span {measured:.2f} cm "
                f"({lo:.2f} to {hi:.2f}); geometry allows {expected:.1f} cm"),
    )


def acceptance_cutoff_check(tracks, geom: DetectorGeometry, tol: float = 0.15) -> Check:
    """The |tan| distribution must die by the geometric limit width/dz.

    A mismatch means the layer separation (31.5 cm) or the active width
    (38.4 cm) is wrong. This is the absolute angular scale, so it is a stop
    condition, not a warning.
    """
    t = np.hypot(tracks.tan_x[tracks.valid], tracks.tan_y[tracks.valid])
    if t.size < 100:
        return Check("acceptance_cutoff", False, 0.0, "n/a", "too few valid tracks")
    measured = float(np.quantile(t, 0.999))
    limit = geom.max_tan()
    return Check(
        name="acceptance_cutoff",
        passed=measured <= limit * (1.0 + tol),
        measured=measured,
        expected=f"<= {limit:.3f} (= {geom.bar.active_width_cm} cm / {geom.dz_cm('x')} cm)",
        detail=f"99.9th percentile of |tan theta| is {measured:.3f}, geometric limit {limit:.3f}",
    )


def unmapped_loss_check(hits: LayerHits) -> Check:
    """Acceptance lost to events touching unmapped channels (spec 1.3.1)."""
    n = hits.reject.shape[0]
    lost = int((hits.reject == REJECT_UNMAPPED).any(axis=1).sum())
    frac = lost / n if n else 0.0
    return Check(
        name="unmapped_loss",
        passed=True,   # reported, not gated - the cause is an open question
        measured=frac,
        expected="reported only",
        detail=f"{lost}/{n} events rejected for touching an unmapped channel",
    )


def run_all(chunk: EventChunk, geom: DetectorGeometry,
            cal: ChannelCalibration) -> list[Check]:
    hits = find_hits(chunk, geom, cal)
    tracks = fit_tracks(hits, geom)
    checks = list(adjacency_check(hits))
    checks.append(coordinate_pairing_check(hits, geom))
    checks.append(active_width_check(hits, geom))
    checks.append(acceptance_cutoff_check(tracks, geom))
    checks.append(unmapped_loss_check(hits))
    return checks


def format_report(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"[{status}] {c.name:24s} measured={c.measured:.4f}  expected {c.expected}")
        lines.append(f"         {c.detail}")
    n_fail = sum(1 for c in checks if not c.passed)
    lines.append(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validate.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add megido/validate.py tests/test_validate.py
git commit -m "feat: S0-det falsification tests on supplied constants"
```

---

## Task 9: Angular histograms and the counts artifact

`counts_<exp>.npz` is the ingest seam: `cafeteria_3d_modeling/muontomo/io.load_phantom_dir` reads exactly this format, so the ported solver needs no changes.

**Files:**
- Create: `megido/anghist.py`, `tests/test_anghist.py`

**Interfaces:**
- Consumes: `megido.tracks.Tracks`, `megido.config.Binning`
- Produces:
  ```python
  @dataclass(frozen=True)
  class AngularHist:
      values: np.ndarray    # int64 [n_bins, n_bins], tan_x along axis 0
      xedges: np.ndarray    # float64 [n_bins+1]
      yedges: np.ndarray    # float64 [n_bins+1]
      name: str = "txty"
      .total -> int
      .__add__(other) -> AngularHist

  histogram_tracks(tracks: Tracks, binning: Binning) -> AngularHist
  save_counts(hist: AngularHist, out_dir: Path, exposure_id: str,
              meta: dict) -> Path
  load_counts(path: Path) -> AngularHist
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_anghist.py`:

```python
import json

import numpy as np
import pytest

from megido.anghist import AngularHist, histogram_tracks, load_counts, save_counts
from megido.config import Binning
from megido.tracks import Tracks


def _tracks(tx, ty):
    tx = np.asarray(tx, float)
    ty = np.asarray(ty, float)
    return Tracks(tan_x=tx, tan_y=ty, x_bot_cm=np.zeros_like(tx),
                  y_bot_cm=np.zeros_like(tx), valid=np.isfinite(tx) & np.isfinite(ty))


def test_binning_shape_and_edges():
    h = histogram_tracks(_tracks([0.0], [0.0]), Binning())
    assert h.values.shape == (500, 500)
    assert h.values.dtype == np.int64
    assert len(h.xedges) == 501
    assert h.xedges[0] == pytest.approx(-1.25)
    assert h.xedges[-1] == pytest.approx(1.25)


def test_a_track_lands_in_the_expected_bin():
    b = Binning()
    h = histogram_tracks(_tracks([0.0], [0.0]), b)
    assert h.total == 1
    i = np.searchsorted(h.xedges, 0.0) - 1
    j = np.searchsorted(h.yedges, 0.0) - 1
    assert h.values[i, j] == 1


def test_tan_x_runs_along_axis_zero():
    b = Binning()
    h = histogram_tracks(_tracks([0.5], [-0.5]), b)
    i, j = np.unravel_index(int(np.argmax(h.values)), h.values.shape)
    assert h.xedges[i] <= 0.5 < h.xedges[i + 1]
    assert h.yedges[j] <= -0.5 < h.yedges[j + 1]


def test_invalid_tracks_are_excluded():
    h = histogram_tracks(_tracks([0.0, np.nan, 0.1], [0.0, 0.0, np.nan]), Binning())
    assert h.total == 1


def test_tracks_outside_the_range_are_dropped_not_clipped():
    h = histogram_tracks(_tracks([5.0, 0.0], [0.0, 0.0]), Binning())
    assert h.total == 1, "an out-of-range track must not pile up on the edge bin"


def test_counts_are_integers_and_additive():
    b = Binning()
    a = histogram_tracks(_tracks([0.0, 0.0], [0.0, 0.0]), b)
    c = histogram_tracks(_tracks([0.0], [0.0]), b)
    s = a + c
    assert s.total == 3
    assert s.values.dtype == np.int64


def test_save_and_load_roundtrip(tmp_path):
    b = Binning()
    h = histogram_tracks(_tracks([0.1, -0.2], [0.3, 0.4]), b)
    p = save_counts(h, tmp_path, "P0", meta={"exposure": "P0", "n_files": 21})
    assert p.name == "counts_P0.npz"

    back = load_counts(p)
    assert np.array_equal(back.values, h.values)
    assert np.allclose(back.xedges, h.xedges)
    assert back.name == "txty"

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["exposures"]["P0"]["n_files"] == 21


def test_saved_npz_has_the_load_phantom_dir_keys(tmp_path):
    """cafeteria io.load_phantom_dir requires exactly these array names."""
    h = histogram_tracks(_tracks([0.0], [0.0]), Binning())
    p = save_counts(h, tmp_path, "P0", meta={})
    with np.load(p) as d:
        assert set(d.files) >= {"values", "xedges", "yedges", "name"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_anghist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.anghist'`

- [ ] **Step 3: Implement `megido/anghist.py`**

```python
"""Angular histograms and the counts_<exp>.npz artifact.

This npz layout is the ingest seam: cafeteria_3d_modeling's
muontomo.io.load_phantom_dir reads {values, xedges, yedges, name} plus a
meta.json, so the ported solver consumes these files unchanged.

Counts stay raw int64. The Poisson bootstrap and MLEM both need un-normalised
integers; do not scale them here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.config import Binning
from megido.tracks import Tracks


@dataclass(frozen=True)
class AngularHist:
    values: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    name: str = "txty"

    @property
    def total(self) -> int:
        return int(self.values.sum())

    def __add__(self, other: "AngularHist") -> "AngularHist":
        if not np.allclose(self.xedges, other.xedges) or not np.allclose(self.yedges, other.yedges):
            raise ValueError("cannot add histograms with different binning")
        return AngularHist(values=self.values + other.values, xedges=self.xedges,
                           yedges=self.yedges, name=self.name)


def histogram_tracks(tracks: Tracks, binning: Binning) -> AngularHist:
    edges = binning.edges()
    v = tracks.valid & np.isfinite(tracks.tan_x) & np.isfinite(tracks.tan_y)
    values, xedges, yedges = np.histogram2d(
        tracks.tan_x[v], tracks.tan_y[v], bins=[edges, edges]
    )
    return AngularHist(values=values.astype(np.int64), xedges=xedges, yedges=yedges)


def save_counts(hist: AngularHist, out_dir: Path, exposure_id: str, meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"counts_{exposure_id}.npz"
    np.savez_compressed(path, values=hist.values, xedges=hist.xedges,
                        yedges=hist.yedges, name=np.array(hist.name))

    meta_path = out_dir / "meta.json"
    doc = json.loads(meta_path.read_text()) if meta_path.exists() else {"exposures": {}}
    doc.setdefault("exposures", {})[exposure_id] = dict(meta, total_counts=hist.total)
    meta_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


def load_counts(path: Path) -> AngularHist:
    with np.load(path) as d:
        return AngularHist(values=d["values"], xedges=d["xedges"],
                           yedges=d["yedges"], name=str(d["name"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_anghist.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add megido/anghist.py tests/test_anghist.py
git commit -m "feat: angular histograms and counts npz ingest seam"
```

---

## Task 10: Tracks Parquet artifact

Emits the 7-column schema `layers_fit_calibration` requires, so its self-supervised S-curve corrector plugs in with no integration work.

**Files:**
- Create: `megido/trackfile.py`, `tests/test_trackfile.py`

**Interfaces:**
- Consumes: `megido.hits.LayerHits`, `megido.detector.DetectorGeometry`
- Produces:
  ```python
  TRACK_SCHEMA: tuple[str, ...] = ("track_id", "layer", "bar_index", "x_formula",
                                   "charge_n", "charge_N", "z_layer")
  tracks_to_table(hits, geom, first_track_id: int = 0) -> pa.Table
  open_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_trackfile.py`:

```python
import numpy as np
import pyarrow.parquet as pq
import pytest

from megido.detector import DetectorGeometry
from megido.hits import REJECT_NO_HIT, REJECT_OK, LayerHits
from megido.trackfile import TRACK_SCHEMA, open_writer, tracks_to_table


@pytest.fixture
def geom():
    return DetectorGeometry.megiddo()


def _hits(n=2, reject=None):
    rej = np.full((n, 4), REJECT_OK, np.uint8) if reject is None else reject
    return LayerHits(
        position_cm=np.full((n, 4), 19.2),
        bar_lo=np.full((n, 4), 11, np.int16),
        charge_lo=np.full((n, 4), 3000.0),
        charge_hi=np.full((n, 4), 1000.0),
        reject=rej,
    )


def test_schema_matches_layers_fit_calibration(geom):
    t = tracks_to_table(_hits(), geom)
    assert tuple(t.column_names) == TRACK_SCHEMA


def test_one_row_per_layer_per_valid_event(geom):
    t = tracks_to_table(_hits(n=3), geom)
    assert t.num_rows == 3 * 4


def test_invalid_events_are_excluded(geom):
    rej = np.full((3, 4), REJECT_OK, np.uint8)
    rej[1, 2] = REJECT_NO_HIT
    t = tracks_to_table(_hits(n=3, reject=rej), geom)
    assert t.num_rows == 2 * 4


def test_bar_index_is_the_physical_bar_not_the_channel(geom):
    """The whole point of the map: bar_index must never be a DAQ channel."""
    t = tracks_to_table(_hits(), geom).to_pydict()
    assert set(t["bar_index"]) == {11}
    # channel 11 maps to a different bar on every ASIC, so a channel would leak through
    assert geom.channel_to_bar(0, 11) != 11


def test_z_layer_uses_supplied_geometry(geom):
    t = tracks_to_table(_hits(n=1), geom).to_pydict()
    assert sorted(set(t["z_layer"])) == [0.0, 6.2, 31.5, 37.7]


def test_layer_column_is_the_layer_index_not_the_asic(geom):
    t = tracks_to_table(_hits(n=1), geom).to_pydict()
    assert sorted(set(t["layer"])) == [0, 1, 2, 3]


def test_track_ids_are_unique_and_offset(geom):
    t = tracks_to_table(_hits(n=2), geom, first_track_id=100).to_pydict()
    assert sorted(set(t["track_id"])) == [100, 101]


def test_roundtrip_through_parquet(tmp_path, geom):
    p = tmp_path / "tracks_P0.parquet"
    table = tracks_to_table(_hits(n=5), geom)
    w = open_writer(p, table.schema)
    w.write_table(table)
    w.close()

    back = pq.read_table(p)
    assert back.num_rows == 20
    assert tuple(back.column_names) == TRACK_SCHEMA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trackfile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.trackfile'`

- [ ] **Step 3: Implement `megido/trackfile.py`**

```python
"""tracks_<exp>.parquet - the 7-column schema layers_fit_calibration consumes.

Emitting this schema means its self-supervised S-curve corrector (LOLO) plugs in
with no integration work: it requires exactly
{track_id, layer, bar_index, x_formula, charge_n, charge_N, z_layer}.

bar_index is the PHYSICAL bar from the channel map, never a DAQ channel number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from megido.detector import DetectorGeometry
from megido.hits import REJECT_OK, LayerHits

TRACK_SCHEMA: tuple[str, ...] = (
    "track_id", "layer", "bar_index", "x_formula", "charge_n", "charge_N", "z_layer",
)


def tracks_to_table(hits: LayerHits, geom: DetectorGeometry,
                    first_track_id: int = 0) -> pa.Table:
    valid = (hits.reject == REJECT_OK).all(axis=1)
    ev = np.flatnonzero(valid)
    n = ev.size

    track_id = np.repeat(first_track_id + np.arange(n, dtype=np.int64), 4)
    asics = np.tile(np.arange(4), n)

    layer = np.array([geom.layer_of_asic(int(a)) for a in asics], dtype=np.int16)
    z_layer = np.array([geom.z_of_asic(int(a)) for a in asics], dtype=np.float64)

    bar_index = hits.bar_lo[ev].reshape(-1).astype(np.int16)
    x_formula = hits.position_cm[ev].reshape(-1).astype(np.float64)
    charge_n = hits.charge_lo[ev].reshape(-1).astype(np.float64)
    charge_N = hits.charge_hi[ev].reshape(-1).astype(np.float64)

    return pa.table({
        "track_id": track_id,
        "layer": layer,
        "bar_index": bar_index,
        "x_formula": x_formula,
        "charge_n": charge_n,
        "charge_N": charge_N,
        "z_layer": z_layer,
    })


def open_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, schema, compression="zstd")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trackfile.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add megido/trackfile.py tests/test_trackfile.py
git commit -m "feat: tracks parquet in the layers_fit_calibration schema"
```

---

## Task 11: Pipeline orchestration and artifact cache

Adding one exposure must not reprocess 6.2 GB. Artifacts are keyed by `sha256(input file list + mtimes + relevant config)`.

**Files:**
- Create: `megido/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  ```python
  def exposure_key(cfg: SiteConfig, eid: str) -> str          # 16-hex digest
  @dataclass(frozen=True)
  class ExposureResult:
      exposure_id: str
      key: str
      n_events: int
      n_valid: int
      counts_path: Path
      tracks_path: Path
      cached: bool
  process_exposure(cfg, eid, out_dir: Path, geom=None, force=False) -> ExposureResult
  process_all(cfg, out_dir: Path, force=False) -> list[ExposureResult]
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pytest
import yaml

from megido.config import load_site_config
from megido.detector import DetectorGeometry
from megido.pipeline import exposure_key, process_exposure
from megido.sim import simulate, write_raw_file


@pytest.fixture
def fake_site(tmp_path):
    """A two-run synthetic exposure plus a config pointing at it."""
    geom = DetectorGeometry.megiddo()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for rid in (200001, 200002):
        truth = simulate(geom, n_events=3000, seed=rid)
        write_raw_file(truth, data_dir / f"DET{rid}_BEAM_20260101_000000_filter.data")

    cfg_path = tmp_path / "site.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(data_dir),
        "frame": {"origin": "E0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 100},
        "exposures": [{
            "id": "E0",
            "runs": "DET200001-DET200002",
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241},
        }],
    }))
    return cfg_path, tmp_path / "out", data_dir


def test_processes_all_files_in_an_exposure(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    r = process_exposure(cfg, "E0", out)
    assert r.n_events == 6000
    assert r.n_valid > 0
    assert r.counts_path.exists()
    assert r.tracks_path.exists()
    assert not r.cached


def test_second_run_hits_the_cache(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    first = process_exposure(cfg, "E0", out)
    second = process_exposure(cfg, "E0", out)
    assert second.cached
    assert second.key == first.key
    assert second.n_valid == first.n_valid


def test_force_bypasses_the_cache(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    process_exposure(cfg, "E0", out)
    again = process_exposure(cfg, "E0", out, force=True)
    assert not again.cached


def test_key_changes_when_binning_changes(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    k1 = exposure_key(cfg, "E0")

    raw = yaml.safe_load(cfg_path.read_text())
    raw["binning"]["n_bins"] = 200
    cfg_path.write_text(yaml.safe_dump(raw))
    k2 = exposure_key(load_site_config(cfg_path), "E0")

    assert k1 != k2, "changing binning must invalidate the artifact"


def test_key_changes_when_a_new_file_appears(fake_site):
    cfg_path, out, data_dir = fake_site
    cfg = load_site_config(cfg_path)
    k1 = exposure_key(cfg, "E0")

    geom = DetectorGeometry.megiddo()
    truth = simulate(geom, n_events=100, seed=999)
    write_raw_file(truth, data_dir / "DET200002_BEAM_20260102_000000_filter.data")

    assert exposure_key(load_site_config(cfg_path), "E0") != k1


def test_counts_total_matches_valid_tracks_in_range(fake_site):
    cfg_path, out, _ = fake_site
    cfg = load_site_config(cfg_path)
    r = process_exposure(cfg, "E0", out)
    from megido.anghist import load_counts
    h = load_counts(r.counts_path)
    assert 0 < h.total <= r.n_valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.pipeline'`

- [ ] **Step 3: Implement `megido/pipeline.py`**

```python
"""Per-exposure orchestration with a content-addressed artifact cache.

6.2 GB of ASCII makes full reprocessing unacceptable for a one-exposure
addition, so artifacts are keyed by the input file list, their sizes and mtimes,
and the config fields that actually affect the output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from megido.anghist import AngularHist, histogram_tracks, save_counts
from megido.calib import calibrate
from megido.config import SiteConfig
from megido.detector import DetectorGeometry
from megido.hits import find_hits
from megido.reader import read_chunks
from megido.trackfile import open_writer, tracks_to_table
from megido.tracks import fit_tracks


@dataclass(frozen=True)
class ExposureResult:
    exposure_id: str
    key: str
    n_events: int
    n_valid: int
    counts_path: Path
    tracks_path: Path
    cached: bool


def exposure_key(cfg: SiteConfig, eid: str) -> str:
    exp = cfg.exposure(eid)
    parts = [eid, repr(exp.pose), cfg.binning.t_max, cfg.binning.n_bins]
    for f in cfg.files_for(eid):
        st = f.stat()
        parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
    blob = "|".join(str(p) for p in parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def process_exposure(cfg: SiteConfig, eid: str, out_dir: Path,
                     geom: DetectorGeometry | None = None,
                     force: bool = False) -> ExposureResult:
    geom = geom or DetectorGeometry.megiddo()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = exposure_key(cfg, eid)
    stamp = out_dir / f".key_{eid}.json"
    counts_path = out_dir / f"counts_{eid}.npz"
    tracks_path = out_dir / f"tracks_{eid}.parquet"

    if not force and stamp.exists() and counts_path.exists() and tracks_path.exists():
        prev = json.loads(stamp.read_text())
        if prev.get("key") == key:
            return ExposureResult(eid, key, prev["n_events"], prev["n_valid"],
                                  counts_path, tracks_path, cached=True)

    files = cfg.files_for(eid)
    if not files:
        raise FileNotFoundError(f"exposure {eid!r} has no data files in {cfg.data_dir}")

    # Pass 1: calibration needs the whole exposure before hits can be found.
    cal = calibrate(chunk for f in files for chunk in read_chunks(f))

    # Pass 2: hits, tracks, artifacts.
    edges = cfg.binning.edges()
    total = AngularHist(values=np.zeros((cfg.binning.n_bins, cfg.binning.n_bins), np.int64),
                        xedges=edges, yedges=edges)
    n_events = n_valid = 0
    writer = None
    try:
        for f in files:
            for chunk in read_chunks(f):
                hits = find_hits(chunk, geom, cal)
                tracks = fit_tracks(hits, geom)
                total = total + histogram_tracks(tracks, cfg.binning)

                table = tracks_to_table(hits, geom, first_track_id=n_valid)
                if writer is None:
                    writer = open_writer(tracks_path, table.schema)
                if table.num_rows:
                    writer.write_table(table)

                n_events += chunk.n_events
                n_valid += tracks.n_valid
    finally:
        if writer is not None:
            writer.close()

    cal.save(out_dir / f"calib_{eid}.npz")
    save_counts(total, out_dir, eid, meta={
        "exposure": eid,
        "n_files": len(files),
        "n_events": n_events,
        "n_valid_tracks": n_valid,
        "pose": cfg.exposure(eid).pose.__dict__,
        "norm_group": cfg.exposure(eid).norm_group,
    })
    stamp.write_text(json.dumps({"key": key, "n_events": n_events, "n_valid": n_valid}))

    return ExposureResult(eid, key, n_events, n_valid, counts_path, tracks_path, cached=False)


def process_all(cfg: SiteConfig, out_dir: Path, force: bool = False) -> list[ExposureResult]:
    return [process_exposure(cfg, e.id, out_dir, force=force) for e in cfg.exposures]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add megido/pipeline.py tests/test_pipeline.py
git commit -m "feat: exposure pipeline with content-addressed artifact cache"
```

---

## Task 12: CLI, real-data run, and the Phase 1 gate report

**Files:**
- Create: `megido/cli.py`, `tests/test_cli.py`, `docs/phase1-validation-report.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `megido.pipeline`, `megido.validate`
- Produces: `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import pytest
import yaml

from megido.cli import main
from megido.detector import DetectorGeometry
from megido.sim import simulate, write_raw_file


@pytest.fixture
def site(tmp_path):
    geom = DetectorGeometry.megiddo()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    truth = simulate(geom, n_events=3000, seed=11)
    write_raw_file(truth, data_dir / "DET200001_BEAM_20260101_000000_filter.data")

    cfg = tmp_path / "site.yaml"
    cfg.write_text(yaml.safe_dump({
        "site": "test",
        "data_dir": str(data_dir),
        "frame": {"origin": "E0", "x_axis_bearing_deg": 241},
        "binning": {"t_max": 1.25, "n_bins": 100},
        "exposures": [{
            "id": "E0",
            "runs": "DET200001-DET200001",
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "tilt_deg": 0, "az_deg": 241},
        }],
    }))
    return cfg, tmp_path / "out"


def test_ingest_returns_zero_and_writes_artifacts(site, capsys):
    cfg, out = site
    rc = main(["ingest", "--config", str(cfg), "--out", str(out)])
    assert rc == 0
    assert (out / "counts_E0.npz").exists()
    assert (out / "tracks_E0.parquet").exists()
    assert "E0" in capsys.readouterr().out


def test_validate_prints_a_report(site, capsys):
    cfg, out = site
    rc = main(["validate", "--config", str(cfg), "--exposure", "E0"])
    captured = capsys.readouterr().out
    assert "adjacency" in captured
    assert "acceptance_cutoff" in captured
    assert rc in (0, 1)


def test_validate_returns_one_when_a_check_fails(site, capsys, monkeypatch):
    cfg, _ = site
    from megido import validate as V
    monkeypatch.setattr(
        V, "run_all",
        lambda *a, **k: [V.Check("forced", False, 0.0, "never", "forced failure")],
    )
    assert main(["validate", "--config", str(cfg), "--exposure", "E0"]) == 1


def test_unknown_command_returns_nonzero():
    with pytest.raises(SystemExit):
        main(["nonsense"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'megido.cli'`

- [ ] **Step 3: Implement `megido/cli.py`**

```python
"""python -m megido.cli validate|ingest"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from megido import validate as V
from megido.calib import calibrate
from megido.config import load_site_config
from megido.detector import DetectorGeometry
from megido.pipeline import process_all
from megido.reader import read_chunks

_VALIDATE_EVENTS = 200_000


def _cmd_validate(args) -> int:
    cfg = load_site_config(args.config)
    geom = DetectorGeometry.megiddo()
    files = cfg.files_for(args.exposure)
    if not files:
        print(f"no files for exposure {args.exposure!r}")
        return 1

    chunks = list(itertools.islice(read_chunks(files[0]), args.max_chunks))
    cal = calibrate(chunks)

    import numpy as np
    from megido.reader import EventChunk
    merged = EventChunk(hit=np.concatenate([c.hit for c in chunks]),
                        charge=np.concatenate([c.charge for c in chunks]))

    checks = V.run_all(merged, geom, cal)
    print(f"Exposure {args.exposure}  file {files[0].name}  events {merged.n_events}\n")
    print(V.format_report(checks))
    return 0 if all(c.passed for c in checks) else 1


def _cmd_ingest(args) -> int:
    cfg = load_site_config(args.config)
    for r in process_all(cfg, Path(args.out), force=args.force):
        tag = "cached" if r.cached else "built"
        rate = r.n_valid / r.n_events if r.n_events else 0.0
        print(f"{r.exposure_id:6s} {tag:6s} key={r.key}  events={r.n_events:>9,}  "
              f"valid={r.n_valid:>9,} ({rate:.1%})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="megido")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="S0-det checks on the supplied constants")
    v.add_argument("--config", default="configs/megido.yaml")
    v.add_argument("--exposure", default="P0")
    v.add_argument("--max-chunks", type=int, default=1)
    v.set_defaults(func=_cmd_validate)

    i = sub.add_parser("ingest", help="S0-exp + S1 for every exposure")
    i.add_argument("--config", default="configs/megido.yaml")
    i.add_argument("--out", default="runs/ingest")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=_cmd_ingest)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass (approximately 80 across 11 modules)

- [ ] **Step 6: Run validation against real P0 data — the Phase 1 gate**

Run:
```bash
uv run python -m megido.cli validate --config configs/megido.yaml --exposure P0 --max-chunks 1
```

Expected, from the measurements already taken during design:
- `adjacency.asic0..3` — PASS, each measured 0.40–0.55
- `coordinate_pairing` — PASS, positive margin
- `active_width` — measured ≤ 38.4 cm × 1.10
- `acceptance_cutoff` — measured ≤ 1.219 × 1.15
- `unmapped_loss` — reports roughly 0.20

**If `acceptance_cutoff` fails, STOP.** It means the 31.5 cm layer separation or the 38.4 cm active width is wrong, and that is the absolute angular scale of everything downstream. Escalate to the engineers rather than adjusting the constant.

**If `adjacency` fails**, the channel map is not being applied correctly — check that `bar_index`, not channel number, reaches `hit_position_cm`.

- [ ] **Step 7: Write the gate report**

Create `docs/phase1-validation-report.md` containing: the command run, the full `format_report` output pasted verbatim, the measured adjacency rate per ASIC, the measured acceptance cutoff against the 1.219 geometric limit, the unmapped-channel acceptance loss, and one paragraph stating whether Phase 1's exit gate is met.

- [ ] **Step 8: Full ingest over all four exposures**

Run:
```bash
uv run python -m megido.cli ingest --config configs/megido.yaml --out runs/ingest
```

Expected: four lines, one per exposure, with `events` summing to roughly 3.9 M and `valid` a substantial fraction. Runtime is tens of minutes; artifacts are cached afterwards. Confirm `runs/ingest/` holds `counts_{P0,T20a,T20b,P1}.npz`, matching `tracks_*.parquet`, `calib_*.npz`, and one `meta.json`.

- [ ] **Step 9: Update README**

Replace `README.md` with the project name, a one-paragraph description, a pointer to the spec and this plan, the install commands (`uv venv --python 3.12`, `uv pip install -e ".[dev]"`), and the two CLI invocations.

- [ ] **Step 10: Commit**

```bash
git add megido/cli.py tests/test_cli.py docs/phase1-validation-report.md README.md
git commit -m "feat: CLI, real-data validation gate, Phase 1 report"
```

---

## Phase 1 exit criteria

- [ ] `uv run pytest` green
- [ ] `adjacency` ≥ 40 % on all four ASICs against real P0 data
- [ ] `active_width` consistent with 23 bars at 1.6 cm pitch (38.4 cm)
- [ ] `acceptance_cutoff` consistent with 38.4 cm / 31.5 cm = 1.219
- [ ] `coordinate_pairing` margin positive
- [ ] Simulator round-trip recovers injected angles with no bias and σ(tan) < 0.04
- [ ] `counts_<exp>.npz` written for P0, T20a, T20b, P1, readable by `load_phantom_dir`
- [ ] `tracks_<exp>.parquet` matches the `layers_fit_calibration` 7-column schema
- [ ] Unmapped-channel acceptance loss quantified and recorded
- [ ] `docs/phase1-validation-report.md` committed

Phase 2 (baseline solve) begins from `counts_<exp>.npz` and the measured acceptance cutoff.
