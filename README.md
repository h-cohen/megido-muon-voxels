# megido-muon-voxels

Phase 1 of a muon-tomography analysis pipeline for the Megiddo detector: it
turns raw CAEN DT5550W cluster dumps into calibrated, geometry-validated
angular tracks and content-addressed exposure artifacts (`counts_<exp>.npz`,
`tracks_<exp>.parquet`) ready for the reconstruction solver. It falsifies the
engineer-supplied detector constants (bar pitch, active width, layer
separation) against real data before anything downstream is allowed to trust
them.

See the design and plan for the full context:

- Design spec: `docs/superpowers/specs/2026-09-16-megido-muon-voxels-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-17-phase1-raw-to-angles.md`
- Phase 1 exit-gate report: `docs/phase1-validation-report.md`

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Usage

Run the S0-det validation gate against a single exposure's data (fast, reads
one chunk):

```bash
uv run python -m megido.cli validate --config configs/megido.yaml --exposure P0 --max-chunks 1
```

Run the full ingest pipeline (S0-exp + S1) over every configured exposure,
writing content-addressed artifacts to an output directory (slow — reads
every file twice; cached on repeat runs):

```bash
uv run python -m megido.cli ingest --config configs/megido.yaml --out runs/ingest
```

Run the test suite:

```bash
uv run pytest
```

## Adding a new detector exposure

Adding a new exposure is a data change, not a code change: drop the raw
`.data` files in the configured `data_dir`, append one exposure block (id,
run range, pose) to `configs/megido.yaml`, and re-run `ingest`.
