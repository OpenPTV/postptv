# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`postptv` publishes as the **`flowtracks`** package: post-processing of 3D
Particle Tracking Velocimetry (3D-PTV) trajectory databases — reading
per-frame particle data, building trajectories, interpolating Lagrangian data
onto Eulerian grids, phase/ensemble averaging for periodic flows, smoothing,
stitching, and VTK export. It is one stage of a larger multi-repo pipeline;
see `~/projects/CLAUDE.md` for how this repo relates to `openptv2`,
`openptv-cloud`, `flowtracks_examples`, and `matlab_to_python_3dptv`.

## Commands

Uses **uv**.

```bash
uv sync --group dev          # installs pytest, pytest-benchmark
uv run pytest                # full suite (testpaths = tests/)
uv run pytest tests/test_scene.py::test_name -v   # single test
uv run pytest tests/test_speedup_equivalence.py -v  # perf-vectorization equivalence tests
uv run pytest --benchmark-only    # pytest-benchmark suite (see .benchmarks/)
```

Ruff is configured in `pyproject.toml` (`E`/`W`/`F`/`I`; `E501`/`E701`/`E741`
ignored) and runs in CI — `ruff check .` must pass before merge. There is no
build step (pure Python, no Cython here — unlike `openptv2`).

## Architecture

### Two parallel data backends, one duck-typed interface

The library has two generations of "give me particle data" reader, and newer
code (`eulerian.py`, `phase_average.py`) is written to work against either:

- **`scene.py` (`Scene`)** — the original backend. Wraps a PyTables HDF5 file
  with a `/particles` table plus a `/bounds` (or similar) trajectory-index
  table. Frame/trajectory queries are built as PyTables `read_where()`
  query strings via `gen_query_string()`.
- **`zarr_scene.py` (`ZarrScene`)** — a Zarr-backed counterpart with the same
  public methods (duck-typed, not a subclass — the backing stores are too
  different to share a base class). Reads a `trajectories/` group with
  `pos`/`vel`/`time`/`trajid`(/`accel`) arrays. This layout is shared with
  `openptv2`'s `storage.seal()` run-store output, so `ZarrScene` transparently
  reads either a flowtracks-written Zarr export (`io.save_zarr_trajectories`)
  or an openptv2 RunStore — no conversion step. `collect()`'s `where`
  filtering is reimplemented in plain numpy instead of a query-string DSL,
  since there's no query engine to target.

When touching either backend, check whether the change needs to be mirrored
in the other — callers like `eulerian.py`'s `scene.collect(...)` fast path
are written to accept both.

### Trajectory data model

`trajectory.py` defines `ParticleSet` (the base class: an array of `pos`,
`velocity`, plus arbitrary per-timepoint keyword properties, each auto-getting
a getter/setter pair via `create_property`), and `Trajectory` /
`ParticleSnapshot` on top of it. `Frame` is a plain data-holder with
`.particles` / `.tracers` attributes.

### proPTV import (`proptv.py`)

`proptv.py` reads a proPTV `tracks*.hdf5` file into flowtracks Trajectories
and converts it to a Zarr store readable by `ZarrScene` natively — the same
interop role `myptv.py` plays for MyPTV, letting proPTV and flowtracks
outputs be exchanged. Each track dataset `track_{t}_{key}` holds
`[frame, x, y, z, vx, vy, vz, ax, ay, az, <cx,cy>*n_cameras]` (mm, integers
in floats); camera columns are ignored.

### MyPTV import (`myptv.py`)

`myptv.py` reads MyPTV TSV trajectory files (tracked / smoothed / stitched)
into flowtracks Trajectories and converts them to a Zarr store readable by
`ZarrScene` natively — the bridge that lets MyPTV and openptv/openptv-cloud
outputs be combined in postptv. Smoothed/stitched files (11 cols:
`id,x,y,z,vx,vy,vz,ax,ay,az,frame`) are used as stored; raw tracked files
(5 or 9 cols) get forward-difference velocity over the frame column
(metres per frame; no accel is guessed). Unlinked samples (trajid == -1) are
kept as one-sample trajectories.

### I/O module (`io.py`)

Central entry points: `trajectories()` / `iter_trajectories()` for reading,
`save_trajectories()` (legacy NPZ-directory format) and
`save_particles_table()` (HDF5, the recommended format) for writing. Format is
inferred from the file name via `infer_format()`. Zarr save/read paths
(`save_zarr_trajectories`, `read_zarr_trajectories`) live alongside these.

### xarray/YAML-recipe modules (newer, parallel implementation track)

`phase_average.py` and `eulerian.py` are **xarray re-expressions** of older
imperative scripts (`phase_average_fluctuations.py`,
`batch_Lagrangian_to_Eulerian.py`, `turbulent_statistics.py`,
`sample_vtkcode.py` — some of these live in `examples/`, not in the package).
They operate as pure `Dataset -> Dataset` stages driven by a YAML recipe, and
are meant to consolidate the domain math (often a one-liner) while making the
data contract (dims, coords, attrs) explicit and self-describing. When adding
post-processing logic, prefer extending these over the older imperative
scripts they replace.

### `pipeline.py`

Thin per-stage wrapper functions (`ptv_is_to_lagrangian`,
`lagrangian_to_eulerian`, `phase_average_all_sets`, ...) — each one is
independently runnable and is the unit the cloud orchestrator (in
`openptv-cloud`) distributes across runs. Stages import their real logic from
the xr modules or `examples/` scripts; nothing is implemented twice here.

### Other modules

- `stitching.py` — reconnects broken trajectory segments across short frame
  gaps (linear-sum-assignment candidate matching).
- `smoothing.py`, `interpolation.py` — trajectory smoothing and RBF/IDW
  interpolation onto grids.
- `analysis.py`, `combine.py`, `pairs.py` — trajectory-level statistics and
  multi-run combination.
- `writers.py`, `graphics.py`, `nhist.py` — visualization/export helpers
  (ParaView export goes through `writers.write_eulerian_series` /
  `write_trajectories_vtp`; the legacy `vtk_export.py` module is removed).

### Performance work in flight

Several modules carry "vectorize X, verify equivalence" pairs: a vectorized
implementation plus a `tests/test_*_equivalence.py` test asserting it matches
the original loop-based output bit-for-bit (or within tolerance) on the same
inputs. `SPEEDUP_PLAN.md` and `future_plan.md` track the broader roadmap
(Zarr/xarray/Dask migration, turbulence statistics, RTS Kalman smoothing,
CLI, GPU hooks) — check these before assuming a described feature already
exists; several are aspirational, not built.

## Notes

- Python ≥3.11. Some docstrings/comments still carry Python-2-era artifacts
  (`from builtins import object`, `# from past.builtins import range`) —
  harmless leftovers, not evidence the code needs to support Python 2.
- `examples/` contains runnable scripts and marimo/Jupyter notebooks that
  exercise the pipeline end-to-end (`marimo_*`, `batch_Lagrangian_to_Eulerian.py`,
  `run_pipeline.py`) — useful as real usage references when a docstring is
  unclear.
