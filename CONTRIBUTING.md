# Contributing to postptv (flowtracks)

## Setup

```bash
git clone https://github.com/OpenPTV/postptv.git
cd postptv
uv sync --group dev
uv run pytest
```

## Module map

| Module | What it does |
|--------|-------------|
| `flowtracks/io.py` | Read/write trajectories (HDF5, Zarr, NPZ, PTVIS, …) |
| `flowtracks/scene.py` | HDF5 scene backend (PyTables, optional `legacy-io`) |
| `flowtracks/zarr_scene.py` | Zarr-backed scene backend (recommended) |
| `flowtracks/myptv.py` | MyPTV TSV import (read / convert to Zarr) |
| `flowtracks/trajectory.py` | Trajectory data model (ParticleSet, Trajectory, ParticleSnapshot) |
| `flowtracks/particle.py` | `Particle` |
| `flowtracks/sequence.py` | Text-format sequence iteration (non-HDF formats) |
| `flowtracks/eulerian.py` | Lagrangian→Eulerian binning, derived fields, Zarr/NetCDF save |
| `flowtracks/phase_average.py` | Phase averaging and fluctuation analysis (pure Dataset→Dataset) |
| `flowtracks/phasing.py` | Canonical phase-bin assignment (shared with OpenPTV family) |
| `flowtracks/pipeline.py` | Thin per-stage wrappers (ptv_is→Lagrangian→Eulerian) |
| `flowtracks/combine.py` | Multi-run combination (per-run or pooled moments) + `postptv-combine` CLI |
| `flowtracks/stitching.py` | Reconnect broken trajectory segments |
| `flowtracks/smoothing.py` | Savitzky-Golay / smoothing-spline filtering |
| `flowtracks/repair.py` | Trajectory repair (cut, attach, join) |
| `flowtracks/clean.py` | Data cleaning chain (repair + smooth + stitch) |
| `flowtracks/interpolation.py` | RBF/IDW interpolation onto grids |
| `flowtracks/graphics.py` | Plotting helpers (distributions, vectors, 3D trajectories) |
| `flowtracks/pairs.py` | Nearest-neighbour trajectory pairing |
| `flowtracks/merge.py` | Merge two camera systems (rigid map, overlap, duplicates) |
| `flowtracks/lyapunov.py` | FTLE / FSLE / Eulerian FTLE / ridge extraction |
| `flowtracks/vortex.py` | Vortex identification (Q-criterion, lambda2, enstrophy) |
| `flowtracks/wall_shear.py` | Near-wall shear stress (TAWSS/OSI/RRT) |
| `flowtracks/flux.py` | Vessel cross-section flow rate |
| `flowtracks/nhist.py` | Normalized histograms |
| `flowtracks/provenance.py` | Run provenance recording (runs in `run.zarr`) |
| `flowtracks/writers.py` | ParaView export (PVD, VTP, VTI, VTR) |
| `flowtracks/analysis.py` | Trajectory-level analysis helpers |
| `flowtracks/format_docstrings.py` | Sphinx docstring formatting extension |

## Key concepts

- **Trajectory**: a sequence of particle positions/velocities over time, identified by `trajid`
- **Scene**: a reader backend (HDF5 `Scene` or Zarr `ZarrScene`) providing `collect()`
- **Eulerian grid**: regular (x, y, z, phase) grid of mean velocities from binning particles
- **Phase**: position in a periodic cycle (0..N-1), assigned by `ti = ((time - base) % cycletime) // zaman`
- **Ensemble (per-run)**: multiple runs binned individually and stacked on a `set` dimension, averaged with count weighting; second moments reflect run-to-run spread
- **Pooled (per-voxel moments)**: each run records unmasked counts, distinct trajectories (`n_traj`), means (`*_m1`), and centred second moments (`*_m2`), combined exactly across runs with `combine_moments()` / `moment_statistics()` (see [docs/eulerian_statistics.md](docs/eulerian_statistics.md))
- **Pipeline stage**: one independently runnable step (see `flowtracks/pipeline.py`) — the unit the cloud orchestrator distributes

## Architecture

```
io.py / scene.py / zarr_scene.py  →  read trajectories
         ↓
stitching.py / repair.py / smoothing.py / clean.py  →  clean trajectories
         ↓
eulerian.py  →  bin to Eulerian grid (per run or pooled)
         ↓
phase_average.py  →  phase/ensemble averaging
         ↓
derived_fields()  →  MKE, TKE, VSS, etc.
         ↓
writers.py  →  VTK/ParaView export
```

## Tests

```bash
uv run pytest                          # all tests
uv run pytest tests/test_eulerian_grid.py  # specific file
uv run pytest --benchmark-only         # benchmarks
```

## Lint

Ruff is configured in `pyproject.toml` (`E`/`W`/`F`/`I`; `E501`/`E701`/`E741`
ignored) and runs in CI — `ruff check .` must pass before merge.
