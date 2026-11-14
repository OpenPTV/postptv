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
| `flowtracks/io.py` | Read/write trajectories (HDF5, Zarr, NPZ) |
| `flowtracks/scene.py` | HDF5 scene backend (PyTables) |
| `flowtracks/zarr_scene.py` | Zarr scene backend (duck-typed) |
| `flowtracks/trajectory.py` | Trajectory data model (ParticleSet, Trajectory) |
| `flowtracks/eulerian.py` | Lagrangian→Eulerian binning, phase/ensemble averaging, derived fields |
| `flowtracks/phase_average.py` | Phase averaging as pure Dataset→Dataset functions |
| `flowtracks/phase_average_xr.py` | YAML-recipe phase averaging |
| `flowtracks/post_analysis_xr.py` | YAML-recipe post-analysis (binning + averaging + derived) |
| `flowtracks/pipeline.py` | Thin per-stage wrappers (ptv_is→Lagrangian→Eulerian) |
| `flowtracks/combine.py` | Multi-run combination (per-run or pooled) |
| `flowtracks/stitching.py` | Reconnect broken trajectory segments |
| `flowtracks/smoothing.py` | Trajectory smoothing (Savitzky-Golay) |
| `flowtracks/repair.py` | Trajectory repair (cut, attach, join) |
| `flowtracks/writers.py` | ParaView export (VTK, VTI, VTR, VTP, PVD) |
| `flowtracks/vortex.py` | Vortex identification (Q-criterion, lambda2) |
| `flowtracks/interpolation.py` | RBF/IDW interpolation onto grids |
| `flowtracks/phasing.py` | Phase bin assignment (shared with openptv-cloud) |
| `flowtracks/analysis.py` | Trajectory-level statistics |
| `flowtracks/flux.py` | Flux calculations |
| `flowtracks/graphics.py` | Visualization helpers |
| `flowtracks/nhist.py` | Normalized histograms |
| `flowtracks/pairs.py` | Trajectory pairing |
| `flowtracks/merge.py` | Merge trajectories |
| `flowtracks/clean.py` | Data cleaning |
| `flowtracks/provenance.py` | Run provenance tracking |
| `flowtracks/format_docstrings.py` | Docstring formatting |

## Key concepts

- **Trajectory**: A sequence of particle positions/velocities over time, identified by `trajid`
- **Scene**: A reader backend (HDF5 or Zarr) providing `collect(["pos", "velocity", "time"])`
- **Eulerian grid**: Regular (x,y,z,phase) grid of mean velocities from binning particles
- **Phase**: Position in a periodic cycle (0..N-1), assigned by `ti = ((time - base) % cycletime) // zaman`
- **Ensemble**: Multiple runs stacked on a `set` dimension, averaged with count weighting
- **Pooled**: All samples from all runs collected together before binning

## Architecture

```
io.py / scene.py / zarr_scene.py  →  read trajectories
         ↓
stitching.py / repair.py / smoothing.py  →  clean trajectories
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
uv run pytest tests/test_eulerian_grid.py  # specific
uv run pytest --benchmark-only         # benchmarks
```