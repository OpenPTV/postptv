# PostPTV (Flowtracks) — Future Development Plan

Strategic roadmap for **PostPTV** (`flowtracks`), a Python package for
post-processing 3D Particle Tracking Velocimetry (3D-PTV) and Lagrangian
Particle Tracking (LPT) datasets. This document tracks what is shipped vs.
what remains openly aspirational.

> Status note: an earlier version of this file was stale. The Zarr backend
> and the VTK writers are **not** future work — they are shipped today. The
> milestone table at the bottom reflects the current state.

---

## 🎯 Executive Summary & Objectives

PostPTV provides core functionality for reading particle trajectories,
interpolating Lagrangian velocity fields onto Eulerian grids, smoothing
trajectories, and exporting datasets. The roadmap modernizes PostPTV into a
high-performance, cloud-native, and feature-rich post-processing framework
for experimental fluid dynamics.

---

## Shipped

- **Zarr + xarray post-processing track** — `zarr_scene.py`, `eulerian.py`,
  `phase_average.py`, `phasing.py`, `pipeline.py` (all shipped). Zarr is the
  recommended backend; `openptv2`'s RunStore writes the same `trajectories/`
  layout, so `ZarrScene` reads openptv2 output with no conversion step.
- **Interactive VTK/ParaView export** — `writers.py` (`write_pvd`,
  `write_trajectories_vtp`, `write_eulerian_series`, `export_run_to_paraview`)
  replaces the removed `vtk_export.py`. See [WRITERS_PLAN.md](WRITERS_PLAN.md)
  for the completed design.
- **FTLE/FSLE / Lyapunov exponents** — `lyapunov.py` (pair FTLE, FSLE, Eulerian
  FTLE, ridge extraction). See `doc/lyapunov.rst`.
- **Near-wall shear, flux, vortex** — `wall_shear.py` (TAWSS/OSI/RRT),
  `flux.py` (cross-section flow rate), `vortex.py` (Q-criterion, lambda2).
- **Pipeline stages** — `pipeline.py` wraps each stage as an independently
  runnable unit for the cloud orchestrator.
- **Provenance** — `provenance.py` appends run records to `run.zarr`'s
  `meta.attrs["runs"]`.

---

## Open aspirations

### 1. Dask Out-of-Core Processing
* **Goal**: enable delayed/chunked frame-by-frame processing for
  multi-gigabyte or distributed datasets without fitting everything in RAM.
* **Features**: Dask arrays over Zarr for the Eulerian pipeline; chunked
  reads in `ZarrScene`.

### 2. Turbulence Statistics Module
* **Goal**: deliver the metrics that are still hand-rolled per-project.
* **Features**: structure functions $D_{LL}, D_{NN}$; Lagrangian velocity
  auto-correlation $R_{ij}(\tau)$ and integral time scale $T_L$; turbulent
  dissipation rate $\varepsilon = 2\nu \langle s_{ij}s_{ij}\rangle$; material
  pressure-gradient reconstruction.

### 3. Kalman / RTS Smoother & Outlier Filtering
* **Goal**: reduce the number of manual repair passes.
* **Features**: forward-backward RTS Kalman smoother in `smoothing.py`;
  universal outlier detection (UOD) for unphysical acceleration spikes.

### 4. CLI & GPU Acceleration Hooks
* **Goal**: streamlined command-line operation and optional hardware
  acceleration.
* **Features**: a `postptv` CLI (`postptv info`, `postptv interpolate`,
  `postptv export`); optional CuPy paths for KD-tree neighbor queries and RBF
  solves.

---

## 🗺️ Milestone Status

| Milestone | Deliverable / Feature Module | Target Scope | Priority | Status |
|---|---|---|---|---|
| M1 | **Zarr + Xarray Data Engine** — Zarr backend and pipeline shipped as `zarr_scene.py` + `eulerian.py`; Dask out-of-core still open | `flowtracks/zarr_scene.py` (+ `eulerian.py`) + future `flowtracks/` Dask path | P1 | Partially shipped (Zarr/xarray) / open (Dask) |
| M2 | **VTK Export** — shipped in `flowtracks/writers.py` | `flowtracks/writers.py` | P2 | **Shipped** (see [WRITERS_PLAN.md](WRITERS_PLAN.md)) |
| M3 | **RTS Kalman Smoothing & Filtering** | `flowtracks/smoothing.py` + UOD | P3 | Open |
| M4 | **Turbulence Statistics Module** | e.g. `flowtracks/turbulence.py` ($D_{LL}, R_{ij}, T_L$, $\varepsilon$) | P4 | Open (some metrics shipped in `lyapunov`/`wall_shear`/`flux`/`vortex`) |
| M5 | **CLI & Acceleration Hooks** | `postptv` CLI + optional CuPy GPU paths | P5 | Open |
