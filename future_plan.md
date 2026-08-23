# PostPTV (Flowtracks) — Future Development Plan

This document outlines the strategic roadmap and future feature specifications for **PostPTV** (`flowtracks`), a Python package for post-processing 3D Particle Tracking Velocimetry (3D-PTV) and Lagrangian Particle Tracking (LPT) datasets.

---

## 🎯 Executive Summary & Objectives

PostPTV provides core functionality for reading particle trajectories, interpolating Lagrangian velocity fields onto Eulerian grids, smoothing trajectories, and exporting datasets. This development plan aims to modernize PostPTV into a high-performance, cloud-native, and feature-rich post-processing framework for experimental fluid dynamics.

---

## 📋 Key Strategic Initiatives

### 1. Next-Generation Cloud-Native Storage (Xarray + Zarr Engine)
* **Goal**: Transition from restrictive single-file HDF5/PyTables backends to scalable, cloud-native array formats.
* **Features**:
  * **First-Class Zarr & Parquet I/O**: Add `.to_zarr()`, `.from_zarr()`, `.to_parquet()` methods to `Scene` and `Trajectory` datasets with ragged array encoding.
  * **Xarray Data Model Standard**: Standardize internal data representations on `xarray.Dataset` with named coordinates (`trajectory`, `time`, `component`).
  * **Dask Out-of-Core Processing**: Enable delayed/chunked frame-by-frame processing for multi-gigabyte or distributed dataset analyses without fitting all frames in RAM.

---

### 2. Advanced Fluid Dynamics & Lagrangian Turbulence Statistics
* **Goal**: Equip PostPTV with advanced fluid turbulence metrics, coherent structure identification, and physical field reconstructions.
* **Features**:
  * **Finite-Time Lyapunov Exponents (FTLE) & Lagrangian Coherent Structures (LCS)**:
    * Compute forward and backward FTLE fields to identify Lagrangian transport barriers, mixing fronts, and vortex boundaries from trajectories $\mathbf{x}(t; \mathbf{x}_0, t_0)$.
  * **Velocity Structure Functions ($D_{LL}, D_{NN}$)**:
    * Calculate longitudinal structure functions $D_{LL}(r) = \langle [(\mathbf{u}(\mathbf{x}+\mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot \hat{\mathbf{r}}]^2 \rangle$ and transverse structure functions $D_{NN}(r)$ across separation distances $r$.
  * **Lagrangian Velocity Auto-Correlation & Integral Time Scales**:
    * Compute velocity auto-correlation tensor $R_{ij}(\tau) = \frac{\langle u_i(t) u_j(t+\tau) \rangle}{\sigma_{u_i} \sigma_{u_j}}$ and Lagrangian integral time scale $T_L = \int_0^\infty R(t) dt$.
  * **Turbulent Dissipation Rate ($\varepsilon$) & Pressure Gradient Reconstruction**:
    * Estimate kinetic energy dissipation rate $\varepsilon = 2 \nu \langle s_{ij} s_{ij} \rangle$ from Eulerian strain rate tensors $s_{ij}$.
    * Reconstruct material pressure gradients $\nabla p = -\rho \left( \frac{D\mathbf{u}}{Dt} - \mathbf{g} \right)$ via a Poisson equation solver using particle accelerations.

---

### 3. Trajectory Processing, Physics-Constrained Gap Filling & Filtering
* **Goal**: Improve trajectory quality, handle missing particle observations, and eliminate tracking artifacts.
* **Features**:
  * **Kalman Filtering & Rauch-Tung-Striebel (RTS) Smoother**:
    * Forward-backward RTS Kalman smoothing for optimal simultaneous estimation of position, velocity, acceleration, and state uncertainty.
  * **Physics-Informed Trajectory Stitching**:
    * Multi-frame gap filling (3–10 frames) using constant acceleration or cubic spline physical constraints.
  * **Universal Outlier Detection (UOD) & Kinematic Filtering**:
    * Automatic detection and filtering of unphysical acceleration spikes caused by optical occlusion or tracking noise.

---

### 4. Interactive 3D Visualization & Modern ParaView Integration
* **Goal**: Deliver modern interactive graphics for Jupyter notebooks and enhanced export formats for ParaView.
* **Features**:
  * **PyVista & Plotly 3D Integration**:
    * Interactive 3D trajectory rendering inside Jupyter notebooks with orbit controls, velocity/acceleration color-mapping, and animated particle motion.
  * **Modern VTK PolyData (`.vtp` / `.vtm`) Exporters**:
    * XML-based multi-block VTK exports containing scalar and vector arrays (vorticity, kinetic energy, trajectory ID) ready for ParaView visual rendering.

---

### 5. CLI & GPU Acceleration Hooks
* **Goal**: Provide streamlined command-line operation and optional hardware acceleration.
* **Features**:
  * **Rich Command-Line Interface (CLI)**:
    * A `postptv` CLI tool (`postptv info`, `postptv interpolate`, `postptv export`).
  * **GPU Acceleration (CuPy / PyTorch)**:
    * Optional GPU execution paths for KD-Tree neighbor queries and RBF linear system solves during dense Eulerian interpolation.

---

## 🗺️ Implementation Roadmap & Milestones

| Milestone | Deliverable / Feature Module | Target Scope | Priority |
|---|---|---|---|
| **M1** | **Zarr & Xarray Data Engine** | Productionize `future_idea_xarray_dask_zarr.py` into `flowtracks/zarr_io.py` | P1 |
| **M2** | **Interactive 3D Viz & VTK Export** | PyVista/Plotly integrations in `flowtracks/graphics.py` & XML VTK exporter | P2 |
| **M3** | **RTS Kalman Smoothing & Filtering** | RTS smoother, kinematic outlier filter in `flowtracks/smoothing.py` | P3 |
| **M4** | **Turbulence Statistics Module** | New `flowtracks/turbulence.py` module ($D_{LL}, R_{ij}, T_L$, FTLE, Pressure) | P4 |
| **M5** | **CLI & Acceleration Hooks** | `postptv` CLI entry points and optional CuPy GPU interpolation paths | P5 |
