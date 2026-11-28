.. Flowtracks documentation master file.
   Built from the doc/ directory:
   ``python -m sphinx -b html doc _build/html``

Welcome to Flowtracks
=====================

Flowtracks is a Python package for post-processing 3D Particle Tracking
Velocimetry (3D-PTV) trajectory databases. It reads trajectory data in the
formats the 3D-PTV community produces, turns it into Eulerian velocity
fields, performs phase and ensemble averaging for periodic flows, and
exports results as Zarr and ParaView (VTK) files.

.. contents::
   :depth: 2

.. toctree::
   :maxdepth: 2

   getting_started
   myptv
   proptv
   openlpt
   datastruct
   particle
   io
   scene
   zarr_scene
   pipeline
   writers
   analysis
   clean
   repair
   merge
   stitching
   smoothing
   interpolation
   graphics
   pairs
   sequence
   lyapunov
   vortex_wall_flux
   combine
   provenance

Getting Started
===============

Install the latest release from PyPI:

.. code:: bash

   pip install flowtracks

Optional extras:

.. code:: bash

   pip install "flowtracks[vtk]"       # ParaView export (pyvista/vtk)
   pip install "flowtracks[legacy-io]" # legacy PyTables HDF5 reader
   pip install "flowtracks[flux]"      # centerline flux (scikit-image)

Requires Python ≥ 3.11. For development, clone the repository and run
``uv sync --group dev`` (or ``pip install -e ".[vtk]"``).

.. code:: python

   from flowtracks.io import trajectories
   from flowtracks.eulerian import eulerian_grid

   # Read trajectory data (format inferred from the file name)
   trajs = trajectories("data/my_run.zarr")

   # Bin a run to an Eulerian grid
   ds = eulerian_grid("data/my_run.zarr", grid_params, 0, 1)

The :doc:`pipeline` page describes the full xarray-based pipeline; the
:doc:`writers` page covers export to ParaView.

Two backend families
====================
Flowtracks ships two duck-typed trajectory readers:

- **HDF5 / PyTables** — :class:`~flowtracks.scene.Scene` and
  :class:`~flowtracks.scene.DualScene` (optional ``legacy-io`` extra).
- **Zarr** — :class:`~flowtracks.zarr_scene.ZarrScene`, the recommended
  backend. It reads the ``trajectories/`` group written by
  ``openptv2``'s RunStore directly, with no conversion step.

Both expose the same public methods, so callers (and the xarray modules)
work against either.

Where the docs live
===================
This page is the top-level index. The more narrative material — recipes and
math notes for Eulerian statistics, plus a static "story" produced with
marimo — is in the repository's ``docs/`` directory, not here:

- ``docs/eulerian_statistics.md`` — per-run vs pooled statistics,
  the math, and usage recipes (requires a YAML recipe in the repo).
- ``docs/story/`` — a marimo-studio HTML story of the pipeline.

Source code lives in ``flowtracks/`` and examples in ``examples/``.
