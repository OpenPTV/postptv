Getting Started
================

Installation
--------------

.. code:: bash

   pip install flowtracks

Optional extras (install the ones you need):

.. code:: bash

   pip install "flowtracks[vtk]"       # ParaView export (pyvista + vtk)
   pip install "flowtracks[legacy-io]" # legacy PyTables HDF5 reader (``tables``)
   pip install "flowtracks[flux]"      # vessel centerline flux (scikit-image)
   pip install "flowtracks[netcdf]"    # NetCDF export (netcdf4)
   pip install "flowtracks[notebooks]" # marimo/Jupyter notebooks environment

The core install requires Python ≥ 3.11 and pulls in numpy, scipy, h5py,
pyyaml, zarr ≥ 3.0.0, xarray ≥ 2024.1.0 and matplotlib.

For development, clone the repository and install it editable:

.. code:: bash

   git clone https://github.com/OpenPTV/postptv
   cd postptv
   uv sync --group dev      # or: pip install -e ".[vtk]"

Quickstart
----------

Reading and saving trajectories:

.. code:: python

   from flowtracks.io import trajectories, save_zarr_trajectories

   trajs = trajectories("my_data.zarr")       # format inferred from filename
   for traj in trajs:
       print(traj.trajid, len(traj))

   save_zarr_trajectories(trajs, "clean.zarr", overwrite=True)

A standard post-processing flow (Lagrangian → Eulerian → phase average):

.. code:: python

   from flowtracks.zarr_scene import ZarrScene
   from flowtracks.eulerian import eulerian_grid, save_zarr

   scene = ZarrScene("my_data.zarr")
   ds = eulerian_grid(scene, grid_params, 0, 2)
   save_zarr(ds, "grid.zarr", overwrite=True)

Exporting to ParaView (requires the ``vtk`` extra):

.. code:: python

   from flowtracks.writers import write_eulerian_series

   write_eulerian_series(ds, "paraview_out")

Running the test suite
----------------------

.. code:: bash

   uv run pytest
