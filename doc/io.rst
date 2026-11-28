

Input and Output Routines
=========================

:mod:`flowtracks.io` reads and writes particle data. Supported input
formats (inferred from the file name by :func:`flowtracks.io.infer_format`):

* **ptv_is** — format output by OpenPTV code (and the older widely used 3DPTV): one file per frame with position, previous/next-frame index and particle number.
* **xuap** — one file per frame with position, velocity and acceleration for the particle and the surrounding fluid.
* **acc** — like xuap, with the time step relative to the trajectory start.
* **mat** — Matlab file of trajectory structure-arrays.
* **hdf** — Flowtracks' own HDF5 format (PyTables). The legacy reader needs the ``legacy-io`` extra.
* **npz** — NumPy archive of trajectory data.
* **zarr** — Zarr ``trajectories/`` group (see :doc:`zarr_scene`).

``flowtracks.io`` also provides the Zarr save/read pair
:func:`~flowtracks.io.save_zarr_trajectories` and
:func:`~flowtracks.io.read_zarr_trajectories`, and the HDF5 table writer
:func:`~flowtracks.io.save_particles_table`.

.. automodule:: flowtracks.io
   :members: 
   :exclude-members: collect_particles, collect_particles_mat, collect_particles_generic