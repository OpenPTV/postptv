Writers (ParaView export)
=========================

:mod:`flowtracks.writers` exports run results to ParaView-readable formats
via pyvista/vtk. It replaces the removed ``vtk_export`` module: the grid-type
rule — **vtp** for trajectories, **vti** for uniform Eulerian volumes,
**vtr** for non-uniform volumes, **pvd** for time series — is applied here.

:func:`flowtracks.writers.write_pvd` writes a ``.pvd`` time-series file.
:func:`flowtracks.writers.write_trajectories_vtp` writes trajectory polydata
(``.vtp``). :func:`flowtracks.writers.write_eulerian_series` writes a
sequence of Eulerian volume files plus a ``.pvd`` — the single entry point
for "export a run". :func:`flowtracks.writers.export_run_to_paraview` does
the same from a ``run.zarr`` store, reading it with
:class:`~flowtracks.zarr_scene.ZarrScene`.

.. automodule:: flowtracks.writers
   :members:
