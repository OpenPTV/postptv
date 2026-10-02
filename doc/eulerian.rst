Eulerian gridding and turbulence statistics
=============================================

Pure ``Dataset -> Dataset`` xarray stages driven by a YAML recipe:
bin Lagrangian samples onto a regular grid
(:func:`~flowtracks.eulerian.eulerian_grid`,
:func:`~flowtracks.eulerian.eulerian_windowed`), then add fluctuations,
turbulent statistics and derived fields (TKE, vorticity, ...).

.. automodule:: flowtracks.eulerian
   :members:
