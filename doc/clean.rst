Cleaning chain
==============

:func:`flowtracks.clean.clean_trajectories` turns raw tracker output (one row
per point with a trajectory id, positions in metres) into smooth trajectories
in one call: :doc:`repair <repair>`, Savitzky-Golay smoothing, a split at
fast violent turns the repair missed, two stitching passes (the second
tighter) with smoothing in between, and a final smoothing.

Every setting has a default in :data:`flowtracks.clean.CLEAN_DEFAULTS` except
the frame rate, which must be stated. :func:`flowtracks.clean.resolve_settings`
merges a parameter-file section and explicit overrides onto the defaults and
reports where each value came from, so nothing is hidden in a script:

.. code:: python

   from flowtracks.clean import clean_trajectories, resolve_settings
   from flowtracks.io import save_zarr_trajectories

   settings, sources = resolve_settings(yaml_section, {"gap1": 8})
   trajs = clean_trajectories(trajid, frame, pos, settings)
   save_zarr_trajectories(trajs, "clean.zarr", overwrite=True)

The pieces are usable on their own: :func:`flowtracks.clean.trajectories_from_arrays`
(flat arrays to :class:`~flowtracks.trajectory.Trajectory` objects) and
:func:`flowtracks.clean.kink_split`.

.. automodule:: flowtracks.clean
   :members:
