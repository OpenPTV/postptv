Importing MyPTV data
====================

:mod:`flowtracks.myptv` reads MyPTV trajectory files (produced by the
`MyPTV <https://github.com/ronshnapp/MyPTV>`_ 3D-PTV pipeline, e.g.
``example/trajecotries_stitched``) and converts them into flowtracks
:class:`~flowtracks.trajectory.Trajectory` objects — and optionally into a
``trajectories`` Zarr store that :class:`~flowtracks.zarr_scene.ZarrScene`
reads natively, with no further conversion. This is how the two pipelines
interoperate: MyPTV tracks, postptv post-processes.

Two file forms are recognized (detected from the column count):

- **Smoothed / stitched** (11 columns): ``[trajid, x, y, z, vx, vy, vz, ax, ay, az, frame]``
  — the final output of the MyPTV ``smoothing`` and ``stitching`` stages.
- **Tracked** (5 or 9 columns): raw output of MyPTV's tracking
  (``myptv.tracking_mod.save_results``). Columns are
  ``[trajid, x, y, z, <per-camera blob index columns>, <per-camera error>, frame]``;
  the last column is always the frame number. Velocity is not stored here;
  the translator re-derives it by finite differences over the frame column,
  exactly as ``traj_stitching_mod`` does — in metres per frame.
  Acceleration is not inferred and stays absent.

Rows with trajectory id ``-1`` (unlinked samples the tracker could not link)
are kept as one-sample trajectories, so no data is dropped.

.. code:: python

   from flowtracks.myptv import read_myptv_trajectories, convert_myptv_to_zarr

   trajects = read_myptv_trajectories("smoothed_trajectories")
   convert_myptv_to_zarr("smoothed_trajectories", "run.zarr")

   # ZarrScene opens the converted store with no conversion step:
   from flowtracks.zarr_scene import ZarrScene
   scene = ZarrScene("run.zarr")

``convert_myptv_to_zarr`` writes a ``trajectories`` group with ``pos``,
``vel``, ``time``, ``trajid`` (and ``accel`` when present), the same layout
:func:`flowtracks.io.save_zarr_trajectories` and ``openptv2``'s RunStore
write. Units default to honest per-frame units (``m frame-1``, never
``m s-1``) because a MyPTV file carries no frame rate; pass ``units=`` to
state absolute units explicitly. A ``source_format``/``source_file``
attribute pair is recorded on the group for provenance.

**Note:** MyPTV units are not assumed by the translator. Run the pipeline
against the same unit system the source file was produced with, and scale
the input (or pass ``units=`` explicitly) if you need SI units.

.. automodule:: flowtracks.myptv
   :members:
   :private-members:
