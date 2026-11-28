ZarrScene
=========

:class:`~flowtracks.zarr_scene.ZarrScene` is the Zarr-backed counterpart of
:class:`~flowtracks.scene.Scene`. It reads a ``trajectories/`` group with
``pos``/``vel``/``time``/``trajid`` (and optional ``accel``) arrays — the same
layout written by ``openptv2``'s run store (``openptv2.storage.seal.seal()``)
and by :func:`flowtracks.io.save_zarr_trajectories` — so either works with no
conversion step.

It implements the same public surface as ``Scene`` (duck-typed, not a
subclass): ``collect(["pos", "vel", ...])``, ``times``, ``len``,
``__iter__`` and ``collect_frames``. Where the HDF5 ``Scene`` builds PyTables
``read_where()`` query strings, ``ZarrScene`` reimplements ``collect``'s
``where`` filtering in plain numpy.

.. automodule:: flowtracks.zarr_scene
   :members:
