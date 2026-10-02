Zarr-backed scenes
==================

A Zarr-backed counterpart to :class:`flowtracks.scene.Scene` with the same
public methods (duck-typed, not a subclass). It reads the ``trajectories/``
group layout shared with ``openptv2`` run stores, so no conversion step is
needed. Where-filtering is done in plain NumPy instead of a query-string DSL.

.. automodule:: flowtracks.zarr_scene
   :members:
