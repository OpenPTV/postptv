Stitching
=========

:mod:`flowtracks.stitching` reconnects broken trajectory segments across
short frame gaps: candidates are matched between the end of one segment and
the start of the next, then accepted when the candidate is the unique nearest
neighbour — a linear-sum-assignment-based matching.

- :func:`flowtracks.stitching.stitch_trajectories` — the standard pass
- :func:`flowtracks.stitching.stitch_trajectories_fast` — a fast variant
  that skips per-candidate validation

Stitching is one of the steps in :mod:`flowtracks.clean`; it is also
available standalone.

.. automodule:: flowtracks.stitching
   :members:
