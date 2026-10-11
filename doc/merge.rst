Merging two camera systems
==========================

When two camera systems (two rigs, or two recordings) see a common volume,
each produces its own trajectories in its own frame and the particles in the
overlap appear twice. :mod:`flowtracks.merge` provides the generic steps:

1. :func:`~flowtracks.merge.estimate_rigid_map` measures the rigid map between
   the two frames from particles both systems see in the same frame. For each
   candidate rotation the same-frame neighbour differences pile up at the true
   offset; the sharpest pile wins and the offset is refined on tight pairs. With
   no sharp pile it refuses (``ValueError``) rather than guess.
2. :func:`~flowtracks.merge.apply_rigid` maps the second system's positions
   (and, with zero offset, velocities and accelerations) into the first.
3. :func:`~flowtracks.merge.overlap_counts` counts the same-frame shared points
   of every trajectory pair; :func:`~flowtracks.merge.best_partners` gives each
   trajectory its partner and :func:`~flowtracks.merge.duplicates` the
   second-system trajectories a keep-the-longer rule removes.

.. code:: python

   from flowtracks.merge import (apply_rigid, best_partners, duplicates,
                                 estimate_rigid_map, overlap_counts)

   rig = estimate_rigid_map(same_frame_point_pairs,
                            [("0", np.eye(3)), ("180", np.diag([-1, 1, -1]))])
   pos_b_in_a = apply_rigid(pos_b, rig["matrix"], rig["offset"])
   shared = overlap_counts(tracks_a, tracks_b, tol=20.0)

A complete use, with the facility's own frame, walls and output layout, is
``scripts/illmenau/merge_dual_rig.py`` in openptv2.

.. automodule:: flowtracks.merge
   :members:
