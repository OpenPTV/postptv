Trajectory repair
=================

A tracker leaves three kinds of damage in a trajectory database: links to the
wrong particle, trajectories broken by a missed detection or a rejected link,
and particles left unlinked as single points. :func:`flowtracks.repair.repair_trajectories`
repairs all three in one call, before any smoothing or averaging:

1. **cut** -- every link inside a trajectory gets a two-sided gap check: a
   straight line is fitted to up to ``window`` points before the link and,
   separately, to up to ``window`` points after it; both are evaluated at the
   middle of the link. A wrong link jumps by about the particle spacing, a
   correct one only by noise. Clear outliers are cut.
2. **attach** -- a single point is appended to the end or start of a
   trajectory whose straight-line prediction (up to ``max_gap + 1`` frames
   away) it matches.
3. **join** -- a trajectory end is joined to a trajectory start up to
   ``max_gap`` missing frames later when the same gap check passes. Each end
   and each start joins at most once, so no piece is ever duplicated.

The check value is the mismatch divided by the size its noise should have, and
both the noise scale and the acceptance threshold are calibrated on the data
(links inside long trajectories), so no distance or velocity tolerance has to
be guessed. No positions are invented: a joined gap stays a gap, and the
returned trajectories carry zero velocity for the caller to differentiate over
the real frame numbers.

.. code:: python

   from flowtracks.repair import repair_trajectories

   repaired, report = repair_trajectories(trajectories, max_gap=3)
   print(report)   # links_cut, points_attached, joins, threshold, noise_scale, ...

For millions of points use the array form, which is what the list form wraps:

.. code:: python

   from flowtracks.repair import repair_arrays

   new_trajid, report = repair_arrays(trajid, frame, pos)   # one row per position

Candidate searches are single KD-tree queries over (x, y, z, frame) and
matches are resolved greedily in order of increasing check value, so a
5000-frame, 6-million-point run is repaired in tens of seconds.

Use it before :mod:`flowtracks.smoothing`: smoothing a trajectory that contains
a wrong link spreads the jump over the whole smoothing window.

It was validated against synthetic ground truth matched to a 4-camera
experiment (openptv-analysis, logs 025-026): attachments 99.3-99.9 % correct,
joins 99.9 % correct.

.. automodule:: flowtracks.repair
   :members:
