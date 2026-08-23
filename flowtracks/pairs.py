# -*- coding: utf-8 -*-
#Created on Thu Aug 15 14:38:58 2013

"""
Pair particles to closest tracers.
"""

import numpy as np
from scipy.spatial import cKDTree
from .trajectory import trajectories_in_frame, take_snapshot

def particle_pairs(primary_trajects, secondary_trajects, trajids, time_points):
    """
    For each of a set of select particles in the primary trajectories, find
    the closest particle in the secondary set.

    Arguments:
    primary_trajects - a list of Trajectory objects, some of which contain the
        source points.
    secondary_trajects - a list of Trajectory objects, in which to look for the
        pair points.
    trajid, time_points - each an n-length array for n pairs to produce, 
        holding correspondingly the trajectory id and index into the trajectory
        of the points in the primary set to which a pair is sought.

    Returns:
    pair_trid, pair_time - coordinates of the found pairs, element i describes
        the pair of particle i in (trajid, time_points). Format is the same as 
        that of ``trajid``, ``time_points``. For particles without a match, 
        returns -1 as the pair_time value.
    """
    # Output buffers:
    pair_trids = np.empty_like(time_points)
    pair_time = np.empty_like(time_points)

    # Index the trajectories once, by id, so membership tests are O(1).
    by_id_prim = {int(t.trajid()): t for t in primary_trajects}
    by_id_sec = {int(t.trajid()): t for t in secondary_trajects}

    # Filter the primary set to only contain the trajectories actually required
    unique_prim = np.unique(trajids)
    prim_traj = [by_id_prim[int(tr)] for tr in unique_prim \
        if int(tr) in by_id_prim]
    frames = np.empty_like(time_points)

    # Typify primary/secondary on a per trajectory basis before combining them
    # into a single snapshot.
    for traj in prim_traj:
        traj_coords = trajids == traj.trajid()
        frames[traj_coords] = traj.time(time_points[traj_coords])

    unique_frames = np.unique(frames)
    schema = prim_traj[0].schema()

    # Hoist the secondary trajectories' start/end times so they are not
    # recomputed inside ``trajectories_in_frame`` for every frame.
    sec_start = np.array([t.time()[0] for t in secondary_trajects])
    sec_end = np.array([t.time()[-1] for t in secondary_trajects])

    # For each frame, create snapshots and compare positions.
    for frame_num in unique_frames:
        coord_locator = frames == frame_num
        prim_in_frame_ids = np.unique(trajids[coord_locator])
        prim_in_frame = [by_id_prim[int(tr)] for tr in prim_in_frame_ids \
            if int(tr) in by_id_prim]
        prim_parts = take_snapshot(prim_in_frame, frame_num, schema)

        sec_in_frame_ixs = trajectories_in_frame(secondary_trajects, frame_num,
            start_times=sec_start, end_times=sec_end, segs=True)
        sec_in_frame = [secondary_trajects[tix] for tix in sec_in_frame_ixs]

        if len(sec_in_frame) == 0:
            pair_trids[coord_locator] = -1
            pair_time[coord_locator] = -1
            continue

        sec_parts = take_snapshot(sec_in_frame, frame_num, schema)

        # Nearest secondary particle to each primary particle via a KD-tree
        # (equivalent to argmin of squared distances, modulo tie-breaking on
        # exactly equidistant points).
        _, pair_ixs = cKDTree(sec_parts.pos()).query(prim_parts.pos())
        pair_trids[coord_locator] = sec_parts.trajid(pair_ixs)
        pair_time[coord_locator] = frame_num # later transformed.

    # Transform frame numbers back into time index in the output array.
    unique_sec = np.unique(pair_trids)
    unique_sec = unique_sec[unique_sec >= 0]  # skip the "no match" marker.

    for trid in unique_sec:
        traj = by_id_sec.get(int(trid))
        if traj is None:
            continue
        pair_time[pair_trids == trid] -= traj.time(0)

    return pair_trids, pair_time
