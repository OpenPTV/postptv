# -*- coding: utf-8 -*-
#Created on Thu Aug 15 14:38:58 2013

"""
Pair particles to closest tracers.
"""

import numpy as np
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
    
    # Filter the primary set to only contain the trajectories actually required
    unique_prim = np.unique(trajids)
    prim_traj = [t for t in primary_trajects if t.trajid() in unique_prim]
    frames = np.empty_like(time_points)
    
    # Typify primary/secondary on a per trajectory basis before combining them
    # into a single snapshot.
    for traj in prim_traj:
        traj_coords = trajids == traj.trajid()
        frames[traj_coords] = traj.time(time_points[traj_coords])

    unique_frames = np.unique(frames)
    schema = prim_traj[0].schema()
    
    # For each frame, create snapshots and compare positions.
    for frame_num in unique_frames:
        coord_locator = frames == frame_num
        prim_in_frame_ids = np.unique(trajids[coord_locator])
        prim_in_frame = [t for t in prim_traj if t.trajid() in prim_in_frame_ids]
        prim_parts = take_snapshot(prim_in_frame, frame_num, schema)
        
        sec_in_frame_ixs = trajectories_in_frame(secondary_trajects, frame_num,
            segs=True)
        sec_in_frame = [secondary_trajects[tix] for tix in sec_in_frame_ixs]
        
        if len(sec_in_frame) == 0:
            pair_trids[coord_locator] = -1
            pair_time[coord_locator] = -1
            continue
            
        sec_parts = take_snapshot(sec_in_frame, frame_num, schema)
        
        dists_sq = np.sum(
            (prim_parts.pos()[:,None,:] - sec_parts.pos()[None,:,:])**2,
            axis=2)
        pair_ixs = np.argmin(dists_sq, axis=1)
        pair_trids[coord_locator] = sec_parts.trajid(pair_ixs)
        pair_time[coord_locator] = frame_num # later transformed.
    
    # Transform frame numbers back into time index in the output array.
    unique_sec = np.unique(pair_trids)
    
    for traj in secondary_trajects:
        trid = traj.trajid()
        if trid not in unique_sec: continue
        pair_time[pair_trids == trid] -= traj.time(0)
        
    return pair_trids, pair_time
