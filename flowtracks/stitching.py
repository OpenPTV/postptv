"""
Trajectory stitching routines for reconnecting broken trajectory segments.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from flowtracks.trajectory import Trajectory


def stitch_trajectories(
    trajs,
    fps=1.0,
    max_gap=5,
    max_distance=5.0,
    max_vel_diff=10.0,
    dt=None,
):
    """Stitch (relink) broken trajectory segments across short frame gaps.

    Parameters
    ----------
    trajs : list of Trajectory
        List of trajectory objects to be stitched.
    fps : float, optional
        Frames per second (or sampling frequency). Default is 1.0.
    max_gap : int, optional
        Maximum number of missing frames allowed between track end and start.
        e.g., max_gap=5 means a gap of 1 to 5 missing frames can be bridged.
    max_distance : float, optional
        Maximum spatial distance between the extrapolated end of Track A and
        start of Track B.
    max_vel_diff : float, optional
        Maximum magnitude of velocity difference between Track A end and Track B start.
    dt : float, optional
        Time step delta between consecutive frames. If None, derived as 1.0 / fps.

    Returns
    -------
    stitched_trajs : list of Trajectory
        List of stitched Trajectory objects with gaps interpolated.
    """
    if dt is None:
        dt = 1.0 / fps

    if not trajs:
        return []

    active_trajs = list(trajs)

    while True:
        N = len(active_trajs)
        if N < 2:
            break

        ends = []
        starts = []

        for tr in active_trajs:
            t_tr = tr.time()
            pos_tr = tr.pos()
            vel_tr = tr.velocity()

            ends.append({
                't': t_tr[-1],
                'pos': pos_tr[-1],
                'vel': vel_tr[-1],
            })

            starts.append({
                't': t_tr[0],
                'pos': pos_tr[0],
                'vel': vel_tr[0],
            })

        # Bucket both ends and starts by frame, then evaluate one (end-frame,
        # start-frame) bucket pair at a time as a single vectorised numpy op
        # instead of a Python-level np.linalg.norm call per pair. A real
        # dataset's trajectory count (tens of thousands after tracking) makes
        # both a full N^2 pair scan AND N^2 individual small-array norm()
        # calls too slow to finish - per-pair Python/numpy dispatch overhead
        # dominates even once the candidate SET itself is bucketed down.
        # Batching within each bucket pair changes neither the candidate set
        # nor the costs, only how they get computed.
        starts_by_frame: dict[int, list[int]] = {}
        for j, s in enumerate(starts):
            starts_by_frame.setdefault(int(round(s['t'])), []).append(j)
        ends_by_frame: dict[int, list[int]] = {}
        for i, e in enumerate(ends):
            ends_by_frame.setdefault(int(round(e['t'])), []).append(i)

        end_pos = np.array([e['pos'] for e in ends])
        end_vel = np.array([e['vel'] for e in ends])
        start_pos = np.array([s['pos'] for s in starts])
        start_vel = np.array([s['vel'] for s in starts])

        candidates = []
        for t_end_frame, end_idx in ends_by_frame.items():
            end_idx = np.asarray(end_idx)
            p_end = end_pos[end_idx][:, None, :]  # (m,1,3)
            v_end = end_vel[end_idx][:, None, :]  # (m,1,3)

            for gap_frames in range(0, max_gap + 1):
                start_idx = starts_by_frame.get(t_end_frame + gap_frames + 1)
                if not start_idx:
                    continue
                start_idx = np.asarray(start_idx)
                p_start = start_pos[start_idx][None, :, :]  # (1,n,3)
                v_start = start_vel[start_idx][None, :, :]  # (1,n,3)

                delta_t = (gap_frames + 1) * dt
                dist_fwd = np.linalg.norm(p_end + v_end * delta_t - p_start, axis=2)
                dist_bwd = np.linalg.norm(p_start - v_start * delta_t - p_end, axis=2)
                dist = 0.5 * (dist_fwd + dist_bwd)  # (m,n)
                vel_diff = np.linalg.norm(v_end - v_start, axis=2)  # (m,n)

                rows, cols = np.nonzero((dist <= max_distance) & (vel_diff <= max_vel_diff))
                for r, c in zip(rows, cols):
                    gi, gj = int(end_idx[r]), int(start_idx[c])
                    if gi == gj:
                        continue
                    candidates.append((gi, gj, float(dist[r, c] + 0.1 * vel_diff[r, c])))

        if not candidates:
            break

        # Solve the assignment only over the rows/columns that actually have
        # a candidate - a dense NxN matrix (and linear_sum_assignment's cost
        # on it) is the other place N^2 previously showed up, and real gap
        # candidates are a small fraction of all trajectories.
        rows = sorted({i for i, _, _ in candidates})
        cols = sorted({j for _, j, _ in candidates})
        row_pos = {i: r for r, i in enumerate(rows)}
        col_pos = {j: c for c, j in enumerate(cols)}

        BIG_COST = 1e9
        cost_matrix = np.full((len(rows), len(cols)), BIG_COST, dtype=np.float64)
        for i, j, c in candidates:
            cost_matrix[row_pos[i], col_pos[j]] = min(cost_matrix[row_pos[i], col_pos[j]], c)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for r, c in zip(row_ind, col_ind):
            cost_val = cost_matrix[r, c]
            if cost_val < BIG_COST / 2:
                matches.append((rows[r], cols[c]))

        if not matches:
            break

        merged_mask = np.zeros(N, dtype=bool)
        new_active_trajs = []

        for i, j in matches:
            merged_mask[i] = True
            merged_mask[j] = True

            trA = active_trajs[i]
            trB = active_trajs[j]

            tA = trA.time()
            tB = trB.time()

            t_end = tA[-1]
            t_start = tB[0]
            gap_count = int(round(t_start - t_end - 1))

            if gap_count > 0:
                gap_times = np.arange(t_end + 1, t_start)
                p_end = trA.pos()[-1]
                p_start = trB.pos()[0]
                v_end = trA.velocity()[-1]
                v_start = trB.velocity()[0]

                s = np.linspace(0, 1, gap_count + 2)[1:-1]
                s2 = s**2
                s3 = s**3

                h00 = 2 * s3 - 3 * s2 + 1
                h10 = s3 - 2 * s2 + s
                h01 = -2 * s3 + 3 * s2
                h11 = s3 - s2

                gap_dt = (t_start - t_end) * dt
                gap_pos = (
                    h00[:, None] * p_end
                    + h10[:, None] * (v_end * gap_dt)
                    + h01[:, None] * p_start
                    + h11[:, None] * (v_start * gap_dt)
                )

                dh00 = 6 * s2 - 6 * s
                dh10 = 3 * s2 - 4 * s + 1
                dh01 = -6 * s2 + 6 * s
                dh11 = 3 * s2 - 2 * s

                gap_vel = (
                    dh00[:, None] * p_end
                    + dh10[:, None] * (v_end * gap_dt)
                    + dh01[:, None] * p_start
                    + dh11[:, None] * (v_start * gap_dt)
                ) / gap_dt

                merged_pos = np.concatenate([trA.pos(), gap_pos, trB.pos()], axis=0)
                merged_vel = np.concatenate([trA.velocity(), gap_vel, trB.velocity()], axis=0)
                merged_time = np.concatenate([tA, gap_times, tB], axis=0)
            else:
                merged_pos = np.concatenate([trA.pos(), trB.pos()], axis=0)
                merged_vel = np.concatenate([trA.velocity(), trB.velocity()], axis=0)
                merged_time = np.concatenate([tA, tB], axis=0)

            merged_traj = Trajectory(merged_pos, merged_vel, merged_time, trA.trajid())

            for k in trA.as_dict():
                if k not in ('pos', 'velocity', 'time', 'trajid') and trB.has_property(k):
                    propA = trA.as_dict()[k]
                    propB = trB.as_dict()[k]
                    if gap_count > 0:
                        prop_end = propA[-1]
                        prop_start = propB[0]
                        gap_prop = np.array([prop_end + (prop_start - prop_end) * si for si in s])
                        merged_prop = np.concatenate([propA, gap_prop, propB], axis=0)
                    else:
                        merged_prop = np.concatenate([propA, propB], axis=0)
                    merged_traj.create_property(k, merged_prop)

            new_active_trajs.append(merged_traj)

        for idx in range(N):
            if not merged_mask[idx]:
                new_active_trajs.append(active_trajs[idx])

        active_trajs = new_active_trajs

    return active_trajs
