"""
Tests for flowtracks.repair.repair_trajectories.

Synthetic straight tracks with Gaussian position noise are broken the way a
tracker breaks them (missing frames, isolated single points) and one pair of
neighbouring tracks is cross-linked (a wrong link). The particle a point
belongs to is recoverable from its y coordinate, so every output trajectory
can be checked for purity without extra bookkeeping.
"""

import unittest

import numpy as np

from flowtracks.repair import _greedy_one_to_one, repair_arrays, repair_trajectories
from flowtracks.trajectory import Trajectory

N_PART, N_FRAMES, NOISE = 40, 80, 0.01
SPACING = 0.5  # y distance between neighbouring particles (>> noise, >> step)
VEL = np.array([0.1, 0.0, 0.02])


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N_FRAMES)
    tracks = []
    for p in range(N_PART):
        base = np.array([0.0, p * SPACING, 0.0])
        pos = base + t[:, None] * VEL + rng.normal(0.0, NOISE, (N_FRAMES, 3))
        tracks.append((t, pos))
    return tracks


def _pid(pos):
    return np.rint(pos[:, 1] / SPACING).astype(int)


def _traj(t, pos, tid):
    return Trajectory(pos, np.zeros_like(pos), t, tid)


def _broken(tracks):
    """Break tracks like a tracker would, and cross-link particles 10 and 11."""
    trajs, tid = [], 0
    for p, (t, pos) in enumerate(tracks):
        if p in (10, 11):
            continue
        # a 2-frame gap, a 1-frame gap, and one isolated point in between
        keep = np.ones(N_FRAMES, bool)
        keep[[20, 21, 50]] = False
        cuts = [0, 20, 22, 35, 36, 50, 51, N_FRAMES]
        for a, b in zip(cuts[:-1], cuts[1:]):
            idx = np.arange(a, b)
            idx = idx[keep[idx]]
            if len(idx):
                trajs.append(_traj(t[idx], pos[idx], tid))
                tid += 1
    (t, p10), (_, p11) = tracks[10], tracks[11]
    h = N_FRAMES // 2
    trajs.append(_traj(t, np.vstack([p10[:h], p11[h:]]), tid))      # wrong link at h
    trajs.append(_traj(t, np.vstack([p11[:h], p10[h:]]), tid + 1))  # and its mirror
    return trajs


class TestRepairTrajectories(unittest.TestCase):
    def test_rejoins_breaks_attaches_singles_and_cuts_wrong_links(self):
        trajs = _broken(_truth())
        n_points = sum(len(tr.time()) for tr in trajs)

        out, report = repair_trajectories(trajs, max_gap=3, link_percentile=99.9)

        # nothing lost, nothing duplicated
        self.assertEqual(sum(len(tr.time()) for tr in out), n_points)
        # every output trajectory holds exactly one particle, and one per particle
        pids = [set(_pid(tr.pos())) for tr in out]
        self.assertTrue(all(len(s) == 1 for s in pids), pids)
        self.assertEqual(len(out), N_PART)
        # the two cross-linked tracks were cut and rejoined correctly
        self.assertGreaterEqual(report["links_cut"], 2)
        self.assertGreater(report["joins"], 0)
        self.assertGreater(report["points_attached"], 0)
        # gaps stay gaps: no invented frames
        for tr in out:
            self.assertTrue(np.all(np.diff(tr.time()) >= 1))

    def test_clean_tracks_come_back_unchanged(self):
        trajs = [_traj(t, pos, k) for k, (t, pos) in enumerate(_truth(seed=1))]

        out, report = repair_trajectories(trajs)

        self.assertEqual(len(out), N_PART)
        self.assertEqual(sorted(len(tr.time()) for tr in out), [N_FRAMES] * N_PART)
        self.assertTrue(all(len(set(_pid(tr.pos()))) == 1 for tr in out))
        # a 99.9th-percentile threshold may cut a noise spike, but it is rejoined
        self.assertLessEqual(report["links_cut"], report["joins"])

    def test_too_little_data_is_skipped_not_guessed(self):
        t = np.arange(4)
        trajs = [_traj(t, np.c_[t, t, t].astype(float), 0)]

        out, report = repair_trajectories(trajs)

        self.assertIn("skipped", report)
        self.assertEqual(len(out), 1)

    def test_non_monotonic_bucket_passes_through(self):
        trajs = [_traj(t, pos, k) for k, (t, pos) in enumerate(_truth(seed=2))]
        bucket = Trajectory(np.zeros((3, 3)), np.zeros((3, 3)), np.array([5, 5, 6]), 999)

        out, report = repair_trajectories(trajs + [bucket])

        self.assertIn(999, [int(np.ravel(tr.trajid())[0]) for tr in out])
        self.assertEqual(report["n_out"], N_PART + 1)


class TestRepairArrays(unittest.TestCase):
    def test_arrays_api_repairs_in_input_row_order(self):
        trajs = _broken(_truth(seed=3))
        tid = np.concatenate([np.full(len(tr.time()), k) for k, tr in enumerate(trajs)])
        time = np.concatenate([tr.time() for tr in trajs])
        pos = np.concatenate([tr.pos() for tr in trajs])
        shuffle = np.random.default_rng(0).permutation(len(tid))

        new_tid, report = repair_arrays(tid[shuffle], time[shuffle], pos[shuffle])

        self.assertEqual(len(new_tid), len(tid))
        pid = _pid(pos[shuffle])
        for t in np.unique(new_tid):
            self.assertEqual(len(set(pid[new_tid == t])), 1)
        self.assertEqual(len(np.unique(new_tid)), N_PART)
        self.assertEqual(report["n_out"], N_PART)

    def test_greedy_matching_equals_a_sequential_scan(self):
        rng = np.random.default_rng(4)
        chi = rng.random(400)
        left, right = rng.integers(0, 60, 400), rng.integers(0, 60, 400)

        got = set(zip(*_greedy_one_to_one(chi, left, right)))

        used_l, used_r, want = set(), set(), set()
        for i in np.argsort(chi, kind="stable"):
            if left[i] in used_l or right[i] in used_r:
                continue
            used_l.add(left[i])
            used_r.add(right[i])
            want.add((left[i], right[i]))
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
