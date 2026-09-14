"""
Tests for flowtracks.stitching.stitch_trajectories.
"""

import unittest
import numpy as np
from flowtracks.trajectory import Trajectory
from flowtracks import stitching


class TestTrajectoryStitching(unittest.TestCase):
    def test_stitching_broken_linear_trajectory(self):
        """Stitches a linear trajectory split into two segments with a 2-frame gap."""
        fps = 10.0
        v = np.array([2.0, 0.0, 0.0])

        # Segment 1: frames 0..4 (5 frames)
        t1 = np.arange(0, 5, dtype=np.float64)
        pos1 = t1[:, None] * (v / fps)
        vel1 = np.tile(v, (len(t1), 1))
        tr1 = Trajectory(pos1, vel1, t1, trajid=101)

        # Segment 2: frames 7..11 (5 frames), 2-frame gap at 5, 6
        t2 = np.arange(7, 12, dtype=np.float64)
        pos2 = t2[:, None] * (v / fps)
        vel2 = np.tile(v, (len(t2), 1))
        tr2 = Trajectory(pos2, vel2, t2, trajid=102)

        stitched = stitching.stitch_trajectories([tr1, tr2], fps=fps, max_gap=3, max_distance=1.0)

        # Should produce exactly 1 merged trajectory
        self.assertEqual(len(stitched), 1)
        merged = stitched[0]

        # Total frames: 5 + 2 + 5 = 12 (0..11)
        self.assertEqual(len(merged), 12)
        np.testing.assert_array_equal(merged.time(), np.arange(12))

        # Position should accurately reconstruct linear motion
        expected_pos = np.arange(12)[:, None] * (v / fps)
        np.testing.assert_allclose(merged.pos(), expected_pos, atol=1e-5)

    def test_stitching_rejects_large_gap(self):
        """Rejects stitching if the frame gap exceeds max_gap."""
        fps = 1.0
        t1 = np.arange(0, 5, dtype=np.float64)
        pos1 = np.zeros((5, 3))
        tr1 = Trajectory(pos1, np.zeros((5, 3)), t1, 1)

        # Gap of 10 frames
        t2 = np.arange(15, 20, dtype=np.float64)
        pos2 = np.zeros((5, 3))
        tr2 = Trajectory(pos2, np.zeros((5, 3)), t2, 2)

        stitched = stitching.stitch_trajectories([tr1, tr2], fps=fps, max_gap=3)
        self.assertEqual(len(stitched), 2)

    def test_stitching_rejects_distant_particles(self):
        """Rejects stitching if spatial gap exceeds max_distance."""
        fps = 1.0
        t1 = np.arange(0, 5, dtype=np.float64)
        pos1 = np.zeros((5, 3))
        tr1 = Trajectory(pos1, np.zeros((5, 3)), t1, 1)

        t2 = np.arange(6, 11, dtype=np.float64)
        pos2 = np.ones((5, 3)) * 100.0  # Far away
        tr2 = Trajectory(pos2, np.zeros((5, 3)), t2, 2)

        stitched = stitching.stitch_trajectories([tr1, tr2], fps=fps, max_gap=3, max_distance=5.0)
        self.assertEqual(len(stitched), 2)


if __name__ == '__main__':
    unittest.main()


class TestFastStitching(unittest.TestCase):
    def _segment(self, trajid, t0, t1, v, fps):
        t = np.arange(t0, t1, dtype=np.float64)
        pos = t[:, None] * (np.asarray(v) / fps)
        vel = np.tile(np.asarray(v, dtype=float), (len(t), 1))
        return Trajectory(pos, vel, t, trajid)

    def test_fast_matches_reference_on_broken_linear(self):
        """Same single merge as stitch_trajectories on the canonical case."""
        fps = 10.0
        tr1 = self._segment(101, 0, 5, [2.0, 0.0, 0.0], fps)
        tr2 = self._segment(102, 7, 12, [2.0, 0.0, 0.0], fps)
        ref = stitching.stitch_trajectories([tr1, tr2], fps=fps, max_gap=3, max_distance=1.0)
        fast = stitching.stitch_trajectories_fast([tr1, tr2], fps=fps, max_gap=3, max_distance=1.0)
        self.assertEqual(len(fast), 1)
        np.testing.assert_array_equal(fast[0].time(), np.arange(12))
        np.testing.assert_allclose(
            fast[0].pos(), np.tile([0.2, 0.0, 0.0], (12, 1)) * np.arange(12)[:, None],
            atol=1e-5,
        )

    def test_fast_equivalence_random_fragments(self):
        """Greedy matching agrees with the reference on unambiguous fragments.

        Fragments are spaced far apart in space so at most one candidate join
        exists per endpoint; there the greedy and Hungarian results must be
        identical.
        """
        fps = 10.0

        # max_distance tiny -> nothing joins; both implementations agree
        rng = np.random.default_rng(42)
        trajs = []
        tid = 1
        for _ in range(60):
            p0 = rng.uniform(-1000, 1000, size=3)
            v = rng.uniform(-1, 1, size=3)
            ta = np.arange(0, 4, dtype=np.float64)
            tb = np.arange(6, 10, dtype=np.float64)
            trajs.append(Trajectory(
                p0[None, :] + ta[:, None] * (v / fps)[None, :],
                np.tile(v, (4, 1)), ta, tid))
            tid += 1
            trajs.append(Trajectory(
                p0[None, :] + tb[:, None] * (v / fps)[None, :],
                np.tile(v, (4, 1)), tb, tid))
            tid += 1
        kwargs = dict(fps=fps, max_gap=3, max_distance=1e-6, max_vel_diff=1e9)
        self.assertEqual(
            len(stitching.stitch_trajectories(list(trajs), **kwargs)),
            len(stitching.stitch_trajectories_fast(list(trajs), **kwargs)),
        )

        # now make each broken pair joinable: 50 far-apart linear tracks,
        # each split into two segments with a 2-frame gap
        trajs = []
        fps_v = np.array([1.0, 0.0, 0.0])
        for i in range(50):
            p0 = np.array([10.0 * i, 0.0, 0.0])
            ta = np.arange(0, 4, dtype=np.float64)
            pos_a = p0[None, :] + ta[:, None] * (fps_v / fps)[None, :]
            a = Trajectory(pos_a, np.tile(fps_v, (4, 1)), ta, 2 * i)
            tb = np.arange(6, 10, dtype=np.float64)
            pos_b = p0[None, :] + tb[:, None] * (fps_v / fps)[None, :]
            b = Trajectory(pos_b, np.tile(fps_v, (4, 1)), tb, 2 * i + 1)
            trajs.extend([a, b])
        kwargs = dict(fps=fps, max_gap=3, max_distance=0.5, max_vel_diff=1e-6)
        ref = stitching.stitch_trajectories(list(trajs), **kwargs)
        fast = stitching.stitch_trajectories_fast(list(trajs), **kwargs)
        self.assertEqual(len(ref), 50)
        self.assertEqual(len(fast), 50)
        for r, f in zip(sorted(ref, key=lambda x: x.trajid()), sorted(fast, key=lambda x: x.trajid())):
            np.testing.assert_allclose(r.pos(), f.pos(), atol=1e-9)
            np.testing.assert_array_equal(r.time(), f.time())

    def test_fast_rejects_large_gap_and_distance(self):
        """Criteria behave exactly as documented for the reference."""
        fps = 1.0
        far = Trajectory(np.ones((5, 3)) * 100.0, np.zeros((5, 3)), np.arange(5.0), 2)
        near_gap = Trajectory(np.zeros((5, 3)), np.zeros((5, 3)), np.arange(15, 20.0), 3)
        base = Trajectory(np.zeros((5, 3)), np.zeros((5, 3)), np.arange(5.0), 1)
        self.assertEqual(len(stitching.stitch_trajectories_fast([base, far], max_gap=3, max_distance=5.0)), 2)
        self.assertEqual(len(stitching.stitch_trajectories_fast([base, near_gap], max_gap=3, max_distance=5.0)), 2)

    def test_chain_of_three_segments_is_not_duplicated(self):
        """A->B->C of one particle gives ONE trajectory; B must not be copied into two."""
        fps = 1.0
        segs = [self._segment(i, t0, t0 + 5, [1.0, 0, 0], fps) for i, t0 in enumerate((0, 6, 12))]
        kwargs = dict(fps=fps, max_gap=2, max_distance=0.5, max_vel_diff=0.5)
        for fn in (stitching.stitch_trajectories, stitching.stitch_trajectories_fast):
            out = fn(list(segs), **kwargs)
            self.assertEqual(len(out), 1, fn.__name__)
            np.testing.assert_array_equal(out[0].time(), np.arange(17))

    def test_fast_passes_through_non_monotonic_segments(self):
        """Unlinked-particle buckets (repeated times) are not stitched."""
        fps = 1.0
        bucket = Trajectory(np.zeros((4, 3)), np.zeros((4, 3)), np.array([1, 1, 2, 2.0]), 0)
        normal = self._segment(7, 3, 6, [1.0, 0, 0], fps)
        out = stitching.stitch_trajectories_fast([bucket, normal], fps=fps, max_gap=5, max_distance=100.0)
        self.assertEqual(len(out), 2)


if __name__ == '__main__':
    unittest.main()
