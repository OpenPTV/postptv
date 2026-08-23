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
