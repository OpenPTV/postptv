"""Tests for flowtracks.pairs.particle_pairs."""

import numpy as np

from flowtracks.pairs import particle_pairs
from flowtracks.trajectory import Trajectory


def test_particle_pairs_closest_and_unmatched():
    # Primary trajectory id=1 present at frames 10, 11, 12
    p1 = Trajectory(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.zeros((3, 3)),
        np.array([10, 11, 12]),
        trajid=1,
    )
    # Secondary trajectories (segs=True requires presence at frame and frame+1):
    # s1 (id=101) at frames 10, 11, 12 near (0.1, 0, 0) at f=10 and (5.0, 0, 0) at f=11
    s1 = Trajectory(
        np.array([[0.1, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        np.zeros((3, 3)),
        np.array([10, 11, 12]),
        trajid=101,
    )
    # s2 (id=102) at frames 10, 11, 12 near (3.0, 0, 0) at f=10 and (1.05, 0, 0) at f=11
    # (at frame 12 neither secondary trajectory has a segment to frame 13 -> unmatched)
    s2 = Trajectory(
        np.array([[3.0, 0.0, 0.0], [1.05, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.zeros((3, 3)),
        np.array([10, 11, 12]),
        trajid=102,
    )

    trajids = np.array([1, 1, 1])
    time_points = np.array([0, 1, 2])  # frames 10, 11, 12
    pair_trids, pair_time = particle_pairs([p1], [s1, s2], trajids, time_points)

    # Frame 10 -> closest is s1 (id=101) at local index 0
    # Frame 11 -> closest is s2 (id=102) at local index 1
    # Frame 12 -> no secondary trajectories -> (-1, -1)
    np.testing.assert_array_equal(pair_trids, [101, 102, -1])
    np.testing.assert_array_equal(pair_time, [0, 1, -1])
