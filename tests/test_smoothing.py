"""
Tests for flowtracks.smoothing.savitzky_golay.

Besides checking the filter on a polynomial trajectory whose derivatives are
known exactly, it also verifies the vectorized windowed-matmul implementation
against the original per-component ``np.convolve`` loop (regression guard for
the refactor in SPEEDUP_PLAN Phase 3.4).
"""

import unittest
import numpy as np
from flowtracks.trajectory import Trajectory
from flowtracks import smoothing


def _old_savitzky_golay(trajs, fps, window_size, order):
    """Faithful copy of the pre-refactor per-component convolve loop."""
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    b = np.array([[k**i for i in order_range]
                  for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b)
    m_pos = m[0]
    m_vel = m[1] * fps
    m_acc = m[2] * (fps**2 * 2)
    m_jerk = m[3] * (fps**3 * 6)

    smoothed_keys = ['pos', 'velocity', 'accel', 'acc_pp', 'time', 'trajid']

    new_trajs = []
    for traj in trajs:
        if len(traj) < window_size:
            continue

        newpos, newvel, newacc, jerk = [], [], [], []
        for y in traj.pos().T:
            firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
            lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            newpos.append(np.convolve(m_pos[::-1], y, mode='valid'))
            newvel.append(np.convolve(m_vel[::-1], y, mode='valid'))
            newacc.append(np.convolve(m_acc[::-1], y, mode='valid'))
            jerk.append(np.convolve(m_jerk[::-1], y, mode='valid'))

        newpos = np.r_[newpos].T
        newvel = np.r_[newvel].T
        newacc = np.r_[newacc].T
        jerk = np.r_[jerk].T

        nextvel = np.vstack((np.zeros(3), newvel + newacc / fps +
                             jerk / 2. / fps**2))[:-1]
        nextacc = np.vstack((np.zeros(3), newacc + jerk / fps))[:-1]

        newtraj = Trajectory(newpos, newvel, traj.time(), traj.trajid(),
                             accel=newacc, vel_pp=nextvel, acc_pp=nextacc)
        for k, v in traj.as_dict().items():
            if k not in smoothed_keys:
                newtraj.create_property(k, v)
        new_trajs.append(newtraj)
    return new_trajs


def _poly_trajectory(coeffs, n=80, fps=1., trajid=0):
    t = np.arange(n, dtype=np.float64)
    pos = np.stack([sum(coeffs[c, d] * t**d for d in range(4))
                    for c in range(3)], axis=1)
    return Trajectory(pos, np.zeros_like(pos), t, trajid)


class TestSavitzkyGolay(unittest.TestCase):
    def test_polynomial_exact(self):
        """SG of order >= polynomial degree reproduces it exactly (interior)."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 4))
        traj = _poly_trajectory(a, n=80)
        smoothed = smoothing.savitzky_golay([traj], fps=1., window_size=9,
                                            order=3)[0]

        # pos is a cubic -> reproduced exactly in the interior (the mirror
        # padding at the very ends is not a polynomial, so boundaries differ,
        # exactly as the original per-component convolve loop did).
        si = slice(10, -10)
        np.testing.assert_allclose(smoothed.pos()[si], traj.pos()[si], atol=1e-9)

        t = traj.time()
        vel = np.stack([a[c, 1] + 2 * a[c, 2] * t + 3 * a[c, 3] * t**2
                        for c in range(3)], axis=1)
        acc = np.stack([2 * a[c, 2] + 6 * a[c, 3] * t for c in range(3)],
                      axis=1)

        # Derivatives are exact away from the mirrored boundaries.
        sl = slice(10, -10)
        np.testing.assert_allclose(smoothed.velocity()[sl], vel[sl], atol=1e-8)
        np.testing.assert_allclose(smoothed.accel()[sl], acc[sl], atol=1e-8)

    def test_matches_old_implementation(self):
        """Vectorized version == original per-component convolve loop."""
        rng = np.random.default_rng(7)
        trajs = []
        for trid in range(4):
            n = 40 + trid * 5
            t = np.arange(n, dtype=np.float64)
            pos = rng.normal(size=(n, 3)).cumsum(axis=0)
            trajs.append(Trajectory(pos, np.zeros_like(pos), t, trid))

        new = smoothing.savitzky_golay(trajs, fps=30., window_size=11, order=3)
        old = _old_savitzky_golay(trajs, fps=30., window_size=11, order=3)

        self.assertEqual(len(new), len(old))
        for nt, ot in zip(new, old):
            np.testing.assert_allclose(nt.pos(), ot.pos(), atol=1e-8)
            np.testing.assert_allclose(nt.velocity(), ot.velocity(), atol=1e-8)
            np.testing.assert_allclose(nt.accel(), ot.accel(), atol=1e-8)

    def test_short_trajectory_dropped(self):
        t = np.arange(3, dtype=np.float64)
        traj = Trajectory(np.random.default_rng(1).normal(size=(3, 3)),
                          np.zeros((3, 3)), t, 0)
        self.assertEqual(
            smoothing.savitzky_golay([traj], fps=1., window_size=9, order=3),
            [])


if __name__ == '__main__':
    unittest.main()
