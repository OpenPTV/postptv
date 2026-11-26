"""
Tests for flowtracks.lyapunov (issue #65), on synthetic exponentially
separating pairs whose exponent is known exactly.
"""

import unittest

import numpy as np

from flowtracks import lyapunov
from flowtracks.trajectory import Trajectory


def _traj(pos, frames, trajid):
    pos = np.asarray(pos, dtype=float)
    return Trajectory(pos, np.zeros_like(pos), np.asarray(frames, float),
                      trajid)


def _exp_pair(lam, r0, dt, nframes, base_id, origin=(0., 0., 0.),
              direction=(1., 0., 0.), start=0):
    """Two particles; separation r0 * exp(lam * t) along ``direction``."""
    frames = np.arange(start, start + nframes)
    t = (frames - start) * dt
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    origin = np.asarray(origin, float)
    a = np.tile(origin, (nframes, 1))
    b = origin + (r0 * np.exp(lam * t))[:, None] * d
    return _traj(a, frames, base_id), _traj(b, frames, base_id + 1)


class TestFindPairs(unittest.TestCase):
    def test_range_and_overlap(self):
        a, b = _exp_pair(0.0, 0.01, 0.1, 10, 0)
        far = _traj(np.tile([5., 5., 5.], (10, 1)), np.arange(10), 2)
        pairs, f0 = lyapunov.find_pairs([a, b, far], 0.005, 0.02)
        np.testing.assert_array_equal(pairs, [[0, 1]])
        np.testing.assert_array_equal(f0, [0])

    def test_min_separation_excludes_close_pairs(self):
        a, b = _exp_pair(0.0, 0.001, 0.1, 10, 0)
        pairs, _ = lyapunov.find_pairs([a, b], 0.005, 0.02)
        self.assertEqual(len(pairs), 0)

    def test_no_temporal_overlap(self):
        a = _traj(np.zeros((5, 3)), np.arange(5), 0)
        b = _traj(np.full((5, 3), 0.01), np.arange(10, 15), 1)
        pairs, _ = lyapunov.find_pairs([a, b], 0.0, 0.1)
        self.assertEqual(len(pairs), 0)

    def test_bad_range(self):
        with self.assertRaises(ValueError):
            lyapunov.find_pairs([], 0.1, 0.05)

    def test_empty(self):
        pairs, f0 = lyapunov.find_pairs([], 0.0, 1.0)
        self.assertEqual(pairs.shape, (0, 2))
        self.assertEqual(len(f0), 0)


class TestFiniteTime(unittest.TestCase):
    def test_recovers_exponent(self):
        lam, dt = 0.5, 0.1
        trajs = []
        # Several pairs, different initial separations in range and
        # different locations/directions, starting at different frames.
        for k, (r0, d) in enumerate([(0.010, (1, 0, 0)), (0.012, (0, 1, 1)),
                                     (0.015, (1, 1, 1))]):
            a, b = _exp_pair(lam, r0, dt, 30, 2 * k,
                             origin=(k * 1.0, 0, 0), direction=d, start=k)
            trajs += [a, b]
        res = lyapunov.finite_time_lyapunov(trajs, 0.008, 0.02, dt=dt)
        self.assertAlmostEqual(float(res.lyapunov), lam, places=6)
        self.assertAlmostEqual(res.attrs['lyapunov_exponent'], lam, places=6)
        self.assertEqual(res.attrs['npairs_total'], 3)
        self.assertEqual(int(res.npairs[0]), 3)

    def test_fit_window_and_max_lag(self):
        lam, dt = 0.3, 0.05
        a, b = _exp_pair(lam, 0.01, dt, 40, 0)
        res = lyapunov.finite_time_lyapunov(
            [a, b], 0.005, 0.02, dt=dt, max_lag=10, fit_lags=(2, 8))
        self.assertEqual(res.sizes['lag'], 11)
        self.assertAlmostEqual(float(res.lyapunov), lam, places=6)

    def test_no_pairs_gives_nan(self):
        a = _traj(np.zeros((5, 3)), np.arange(5), 0)
        res = lyapunov.finite_time_lyapunov([a], 0.01, 0.02)
        self.assertTrue(np.isnan(float(res.lyapunov)))


class TestFiniteSize(unittest.TestCase):
    def test_recovers_exponent_at_all_scales(self):
        lam, dt = 0.5, 0.01
        a, b = _exp_pair(lam, 0.01, dt, 600, 0)
        res = lyapunov.finite_size_lyapunov(
            [a, b], 0.005, 0.02, ratio=2.0, num_scales=4, dt=dt)
        # Scales that start below the pair's initial separation cannot be
        # timed; the others recover lam (up to linear-interp error).
        ok = res.fsle.dropna('delta')
        self.assertGreaterEqual(len(ok), 2)
        np.testing.assert_allclose(ok.values, lam, rtol=0.02)
        self.assertTrue((res['count'].values[:1] == 0).all())

    def test_bad_ratio(self):
        with self.assertRaises(ValueError):
            lyapunov.finite_size_lyapunov([], 0.01, 0.02, ratio=1.0)


if __name__ == '__main__':
    unittest.main()
