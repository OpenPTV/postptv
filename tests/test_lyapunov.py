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


def _flowmap_trajs(x0, x1, t_start=0, t_end=5):
    """One two-point trajectory per row: x0 at t_start, x1 at t_end."""
    return [_traj([a, b], [t_start, t_end], i)
            for i, (a, b) in enumerate(zip(x0, x1))]


class TestEulerianFTLE(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.x0 = self.rng.uniform(-0.02, 0.02, size=(4000, 3))
        self.grid = (np.linspace(-0.01, 0.01, 4),) * 3
        self.dt, self.n = 0.01, 5   # T = 0.05 s

    def _check_linear(self, fmat, noise=0.0, **kwds):
        x1 = self.x0 @ fmat.T + self.rng.normal(scale=noise,
                                                size=self.x0.shape)
        trajs = _flowmap_trajs(x1=x1, x0=self.x0, t_end=self.n)
        res = lyapunov.eulerian_ftle(trajs, self.grid, 0, self.n,
                                     radius=0.01, dt=self.dt, **kwds)
        T = self.n * self.dt
        exact = 0.5 * np.log(np.linalg.eigvalsh(fmat.T @ fmat)[-1]) / T
        return res, exact

    def test_pure_strain(self):
        a, T = 8.0, 0.05
        fmat = np.diag([np.exp(a * T), np.exp(-a * T), 1.0])
        res, exact = self._check_linear(fmat)
        self.assertAlmostEqual(exact, a, places=9)
        self.assertTrue(res.valid.all())
        np.testing.assert_allclose(res.ftle.values, a, rtol=1e-8)
        np.testing.assert_allclose(res.det_F.values, 1.0, rtol=1e-8)

    def test_shear(self):
        gamma_t = 0.4
        fmat = np.array([[1, gamma_t, 0], [0, 1, 0], [0, 0, 1.]])
        res, exact = self._check_linear(fmat)
        np.testing.assert_allclose(res.ftle.values, exact, rtol=1e-8)
        np.testing.assert_allclose(res.det_F.values, 1.0, rtol=1e-8)

    def test_noise_floor_and_accuracy(self):
        a, T = 8.0, 0.05
        fmat = np.diag([np.exp(a * T), np.exp(-a * T), 1.0])
        sigma = 2e-5
        res, _ = self._check_linear(fmat, noise=sigma, position_error=sigma)
        err = (res.ftle - a).values[res.valid.values]
        # Estimated noise floor is the right order of magnitude of the
        # actual scatter, and well below the signal.
        floor = float(res.ftle_noise_floor.where(res.valid).mean())
        self.assertLess(floor, 0.1 * a)
        self.assertLess(np.abs(err).max(), 5 * floor)
        self.assertAlmostEqual(float(np.std(err)) / floor, 1.0, delta=0.8)

    def test_backward_window(self):
        a, T = 8.0, 0.05
        fmat = np.diag([np.exp(a * T), np.exp(-a * T), 1.0])
        # Backward map: positions at frame 5 -> frame 0, i.e. inverse of F.
        x_end = self.x0 @ fmat.T
        trajs = _flowmap_trajs(self.x0, x_end, t_end=5)
        res = lyapunov.eulerian_ftle(
            trajs, self.grid, 5, 0, radius=0.01, dt=self.dt)
        # Inverse map has the same singular values magnitude on the
        # contracting axis -> backward FTLE = a for this symmetric strain.
        self.assertTrue(res.valid.all())
        np.testing.assert_allclose(res.ftle.values, a, rtol=1e-6)

    def test_only_spanning_trajectories_used(self):
        x1 = self.x0 * 1.0
        trajs = _flowmap_trajs(self.x0, x1, t_end=5)
        # Add trajectories that do not reach t_end.
        trajs += [_traj([[0, 0, 0], [0, 0, 0]], [0, 3], 10_000 + i)
                  for i in range(50)]
        x0, _, ids = lyapunov.flow_map(trajs, 0, 5)
        self.assertEqual(len(x0), 4000)
        self.assertTrue((ids < 4000).all())

    def test_sparse_and_edge_nodes_masked(self):
        fmat = np.eye(3)
        x0 = self.x0[self.x0[:, 0] < 0.0]            # data only for x < 0
        trajs = _flowmap_trajs(x0, x0 @ fmat.T)
        grid = (np.array([-0.01, 0.0, 0.015]), np.array([0.0]),
                np.array([0.0]))
        res = lyapunov.eulerian_ftle(trajs, grid, 0, 5, radius=0.01, dt=0.01)
        valid = res.valid.values[:, 0, 0]
        self.assertTrue(valid[0])          # inside the data
        self.assertFalse(valid[1])         # at the edge: one-sided neighbours
        self.assertFalse(valid[2])         # no data at all
        self.assertTrue(np.isnan(res.ftle.values[2, 0, 0]))

    def test_degenerate_planar_cloud_masked(self):
        x0 = self.x0.copy()
        x0[:, 2] = 0.0                      # all in a plane: F not defined
        trajs = _flowmap_trajs(x0, x0)
        res = lyapunov.eulerian_ftle(
            trajs, self.grid, 0, 5, radius=0.01, dt=0.01)
        self.assertFalse(res.valid.any())

    def test_warns_when_no_trajectory_spans_window(self):
        trajs = _flowmap_trajs(self.x0[:20], self.x0[:20], t_end=5)
        with self.assertWarns(UserWarning):
            res = lyapunov.eulerian_ftle(
                trajs, self.grid, 0, 9, radius=0.01, dt=0.01)
        self.assertFalse(res.valid.any())

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            lyapunov.eulerian_ftle([], self.grid, 3, 3, radius=0.01)
        with self.assertRaises(ValueError):
            lyapunov.eulerian_ftle([], self.grid, 0, 3, radius=-1)
        with self.assertRaises(ValueError):
            lyapunov.eulerian_ftle([], self.grid, 0, 3, radius=0.01,
                                   min_neighbors=3)
        with self.assertRaises(ValueError):
            lyapunov.eulerian_ftle([], self.grid[:2], 0, 3, radius=0.01)
