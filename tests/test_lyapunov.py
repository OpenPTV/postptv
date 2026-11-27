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


# ---------------------------------------------------------------------------
# Ridge extraction


def _field(ftle, valid=None, axes=None):
    """Wrap an (nx, ny[, nz]) array as an eulerian_ftle-like Dataset."""
    import xarray as xr
    if ftle.ndim == 2:
        ftle = ftle[:, :, None]
    if valid is not None and valid.ndim == 2:
        valid = valid[:, :, None]
    nx, ny, nz = ftle.shape
    if axes is None:
        axes = (np.linspace(0, 1, nx), np.linspace(0, 1, ny),
                np.linspace(0, 1, nz) if nz > 1 else np.array([0.0]))
    data = {'ftle': (('x', 'y', 'z'), ftle)}
    if valid is not None:
        data['valid'] = (('x', 'y', 'z'), valid)
    return xr.Dataset(data, coords=dict(zip('xyz', axes)))


def _ridge_quality(ridge_xy, dist_to_truth, tol, truth_xy_dist_fn=None):
    """Fraction of ridge nodes within tol of the truth (precision)."""
    return float((dist_to_truth[ridge_xy] <= tol).mean())


class TestFTLERidges(unittest.TestCase):
    def setUp(self):
        n = 81
        self.g = np.linspace(0, 1, n)
        self.X, self.Y = np.meshgrid(self.g, self.g, indexing='ij')
        self.h = self.g[1] - self.g[0]

    def test_single_line_ridge_located(self):
        w = 0.04
        f = 5 * np.exp(-0.5 * ((self.Y - 0.4) / w) ** 2)
        res = lyapunov.ftle_ridges(_field(f), smooth=0.0, min_ftle=1.0)
        r = res.ridge.values[:, :, 0]
        # Ridge nodes lie on y = 0.4 (to within one grid step), and run the
        # whole way along x. The outermost nodes never carry a ridge, so the
        # line spans the 79 interior columns.
        ys = self.Y[r]
        self.assertGreater(r.sum(), 60)
        self.assertLessEqual(np.abs(ys - 0.4).max(), self.h)
        self.assertEqual(len(np.unique(self.X[r])), 79)
        self.assertFalse(r[0].any() or r[-1].any())

    def test_ridge_is_one_node_thick(self):
        f = 5 * np.exp(-0.5 * ((self.Y - 0.4) / 0.04) ** 2)
        r = lyapunov.ftle_ridges(
            _field(f), smooth=0.0, min_ftle=1.0).ridge.values[:, :, 0]
        self.assertLessEqual(r.sum(axis=1).max(), 1)

    def test_tilted_line_and_noise(self):
        rng = np.random.default_rng(0)
        d = (self.Y - self.X - 0.1) / np.sqrt(2)      # distance to y = x+0.1
        f = 5 * np.exp(-0.5 * (d / 0.04) ** 2)
        f = f + rng.normal(scale=0.05, size=f.shape)
        res = lyapunov.ftle_ridges(_field(f), smooth=1.5, min_ftle=2.0)
        r = res.ridge.values[:, :, 0]
        dist = np.abs(d)
        self.assertGreater(r.sum(), 40)
        # Precision: nearly every ridge node is on the true line.
        self.assertGreater((dist[r] <= 2 * self.h).mean(), 0.97)
        # Nothing in the background, far from the line.
        self.assertFalse(r[dist > 0.15].any())

    def test_cross_of_two_ridges(self):
        f = (4 * np.exp(-0.5 * ((self.X - 0.5) / 0.04) ** 2)
             + 4 * np.exp(-0.5 * ((self.Y - 0.5) / 0.04) ** 2))
        res = lyapunov.ftle_ridges(_field(f), smooth=1.0, min_ftle=2.0)
        r = res.ridge.values[:, :, 0]
        near = (np.abs(self.X - 0.5) <= 2 * self.h) | \
               (np.abs(self.Y - 0.5) <= 2 * self.h)
        self.assertGreater((r & near).sum(), 100)
        self.assertGreater((r & near).sum() / r.sum(), 0.97)

    def test_default_threshold_uses_percentile(self):
        f = 5 * np.exp(-0.5 * ((self.Y - 0.4) / 0.04) ** 2)
        res = lyapunov.ftle_ridges(_field(f), smooth=0.0, min_percentile=90)
        self.assertAlmostEqual(
            res.attrs['min_ftle_used'],
            np.percentile(res.ftle_smooth.values[np.isfinite(
                res.ftle_smooth.values)], 90))
        self.assertGreater(res.ridge.sum(), 50)

    def test_flat_and_monotonic_fields_have_no_ridges(self):
        flat = np.full_like(self.X, 3.0)
        ramp = 2.0 * self.X + 1.0
        for f in (flat, ramp):
            res = lyapunov.ftle_ridges(_field(f), smooth=0.0, min_ftle=0.0)
            self.assertFalse(res.ridge.values.any())

    def test_valleys_are_not_ridges(self):
        f = 5 - 5 * np.exp(-0.5 * ((self.Y - 0.4) / 0.04) ** 2)
        res = lyapunov.ftle_ridges(_field(f), smooth=0.0, min_ftle=0.0)
        r = res.ridge.values[:, :, 0]
        # A valley bottom is a minimum across; any "ridges" found must be on
        # the shoulders, never at the valley bottom.
        self.assertFalse(r[np.abs(self.Y - 0.4) < self.h].any())

    def test_invalid_nodes_are_excluded(self):
        f = 5 * np.exp(-0.5 * ((self.Y - 0.4) / 0.04) ** 2)
        valid = np.ones_like(f, dtype=bool)
        valid[30:50, :] = False
        res = lyapunov.ftle_ridges(_field(f, valid), smooth=1.0, min_ftle=1.0)
        r = res.ridge.values[:, :, 0]
        self.assertFalse(r[valid == False].any())     # noqa: E712
        self.assertTrue(r[:25].any() and r[55:].any())

    def test_3d_planar_ridge(self):
        g = np.linspace(0, 1, 31)
        X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
        f = 5 * np.exp(-0.5 * ((X - 0.5) / 0.08) ** 2)
        res = lyapunov.ftle_ridges(_field(f), smooth=0.0, min_ftle=1.0)
        r = res.ridge.values
        # A plane across the volume, over the 29 x 29 interior nodes.
        self.assertGreater(r.sum(), 0.9 * 29 * 29)
        self.assertLessEqual(np.abs(X[r] - 0.5).max(), g[1] - g[0])
        self.assertFalse(r[0].any() or r[-1].any() or r[:, 0].any()
                         or r[:, :, -1].any())

    def test_nonuniform_grid_rejected(self):
        f = np.zeros((10, 10))
        ax = (np.r_[0, 1, 2, 4, 5, 6, 7, 8, 9, 10.], np.arange(10.),
              np.array([0.0]))
        with self.assertRaises(ValueError):
            lyapunov.ftle_ridges(_field(f, axes=ax))

    def test_argument_validation(self):
        f = _field(np.zeros((10, 10)))
        with self.assertRaises(ValueError):
            lyapunov.ftle_ridges(f, min_percentile=120)
        with self.assertRaises(ValueError):
            lyapunov.ftle_ridges(f, smooth=-1)
        with self.assertRaises(ValueError):
            lyapunov.ftle_ridges(f.rename({'ftle': 'other'}))
        # A line of nodes is not enough; neither is a 2-node dimension.
        with self.assertRaises(ValueError):
            lyapunov.ftle_ridges(_field(np.zeros((10, 1))))
        with self.assertRaises(ValueError):
            lyapunov.ftle_ridges(_field(np.zeros((10, 10, 2))))


def _taylor_green_map(x0, k, amp, T, nsteps=200):
    """Flow map of u = amp*(-sin kx cos ky, cos kx sin ky, 0), by RK4."""
    def vel(p):
        out = np.zeros_like(p)
        out[:, 0] = -amp * np.sin(k * p[:, 0]) * np.cos(k * p[:, 1])
        out[:, 1] = amp * np.cos(k * p[:, 0]) * np.sin(k * p[:, 1])
        return out
    p, h = x0.copy(), T / nsteps
    for _ in range(nsteps):
        k1 = vel(p)
        k2 = vel(p + 0.5 * h * k1)
        k3 = vel(p + 0.5 * h * k2)
        k4 = vel(p + h * k3)
        p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


class TestRidgesEndToEnd(unittest.TestCase):
    """
    Seeds noisy particles in a steady Taylor-Green flow, estimates the Eulerian
    FTLE from the scattered flow map, extracts ridges, and checks them against
    the known separatrices of the flow (the lines x = L/2 and y = L/2).
    """
    L = 0.04

    def _ridges(self, radius=3e-3, sigma=20e-6, npart=30000, seed=2):
        rng = np.random.default_rng(seed)
        L, amp, T, nfr = self.L, 0.05, 0.4, 10
        k = 2 * np.pi / L
        x0 = np.column_stack([rng.uniform(-0.01, L + 0.01, npart),
                              rng.uniform(-0.01, L + 0.01, npart),
                              rng.uniform(-0.005, 0.005, npart)])
        x1 = _taylor_green_map(x0, k, amp, T) + rng.normal(
            scale=sigma, size=x0.shape)
        x0 = x0 + rng.normal(scale=sigma, size=x0.shape)
        trajs = [Trajectory(np.array([a, b]), np.zeros((2, 3)),
                            np.array([0., nfr]), i)
                 for i, (a, b) in enumerate(zip(x0, x1))]
        g = np.linspace(0.002, L - 0.002, 46)
        ftle = lyapunov.eulerian_ftle(
            trajs, (g, g, np.array([0.0])), 0, nfr, radius=radius,
            dt=T / nfr, position_error=sigma)
        return g, ftle, lyapunov.ftle_ridges(ftle, smooth=1.0,
                                             min_percentile=80)

    def test_ridges_are_the_separatrices(self):
        radius = 3e-3
        g, ftle, rid = self._ridges(radius=radius)
        r = rid.ridge.values[:, :, 0]
        X, Y = np.meshgrid(g, g, indexing='ij')
        L = self.L
        # The steady flow u = amp*(-sin kx cos ky, cos kx sin ky) with
        # k = 2*pi/L has its separatrices on x or y in {0, L/2, L}; the grid
        # covers 2..38 mm, so the outer two are just outside it.
        lines = np.array([0.0, L / 2, L])
        dist = np.minimum(np.abs(X[..., None] - lines).min(-1),
                          np.abs(Y[..., None] - lines).min(-1))
        h = g[1] - g[0]

        self.assertGreater(r.sum(), 30)
        # The location is only determined to about the fit radius.
        precision = (dist[r] <= radius).mean()
        self.assertGreater(precision, 0.95)
        # ... and typically much better than that.
        self.assertLess(np.median(dist[r]), 0.5 * radius)
        # The test is not vacuous: random nodes would mostly fail it.
        self.assertLess((dist <= radius).mean(), 0.6)

        # Recall, on the interior separatrices (x = L/2 and y = L/2): most
        # of their nodes have a ridge node within one radius.
        on_line = (np.abs(X - L / 2) <= h) | (np.abs(Y - L / 2) <= h)
        self.assertGreater(on_line.sum(), 50)
        reach = int(np.ceil(radius / h))
        near_ridge = np.zeros_like(r)
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                near_ridge |= np.roll(np.roll(r, di, 0), dj, 1)
        self.assertGreater(near_ridge[on_line].mean(), 0.7)

    def test_no_ridges_without_signal(self):
        # Zero-flow field: FTLE is noise-only; with an absolute threshold
        # above the noise floor, nothing is reported.
        g, ftle, _ = self._ridges()
        rid = lyapunov.ftle_ridges(ftle, smooth=1.0, min_ftle=100.0)
        self.assertFalse(rid.ridge.values.any())
