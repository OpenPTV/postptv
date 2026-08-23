"""
Interpolation benchmarks (SPEEDUP_PLAN Phase 0 / Phase 1).

Run with::

    uv run pytest benchmarks/ --benchmark-save=<phase>

All RNGs are seeded with ``np.random.default_rng(42)`` for reproducibility.
"""

import numpy as np
import pytest
from flowtracks.interpolation import InverseDistanceWeighter, Interpolant


def _make(n, m, d, rng):
    tracer_pos = rng.random((n, 3))
    interp_points = rng.random((m, 3))
    data = rng.random((n, d))
    return tracer_pos, interp_points, data


def test_idw_call(benchmark):
    """InverseDistanceWeighter.__call__ at n=5000, m=2000, d=3."""
    rng = np.random.default_rng(42)
    tracer_pos, interp_points, data = _make(5000, 2000, 3, rng)
    w = InverseDistanceWeighter(num_neighbs=4)

    def run():
        return w(tracer_pos, interp_points, data)
    benchmark(run)


def test_idw_scene(benchmark):
    """IDW scene path: set_scene + interpolate() at n=5000, m=2000."""
    rng = np.random.default_rng(42)
    tracer_pos, interp_points, data = _make(5000, 2000, 3, rng)
    w = Interpolant('inv', num_neighbs=4)
    w.set_scene(tracer_pos, interp_points, data)

    benchmark(w.interpolate)


def test_rbf_call(benchmark):
    """GeneralInterpolant rbf __call__ at n=2000, m=500 (dense path)."""
    rng = np.random.default_rng(42)
    tracer_pos, interp_points, data = _make(2000, 500, 3, rng)
    rbf = Interpolant('rbf', num_neighbs=7, param=1e5)

    def run():
        return rbf(tracer_pos, interp_points, data)
    benchmark(run)


def test_neighb_dists(benchmark):
    """neighb_dists at n=5000, m=2000."""
    rng = np.random.default_rng(42)
    tracer_pos, interp_points, data = _make(5000, 2000, 3, rng)
    w = InverseDistanceWeighter(num_neighbs=4)

    def run():
        return w.neighb_dists(tracer_pos, interp_points)
    benchmark(run)
