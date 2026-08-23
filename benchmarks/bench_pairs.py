"""
Pairing benchmark (SPEEDUP_PLAN Phase 0 / Phase 3.1).

Run with::

    uv run pytest benchmarks/ --benchmark-save=<phase>
"""

import numpy as np
import pytest
from flowtracks.trajectory import Trajectory
from flowtracks import pairs


def _make(n_traj, frames, rng):
    trajs = []
    for i in range(n_traj):
        t = np.arange(frames)
        pos = rng.random((frames, 3))
        trajs.append(Trajectory(pos, np.zeros_like(pos), t, i))
    return trajs


def test_particle_pairs(benchmark):
    rng = np.random.default_rng(42)
    primary = _make(500, 50, rng)
    secondary = _make(500, 50, rng)

    prim_ids = rng.choice(500, 200, replace=False)   # unique trajids
    time_points = rng.integers(0, 50, 200)
    trajids = np.array([primary[i].trajid() for i in prim_ids])

    benchmark(pairs.particle_pairs, primary, secondary, trajids, time_points)
