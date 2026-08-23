"""Vectorized single-read binning must reproduce the legacy per-frame algorithm."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from flowtracks.eulerian import VEL_VARS, eulerian_grid

GRID = {"stepx": 4, "stepy": 3, "stepz": 2,
        "min_x": -0.01, "max_x": 0.02, "min_y": 0.0, "max_y": 0.05,
        "min_z": -0.03, "max_z": 0.03}


class RandomScene:
    """Random particles per frame, some outside the domain; both APIs."""

    def __init__(self, first, last, seed=0):
        rng = np.random.default_rng(seed)
        self.frames = {}
        for t in range(first, last + 1):
            n = rng.integers(0, 40)
            pos = rng.uniform([-0.02, -0.01, -0.04], [0.03, 0.06, 0.04], (n, 3))
            vel = rng.normal(size=(n, 3))
            self.frames[t] = (pos, vel)

    def frame_by_time(self, t):
        pos, vel = self.frames.get(t, (np.empty((0, 3)), np.empty((0, 3))))

        class Frame:
            def pos(self):
                return pos

            def velocity(self):
                return vel

        return Frame()

    def collect(self, keys):
        ts = sorted(self.frames)
        cols = {
            "pos": np.concatenate([self.frames[t][0] for t in ts]),
            "velocity": np.concatenate([self.frames[t][1] for t in ts]),
            "time": np.concatenate(
                [np.full(len(self.frames[t][0]), t) for t in ts]),
        }
        return [cols[k] for k in keys]


def legacy_eulerian_grid(scene, grid_params, first, last, cycletime,
                         deltat, base_time, min_count):
    """Verbatim port of batch_Lagrangian_to_Eulerian.eulerian_grid math."""
    zaman = deltat * 2 + 1
    fin = int(np.ceil(cycletime / zaman))
    steps = [grid_params[f"step{d}"] for d in "xyz"]
    bins = [np.linspace(grid_params[f"min_{d}"], grid_params[f"max_{d}"], n + 1)
            for d, n in zip("xyz", steps)]
    counts = np.zeros((*steps, fin), dtype=np.int64)
    sums = [np.zeros((*steps, fin)) for _ in range(3)]
    for frame_num in range(first, last + 1):
        frame = scene.frame_by_time(frame_num)
        pos = np.asarray(frame.pos(), dtype=float)
        vel = np.asarray(frame.velocity(), dtype=float)
        if pos.shape[0] == 0:
            continue
        mask = np.ones(pos.shape[0], dtype=bool)
        for d, b in enumerate(bins):
            mask &= (pos[:, d] >= b[0]) & (pos[:, d] < b[-1])
        pos, vel = pos[mask], vel[mask]
        if pos.shape[0] == 0:
            continue
        hc = int(np.ceil((frame_num - base_time) / float(cycletime)))
        cycle_start = cycletime * (hc - 1) + base_time
        ti = int(np.clip(np.ceil((frame_num - cycle_start) / float(zaman)) - 1,
                         0, fin - 1))
        idx = tuple(np.clip(np.digitize(pos[:, d], bins[d]) - 1, 0, steps[d] - 1)
                    for d in range(3))
        np.add.at(counts, (*idx, ti), 1)
        for d in range(3):
            np.add.at(sums[d], (*idx, ti), vel[:, d])
    low = counts < min_count
    counts[low] = 0
    means = []
    for d in range(3):
        sums[d][low] = 0.0
        means.append(np.divide(sums[d], counts, out=np.zeros_like(sums[d]),
                               where=counts != 0))
    return means, counts


def _compare(first, last, cycletime, deltat, min_count):
    scene = RandomScene(first, last)
    new = eulerian_grid(scene, GRID, first, last, cycletime,
                        deltat=deltat, min_count=min_count)
    means, counts = legacy_eulerian_grid(scene, GRID, first, last, cycletime,
                                         deltat, 100000, min_count)
    np.testing.assert_array_equal(new["par_ave2"].values, counts)
    for d, v in enumerate(VEL_VARS):
        np.testing.assert_allclose(new[v].values, means[d], rtol=1e-12, atol=0)


def test_equivalence_basic():
    _compare(first=100001, last=100200, cycletime=50, deltat=5, min_count=3)


def test_equivalence_cycle_not_divisible_by_zaman():
    # cycletime % zaman != 0: last bin of each cycle clips — the tricky path
    _compare(first=100001, last=100150, cycletime=37, deltat=4, min_count=1)


def test_equivalence_range_not_starting_at_cycle_boundary():
    _compare(first=100013, last=100160, cycletime=29, deltat=3, min_count=2)


def test_equivalence_float_cycletime():
    # main() computes cycletime via np.ceil -> float; both paths must agree
    _compare(first=100001, last=100100, cycletime=float(np.ceil(4286 / 100)),
             deltat=6, min_count=1)
