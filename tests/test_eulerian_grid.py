import sys
from pathlib import Path
import numpy as np
import h5py
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from helpers import FakeScene
from flowtracks import eulerian as lag


GRID = {
    'stepx': 2, 'stepy': 2, 'stepz': 2,
    'min_x': -1.0, 'max_x': 1.0,
    'min_y': -1.0, 'max_y': 1.0,
    'min_z': -1.0, 'max_z': 1.0,
}


def _fake_scene(vel=(1.0, 2.0, 3.0), n_particles=5):
    # All particles sit at (0,0,0) -> bin index (1,1,1); constant velocity.
    pos = np.zeros((n_particles, 3))
    velocity = np.tile(np.array(vel, dtype=float), (n_particles, 1))
    return FakeScene(pos, velocity)


def test_eulerian_grid_shapes_and_means():
    scene = _fake_scene()
    ds = lag.eulerian_grid(
        scene, GRID, first=100001, last=100005, cycletime=100,
        deltat=90, base_time=100000, min_count=1,
    )
    u = ds['u_ins_mean'].values
    v = ds['v_ins_mean'].values
    w = ds['w_ins_mean'].values
    par = ds['par_ave2'].values

    assert u.shape == (2, 2, 2, 1)
    assert u[1, 1, 1, 0] == pytest.approx(1.0)
    assert v[1, 1, 1, 0] == pytest.approx(2.0)
    assert w[1, 1, 1, 0] == pytest.approx(3.0)
    assert par[1, 1, 1, 0] == 5 * 5  # 5 particles * 5 frames


def test_eulerian_grid_min_count_masking():
    scene = _fake_scene(n_particles=5)
    ds = lag.eulerian_grid(
        scene, GRID, first=100001, last=100005, cycletime=100,
        deltat=90, base_time=100000, min_count=100,
    )
    u = ds['u_ins_mean'].values
    par = ds['par_ave2'].values
    assert u[1, 1, 1, 0] == 0.0
    assert par[1, 1, 1, 0] == 0
