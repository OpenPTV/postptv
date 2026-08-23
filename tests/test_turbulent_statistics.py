import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from helpers import write_grid_h5
from flowtracks import eulerian as ts


GRID = {
    'stepx': 3, 'stepy': 3, 'stepz': 2,
    'min_x': -0.01, 'max_x': 0.02,
    'min_y': 0.0, 'max_y': 0.05,
    'min_z': -0.03, 'max_z': 0.03,
}


def _write_config(tmp_path, shift=0):
    (tmp_path / 'config.yaml').write_text(
        yaml.safe_dump({
            'traj_min_length': 20, 'first': 100001, 'last': 100010,
            'frate': 5000, 'hb': 70, 'data_path': '.', 'set_names': ['wp1', 'wp2'],
            'ptv_res_path': 'test/res/', 'h5_path': '.', 'delta_t': 90,
            'rho': 1000, 'mu': 0.001, 'shift': shift,
        })
    )
    (tmp_path / 'grid.yaml').write_text(yaml.safe_dump(GRID))


def _write_grids_with_fluct(tmp_path, u_f, v_f, w_f, par):
    for s, (uf, vf, wf) in enumerate(zip(u_f, v_f, w_f)):
        p = tmp_path / f'wp{s+1}_grid.h5'
        write_grid_h5(p, GRID, uf, vf, wf, par)
        with h5py.File(p, 'a') as f:
            f.create_dataset('u_fluct', data=uf)
            f.create_dataset('v_fluct', data=vf)
def test_shift_fields_idempotent(tmp_path):
    p = tmp_path / 'a.h5'
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(3, 3, 2, 4))
    with h5py.File(p, 'w') as f:
        f.create_dataset('u', data=arr)

    ts.shift_fields(p, ['u'], 3)
    first = h5py.File(p, 'r')['u'][()]
    assert np.allclose(first, np.roll(arr, 3, axis=-1))

    # A second call must be a no-op (idempotent).
    ts.shift_fields(p, ['u'], 3)
    second = h5py.File(p, 'r')['u'][()]
    assert np.allclose(second, first)
    assert h5py.File(p, 'r').attrs['shift'] == 3
