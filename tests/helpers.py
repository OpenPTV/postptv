"""Shared helpers for pipeline tests: synthetic HDF5 builders and a fake Scene."""
import sys
from pathlib import Path
import numpy as np
import h5py

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))


def build_coordinates(grid):
    """Replicate eulerian_grid()'s coordinate construction."""
    sx, sy, sz = grid['stepx'], grid['stepy'], grid['stepz']
    bins_x = np.linspace(grid['min_x'], grid['max_x'], sx + 1)
    bins_y = np.linspace(grid['min_y'], grid['max_y'], sy + 1)
    bins_z = np.linspace(grid['min_z'], grid['max_z'], sz + 1)
    x_vals = 0.5 * (bins_x[:-1] + bins_x[1:])
    y_vals = 0.5 * (bins_y[:-1] + bins_y[1:])
    z_vals = 0.5 * (bins_z[:-1] + bins_z[1:])
    x3, y3, z3 = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    return x_vals, y_vals, z_vals, x3, y3, z3


def write_grid_h5(path, grid, u, v, w, par_ave2,
                  first=0, last=1, cycletime=1, zaman=1, min_count=1):
    x_vals, y_vals, z_vals, x3, y3, z3 = build_coordinates(grid)
    with h5py.File(path, 'w') as f:
        f.create_dataset('u_ins_mean', data=u)
        f.create_dataset('v_ins_mean', data=v)
        f.create_dataset('w_ins_mean', data=w)
        f.create_dataset('par_ave2', data=par_ave2)
        f.create_dataset('x_vals', data=x_vals)
        f.create_dataset('y_vals', data=y_vals)
        f.create_dataset('z_vals', data=z_vals)
        f.create_dataset('x3', data=x3)
        f.create_dataset('y3', data=y3)
        f.create_dataset('z3', data=z3)
        f.attrs['first'] = first
        f.attrs['last'] = last
        f.attrs['cycletime'] = cycletime
        f.attrs['zaman'] = zaman
        f.attrs['min_count'] = min_count


def write_phase_averaged_h5(path, u, v, w):
    with h5py.File(path, 'w') as f:
        f.create_dataset('u_phase_averaged', data=u)
        f.create_dataset('v_phase_averaged', data=v)
        f.create_dataset('w_phase_averaged', data=w)


def write_turb_stats_h5(path, u_rms, v_rms, w_rms,
                        u_ins_v_ins, u_ins_w_ins, v_ins_w_ins,
                        u_ins_u_ins, v_ins_v_ins, w_ins_w_ins):
    with h5py.File(path, 'w') as f:
        f.create_dataset('u_rms', data=u_rms)
        f.create_dataset('v_rms', data=v_rms)
        f.create_dataset('w_rms', data=w_rms)
        f.create_dataset('u_ins_v_ins', data=u_ins_v_ins)
        f.create_dataset('u_ins_w_ins', data=u_ins_w_ins)
        f.create_dataset('v_ins_w_ins', data=v_ins_w_ins)
        f.create_dataset('u_ins_u_ins', data=u_ins_u_ins)
        f.create_dataset('v_ins_v_ins', data=v_ins_v_ins)
        f.create_dataset('w_ins_w_ins', data=w_ins_w_ins)


class FakeFrame:
    def __init__(self, pos, vel):
        self._pos = np.asarray(pos, dtype=float)
        self._vel = np.asarray(vel, dtype=float)

    def pos(self):
        return self._pos

    def velocity(self):
        return self._vel


class FakeScene:
    """Minimal Scene-like object for eulerian_grid(): ignores the frame index."""

    def __init__(self, pos, vel, frames=range(100001, 100201)):
        self._pos = np.asarray(pos, dtype=float)
        self._vel = np.asarray(vel, dtype=float)
        self._frames = list(frames)

    def frame_by_time(self, t):
        return FakeFrame(self._pos, self._vel)

    def collect(self, keys):
        n, m = len(self._frames), self._pos.shape[0]
        cols = {
            'pos': np.tile(self._pos, (n, 1)),
            'velocity': np.tile(self._vel, (n, 1)),
            'time': np.repeat(np.asarray(self._frames), m),
        }
        return [cols[k] for k in keys]
