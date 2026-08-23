"""Edge-case tests: where the xarray post-analysis functions might break."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from flowtracks.phase_average import fluctuations, open_sets, phase_average
from flowtracks.eulerian import (
    derived_fields,
    eulerian_grid,
    export_vtk,
    shift_phase,
    turbulent_statistics,
)
from test_phase_average_xr import VARS, _write_grid

GRID_1x1 = {"stepx": 1, "stepy": 1, "stepz": 1,
            "min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1,
            "min_z": 0, "max_z": 1}


class SceneOf:
    """Scene stub: same positions/velocities every frame, flowtracks-like API."""

    def __init__(self, pos, vel, frames=range(100001, 100011)):
        self._pos, self._vel = np.atleast_2d(pos), np.atleast_2d(vel)
        self._frames = list(frames)

    def frame_by_time(self, t):
        outer = self

        class Frame:
            def pos(self):
                return outer._pos

            def velocity(self):
                return outer._vel

        return Frame()

    def collect(self, keys):
        n, m = len(self._frames), self._pos.shape[0]
        cols = {
            "pos": np.tile(self._pos, (n, 1)),
            "velocity": np.tile(self._vel, (n, 1)),
            "time": np.repeat(np.asarray(self._frames), m),
        }
        return [cols[k] for k in keys]


# --- open_sets: input contract ---------------------------------------------


def test_open_sets_missing_file_raises(tmp_path):
    _write_grid(tmp_path / "a_grid.h5", 1.0)
    with pytest.raises(FileNotFoundError):
        open_sets(tmp_path, ["a", "missing"], VARS)


def test_open_sets_missing_variable_raises(tmp_path):
    _write_grid(tmp_path / "a_grid.h5", 1.0)
    with pytest.raises(KeyError):
        open_sets(tmp_path, ["a"], ["no_such_var"])


def test_open_sets_mismatched_grids_fail_loud(tmp_path):
    _write_grid(tmp_path / "a_grid.h5", 1.0)
    _write_grid(tmp_path / "b_grid.h5", 2.0)
    with h5py.File(tmp_path / "b_grid.h5", "a") as f:  # move one grid point
        x = f["x_vals"][()]
        x[-1] += 0.5
        del f["x_vals"]
        f["x_vals"] = x
    with pytest.raises(ValueError):
        open_sets(tmp_path, ["a", "b"], VARS)


def test_single_set_average_is_identity(tmp_path):
    _write_grid(tmp_path / "a_grid.h5", 7.0)
    ds = open_sets(tmp_path, ["a"], VARS)
    avg = phase_average(ds)
    assert float(avg["u_ins_mean"].min()) == 7.0
    fluct = fluctuations(ds, avg)
    assert float(np.abs(fluct["u_ins_mean"]).max()) == 0.0


# --- eulerian_grid: binning edges -------------------------------------------


def test_particle_on_min_edge_included_max_edge_excluded():
    on_min = eulerian_grid(SceneOf([0.0, 0.0, 0.0], [1, 1, 1]), GRID_1x1,
                           100001, 100001, cycletime=1, deltat=0, min_count=1)
    assert int(on_min["par_ave2"].sum()) == 1

    on_max = eulerian_grid(SceneOf([1.0, 1.0, 1.0], [1, 1, 1]), GRID_1x1,
                           100001, 100001, cycletime=1, deltat=0, min_count=1)
    assert int(on_max["par_ave2"].sum()) == 0


def test_outside_domain_particles_ignored():
    ds = eulerian_grid(SceneOf([-5.0, 0.5, 0.5], [9, 9, 9]), GRID_1x1,
                       100001, 100001, cycletime=1, deltat=0, min_count=1)
    assert int(ds["par_ave2"].sum()) == 0
    assert float(ds["u_ins_mean"].sum()) == 0.0


def test_min_count_zeroes_sparse_cells():
    scene = SceneOf([0.5, 0.5, 0.5], [2, 2, 2])
    # 3 frames of 1 particle = 3 samples < min_count=5 -> zeroed, no divide
    ds = eulerian_grid(scene, GRID_1x1, 100001, 100003, cycletime=3,
                       deltat=1, min_count=5)
    assert int(ds["par_ave2"].sum()) == 0
    assert float(ds["u_ins_mean"].sum()) == 0.0


def test_empty_scene_gives_zeros_not_nan():
    ds = eulerian_grid(SceneOf(np.empty((0, 3)), np.empty((0, 3))), GRID_1x1,
                       100001, 100010, cycletime=10, deltat=2, min_count=1)
    assert not ds["u_ins_mean"].isnull().any()
    assert float(ds["u_ins_mean"].sum()) == 0.0


def test_phase_index_never_out_of_bounds():
    # cycletime not divisible by zaman: last frames of a cycle clip into final bin
    ds = eulerian_grid(SceneOf([0.5, 0.5, 0.5], [1, 1, 1]), GRID_1x1,
                       100001, 100010, cycletime=7, deltat=1, min_count=1)
    assert int(ds["par_ave2"].sum()) == 10  # every frame landed in a valid bin


# --- shift_phase -------------------------------------------------------------


@pytest.mark.parametrize("shift,expected", [
    (-1, [1.0, 2.0, 3.0, 0.0]),   # negative shift rolls the other way
    (4, [0.0, 1.0, 2.0, 3.0]),    # full cycle = identity
    (5, [3.0, 0.0, 1.0, 2.0]),    # wraps modulo cycle length
])
def test_shift_phase_wrapping(shift, expected):
    ds = xr.Dataset({"u": (("phase",), np.arange(4.0))},
                    coords={"phase": np.arange(4)})
    assert list(shift_phase(ds, shift)["u"].values) == expected


def test_shift_does_not_mutate_input():
    ds = xr.Dataset({"u": (("phase",), np.arange(4.0))},
                    coords={"phase": np.arange(4)})
    shift_phase(ds, 2)
    assert list(ds["u"].values) == [0.0, 1.0, 2.0, 3.0]
    assert "shift" not in ds.attrs


# --- turbulent_statistics ----------------------------------------------------


def _fluct(values_by_set, shape=(2, 2, 1, 3)):
    data = {f"{a}_fluct": (("set", "x", "y", "z", "phase"),
                           np.stack([np.full(shape, v) for v in values_by_set]))
            for a in "uvw"}
    return xr.Dataset(data, coords={"set": list("ab"[: len(values_by_set)])})


def test_stats_single_set_zero_fluct_is_zero_not_nan():
    fluct = _fluct([0.0])
    counts = xr.ones_like(fluct["u_fluct"]).astype(int)
    stats = turbulent_statistics(fluct, counts)
    assert float(stats["u_rms"].max()) == 0.0
    assert not stats["u_rms"].isnull().any()


def test_stats_mixed_zero_count_voxels():
    fluct = _fluct([1.0, -1.0])
    counts = xr.ones_like(fluct["u_fluct"]).astype(int)
    counts[:, 0, 0, 0, 0] = 0  # one voxel unsampled in EVERY set
    stats = turbulent_statistics(fluct, counts)
    assert np.isnan(float(stats["u_rms"][0, 0, 0, 0]))  # unsampled -> NaN
    assert float(stats["u_rms"][1, 1, 0, 1]) == 1.0  # sampled voxels unaffected


def test_stats_weighting_ignores_zero_count_set():
    # set b has zero counts everywhere: its fluct values must not contribute
    fluct = _fluct([2.0, 999.0])
    counts = xr.concat(
        [xr.ones_like(fluct["u_fluct"].isel(set=0, drop=True)).astype(int),
         xr.zeros_like(fluct["u_fluct"].isel(set=0, drop=True)).astype(int)],
        dim=fluct["set"],
    )
    stats = turbulent_statistics(fluct, counts)
    assert float(stats["u_ins_u_ins"].mean()) == 4.0


# --- derived_fields / export_vtk --------------------------------------------


def test_derived_fields_propagate_nan():
    avg = xr.Dataset({v: xr.DataArray(1.0) for v in
                      ["u_ins_mean", "v_ins_mean", "w_ins_mean"]})
    stats = xr.Dataset({f"{a}_ins_{a}_ins": xr.DataArray(np.nan) for a in "uvw"})
    assert np.isnan(float(derived_fields(avg, stats)["TKE"]))


def test_export_vtk_scalar_only_dataset(tmp_path):
    pytest.importorskip("vtk")
    ds = xr.Dataset(
        {"TKE": (("x", "y", "z", "phase"), np.ones((2, 2, 2, 2)))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0],
                "phase": [0, 1]},
    )
    files = export_vtk(ds, tmp_path / "vtk")
    assert [f.name for f in files] == ["phase_000.vtk", "phase_001.vtk"]
    assert all(f.stat().st_size > 0 for f in files)
