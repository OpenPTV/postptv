"""End-to-end tests for flowtracks.combine.run_combine(..., method="pooled")."""

import numpy as np
import pytest
import xarray as xr

from flowtracks import eulerian as eu
from flowtracks.combine import run_combine
from flowtracks.io import save_zarr_trajectories
from flowtracks.trajectory import Trajectory

GRID = {
    "min_x": 0.0, "max_x": 1.0, "stepx": 2,
    "min_y": 0.0, "max_y": 1.0, "stepy": 2,
    "min_z": 0.0, "max_z": 1.0, "stepz": 2,
}


def _write_run(path, velocities, trajids, frames):
    pos = np.full((len(frames), 3), 0.25)
    trajs = []
    for tid, vel in zip(trajids, velocities):
        vel_arr = np.tile(np.asarray(vel, dtype=float), (len(frames), 1))
        trajs.append(Trajectory(pos.copy(), vel_arr, np.asarray(frames), trajid=tid))
    save_zarr_trajectories(trajs, path, group="trajectories")


def test_run_combine_pooled_end_to_end(tmp_path):
    pytest.importorskip("pyvista")

    frames = [1, 2, 3, 4, 5]
    # run1: 2 trajectories * 5 frames = 10 samples (u = 1.0 and 3.0)
    _write_run(tmp_path / "run1" / "run.zarr",
               velocities=[(1.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
               trajids=[1, 2], frames=frames)
    # run2: 1 trajectory * 5 frames = 5 samples (u = 5.0)
    _write_run(tmp_path / "run2" / "run.zarr",
               velocities=[(5.0, 0.0, 0.0)],
               trajids=[1], frames=frames)

    res = run_combine(
        tmp_path,
        method="pooled",
        grid_params=GRID,
        first=1, last=5, cycletime=100, deltat=50, base_time=0, min_count=12,
    )

    zarr_out = tmp_path / "post_analysis_pooled.zarr"
    assert res["output"] == zarr_out
    assert zarr_out.is_dir()
    assert not (tmp_path / "post_analysis_pooled_samples.npz").exists()
    assert (tmp_path / "vtk_output" / "phase_series.pvd").is_file()

    ds = xr.open_zarr(zarr_out).load()
    all_u = np.array([1.0] * 5 + [3.0] * 5 + [5.0] * 5)

    # Pooled total is 15 >= min_count=12 (even though run1=10 and run2=5 are each < 12)
    assert int(ds[eu.N_VAR].values[0, 0, 0, 0]) == 15
    assert int(ds[eu.NTRAJ_VAR].values[0, 0, 0, 0]) == 3
    assert ds["u_ins_mean"].values[0, 0, 0, 0] == pytest.approx(all_u.mean())
    assert ds["u_ins_u_ins"].values[0, 0, 0, 0] == pytest.approx(all_u.var(ddof=1))
    assert ds["u_rms"].values[0, 0, 0, 0] == pytest.approx(all_u.std(ddof=1))
    assert ds["counts_per_run"].sel(run="run1").values[0, 0, 0, 0] == 10
    assert ds["counts_per_run"].sel(run="run2").values[0, 0, 0, 0] == 5
    for field in ("MKE", "TKE", "VEL"):
        assert field in ds
        assert np.isfinite(ds[field].values[0, 0, 0, 0])
