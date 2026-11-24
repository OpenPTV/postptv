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


def test_run_combine_per_run_multi_and_single(tmp_path):
    pytest.importorskip("pyvista")

    frames = [1, 2, 3, 4, 5]
    # Multi-run per_run
    multi = tmp_path / "multi"
    _write_run(multi / "run1" / "run.zarr", [(1.0, 0.0, 0.0)], [1], frames)
    _write_run(multi / "run2" / "run.zarr", [(3.0, 0.0, 0.0)], [1], frames)

    res = run_combine(
        multi,
        method="per_run",
        grid_params=GRID,
        first=1, last=5, cycletime=100, deltat=50, base_time=0, min_count=1,
    )
    assert res["output"] == multi / "post_analysis.zarr"
    ds = xr.open_zarr(res["output"]).load()
    assert ds["u_ins_mean"].values[0, 0, 0, 0] == pytest.approx(2.0)
    assert ds["u_ins_u_ins"].values[0, 0, 0, 0] == pytest.approx(1.0)
    assert ds["u_rms"].values[0, 0, 0, 0] == pytest.approx(1.0)
    assert "TKE" in ds and "MKE" in ds and "VEL" in ds

    # Single-run per_run (returns single grid directly without stacking)
    single = tmp_path / "single"
    _write_run(single / "run.zarr", [(4.0, 0.0, 0.0)], [1], frames)
    res1 = run_combine(
        single,
        method="per_run",
        grid_params=GRID,
        first=1, last=5, cycletime=100, deltat=50, base_time=0, min_count=1,
    )
    ds1 = xr.open_zarr(res1["output"]).load()
    assert ds1["u_ins_mean"].values[0, 0, 0, 0] == pytest.approx(4.0)


def test_run_combine_empty_dir_returns_none(tmp_path):
    assert run_combine(tmp_path, method="pooled") is None


def test_combine_main_cli_dispatches(tmp_path, monkeypatch):
    import sys

    from flowtracks import combine

    called = {}

    def fake_run_combine(data_dir, method="per_run"):
        called["data_dir"] = data_dir
        called["method"] = method

    monkeypatch.setattr(combine, "run_combine", fake_run_combine)
    monkeypatch.setattr(sys, "argv", ["postptv-combine", str(tmp_path), "--method", "pooled"])
    combine.main()
    assert called == {"data_dir": str(tmp_path), "method": "pooled"}
