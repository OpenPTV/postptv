"""End-to-end check that flowtracks works all the way through with zarr data:

trajectories -> save_zarr_trajectories -> real .zarr directory ->
eulerian_grid (via its io.trajectories() fallback for a path with no
.collect()) -> run_post_analysis_ds -> save_dataset(.zarr) -> reload.

Existing zarr tests cover trajectory round-tripping (test_zarr_io.py) and the
streamlined pipeline with a mocked Scene (test_streamlined_zarr_paraview.py)
separately; this closes the gap of a real zarr trajectory store feeding
eulerian_grid directly.
"""
import numpy as np
import xarray as xr

from flowtracks.eulerian import eulerian_grid, run_post_analysis_ds, save_dataset
from flowtracks.io import infer_format, save_zarr_trajectories, trajectories
from flowtracks.trajectory import Trajectory

GRID = {"stepx": 2, "stepy": 2, "stepz": 1,
        "min_x": 0.0, "max_x": 0.02, "min_y": 0.0, "max_y": 0.02,
        "min_z": 0.0, "max_z": 0.01}


def _make_trajectories(vel, n_particles=20, n_frames=10, first=100001, seed=0):
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n_particles):
        pos0 = rng.uniform([0, 0, 0], [0.02, 0.02, 0.01])
        pos = np.tile(pos0, (n_frames, 1))
        velocity = np.tile(np.asarray(vel, dtype=float), (n_frames, 1))
        time = np.arange(first, first + n_frames)
        trajs.append(Trajectory(pos, velocity, time, trajid=i))
    return trajs


def test_zarr_trajectories_feed_eulerian_grid_directly(tmp_path):
    zarr_path = tmp_path / "s1_traj.zarr"
    save_zarr_trajectories(_make_trajectories((1.0, 0.0, 0.0), seed=1), zarr_path)

    assert infer_format(str(zarr_path)) == "zarr"
    assert len(trajectories(str(zarr_path))) == 20

    ds = eulerian_grid(str(zarr_path), GRID, first=100001, last=100010,
                       cycletime=10, deltat=1, base_time=100000, min_count=1)

    assert float(ds["par_ave2"].sum()) > 0
    populated = ds["par_ave2"] > 0
    assert np.allclose(ds["u_ins_mean"].values[populated.values], 1.0)
    assert np.allclose(ds["v_ins_mean"].values[populated.values], 0.0)


def test_full_pipeline_from_zarr_trajectories_to_zarr_output(tmp_path):
    ds_sets = {}
    for set_name, vel in (("s1", (1.0, 0.0, 0.0)), ("s2", (2.0, 0.0, 0.0))):
        zarr_path = tmp_path / f"{set_name}_traj.zarr"
        save_zarr_trajectories(_make_trajectories(vel, seed=hash(set_name) % 1000), zarr_path)
        ds_sets[set_name] = eulerian_grid(
            str(zarr_path), GRID, first=100001, last=100010,
            cycletime=10, deltat=1, base_time=100000, min_count=1)

    recipe = {"sets": ["s1", "s2"], "output": "final.zarr",
             "average": {"weighting": "counts"}, "derived": {"fields": ["MKE", "TKE"]}}
    out = run_post_analysis_ds(ds_sets, recipe)

    out_path = tmp_path / "final.zarr"
    save_dataset(out, out_path)

    reloaded = xr.open_zarr(out_path)
    assert "TKE" in reloaded
    assert "u_ins_mean" in reloaded
    np.testing.assert_allclose(reloaded["u_ins_mean"].values, out["u_ins_mean"].values)
