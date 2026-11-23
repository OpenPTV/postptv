import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from helpers import FakeScene

from flowtracks import eulerian as eu
from flowtracks.phase_average import phase_average

GRID = {
    "stepx": 2, "stepy": 2, "stepz": 2,
    "min_x": -1.0, "max_x": 1.0,
    "min_y": -1.0, "max_y": 1.0,
    "min_z": -1.0, "max_z": 1.0,
}

# 5 frames, 1 phase bin
BIN_KW = dict(first=100001, last=100005, cycletime=100, deltat=90, base_time=100000)


def _scene(vel, n_particles=1):
    pos = np.zeros((n_particles, 3))
    velocity = np.tile(np.array(vel, dtype=float), (n_particles, 1))
    return FakeScene(pos, velocity, frames=range(100001, 100006))


def _pool_scenes(scenes, min_count=1, ddof=0, voxel_rule="half-open"):
    names = list(scenes)
    per = [
        eu.eulerian_grid(scenes[k], GRID, min_count=0, voxel_rule=voxel_rule,
                         moments=True, **BIN_KW)[eu.MOMENT_VARS]
        for k in names
    ]
    stacked = xr.concat(per, dim=xr.DataArray(names, dims="set", name="set"))
    pooled = eu.combine_moments(stacked, "set")
    stats = eu.moment_statistics(pooled, min_count=min_count, ddof=ddof)
    return stacked, pooled, stats


def test_pooled_matches_count_weighted_per_run_mean():
    """Pooled mean equals count-weighted average of per-run means (normal case)."""
    scenes = {"wp1": _scene((1.0, 0.0, 0.0), n_particles=2),  # 10 samples
              "wp2": _scene((3.0, 0.0, 0.0), n_particles=4)}  # 20 samples
    ds1 = eu.eulerian_grid(scenes["wp1"], GRID, min_count=1, **BIN_KW)
    ds2 = eu.eulerian_grid(scenes["wp2"], GRID, min_count=1, **BIN_KW)
    combined = xr.concat([ds1, ds2], dim=xr.DataArray(["wp1", "wp2"], dims="set"))
    avg = phase_average(combined, weights=combined[eu.COUNT_VAR])

    stacked, _, stats = _pool_scenes(scenes, min_count=1)

    assert np.allclose(stats["u_ins_mean"].values, avg["u_ins_mean"].values, equal_nan=True)
    assert np.allclose(stats[eu.N_VAR].values,
                       (ds1[eu.COUNT_VAR] + ds2[eu.COUNT_VAR]).values)
    assert stacked[eu.N_VAR].sel(set="wp1").values[1, 1, 1, 0] == 10
    assert stacked[eu.N_VAR].sel(set="wp2").values[1, 1, 1, 0] == 20


def test_pooled_recovers_bin_masked_out_per_run():
    """Edge case: each run is below min_count alone, pooled total is sufficient."""
    scenes = {"wp1": _scene((1.0, 0.0, 0.0), n_particles=3),  # 15 samples
              "wp2": _scene((3.0, 0.0, 0.0), n_particles=3)}  # 15 samples

    ds1 = eu.eulerian_grid(scenes["wp1"], GRID, min_count=20, **BIN_KW)
    ds2 = eu.eulerian_grid(scenes["wp2"], GRID, min_count=20, **BIN_KW)
    assert np.isnan(ds1["u_ins_mean"].values[1, 1, 1, 0])
    assert np.isnan(ds2["u_ins_mean"].values[1, 1, 1, 0])
    combined = xr.concat([ds1, ds2], dim=xr.DataArray(["wp1", "wp2"], dims="set"))
    avg = phase_average(combined, weights=combined[eu.COUNT_VAR])
    assert np.isnan(avg["u_ins_mean"].values[1, 1, 1, 0])

    stacked, _, stats = _pool_scenes(scenes, min_count=20)
    assert stats["u_ins_mean"].values[1, 1, 1, 0] == pytest.approx(2.0)
    assert stats[eu.N_VAR].values[1, 1, 1, 0] == 30
    assert stacked[eu.N_VAR].sel(set="wp1").values[1, 1, 1, 0] == 15
    assert stacked[eu.N_VAR].sel(set="wp2").values[1, 1, 1, 0] == 15


def test_pooled_single_run():
    """Pooled with one run: mean matches single-run grid."""
    scenes = {"wp1": _scene((2.0, 0.0, 0.0), n_particles=3)}
    ds_single = eu.eulerian_grid(scenes["wp1"], GRID, min_count=1, **BIN_KW)
    stacked, _, stats = _pool_scenes(scenes, min_count=1)
    assert np.allclose(stats["u_ins_mean"].values, ds_single["u_ins_mean"].values, equal_nan=True)
    assert stacked[eu.N_VAR].sel(set="wp1").values[1, 1, 1, 0] == 15


def test_pooled_ceil_voxel_rule():
    """Pooled with ceil voxel rule: binning matches per-run ceil path."""
    pos = np.array([[0.5, 0.0, 0.0]])
    vel = np.array([[1.0, 2.0, 3.0]])
    scene = FakeScene(pos, vel, frames=range(100001, 100006))
    ds_ceil = eu.eulerian_grid(scene, GRID, min_count=1, voxel_rule="ceil", **BIN_KW)
    _, _, stats_ceil = _pool_scenes({"wp1": scene}, min_count=1, voxel_rule="ceil")
    assert np.allclose(stats_ceil["u_ins_mean"].values, ds_ceil["u_ins_mean"].values, equal_nan=True)
    assert np.allclose(stats_ceil[eu.N_VAR].values, ds_ceil[eu.COUNT_VAR].values)


def test_compare_statistics_per_run_vs_pooled_synthetic():
    """Run both methods on one synthetic case and compare statistics."""
    scenes = {"wp1": _scene((1.0, 0.0, 0.0), n_particles=3),  # 15 samples
              "wp2": _scene((4.0, 0.0, 0.0), n_particles=2)}  # 10 samples

    ds1 = eu.eulerian_grid(scenes["wp1"], GRID, min_count=1, **BIN_KW)
    ds2 = eu.eulerian_grid(scenes["wp2"], GRID, min_count=1, **BIN_KW)
    combined = xr.concat([ds1, ds2], dim=xr.DataArray(["wp1", "wp2"], dims="set", name="set"))

    per_run_mean_ds = phase_average(combined, weights=combined[eu.COUNT_VAR])
    fl = combined[["u_ins_mean", "v_ins_mean", "w_ins_mean"]] - per_run_mean_ds
    per_run_u_rms = np.sqrt((fl["u_ins_mean"] ** 2).mean("set"))

    stacked, _, stats = _pool_scenes(scenes, min_count=1, ddof=0)

    pooled_u = np.array([1.0] * 15 + [4.0] * 10)
    expected_pooled_mean = pooled_u.mean()  # 2.2
    expected_pooled_var_pop = pooled_u.var(ddof=0)
    per_run_means = np.array([1.0, 4.0])
    expected_run_to_run_u_rms = np.sqrt(np.mean((per_run_means - expected_pooled_mean) ** 2))

    assert per_run_mean_ds["u_ins_mean"].values[1, 1, 1, 0] == pytest.approx(expected_pooled_mean)
    assert stats["u_ins_mean"].values[1, 1, 1, 0] == pytest.approx(expected_pooled_mean)
    assert stats[eu.N_VAR].values[1, 1, 1, 0] == 25
    assert stacked[eu.N_VAR].sel(set="wp1").values[1, 1, 1, 0] == 15
    assert stacked[eu.N_VAR].sel(set="wp2").values[1, 1, 1, 0] == 10

    assert per_run_u_rms.values[1, 1, 1, 0] == pytest.approx(expected_run_to_run_u_rms)
    assert stats["u_ins_u_ins"].values[1, 1, 1, 0] == pytest.approx(expected_pooled_var_pop)
    assert stats["u_rms"].values[1, 1, 1, 0] == pytest.approx(np.sqrt(expected_pooled_var_pop))
    assert stats["u_rms"].values[1, 1, 1, 0] != pytest.approx(float(per_run_u_rms.values[1, 1, 1, 0]))
