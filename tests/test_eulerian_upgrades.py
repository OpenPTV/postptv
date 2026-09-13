"""Comprehensive correctness + performance tests for the eulerian upgrades.

Covers the smart tricks borrowed from aortic-particle-pipeline (NaN+valid,
QC gates, FD velocity, sliding windows, chunked accumulation) and from the
openptv-analysis Matlab ports (upfront clean, fluidMask>100, PRT=counts*dt,
gradient conventions). Performance tests use wall-clock budgets (not
pytest-benchmark) so they run in the default suite.
"""

import time

import numpy as np
import pytest
import xarray as xr

from flowtracks.eulerian import (
    VEL_VARS,
    clean_field,
    derived_fields,
    eulerian_grid,
    eulerian_windowed,
    finite_difference_velocity,
    fluid_mask,
    qc_mask,
)

GRID = {"stepx": 4, "stepy": 3, "stepz": 2,
        "min_x": -0.01, "max_x": 0.02, "min_y": 0.0, "max_y": 0.05,
        "min_z": -0.03, "max_z": 0.03}
DIMS4 = ("x", "y", "z", "phase")


class ArrayScene:
    def __init__(self, pos, vel, time):
        self._pos = np.asarray(pos, float)
        self._vel = np.asarray(vel, float)
        self._time = np.asarray(time)

    def collect(self, keys):
        cols = {"pos": self._pos, "velocity": self._vel, "time": self._time}
        return [cols[k] for k in keys]


def _const_scene(n=200, vel=(1.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([-0.01, 0.0, -0.03], [0.02, 0.05, 0.03], (n, 3))
    vv = np.tile(np.asarray(vel, float), (n, 1))
    tt = np.full(n, 100001)
    return ArrayScene(pos, vv, tt)


# --- NaN + valid -------------------------------------------------------------

def test_nan_fill_and_valid_flag():
    ds = eulerian_grid(_const_scene(n=5), GRID, 100001, 100001, 50,
                       deltat=5, min_count=1000)
    for v in VEL_VARS:
        assert np.isnan(ds[v].values).all()
    assert (ds["par_ave2"].values == 0).all()
    assert "valid" in ds
    assert not ds["valid"].values.any()


def test_dense_voxel_finite_and_valid():
    ds = eulerian_grid(_const_scene(n=2000), GRID, 100001, 100001, 50,
                       deltat=5, min_count=1)
    assert np.isfinite(ds["u_ins_mean"].values).any()
    assert ds["valid"].values.any()
    # valid == counts >= min_count, par_ave2 zeroed where invalid
    assert ((ds["par_ave2"].values > 0) == ds["valid"].values).all()


def test_counts_convention_matches_legacy_zeros():
    ds = eulerian_grid(_const_scene(n=5), GRID, 100001, 100001, 50,
                       deltat=5, min_count=1000)
    assert ds["par_ave2"].dtype.kind in "iu"
    assert (ds["par_ave2"].values == 0).all()


# --- QC ----------------------------------------------------------------------

def test_qc_speed_filter_drops_fast_particles():
    rng = np.random.default_rng(3)
    pos = rng.uniform([-0.01, 0.0, -0.03], [0.02, 0.05, 0.03], (400, 3))
    vel = np.ones((400, 3))
    vel[:200] *= 100.0  # fast half
    tm = np.full(400, 100001)
    base = eulerian_grid(ArrayScene(pos, vel, tm), GRID, 100001, 100001, 50,
                         deltat=5, min_count=1)
    filt = eulerian_grid(ArrayScene(pos, vel, tm), GRID, 100001, 100001, 50,
                         deltat=5, min_count=1, qc={"max_speed": 10.0})
    assert filt["par_ave2"].values.sum() < base["par_ave2"].values.sum()
    assert filt["par_ave2"].values.sum() > 0


def test_qc_finite_filter():
    pos = np.array([[0.0, 0.01, 0.0], [np.nan, 0.01, 0.0], [0.0, 0.01, 0.0]])
    vel = np.array([[1.0, 0, 0], [1.0, 0, 0], [np.inf, 0, 0]])
    assert qc_mask(pos, vel).tolist() == [True, False, False]
    assert qc_mask(pos, vel, finite_only=False).tolist() == [True, True, True]


def test_finite_difference_velocity():
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [10.0, 0, 0]])
    ids = np.array([7, 7, 7, 9])  # 3-point track + singleton
    vel = finite_difference_velocity(pos, ids)
    np.testing.assert_allclose(vel[0, 0], 1.0)      # forward
    np.testing.assert_allclose(vel[1, 0], 1.5)      # centered
    np.testing.assert_allclose(vel[2, 0], 2.0)      # backward
    assert np.isnan(vel[3]).all()                    # singleton
    assert finite_difference_velocity(
        np.empty((0, 3)), np.empty((0,), int)).shape == (0, 3)


# --- chunked accumulation ----------------------------------------------------

def test_chunked_matches_single_pass():
    rng = np.random.default_rng(11)
    n = 3000
    pos = rng.uniform([-0.01, 0.0, -0.03], [0.02, 0.05, 0.03], (n, 3))
    vel = rng.normal(size=(n, 3))
    tm = rng.integers(100001, 100101, n)
    kw = dict(grid_params=GRID, first=100001, last=100100, cycletime=50,
              deltat=5, min_count=2)
    a = eulerian_grid(ArrayScene(pos, vel, tm), **kw)
    b = eulerian_grid(ArrayScene(pos, vel, tm), chunk_frames=7, **kw)
    np.testing.assert_array_equal(a["par_ave2"].values, b["par_ave2"].values)
    for v in VEL_VARS:
        np.testing.assert_allclose(a[v].values, b[v].values, equal_nan=True)
    np.testing.assert_array_equal(a["valid"].values, b["valid"].values)


# --- sliding windows ----------------------------------------------------------

def test_windowed_constant_field_and_centers():
    ds = eulerian_windowed(_const_scene(n=500, vel=(2.0, 0, 0)), GRID,
                           100001, 100020, window_frames=6, step_frames=5,
                           min_count=1)
    assert ds.sizes["window"] == 3  # centers 100004, 100009, 100014
    assert ds["window_center"].values.tolist() == [100004, 100009, 100014]
    hit = ds["u_ins_mean"].values[np.isfinite(ds["u_ins_mean"].values)]
    assert hit.size > 0
    np.testing.assert_allclose(hit, 2.0, rtol=1e-12)


def test_windowed_searchsorted_matches_brute_force():
    rng = np.random.default_rng(5)
    n = 800
    pos = rng.uniform([-0.01, 0.0, -0.03], [0.02, 0.05, 0.03], (n, 3))
    vel = rng.normal(size=(n, 3))
    tm = rng.integers(100001, 100021, n)
    ds = eulerian_windowed(ArrayScene(pos, vel, tm), GRID, 100001, 100020,
                           window_frames=6, step_frames=10, min_count=1)
    edges = [np.linspace(GRID[f"min_{d}"], GRID[f"max_{d}"],
                         GRID[f"step{d}"] + 1) for d in "xyz"]
    for k, c in enumerate(ds["window_center"].values):
        sel = (tm >= int(c) - 3) & (tm <= int(c) + 3)
        xyz = pos[sel]
        keep = np.ones(len(xyz), bool)
        for d, e in enumerate(edges):
            keep &= (xyz[:, d] >= e[0]) & (xyz[:, d] < e[-1])
        ref = np.histogramdd(xyz[keep], bins=edges)[0]
        np.testing.assert_array_equal(ds["par_ave2"].values[..., k], ref)


# --- clean / fluid mask / PRT / gradients -------------------------------------

def test_clean_field():
    a = np.array([1.0, np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(clean_field(a), [1.0, 0, 0, 0])


def test_fluid_mask_threshold():
    c = xr.DataArray(np.array([[[[49, 50, 100, 101]]]]))
    assert fluid_mask(c, 100).values.ravel().tolist() == [False, False, False, True]
    assert fluid_mask(np.array([0, 101]), 100).tolist() == [False, True]


def _avg_stats(shape=(4, 3, 3, 2), seed=2):
    rng = np.random.default_rng(seed)
    coords = {"x": np.linspace(-0.01, 0.02, shape[0]),
              "y": np.linspace(0.0, 0.05, shape[1]),
              "z": np.linspace(-0.03, 0.03, shape[2]),
              "phase": np.arange(shape[3])}
    avg = xr.Dataset({v: (DIMS4, rng.normal(size=shape)) for v in VEL_VARS},
                     coords=coords)
    names = ["u_ins_u_ins", "v_ins_v_ins", "w_ins_w_ins",
             "u_ins_v_ins", "u_ins_w_ins", "v_ins_w_ins"]
    stats = xr.Dataset({s: (DIMS4, np.abs(rng.normal(size=shape))) for s in names},
                       coords=coords)
    return avg, stats


def test_derived_clean_nan_contains_mask():
    avg, stats = _avg_stats()
    avg["u_ins_mean"].values[1, 1, 1, 0] = np.nan
    # ML uses raw gradient products (no fillna inside, unlike the VSS/RSS
    # eig path) so NaN poisoning is directly observable here.
    dirty = derived_fields(avg, stats, fields=["ML"], clean_nan=False)
    clean = derived_fields(avg, stats, fields=["ML"], clean_nan=True)
    assert np.isnan(dirty["ML"].values[0:3, 0:3, 0:2, 0]).sum() > \
        np.isnan(clean["ML"].values[0:3, 0:3, 0:2, 0]).sum()
    assert np.isfinite(clean["ML"].values[0, 0, 0, 0])


def test_derived_fluid_min_masks_outputs():
    avg, stats = _avg_stats()
    counts = xr.DataArray(np.ones(avg["u_ins_mean"].shape), dims=DIMS4,
                          coords=avg.coords)
    counts.values[..., 0] = 1000  # phase 0 fluid, phase 1 not
    out = derived_fields(avg, stats, fields=["MKE", "TKE"], counts=counts,
                         fluid_min=100)
    assert np.isfinite(out["MKE"].values[..., 0]).all()
    assert np.isnan(out["MKE"].values[..., 1]).all()


def test_derived_prt_counts_mode():
    avg, stats = _avg_stats()
    counts = xr.DataArray(np.full(avg["u_ins_mean"].shape, 7.0), dims=DIMS4,
                          coords=avg.coords)
    out = derived_fields(avg, stats, fields=["PRT"], counts=counts,
                         prt_mode="counts", prt_dt=2.0)
    np.testing.assert_allclose(out["PRT"].values, 14.0)
    with pytest.raises(ValueError):
        derived_fields(avg, stats, fields=["PRT"], prt_mode="counts")
    with pytest.raises(ValueError):
        derived_fields(avg, stats, fields=["PRT"], prt_mode="bogus")


def test_derived_gradient_conventions_both_finite_and_differ():
    avg, stats = _avg_stats(shape=(5, 4, 3, 2))  # anisotropic spacing
    m = derived_fields(avg, stats, fields=["VSS"])
    p = derived_fields(avg, stats, fields=["VSS"],
                       gradient_convention="physical")
    assert np.isfinite(m["VSS"].values).all()
    assert np.isfinite(p["VSS"].values).all()
    assert not np.allclose(m["VSS"].values, p["VSS"].values)
    with pytest.raises(ValueError):
        derived_fields(avg, stats, fields=["VSS"],
                       gradient_convention="bogus")


# --- performance ---------------------------------------------------------------

def test_performance_binning_200k():
    rng = np.random.default_rng(0)
    n = 200_000
    pos = rng.uniform([-0.01, 0.0, -0.03], [0.02, 0.05, 0.03], (n, 3))
    vel = rng.normal(size=(n, 3))
    tm = rng.integers(100001, 100301, n)
    scene = ArrayScene(pos, vel, tm)
    t0 = time.perf_counter()
    ds = eulerian_grid(scene, GRID, 100001, 100300, 4286, deltat=88,
                       min_count=5)
    dt = time.perf_counter() - t0
    assert ds["par_ave2"].values.sum() > 0
    assert dt < 5.0, f"binning 200k took {dt:.2f}s"


def test_performance_chunked_within_3x():
    rng = np.random.default_rng(1)
    n = 100_000
    pos = rng.uniform([-0.01, 0.0, -0.03], [0.02, 0.05, 0.03], (n, 3))
    vel = rng.normal(size=(n, 3))
    tm = rng.integers(100001, 100201, n)
    scene = ArrayScene(pos, vel, tm)
    kw = dict(grid_params=GRID, first=100001, last=100200, cycletime=4286,
              deltat=88, min_count=5)
    t0 = time.perf_counter()
    eulerian_grid(scene, **kw)
    dt_single = time.perf_counter() - t0
    t0 = time.perf_counter()
    eulerian_grid(scene, chunk_frames=25, **kw)
    dt_chunk = time.perf_counter() - t0
    assert dt_chunk < max(10.0, 3 * dt_single + 1.0)


def test_performance_derived_fields():
    avg, stats = _avg_stats(shape=(20, 20, 10, 4))
    t0 = time.perf_counter()
    out = derived_fields(avg, stats, fields=["MKE", "TKE", "VEL", "VSS", "ML"])
    dt = time.perf_counter() - t0
    assert set(out.data_vars) == {"MKE", "TKE", "VEL", "VSS", "ML"}
    assert dt < 5.0, f"derived fields took {dt:.2f}s"
