"""Tests for previously uncovered functions and branches in flowtracks.eulerian
and flowtracks.writers."""

import numpy as np
import pytest
import xarray as xr

from flowtracks import eulerian as eu
from flowtracks.io import save_zarr_trajectories
from flowtracks.trajectory import Trajectory
from flowtracks.writers import (
    _trajectory_arrays,
    export_run_to_paraview,
    trajectory_units,
)

GRID = {
    "min_x": 0.0, "max_x": 1.0, "stepx": 2,
    "min_y": 0.0, "max_y": 1.0, "stepy": 2,
    "min_z": 0.0, "max_z": 1.0, "stepz": 2,
}


def test_fluid_mask_ndarray():
    counts = np.array([0, 50, 100, 101])
    np.testing.assert_array_equal(eu.fluid_mask(counts, threshold=100),
                                  [False, False, False, True])


def test_eulerian_grid_validation_errors():
    class ShortCollectScene:
        def collect(self, keys):
            return [np.zeros((1, 3)), np.zeros((1, 3)), np.array([1])]

    sc = ShortCollectScene()
    with pytest.raises(ValueError, match="needs a period"):
        eu.eulerian_grid(sc, GRID, 1, 5)
    with pytest.raises(ValueError, match="voxel_rule"):
        eu.eulerian_grid(sc, GRID, 1, 5, cycletime=10, voxel_rule="nearest")
    with pytest.raises(ValueError, match="trajid"):
        eu.eulerian_grid(sc, GRID, 1, 5, cycletime=10, moments=True)


def test_eulerian_windowed_file_path_qc_and_short_range(tmp_path):
    """Covers eulerian_windowed() trajectories() branch, qc=, and single-center fallback."""
    t1 = Trajectory(
        np.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
        np.array([1, 2, 3]),
        trajid=1,
    )
    zpath = tmp_path / "win.zarr"
    save_zarr_trajectories([t1], zpath, group="trajectories")

    # window_frames=20 > (last - first)=2 triggers single-center fallback;
    # qc max_speed=10 drops the 50.0 outlier.
    ds = eu.eulerian_windowed(
        str(zpath), GRID, first=1, last=3, window_frames=20, step_frames=5,
        min_count=1, qc={"max_speed": 10.0},
    )
    assert ds.sizes["window"] == 1
    assert int(ds["window_center"].values[0]) == 2
    assert int(ds[eu.COUNT_VAR].values[0, 0, 0, 0]) == 2
    assert ds["u_ins_mean"].values[0, 0, 0, 0] == pytest.approx(1.5)


def test_region_timeseries():
    data = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3)
    da = xr.DataArray(data, dims=("x", "y", "z", "phase"))
    mask = xr.DataArray(
        np.array([[[True, False], [False, False]], [[True, False], [False, False]]]),
        dims=("x", "y", "z"),
    )
    out = eu.region_timeseries({"ke": da}, mask)
    # Selected cells are (0,0,0) and (1,0,0): values at phase 0 are 0.0 and 12.0
    np.testing.assert_allclose(out["ke_mean"].values, [6.0, 7.0, 8.0])
    np.testing.assert_allclose(out["ke_min"].values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(out["ke_max"].values, [12.0, 13.0, 14.0])


def test_y_slab_masks():
    y = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    fluid = np.ones((2, len(y), 2, 3), dtype=bool)
    # start_idx=3 (1-based) -> y0 = y[2] = 0.04; inner: [0.04, 0.09] -> indices 2, 3, 4;
    # outer: (0.09, 0.12] -> indices 5, 6
    inner, outer = eu.y_slab_masks(fluid, y, start_idx=3, slab_length=0.05)
    assert inner.shape == fluid.shape and outer.shape == fluid.shape
    assert not (inner & outer).any()
    np.testing.assert_array_equal(inner[0, :, 0, 0], [False, False, True, True, True, False, False])
    np.testing.assert_array_equal(outer[0, :, 0, 0], [False, False, False, False, False, True, True])


def test_mask_array_zero_to_nan():
    u = np.array([[[[0.0], [1.0]], [[2.0], [3.0]]], [[[4.0], [5.0]], [[6.0], [7.0]]]])
    ds = xr.Dataset({"u_ins_mean": (("x", "y", "z", "phase"), u)})
    roi = np.ones((2, 2, 2), dtype=bool)
    roi[1, 1, 1] = False
    out = eu.mask_array(ds, roi, zero_to_nan=True)
    assert np.isnan(out["u_ins_mean"].values[0, 0, 0, 0])  # exact zero -> NaN
    assert np.isnan(out["u_ins_mean"].values[1, 1, 1, 0])  # outside ROI -> NaN
    assert out["u_ins_mean"].values[0, 0, 1, 0] == pytest.approx(1.0)


def test_reference_derived_solid_body_rotation_and_mask():
    """Analytical solid-body rotation: u = -omega*y, v = omega*x, w = 0.

    Note: reference_derived() preserves MATLAB's dim2-first gradient convention
    (du_dx = dd[1], du_dy = dd[0]), so passing a field with axes (y, x, z)
    gives exact analytical vorticity |omega_z| = 2*omega, Q = omega^2, lambda2 = -omega^2.
    """
    n = 5
    dx = dy = dz = 0.5
    omega = 2.0
    coords = np.arange(n) * dx
    # MATLAB convention: axis 0 is y, axis 1 is x
    Y, X, _ = np.meshgrid(coords, coords, coords, indexing="ij")
    u = -omega * Y
    v = omega * X
    w = np.zeros_like(u)

    mask = np.ones((n, n, n), dtype=bool)
    mask[0, 0, 0] = False
    counts = np.full((n, n, n), 10.0)

    out = eu.reference_derived(u, v, w, dx, dy, dz, mask=mask, rho=1000.0,
                               counts=counts, dt=0.01)
    # Interior voxel (2, 2, 2)
    assert out["vortmag"][2, 2, 2] == pytest.approx(2.0 * omega)
    assert out["ens"][2, 2, 2] == pytest.approx(0.5 * (2.0 * omega) ** 2)
    assert out["q"][2, 2, 2] == pytest.approx(omega**2)
    assert out["l2"][2, 2, 2] == pytest.approx(-(omega**2))
    assert out["vss"][2, 2, 2] == pytest.approx(0.0, abs=1e-12)
    assert out["prt"][2, 2, 2] == pytest.approx(0.1)
    # Masked voxel is NaN everywhere
    assert np.isnan(out["q"][0, 0, 0])
    assert np.isnan(out["u"][0, 0, 0])

    # Without counts -> prt is NaN; central-difference neighbors of a NaN voxel
    # have non-finite gradients and skip eigen-decomposition cleanly
    u_nan = u.copy()
    u_nan[2, 2, 2] = np.nan
    out_nan = eu.reference_derived(u_nan, v, w, dx, dy, dz)
    assert np.isnan(out_nan["prt"]).all()
    assert np.isnan(out_nan["q"][1, 2, 2])
    assert np.isnan(out_nan["l2"][3, 2, 2])


def test_writers_fallback_and_export_with_grid_params(tmp_path):
    with pytest.raises(TypeError, match="Unsupported trajectory source"):
        _trajectory_arrays(12345)

    assert trajectory_units(tmp_path / "does_not_exist.zarr") == {}
    assert trajectory_units(object()) == {}

    pytest.importorskip("pyvista")
    t1 = Trajectory(
        np.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]),
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        np.array([1, 2]),
        trajid=1,
    )
    zpath = tmp_path / "run.zarr"
    save_zarr_trajectories([t1], zpath, group="trajectories")

    out = export_run_to_paraview(
        zpath, tmp_path / "pv_out",
        grid_params=GRID, first=1, last=2, cycletime=10, deltat=5, base_time=0, min_count=1,
    )
    assert out["trajectories"].is_file()
    assert out["eulerian"].is_file()
