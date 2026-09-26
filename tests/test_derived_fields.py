"""Derived fields physical-gradient correctness; masks;
weighted phase average; binary VTK round-trip; netCDF encoding."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from flowtracks.eulerian import (
    VEL_VARS,
    apply_masks,
    derived_fields,
    save_netcdf,
)
from flowtracks.phase_average import phase_average
from flowtracks.writers import write_eulerian_series

RHO, MU = 1000.0, 0.001
DIMS4 = ("x", "y", "z", "phase")


def _random_inputs(seed=1, shape=(4, 3, 3, 5)):
    rng = np.random.default_rng(seed)
    coords = {"x": np.linspace(-0.01, 0.02, shape[0]),
              "y": np.linspace(0.0, 0.05, shape[1]),
              "z": np.linspace(-0.03, 0.03, shape[2]),
              "phase": np.arange(shape[3])}
    avg = xr.Dataset({v: (DIMS4, rng.normal(size=shape)) for v in VEL_VARS},
                     coords=coords)
    names = ["u_ins_u_ins", "v_ins_v_ins", "w_ins_w_ins",
             "u_ins_v_ins", "u_ins_w_ins", "v_ins_w_ins"]
    stats = xr.Dataset({n: (DIMS4, np.abs(rng.normal(size=shape))) for n in names},
                       coords=coords)
    return avg, stats


def _linear_field_inputs(a=1.5):
    """Analytic linear field u = a*x on a fully anisotropic grid.

    True gradients: ux = a, everything else 0 — np.gradient is exact for
    linears, so VSS/ML have closed forms below. Deliberately dx != dy != dz
    so any spacing mix-up (e.g. dx := dz) fails loudly.
    """
    nx, ny, nz, nph = 4, 3, 3, 2
    x = np.linspace(0.0, 0.03, nx)  # dx = 0.01
    y = np.linspace(0.0, 0.04, ny)  # dy = 0.02
    z = np.linspace(0.0, 0.10, nz)  # dz = 0.05
    u = (a * x)[:, None, None, None] * np.ones((1, ny, nz, nph))
    zeros = np.zeros((nx, ny, nz, nph))
    coords = {"x": x, "y": y, "z": z, "phase": np.arange(nph)}
    avg = xr.Dataset({"u_ins_mean": (DIMS4, u),
                      "v_ins_mean": (DIMS4, zeros.copy()),
                      "w_ins_mean": (DIMS4, zeros.copy())}, coords=coords)
    names = ["u_ins_u_ins", "v_ins_v_ins", "w_ins_w_ins",
             "u_ins_v_ins", "u_ins_w_ins", "v_ins_w_ins"]
    stats = xr.Dataset({n: (DIMS4, np.zeros((nx, ny, nz, nph))) for n in names},
                       coords=coords)
    return avg, stats


def test_derived_physical_gradients_linear_field():
    # u = a*x, a = 1.5 -> ux = a, all other gradient components 0.
    # p1 = 2a - 2a/3 = 4a/3 = 2.0; p5 = p9 = off-diagonals = 0.
    # VSS = 0.5*(4a/3)*RHO*MU = 1.0 (eig spread carries a 0.5 factor);
    # ML = (4a/3)^2*MU*RHO = 4.0.
    avg, stats = _linear_field_inputs(a=1.5)
    out = derived_fields(avg, stats, rho=RHO, mu=MU, fields=["VSS", "ML"])
    np.testing.assert_allclose(out["VSS"].values, 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out["ML"].values, 4.0, rtol=1e-12, atol=1e-12)


def test_unknown_derived_field_rejected():
    avg, stats = _random_inputs()
    try:
        derived_fields(avg, stats, fields=["TKE", "nope"])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "nope" in str(e)


# --- masks -------------------------------------------------------------------


def _grid_ds(u, counts):
    np.shape(u)
    return xr.Dataset(
        {**{v: (DIMS4, np.asarray(u, dtype=float)) for v in VEL_VARS},
         "par_ave2": (DIMS4, np.asarray(counts))})


def test_mask_count():
    u = np.ones((2, 2, 1, 1))
    counts = np.array([[[[10]], [[1]]], [[[10]], [[10]]]])
    out = apply_masks(_grid_ds(u, counts), [{"method": "count", "min_count": 5}])
    assert np.isnan(out["u_ins_mean"].values[0, 1, 0, 0])
    assert out["u_ins_mean"].values[0, 0, 0, 0] == 1.0


def test_mask_variance_flags_spike():
    u = np.zeros((5, 5, 1, 1))
    u[2, 2, 0, 0] = 100.0  # lone spike far above domain std
    out = apply_masks(_grid_ds(u, np.ones_like(u, dtype=int)),
                      [{"method": "variance", "k": 2.0}])
    assert np.isnan(out["u_ins_mean"].values[2, 2, 0, 0])
    assert np.isfinite(out["u_ins_mean"].values[0, 0, 0, 0])


def test_mask_outliers_keeps_uniform_field():
    u = np.ones((4, 4, 2, 2))
    out = apply_masks(_grid_ds(u, np.ones_like(u, dtype=int)),
                      ["outliers"])
    assert np.isfinite(out["u_ins_mean"].values).all()


# --- weighted phase average --------------------------------------------------


def test_weighted_phase_average():
    ds = xr.Dataset(
        {v: (("set",), np.array([1.0, 4.0])) for v in VEL_VARS},
        coords={"set": ["a", "b"]})
    weights = xr.DataArray(np.array([3, 1]), dims="set")
    assert float(phase_average(ds, weights=weights)["u_ins_mean"]) == 1.75
    assert float(phase_average(ds)["u_ins_mean"]) == 2.5  # unweighted default


# --- outputs -----------------------------------------------------------------


def test_eulerian_series_round_trip(tmp_path):
    pv = pytest.importorskip("pyvista")

    ds = xr.Dataset(
        {"TKE": (DIMS4, np.arange(8.0).reshape(2, 2, 2, 1))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0], "phase": [0]})
    pvd = write_eulerian_series(ds, tmp_path)
    assert pvd.exists()
    grid = pv.read(tmp_path / "phase_0000.vti")
    back = np.asarray(grid.point_data["TKE"]).reshape((2, 2, 2), order="F")
    np.testing.assert_allclose(back, ds["TKE"].values[..., 0], rtol=1e-6)


def _reference_eig_spread(*comps):
    """Independent per-voxel reference of the symmetric 3x3 spread."""
    clean = [np.nan_to_num(np.asarray(c, dtype=float),
                           nan=0.0, posinf=0.0, neginf=0.0) for c in comps]
    shape = clean[0].shape
    out = np.zeros(shape)
    for idx in np.ndindex(shape):
        a = np.array([[clean[0][idx], clean[1][idx], clean[2][idx]],
                      [clean[3][idx], clean[4][idx], clean[5][idx]],
                      [clean[6][idx], clean[7][idx], clean[8][idx]]])
        r = np.linalg.eigvalsh(a)
        out[idx] = 0.5 * (np.max(r) - np.min(r))
    return out


def test_eig_spread_matches_per_voxel_reference():
    from flowtracks.eulerian import _eig_spread
    rng = np.random.default_rng(42)
    shape = (4, 5, 6, 2)
    comps = [xr.DataArray(rng.normal(size=shape),
                          dims=("x", "y", "z", "phase")) for _ in range(9)]
    comps[0].values[0, 0, 0, 0] = np.nan  # NaN zero-filled like the kernel
    got = _eig_spread(*comps)
    assert got.shape == shape
    np.testing.assert_allclose(
        got.values, _reference_eig_spread(*[c.values for c in comps]),
        rtol=1e-12)


def test_save_netcdf_compression_and_attrs(tmp_path):
    pytest.importorskip("netCDF4")  # optional extra flowtracks[netcdf]
    ds = xr.Dataset(
        {"u_ins_mean": (DIMS4, np.zeros((4, 4, 4, 4)))},
        coords={"x": np.arange(4.0), "y": np.arange(4.0),
                "z": np.arange(4.0), "phase": np.arange(4)})
    path = tmp_path / "out.nc"
    save_netcdf(ds, path)
    back = xr.open_dataset(path)
    assert back["u_ins_mean"].attrs["units"] == "m s-1"
    assert back.attrs["Conventions"] == "CF-1.8"
    assert back["u_ins_mean"].encoding.get("zlib")
