"""Derived fields vs the legacy sample_vtkcode per-slice formulas; masks;
weighted phase average; binary VTK round-trip; netCDF encoding."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from flowtracks.phase_average import phase_average
from flowtracks.eulerian import (
    ALL_DERIVED,
    VEL_VARS,
    apply_masks,
    derived_fields,
    export_vtk,
    save_netcdf,
)
from flowtracks.vtk_export import max_eigenvalue_spread

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


def _legacy_slice(avg, stats, t):
    """Verbatim re-execution of sample_vtkcode.py main() math for one slice."""
    u = avg["u_ins_mean"].values[..., t]
    v = avg["v_ins_mean"].values[..., t]
    w = avg["w_ins_mean"].values[..., t]
    uu = stats["u_ins_u_ins"].values[..., t].copy()
    vv = stats["v_ins_v_ins"].values[..., t].copy()
    ww = stats["w_ins_w_ins"].values[..., t].copy()
    uv = stats["u_ins_v_ins"].values[..., t].copy()
    uw = stats["u_ins_w_ins"].values[..., t].copy()
    vw = stats["v_ins_w_ins"].values[..., t].copy()
    gridx = float(avg.x[1] - avg.x[0])
    dy = float(avg.y[1] - avg.y[0])
    dz = float(avg.z[1] - avg.z[0])
    dx = dz
    out = {}
    out["MKE"] = 0.5 * RHO * (u**2 + v**2 + w**2)
    out["TKE"] = 0.5 * RHO * (uu + vv + ww)
    VEL = np.sqrt(u**2 + v**2 + w**2)
    out["VEL"] = VEL
    out["PRT"] = np.divide(gridx * 1000.0, VEL, out=np.zeros_like(VEL),
                           where=VEL != 0)
    uy, ux, uz = np.gradient(-u, dy, dx, dz)
    vy, vx, vz = np.gradient(v, dy, dx, dz)
    wy, wx, wz = np.gradient(w, dy, dx, dz)
    p2, p3, p6 = uy + vx, uz + wx, vz + wy
    p1 = 2 * ux - (2 / 3) * (ux + uy + uz)
    p5 = 2 * vy - (2 / 3) * (vx + vy + vz)
    p9 = 2 * wz - (2 / 3) * (wx + wy + wz)
    out["VSS"] = max_eigenvalue_spread(
        p1.copy(), p2.copy(), p3.copy(), p2.copy(), p5.copy(), p6.copy(),
        p3.copy(), p6.copy(), p9.copy()) * RHO * MU
    out["RSS"] = max_eigenvalue_spread(
        uu.copy(), uv.copy(), uw.copy(), uv.copy(), vv.copy(), vw.copy(),
        uw.copy(), vw.copy(), ww.copy()) * RHO
    meandiss = (2 * (uy + vx)**2 + 2 * (uz + wx)**2 + 2 * (vz + wy)**2
                + p1**2 + p5**2 + p9**2)
    out["ML"] = meandiss * MU * RHO
    prod = (-uu * ux - vv * vy - ww * wz - uv * uy - uw * uz - vw * vz
            - uv * vx - uw * wx - vw * wy)
    out["TL"] = prod * RHO
    t11 = RHO * MU * 2 * ux - RHO * uu
    t22 = RHO * MU * 2 * vy - RHO * vv
    t33 = RHO * MU * 2 * wz - RHO * ww
    t12 = RHO * MU * (uy + vx) - RHO * uv
    t13 = RHO * MU * (uz + wx) - RHO * uw
    t23 = RHO * MU * (vz + wy) - RHO * vw
    out["ScalarShear"] = (1 / np.sqrt(3)) * np.sqrt(
        t11**2 + t22**2 + t33**2 - (t11 * t22 + t22 * t33 + t11 * t33)
        + 3 * (t12**2 + t23**2 + t13**2))
    w1, w2, w3 = wy - vz, uz - wx, vx - uy
    h1 = w1 * u + w2 * v + w3 * w
    h2 = np.sqrt((w1 * u)**2 + (w2 * v)**2 + (w3 * w)**2)
    out["H1"], out["H2"] = h1, h2
    out["H3"] = np.divide(h1, h2, out=np.zeros_like(h1), where=h2 != 0)
    out["H4"] = np.divide(np.abs(h1), h2, out=np.zeros_like(h1), where=h2 != 0)
    return out


def test_all_derived_fields_match_legacy_per_slice():
    avg, stats = _random_inputs()
    new = derived_fields(avg, stats, rho=RHO, mu=MU, fields=ALL_DERIVED)
    for t in range(avg.sizes["phase"]):
        legacy = _legacy_slice(avg, stats, t)
        for name in ALL_DERIVED:
            np.testing.assert_allclose(
                new[name].values[..., t], legacy[name],
                rtol=1e-10, atol=1e-12, err_msg=f"{name} @ phase {t}")


def test_unknown_derived_field_rejected():
    avg, stats = _random_inputs()
    try:
        derived_fields(avg, stats, fields=["TKE", "nope"])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "nope" in str(e)


# --- masks -------------------------------------------------------------------


def _grid_ds(u, counts):
    shape = np.shape(u)
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


def test_binary_vtk_round_trip(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util import numpy_support

    ds = xr.Dataset(
        {"TKE": (DIMS4, np.arange(8.0).reshape(2, 2, 2, 1))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0], "phase": [0]})
    (path,) = export_vtk(ds, tmp_path)
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput().GetPointData().GetArray("TKE")
    back = numpy_support.vtk_to_numpy(data).reshape((2, 2, 2), order="F")
    np.testing.assert_allclose(back, ds["TKE"].values[..., 0])


def test_save_netcdf_compression_and_attrs(tmp_path):
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
