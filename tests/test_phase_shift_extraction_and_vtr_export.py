"""Self-check for the wp1-4 phase-alignment / ParaView-export additions.

Covers: find_phase_shift recovers a known injected shift, align_phases makes
two differently-shifted synthetic runs agree after shifting, mask_array NaNs
outside a boolean ROI, and export_vtk_rectilinear_series writes a readable
.vtr + .pvd pair (round-tripped with vtk's own reader).
"""
import numpy as np
import xarray as xr

from flowtracks.eulerian import (
    align_phases, export_vtk_rectilinear_series, find_phase_shift,
    apply_masks, shift_phase,
)

N = 24


def _bump_dataset(peak, n=N, nx=3, ny=3, nz=2):
    phase = np.arange(n)
    amp = 0.5 * (1 + np.cos(2 * np.pi * (phase - peak) / n))  # peaks at `peak`
    u = np.ones((nx, ny, nz, n)) * amp
    zeros = np.zeros((nx, ny, nz, n))
    return xr.Dataset(
        {"u_ins_mean": (("x", "y", "z", "phase"), u),
         "v_ins_mean": (("x", "y", "z", "phase"), zeros),
         "w_ins_mean": (("x", "y", "z", "phase"), zeros)},
        coords={"x": np.arange(nx, dtype=float), "y": np.arange(ny, dtype=float),
                "z": np.arange(nz, dtype=float), "phase": phase},
    )


def test_find_phase_shift_recovers_injected_peak():
    ds = _bump_dataset(peak=17)
    shift = find_phase_shift(ds, reference_phase=0)
    aligned = shift_phase(ds, shift)
    curve = aligned["u_ins_mean"].mean(("x", "y", "z"))
    assert int(curve.argmax("phase")) == 0


def test_align_phases_matches_multiple_runs_to_common_reference():
    sets = {"wp1": _bump_dataset(peak=3), "wp2": _bump_dataset(peak=19)}
    aligned = align_phases(sets, reference_phase=5)
    for ds in aligned.values():
        curve = ds["u_ins_mean"].mean(("x", "y", "z"))
        assert int(curve.argmax("phase")) == 5
    np.testing.assert_allclose(
        aligned["wp1"]["u_ins_mean"].values, aligned["wp2"]["u_ins_mean"].values)


def test_mask_array_nans_outside_roi():
    ds = _bump_dataset(peak=0, nx=2, ny=2, nz=1)
    mask = np.array([[[True], [False]], [[True], [True]]])
    out = apply_masks(ds, [{"method": "array", "mask": mask}])
    assert np.isnan(out["u_ins_mean"].isel(x=0, y=1).values).all()
    assert not np.isnan(out["u_ins_mean"].isel(x=0, y=0).values).any()


def test_export_vtk_rectilinear_series_writes_readable_pvd_and_vtr(tmp_path):
    import vtk

    ds = _bump_dataset(peak=0, nx=4, ny=3, nz=2)[["u_ins_mean", "v_ins_mean", "w_ins_mean"]]
    ds = ds.isel(phase=slice(0, 3))
    pvd = export_vtk_rectilinear_series(ds, tmp_path / "vtr_out", prefix="wp1", dt=0.5)

    assert pvd.exists()
    text = pvd.read_text()
    assert 'timestep="0"' in text and 'timestep="1"' in text
    vtr_files = sorted((tmp_path / "vtr_out").glob("wp1_*.vtr"))
    assert len(vtr_files) == 3

    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(str(vtr_files[0]))
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetPointData().GetArray("velocity") is not None
    assert grid.GetNumberOfPoints() == 4 * 3 * 2


if __name__ == "__main__":
    test_find_phase_shift_recovers_injected_peak()
    test_align_phases_matches_multiple_runs_to_common_reference()
    test_mask_array_nans_outside_roi()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_export_vtk_rectilinear_series_writes_readable_pvd_and_vtr(Path(d))
    print("ok")
