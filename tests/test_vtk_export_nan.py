"""Tests for VTK export: NaN/Inf preservation and valid/invalid cell counts.

Verifies that flowtracks.writers.write_eulerian_series:
- Preserves NaN for voxels with insufficient samples (does not zero them)
- Converts Inf to NaN (does not zero or overflow)
- Correctly exports the ``valid`` field (True/False per voxel)
- Correctly exports ``par_ave2`` (sample counts, including sub-threshold)
- Exports all non-finite voxels as NaN in velocity fields
"""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("pyvista")

from flowtracks.writers import _clean, write_eulerian_series


class TestCleanFunction:
    """Unit tests for _clean helper."""

    def test_nan_preserved(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = _clean(arr)
        assert np.isnan(out[1])
        assert out[0] == 1.0 and out[2] == 3.0

    def test_posinf_to_nan(self):
        arr = np.array([1.0, np.inf, 3.0])
        out = _clean(arr)
        assert np.isnan(out[1])
        assert out[0] == 1.0 and out[2] == 3.0

    def test_neginf_to_nan(self):
        arr = np.array([1.0, -np.inf, 3.0])
        out = _clean(arr)
        assert np.isnan(out[1])
        assert out[0] == 1.0 and out[2] == 3.0

    def test_finite_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = _clean(arr)
        np.testing.assert_array_equal(out, arr)

    def test_non_float_passthrough(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        out = _clean(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == arr.dtype

    def test_bool_passthrough(self):
        arr = np.array([True, False, True])
        out = _clean(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == arr.dtype

    def test_mixed_nonfinite(self):
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
        out = _clean(arr)
        assert out[0] == 1.0
        assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
        assert out[4] == 5.0


def _make_dataset_with_invalid_voxels():
    """Dataset with one voxel below min_count (NaN velocity, count=2, valid=False)."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    phase = np.array([0])

    u = np.ones((2, 2, 2, 1)) * 1.0
    u[0, 0, 0, 0] = np.nan  # insufficient samples
    v = np.ones((2, 2, 2, 1)) * 2.0
    w = np.ones((2, 2, 2, 1)) * 3.0
    # par_ave2: actual counts (1 for invalid voxel, 5 for valid)
    counts = np.array([[[[1], [5]], [[5], [5]]], [[[5], [5]], [[5], [5]]]])
    valid = counts >= 4

    return xr.Dataset(
        {
            "u_ins_mean": (("x", "y", "z", "phase"), u),
            "v_ins_mean": (("x", "y", "z", "phase"), v),
            "w_ins_mean": (("x", "y", "z", "phase"), w),
            "par_ave2": (("x", "y", "z", "phase"), counts),
            "valid": (("x", "y", "z", "phase"), valid),
        },
        coords={"x": x, "y": y, "z": z, "phase": phase},
    ), counts, valid


class TestVtkExportNanPreservation:
    """Integration tests for VTK export NaN/Inf and valid/invalid handling."""

    def test_nan_preserved_in_velocity(self, tmp_path):
        ds, counts, valid = _make_dataset_with_invalid_voxels()
        pvd = write_eulerian_series(ds, tmp_path, prefix="test")
        import pyvista as pv

        grid = pv.read(str(tmp_path / "test_0000.vti"))
        vel = grid.point_data["velocity"]
        nan_count = np.isnan(vel).any(axis=1).sum()
        assert nan_count == 1, f"Expected 1 NaN voxel, got {nan_count}"

    def test_valid_field_exported(self, tmp_path):
        ds, counts, valid = _make_dataset_with_invalid_voxels()
        write_eulerian_series(ds, tmp_path, prefix="test")
        import pyvista as pv

        grid = pv.read(str(tmp_path / "test_0000.vti"))
        exported_valid = grid.point_data["valid"]
        assert exported_valid.dtype == bool or exported_valid.dtype == np.bool_
        assert not exported_valid[0], "Invalid voxel should be False"
        assert exported_valid[1:].all(), "Valid voxels should be True"

    def test_par_ave2_counts_exported(self, tmp_path):
        ds, counts, valid = _make_dataset_with_invalid_voxels()
        write_eulerian_series(ds, tmp_path, prefix="test")
        import pyvista as pv

        grid = pv.read(str(tmp_path / "test_0000.vti"))
        exported_counts = grid.point_data["par_ave2"]
        # Invalid voxel should show actual count (1), not 0
        assert exported_counts[0] == 1, f"Expected count 1 for invalid voxel, got {exported_counts[0]}"
        assert exported_counts[1] == 5, f"Expected count 5 for valid voxel, got {exported_counts[1]}"

    def test_inf_converted_to_nan(self, tmp_path):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        phase = np.array([0])

        u = np.array([[[[np.inf], [1.0]], [[2.0], [3.0]]], [[[4.0], [5.0]], [[6.0], [7.0]]]])
        v = np.ones((2, 2, 2, 1)) * 2.0
        w = np.ones((2, 2, 2, 1)) * 3.0

        ds = xr.Dataset(
            {"u_ins_mean": (("x", "y", "z", "phase"), u),
             "v_ins_mean": (("x", "y", "z", "phase"), v),
             "w_ins_mean": (("x", "y", "z", "phase"), w)},
            coords={"x": x, "y": y, "z": z, "phase": phase},
        )
        write_eulerian_series(ds, tmp_path, prefix="test")
        import pyvista as pv

        grid = pv.read(str(tmp_path / "test_0000.vti"))
        vel = grid.point_data["velocity"]
        assert np.isnan(vel[0, 0]), "Inf should be converted to NaN"

    def test_all_fields_present(self, tmp_path):
        ds, counts, valid = _make_dataset_with_invalid_voxels()
        write_eulerian_series(ds, tmp_path, prefix="test")
        import pyvista as pv

        grid = pv.read(str(tmp_path / "test_0000.vti"))
        assert "velocity" in grid.point_data
        assert "par_ave2" in grid.point_data
        assert "valid" in grid.point_data
        # VEL_VARS are combined into "velocity", not written separately