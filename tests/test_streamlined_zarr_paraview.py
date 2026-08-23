"""Comprehensive tests for Zarr storage, CF metadata / ParaView compatibility,
and in-memory streamlined pipeline execution.
"""

from pathlib import Path
import numpy as np
import pytest
import xarray as xr
import yaml

from flowtracks.eulerian import (
    ALL_DERIVED,
    CF_ATTRS,
    apply_cf_metadata,
    derived_fields,
    eulerian_grid,
    export_vtk,
    run_post_analysis_ds,
    save_dataset,
    save_netcdf,
    save_zarr,
)
from flowtracks.pipeline import streamlined_pipeline


class FakeScene:
    """Mock trajectory scene with deterministic particle data."""

    def __init__(self, seed=42, n_particles=50):
        rng = np.random.default_rng(seed)
        frames = np.arange(100001, 100021)
        n_total = len(frames) * n_particles
        self.pos = rng.uniform([0.0, 0.0, 0.0], [0.01, 0.02, 0.01], (n_total, 3))
        self.vel = rng.normal(1.0, 0.2, (n_total, 3))
        self.time = np.repeat(frames, n_particles)

    def collect(self, keys):
        cols = {"pos": self.pos, "velocity": self.vel, "time": self.time}
        return [cols[k] for k in keys]


@pytest.fixture
def sample_dataset():
    """Create a sample post-analysis Dataset with all physical variables."""
    mids = {"x": np.linspace(0, 0.01, 3), "y": np.linspace(0, 0.02, 3),
            "z": np.linspace(0, 0.01, 2), "phase": np.arange(4)}
    shape = (3, 3, 2, 4)
    rng = np.random.default_rng(123)

    avg = xr.Dataset(
        {v: (("x", "y", "z", "phase"), rng.uniform(0.5, 2.0, shape)) for v in ["u_ins_mean", "v_ins_mean", "w_ins_mean"]},
        coords=mids,
    )
    avg["par_ave2"] = (("x", "y", "z", "phase"), rng.integers(10, 100, shape))

    stats = xr.Dataset(
        {f"{a}_ins_{b}_ins": (("x", "y", "z", "phase"), rng.uniform(0.01, 0.1, shape))
         for i, a in enumerate("uvw") for b in "uvw"[i:]},
        coords=mids,
    )
    for a in "uvw":
        stats[f"{a}_rms"] = np.sqrt(stats[f"{a}_ins_{a}_ins"])

    derived = derived_fields(avg, stats, fields=ALL_DERIVED)
    out = xr.merge([avg, stats, derived], combine_attrs="override")
    return apply_cf_metadata(out)


def test_cf_metadata_paraview_compatibility(sample_dataset):
    """Verify standard CF-1.8 attributes for ParaView rectilinear grid recognition."""
    ds = apply_cf_metadata(sample_dataset)

    # Coordinate attributes
    assert ds.x.attrs["axis"] == "X"
    assert ds.x.attrs["units"] == "m"
    assert ds.y.attrs["axis"] == "Y"
    assert ds.y.attrs["units"] == "m"
    assert ds.z.attrs["axis"] == "Z"
    assert ds.z.attrs["units"] == "m"
    assert ds.phase.attrs["axis"] == "T"

    # Variable attributes
    assert ds["u_ins_mean"].attrs["units"] == "m s-1"
    assert ds["TKE"].attrs["units"] == "J m-3"
    assert ds["MKE"].attrs["units"] == "J m-3"
    assert ds["VEL"].attrs["units"] == "m s-1"
    assert ds["VSS"].attrs["units"] == "Pa"
    assert ds["H1"].attrs["units"] == "m s-2"
    assert ds.attrs["Conventions"] == "CF-1.8"


def test_zarr_and_netcdf_roundtrip_equivalence(sample_dataset, tmp_path):
    """Saving to Zarr vs NetCDF must preserve identical data arrays, coords, and metadata."""
    nc_path = tmp_path / "test.nc"
    zarr_path = tmp_path / "test.zarr"

    save_netcdf(sample_dataset, nc_path)
    save_zarr(sample_dataset, zarr_path)

    ds_nc = xr.open_dataset(nc_path)
    ds_zarr = xr.open_zarr(zarr_path)

    # Verify identical data variables and values
    assert set(ds_nc.data_vars) == set(ds_zarr.data_vars)
    for var in ds_nc.data_vars:
        np.testing.assert_allclose(ds_nc[var].values, ds_zarr[var].values, rtol=1e-6)

    # Verify CF attributes match
    assert ds_nc["TKE"].attrs["long_name"] == ds_zarr["TKE"].attrs["long_name"]
    assert ds_nc.x.attrs["axis"] == ds_zarr.x.attrs["axis"]


def test_save_dataset_auto_format_selection(sample_dataset, tmp_path):
    """save_dataset should automatically select Zarr vs NetCDF based on extension."""
    p_nc = tmp_path / "out.nc"
    p_zarr = tmp_path / "out.zarr"

    save_dataset(sample_dataset, p_nc)
    save_dataset(sample_dataset, p_zarr)

    assert p_nc.exists()
    assert p_zarr.exists() and p_zarr.is_dir()

    read_nc = xr.open_dataset(p_nc)
    read_zarr = xr.open_zarr(p_zarr)

    np.testing.assert_allclose(read_nc["u_ins_mean"].values, read_zarr["u_ins_mean"].values)


def test_in_memory_vs_disk_pipeline_equivalence(tmp_path):
    """In-memory run_post_analysis_ds must produce identical output to disk recipe run."""
    grid_params = {"stepx": 2, "stepy": 2, "stepz": 2,
                   "min_x": 0.0, "max_x": 0.01,
                   "min_y": 0.0, "max_y": 0.02,
                   "min_z": 0.0, "max_z": 0.01}

    scene_a = FakeScene(seed=1)
    scene_b = FakeScene(seed=2)

    ds_a = eulerian_grid(scene_a, grid_params, first=100001, last=100020, cycletime=20, deltat=2, min_count=1)
    ds_b = eulerian_grid(scene_b, grid_params, first=100001, last=100020, cycletime=20, deltat=2, min_count=1)

    ds_dict = {"a": ds_a, "b": ds_b}
    recipe = {
        "sets": ["a", "b"],
        "output": "out.zarr",
        "derived": {"fields": ["MKE", "TKE", "VEL"]},
    }

    ds_in_memory = run_post_analysis_ds(ds_dict, recipe)

    # Compare with save_dataset roundtrip
    zarr_file = tmp_path / "out.zarr"
    save_dataset(ds_in_memory, zarr_file)
    ds_from_zarr = xr.open_zarr(zarr_file)

    for var in ds_in_memory.data_vars:
        np.testing.assert_allclose(ds_in_memory[var].values, ds_from_zarr[var].values)


def test_export_vtk_structured_grid(sample_dataset, tmp_path):
    """export_vtk must write valid binary VTK structured grid files for all phases."""
    pytest.importorskip("vtk")
    vtk_dir = tmp_path / "vtk"
    paths = export_vtk(sample_dataset, vtk_dir, prefix="test_phase")

    assert len(paths) == len(sample_dataset.phase)
    for p in paths:
        assert p.exists()
        header = p.read_bytes()[:100].decode("ascii", errors="ignore")
        assert "STRUCTURED_GRID" in header or "vtk" in header.lower()


def test_streamlined_pipeline_full_integration(tmp_path, monkeypatch):
    """Test full streamlined_pipeline execution with both NetCDF and Zarr output."""
    pytest.importorskip("vtk")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "traj_min_length": 20, "first": 100001, "last": 100020,
        "frate": 5000, "hb": 70, "data_path": ".", "set_names": ["s1", "s2"],
        "ptv_res_path": "res", "h5_path": ".", "delta_t": 2,
    }))
    (tmp_path / "grid.yaml").write_text(yaml.safe_dump({
        "stepx": 2, "stepy": 2, "stepz": 2,
        "min_x": 0.0, "max_x": 0.01,
        "min_y": 0.0, "max_y": 0.02,
        "min_z": 0.0, "max_z": 0.01,
    }))
    (tmp_path / "post_recipe.yaml").write_text(yaml.safe_dump({
        "sets": ["s1", "s2"],
        "output": "final.zarr",
        "vtk": {"dir": "vtk_out", "prefix": "phase"},
    }))

    mock_scenes = {"s1": FakeScene(seed=10), "s2": FakeScene(seed=20)}

    def mock_ptv_is_to_lagrangian(set_name, config, base="."):
        return Path(base) / f"{set_name}_traj.h5"

    class MockScene:
        def __init__(self, path):
            name = Path(path).name.split("_")[0]
            self.sc = mock_scenes[name]

        def collect(self, keys):
            return self.sc.collect(keys)

    monkeypatch.setattr("flowtracks.pipeline.ptv_is_to_lagrangian", mock_ptv_is_to_lagrangian)
    monkeypatch.setattr("flowtracks.io.Scene", MockScene)

    out_p = streamlined_pipeline(
        config_path="config.yaml", grid_path="grid.yaml",
        recipe_path="post_recipe.yaml", base=str(tmp_path)
    )

    assert out_p.exists()
    assert out_p.name == "final.zarr"
    ds = xr.open_zarr(out_p)
    assert "TKE" in ds
    assert (tmp_path / "vtk_out" / "phase_000.vtk").exists()
