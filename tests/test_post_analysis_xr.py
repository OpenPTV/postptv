import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from flowtracks.eulerian import (
    derived_fields,
    eulerian_grid,
    run,
    shift_phase,
    turbulent_statistics,
)
from test_phase_average_xr import VARS, _write_grid


def _fluct_ds(values_by_set):
    """Build a fluct Dataset with dims (set, x, y, z, phase) from constants."""
    shape = (3, 3, 2, 4)
    data = {
        f"{a}_fluct": (
            ("set", "x", "y", "z", "phase"),
            np.stack([np.full(shape, v) for v in values_by_set]),
        )
        for a in "uvw"
    }
    return xr.Dataset(data, coords={"set": ["a", "b"]})


def test_turbulent_statistics_matches_legacy_accumulation():
    fluct = _fluct_ds([2.0, -1.0])
    counts = xr.DataArray(
        np.stack([np.full((3, 3, 2, 4), 3), np.full((3, 3, 2, 4), 1)]),
        dims=("set", "x", "y", "z", "phase"),
        coords={"set": ["a", "b"]},
    )
    stats = turbulent_statistics(fluct, counts)

    # legacy: sum(f^2 * n) / sum(n) = (4*3 + 1*1) / 4 = 3.25
    assert float(stats["u_ins_u_ins"].mean()) == 3.25
    assert np.isclose(float(stats["u_rms"].mean()), np.sqrt(3.25))
    # cross term: (2*2*3 + (-1)(-1)*1) / 4 = 3.25 (same constants for u,v,w)
    assert float(stats["u_ins_v_ins"].mean()) == 3.25

    # zero counts -> NaN, as in the legacy script
    stats0 = turbulent_statistics(fluct, xr.zeros_like(counts))
    assert np.isnan(stats0["u_rms"]).all()


def test_shift_phase_rolls_and_records():
    ds = xr.Dataset(
        {"u_ins_mean": (("phase",), np.arange(4.0))},
        coords={"phase": np.arange(4)},
    )
    out = shift_phase(ds, 1)
    assert list(out["u_ins_mean"].values) == [3.0, 0.0, 1.0, 2.0]
    assert out.attrs["shift"] == 1
    assert shift_phase(ds, 0) is ds


def test_derived_fields():
    avg = xr.Dataset({v: xr.DataArray(2.0) for v in
                      ["u_ins_mean", "v_ins_mean", "w_ins_mean"]})
    stats = xr.Dataset({f"{a}_ins_{a}_ins": xr.DataArray(1.0) for a in "uvw"})
    d = derived_fields(avg, stats)
    assert float(d["MKE"]) == 6.0  # 0.5 * 3 * 4
    assert float(d["TKE"]) == 1.5  # 0.5 * 3 * 1


class FakeScene:
    """Two particles per frame at known positions/velocities."""

    POS = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    VEL = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def frame_by_time(self, t):
        outer = self

        class Frame:
            def pos(self):
                return outer.POS

            def velocity(self):
                return outer.VEL

        return Frame()

    def collect(self, keys):
        frames = np.arange(100001, 100011)
        cols = {
            "pos": np.tile(self.POS, (len(frames), 1)),
            "velocity": np.tile(self.VEL, (len(frames), 1)),
            "time": np.repeat(frames, len(self.POS)),
        }
        return [cols[k] for k in keys]


def test_eulerian_grid_bins_particles():
    grid_params = {"stepx": 2, "stepy": 2, "stepz": 2,
                   "min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1,
                   "min_z": 0, "max_z": 1}
    ds = eulerian_grid(FakeScene(), grid_params, first=100001, last=100010,
                       cycletime=10, deltat=2, min_count=1)
    assert ds["u_ins_mean"].dims == ("x", "y", "z", "phase")
    # particle 1 always lands in cell (0,0,0) with u=1
    assert float(ds["u_ins_mean"].isel(x=0, y=0, z=0, phase=0)) == 1.0
    assert float(ds["w_ins_mean"].isel(x=1, y=1, z=1, phase=0)) == 6.0
    assert int(ds["par_ave2"].sum()) == 20  # 2 particles x 10 frames


def test_recipe_run_end_to_end(tmp_path):
    pytest.importorskip("vtk")
    _write_grid(tmp_path / "a_grid.h5", 1.0)
    _write_grid(tmp_path / "b_grid.h5", 3.0)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({
        "sets": ["a", "b"], "output": "post.nc",
        "vtk": {"dir": "vtk", "prefix": "p"},
    }))

    out_path = run(recipe)

    out = xr.open_dataset(out_path)
    for name in ["u_ins_mean", "u_fluct", "u_rms", "u_ins_v_ins", "MKE", "TKE"]:
        assert name in out
    # sets are constants 1 and 3 -> fluct = -/+1, all counts equal -> u'u' = 1
    assert float(out["u_ins_u_ins"].mean()) == 1.0
    assert (tmp_path / "vtk" / "p_000.vtk").exists()


def test_recipe_run_zarr_output(tmp_path):
    _write_grid(tmp_path / "a_grid.h5", 1.0)
    _write_grid(tmp_path / "b_grid.h5", 3.0)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({
        "sets": ["a", "b"], "output": "post.zarr",
    }))

    out_path = run(recipe)
    assert out_path.name == "post.zarr"
    out = xr.open_zarr(out_path)
    assert "u_ins_mean" in out
    assert out["u_ins_mean"].attrs["units"] == "m s-1"

