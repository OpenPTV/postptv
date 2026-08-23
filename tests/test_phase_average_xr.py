import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from flowtracks.phase_average import fluctuations, open_sets, phase_average, run

VARS = ["u_ins_mean", "v_ins_mean", "w_ins_mean"]


def _write_grid(path, value):
    with h5py.File(path, "w") as f:
        for v in VARS:
            f[v] = np.full((3, 3, 2, 4), value, dtype=float)
        f["par_ave2"] = np.full((3, 3, 2, 4), 2, dtype=np.int64)
        f["x_vals"] = np.arange(3.0)
        f["y_vals"] = np.arange(3.0)
        f["z_vals"] = np.arange(2.0)
        f.attrs["cycletime"] = 4286


def test_phase_average_and_fluctuations(tmp_path):
    _write_grid(tmp_path / "a_grid.h5", 1.0)
    _write_grid(tmp_path / "b_grid.h5", 3.0)

    ds = open_sets(tmp_path, ["a", "b"], VARS)
    assert ds["u_ins_mean"].dims == ("set", "x", "y", "z", "phase")

    avg = phase_average(ds)
    assert float(avg["u_ins_mean"].mean()) == 2.0
    assert avg.attrs["cycletime"] == 4286

    fluct = fluctuations(ds, avg)
    # deviations across sets sum to zero at every grid point and phase
    assert float(np.abs(fluct["u_ins_mean"].sum("set")).max()) == 0.0
    assert float(fluct["u_ins_mean"].sel(set="a").mean()) == -1.0


def test_recipe_run(tmp_path):
    _write_grid(tmp_path / "a_grid.h5", 1.0)
    _write_grid(tmp_path / "b_grid.h5", 3.0)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump({"sets": ["a", "b"], "variables": VARS, "output": "out.nc"})
    )

    out_path = run(recipe)

    out = xr.open_dataset(out_path)
    assert float(out["u_phase_averaged"].mean()) == 2.0
    assert out["u_fluct"].dims == ("set", "x", "y", "z", "phase")
