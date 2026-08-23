"""Phase averaging as pure Dataset -> Dataset functions, driven by a YAML recipe.

xarray re-expression of phase_average_fluctuations.py (pyorc-style):
the domain math is two one-liners; everything else is the data contract.

Run:  uv run python src/phase_average_xr.py [recipe.yaml]
"""

from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import yaml

DIMS = ("x", "y", "z", "phase")


def open_grid(path: Path, variables: list[str]) -> xr.Dataset:
    """Read one <set>_grid.h5 into a self-describing Dataset with dims (x, y, z, phase)."""
    with h5py.File(path, "r") as f:
        coords = {
            "x": f["x_vals"][()],
            "y": f["y_vals"][()],
            "z": f["z_vals"][()],
            "phase": np.arange(f[variables[0]].shape[-1]),
        }
        data = {v: (DIMS, f[v][()]) for v in variables}
        attrs = dict(f.attrs)
    return xr.Dataset(data, coords=coords, attrs=attrs)


def open_sets(grid_dir: Path, sets: list[str], variables: list[str]) -> xr.Dataset:
    """Stack per-set grid files along a new 'set' dimension."""
    return xr.concat(
        [open_grid(grid_dir / f"{s}_grid.h5", variables) for s in sets],
        dim=xr.DataArray(sets, dims="set", name="set"),
        join="exact",  # sets binned on different grids must fail, not NaN-pad
    )


def phase_average(ds: xr.Dataset, weights: xr.DataArray | None = None) -> xr.Dataset:
    """Ensemble mean over realizations: (set, x, y, z, phase) -> (x, y, z, phase).

    With weights (e.g. the per-set sample counts par_ave2) the mean is
    sum(w*u)/sum(w) — sets that saw more particles in a voxel count more.
    """
    if weights is None:
        return ds.mean("set", keep_attrs=True)
    return ds.weighted(weights).mean("set", keep_attrs=True)


def fluctuations(ds: xr.Dataset, avg: xr.Dataset) -> xr.Dataset:
    """Per-set deviation from the phase average (broadcasts over 'set')."""
    return ds - avg


def run(recipe_path: Path) -> Path:
    recipe = yaml.safe_load(recipe_path.read_text())
    grid_dir = recipe_path.parent / recipe.get("grid_dir", ".")
    variables = recipe["variables"]

    ds = open_sets(grid_dir, recipe["sets"], variables)
    avg = phase_average(ds)
    fluct = fluctuations(ds, avg)

    out = xr.merge(
        [
            avg.rename({v: f"{v.removesuffix('_ins_mean')}_phase_averaged" for v in variables}),
            fluct.rename({v: f"{v.removesuffix('_ins_mean')}_fluct" for v in variables}),
        ],
        combine_attrs="override",
    )
    out_path = grid_dir / recipe["output"]
    out.to_netcdf(out_path)
    print(f"Saved {list(out.data_vars)} for {len(recipe['sets'])} sets to {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1] if len(sys.argv) > 1 else "phase_recipe.yaml"))
