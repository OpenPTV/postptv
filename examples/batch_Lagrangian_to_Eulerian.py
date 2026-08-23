# %%
from pathlib import Path

import h5py
import numpy as np
import yaml

from flowtracks.eulerian import eulerian_grid as _eulerian_grid_ds


def eulerian_grid(
    scene,
    grid_params,
    first,
    last,
    cycletime,
    out_fname="anyset_grid.h5",
    deltat=90,
    base_time=100000,
    min_count=50,
):
    """Compute a phase-binned Eulerian grid and save to HDF5.

    Thin wrapper: the computation lives in post_analysis_xr.eulerian_grid
    (single whole-table read + vectorized binning); this function only keeps
    the legacy HDF5 file contract for existing callers.
    """
    ds = _eulerian_grid_ds(scene, grid_params, first, last, cycletime,
                           deltat=deltat, base_time=base_time,
                           min_count=min_count)
    with h5py.File(out_fname, "w") as f:
        for name in ("u_ins_mean", "v_ins_mean", "w_ins_mean", "par_ave2"):
            f.create_dataset(name, data=ds[name].values)
        for axis in "xyz":
            f.create_dataset(f"{axis}_vals", data=ds[axis].values)
        for key, val in ds.attrs.items():
            f.attrs[key] = val
    return ds


def main():
    with open("grid.yaml") as f:
        grid_params = yaml.safe_load(f)
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    first = config["first"]
    last = config["last"]
    frate = config["frate"]
    hb = config["hb"]
    delta_t = config["delta_t"]
    traj_min_length = config["traj_min_length"]

    cycletime = np.ceil((60 / hb) * frate)
    print(f"cycle time: {cycletime} frames")
    print(f"Expected number of cycles: {int(np.floor((last - first) / cycletime))}")

    full_h5_path = Path(config["data_path"]) / Path(config["h5_path"])
    full_h5_path.mkdir(parents=True, exist_ok=True)

    from flowtracks.io import Scene

    for set_name in config["set_names"]:
        print(f"Processing set: {set_name}")
        h5file = full_h5_path / f"{set_name}_traj{traj_min_length}.h5"
        eulerian_grid(
            Scene(h5file),
            grid_params,
            first,
            last,
            cycletime,
            out_fname=full_h5_path / f"{set_name}_grid.h5",
            deltat=delta_t,
            base_time=100000,
            min_count=50,
        )


if __name__ == "__main__":
    main()
