"""Modular pipeline for 3DPTV data processing.

Each function is one independently-runnable stage (the decomposition the
cloud orchestrator will distribute). All computation lives in the xr modules
and the stage wrappers — nothing is implemented here twice.
"""
from pathlib import Path

import numpy as np
import yaml


# --- Step 1: PTV-IS to Lagrangian HDF5 ---
def ptv_is_to_lagrangian(set_name, config, base='.'):
    from flowtracks.io import save_particles_table, trajectories_ptvis
    ptv_is_path = Path(base) / config['data_path'] / set_name / config['ptv_res_path']
    full_h5_path = Path(base) / config['data_path'] / config['h5_path']
    h5file = full_h5_path / f"{set_name}_traj{config['traj_min_length']}.h5"
    if not h5file.exists():
        trajectories = trajectories_ptvis(
            ptv_is_path / "ptv_is.%d", first=config['first'],
            last=config['last'], frate=config['frate'],
            traj_min_len=config['traj_min_length'])
        save_particles_table(h5file, trajectories)
    return h5file


# --- Step 2: Lagrangian to Eulerian Grid (per set, distributed) ---
def lagrangian_to_eulerian(set_name, config, grid_params, base='.'):
    from flowtracks.io import Scene

    from batch_Lagrangian_to_Eulerian import eulerian_grid
    full_h5_path = Path(base) / config['data_path'] / config['h5_path']
    h5file = full_h5_path / f"{set_name}_traj{config['traj_min_length']}.h5"
    gridfile = full_h5_path / f"{set_name}_grid.h5"
    if not gridfile.exists():
        cycletime = np.ceil((60 / config['hb']) * config['frate'])
        eulerian_grid(Scene(h5file), grid_params, config['first'],
                      config['last'], cycletime, out_fname=gridfile,
                      deltat=config['delta_t'],
                      min_count=config.get('min_count', 50))
    return gridfile


# --- Step 3: Phase Averaging (central node) ---
def phase_average_all_sets(config, grid_params):
    from phase_average_fluctuations import phase_average_sets
    phase_average_sets()
    return Path(config['data_path']) / config['h5_path'] / "phase_averaged.h5"


# --- Step 4: Compute Fluctuations (per set, distributed) ---
def compute_fluctuations(set_name, config):
    from phase_average_fluctuations import calculate_fluctuations
    calculate_fluctuations()
    return Path(config['data_path']) / config['h5_path'] / f"{set_name}_grid.h5"


# --- Step 5: Turbulent Statistics (central or distributed) ---
def compute_turbulent_statistics(config):
    from turbulent_statistics import main as ts_main
    ts_main()


# --- Step 6: VTK Export (per set or central) ---
def export_vtk(config):
    from flowtracks.vtk_export import main as vtk_main
    vtk_main()


def streamlined_pipeline(config_path="config.yaml", grid_path="grid.yaml", recipe_path="post_recipe.yaml", base="."):
    """Streamlined in-memory pipeline.

    1. Step 1: ptv_is -> Lagrangian HDF5 (via flowtracks).
    2. Step 2: Bin Lagrangian scenes directly into in-memory xarray Datasets.
    3. Steps 3-5: Run phase-averaging, fluctuations, turbulent statistics and derived
       fields directly on xarray Datasets in memory.
    4. Save final output as NetCDF or Zarr with rich CF metadata for ParaView, and
       optionally export binary VTK files.
    """
    from flowtracks.io import Scene
    from flowtracks.eulerian import eulerian_grid, export_vtk, run_post_analysis_ds, save_dataset

    base_path = Path(base)
    with open(base_path / config_path) as f:
        config = yaml.safe_load(f)
    with open(base_path / grid_path) as f:
        grid_params = yaml.safe_load(f)
    with open(base_path / recipe_path) as f:
        recipe = yaml.safe_load(f)

    cycletime = float(np.ceil((60 / config["hb"]) * config["frate"]))
    ds_sets = {}

    for set_name in config["set_names"]:
        h5file = ptv_is_to_lagrangian(set_name, config, base=base)
        scene = Scene(h5file)
        ds_sets[set_name] = eulerian_grid(
            scene,
            grid_params,
            config["first"],
            config["last"],
            cycletime,
            deltat=config["delta_t"],
            base_time=100000,
            min_count=config.get("min_count", 50),
        )

    out_ds = run_post_analysis_ds(ds_sets, recipe)
    grid_dir = base_path / Path(recipe_path).parent / recipe.get("grid_dir", ".")
    out_path = grid_dir / recipe["output"]

    save_dataset(out_ds, out_path)
    print(f"[streamlined_pipeline] Saved final analysis to {out_path}")

    vtk_cfg = recipe.get("vtk")
    if vtk_cfg:
        files = export_vtk(
            out_ds,
            grid_dir / vtk_cfg.get("dir", "vtk_output"),
            prefix=vtk_cfg.get("prefix", "phase"),
        )
        print(f"[streamlined_pipeline] Exported {len(files)} VTK files to {files[0].parent}")

    return out_path


# --- Main orchestrator ---
def main():
    streamlined_pipeline()


if __name__ == "__main__":
    main()

