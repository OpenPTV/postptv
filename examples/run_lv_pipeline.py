"""Full 3D-PTV Post-Processing Pipeline Execution Script for LV Dataset.

Loads wp4/trajectories.h5 and wp5/trajectories.h5 directly, performs in-memory
Eulerian grid binning, phase-averaging, turbulent statistics calculation,
attaches CF metadata for ParaView, and exports Zarr, NetCDF, and binary VTK files.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import yaml

# Ensure project src is in Python path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from flowtracks.io import Scene
from flowtracks.eulerian import eulerian_grid, run_post_analysis_ds, save_dataset, export_vtk


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure detailed console and file logger."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("LV_Pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def run_pipeline(lv_dir: Path) -> None:
    log_file = lv_dir / "logs" / "post_processing_lv.log"
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("Starting Full 3D-PTV Post-Processing Pipeline")
    logger.info(f"Target Directory: {lv_dir}")
    logger.info("=" * 70)

    t0_total = time.perf_counter()

    # 1. Load Configurations
    config_file = lv_dir / "config.yaml"
    grid_file = lv_dir / "grid.yaml"
    recipe_file = lv_dir / "post_recipe.yaml"

    logger.info("Reading configuration files...")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    with open(grid_file) as f:
        grid_params = yaml.safe_load(f)
    with open(recipe_file) as f:
        recipe = yaml.safe_load(f)

    logger.debug(f"Config: {config}")
    logger.debug(f"Grid Params: {grid_params}")
    logger.debug(f"Recipe: {recipe}")

    set_names = config["set_names"]
    cycletime = float(np.ceil((60 / config["hb"]) * config["frate"]))
    logger.info(f"Frame Rate: {config['frate']} Hz | Heart Rate: {config['hb']} bpm")
    logger.info(f"Calculated Cycle Time: {cycletime:.1f} frames per cycle")

    # 2. Eulerian Grid Binning per Set (In Memory)
    ds_sets = {}
    for set_name in set_names:
        traj_h5 = lv_dir / set_name / "trajectories.h5"
        if not traj_h5.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_h5}")

        logger.info("-" * 50)
        logger.info(f"Processing Set: [{set_name}] -> {traj_h5.name}")
        file_mb = traj_h5.stat().st_size / (1024 * 1024)
        logger.info(f"File Size: {file_mb:.2f} MB")

        t0_set = time.perf_counter()
        scene = Scene(str(traj_h5))
        first_frame, last_frame = scene.frame_range()
        logger.info(f"Trajectory Frame Range: {first_frame} .. {last_frame}")

        # Limit frame range to config bounds if specified
        first = max(first_frame, config.get("first", first_frame))
        last = min(last_frame, config.get("last", last_frame))
        logger.info(f"Active Binning Window: frame {first} .. {last}")

        logger.info("Performing 4D spatial and phase binning into xarray.Dataset...")
        ds_set = eulerian_grid(
            scene,
            grid_params,
            first=first,
            last=last,
            cycletime=cycletime,
            deltat=config.get("delta_t", 90),
            base_time=100000,
            min_count=config.get("min_count", 5),
        )
        t_set = time.perf_counter() - t0_set

        ds_sets[set_name] = ds_set
        counts = ds_set["par_ave2"].values
        pop_voxels = np.count_nonzero(counts)
        tot_voxels = counts.size
        logger.info(f"[{set_name}] Binned in {t_set:.2f} seconds.")
        logger.info(f"  - Grid Shape (x,y,z,phase): {counts.shape}")
        logger.info(f"  - Populated Voxels: {pop_voxels:,} / {tot_voxels:,} ({pop_voxels/tot_voxels*100:.1f}%)")
        logger.info(f"  - Max Samples per Voxel-Phase: {counts.max():,}")

    # 3. Central Post-Analysis (Phase Averaging, Fluctuations, Turbulent Statistics)
    logger.info("=" * 70)
    logger.info("Running Central Phase-Averaging & Turbulent Statistics...")
    t0_post = time.perf_counter()
    
    out_ds = run_post_analysis_ds(ds_sets, recipe)
    t_post = time.perf_counter() - t0_post
    logger.info(f"Central Post-Analysis Completed in {t_post:.2f} seconds.")
    logger.info(f"Total Variables Generated: {len(out_ds.data_vars)}")
    logger.info(f"Variables: {sorted(list(out_ds.data_vars))}")

    # Log summary statistics of key physical variables
    for var in ["u_ins_mean", "TKE", "MKE", "VEL", "VSS", "H1"]:
        if var in out_ds:
            vals = out_ds[var].values
            non_nan = vals[~np.isnan(vals)]
            if len(non_nan) > 0:
                logger.info(
                    f"  * {var:12s}: min={non_nan.min():.4e}, mean={non_nan.mean():.4e}, max={non_nan.max():.4e}"
                )

    # 4. Exporting Outputs (Zarr, NetCDF, VTK)
    logger.info("=" * 70)
    logger.info("Saving Deliverables...")

    # Zarr Output
    zarr_out = lv_dir / "post_analysis.zarr"
    logger.info(f"Writing Cloud-Native Zarr Store: {zarr_out}")
    t0_zarr = time.perf_counter()
    save_dataset(out_ds, zarr_out)
    t_zarr = time.perf_counter() - t0_zarr
    logger.info(f"Zarr Store Saved in {t_zarr:.2f} seconds.")

    # NetCDF Output
    nc_out = lv_dir / "post_analysis.nc"
    logger.info(f"Writing CF-1.8 NetCDF File: {nc_out}")
    t0_nc = time.perf_counter()
    save_dataset(out_ds, nc_out)
    t_nc = time.perf_counter() - t0_nc
    logger.info(f"NetCDF File Saved in {t_nc:.2f} seconds ({nc_out.stat().st_size / (1024*1024):.2f} MB).")

    # VTK Output
    vtk_dir = lv_dir / recipe.get("vtk", {}).get("dir", "vtk_output")
    prefix = recipe.get("vtk", {}).get("prefix", "phase")
    logger.info(f"Exporting Binary Structured VTK Files to: {vtk_dir}")
    t0_vtk = time.perf_counter()
    vtk_files = export_vtk(out_ds, vtk_dir, prefix=prefix)
    t_vtk = time.perf_counter() - t0_vtk
    logger.info(f"Exported {len(vtk_files)} VTK Files in {t_vtk:.2f} seconds.")

    t_total = time.perf_counter() - t0_total
    logger.info("=" * 70)
    logger.info(f"SUCCESS: Full Pipeline Execution Finished in {t_total:.2f} seconds!")
    logger.info(f"Detailed Execution Log Saved To: {log_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    target_dir = Path(r"C:\Users\alex\Downloads\hidimaging_test\LV")
    run_pipeline(target_dir)
