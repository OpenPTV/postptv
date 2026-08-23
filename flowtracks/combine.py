"""flowtracks.combine - CLI entrypoint for postptv-combine

Vectorized 3D Eulerian binning, ensemble phase-averaging across realizations/subfolders,
and NetCDF/VTK ParaView export.

Usage:
    postptv-combine [DATA_DIR]
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr

from flowtracks import eulerian as post_analysis_xr
from flowtracks.phase_average import phase_average, fluctuations
from flowtracks.io import Scene


def run_combine(data_dir: str | Path = "."):
    data_path = Path(data_dir)
    res_dir = data_path / "res"
    vtk_dir = data_path / "vtk_output"
    
    # Locate all run.zarr directories or trajectories.h5 files
    all_zarr = list(data_path.rglob("run.zarr")) + list(data_path.rglob("*.zarr"))
    all_h5 = list(data_path.rglob("trajectories.h5"))
    
    all_datasets = sorted(all_zarr + all_h5, key=lambda x: len(x.parts))
    if not all_datasets:
        print(f"[postptv-combine] No Zarr or trajectories.h5 datasets found in {data_path}, skipping stage.")
        return

    seen_parents = set()
    traj_files = []
    for h in all_datasets:
        parent = h.parent if h.parent.name != "res" else h.parent.parent
        if parent not in seen_parents:
            seen_parents.add(parent)
            traj_files.append(h)

    print(f"[postptv-combine] Found {len(traj_files)} realization dataset(s): {[f.relative_to(data_path) for f in traj_files]}")

    grid_params = {
        'min_x': -0.021, 'max_x': 0.070, 'stepx': 45,
        'min_y': -0.071, 'max_y': 0.016, 'stepy': 43,
        'min_z': -0.061, 'max_z': 0.001, 'stepz': 31,
    }

    ds_sets = {}
    set_names = []
    for traj_file in traj_files:
        set_name = traj_file.parent.name if traj_file.parent.name != "res" else traj_file.parent.parent.name
        if set_name == data_path.name:
            set_name = "run1"
        print(f"[postptv-combine] Processing realization '{set_name}' ({traj_file})...")
        if str(traj_file).endswith(".h5"):
            scene = Scene(str(traj_file))
        else:
            scene = str(traj_file)
        ds = post_analysis_xr.eulerian_grid(
            scene, grid_params, first=1, last=5005, cycletime=5005, deltat=50, base_time=1, min_count=2
        )
        ds_sets[set_name] = ds
        set_names.append(set_name)

    if len(ds_sets) == 1:
        final_ds = list(ds_sets.values())[0]
    else:
        print(f"[postptv-combine] Stacking realizations along 'set' dimension: {set_names}")
        combined = xr.concat(list(ds_sets.values()), dim=xr.DataArray(set_names, dims="set", name="set"))
        
        print("[postptv-combine] Computing ensemble phase-averaging across realizations...")
        avg = phase_average(combined, weights=combined["par_ave2"])
        
        print("[postptv-combine] Computing turbulent fluctuations and statistics...")
        fl = fluctuations(combined, avg)
        
        u_rms = np.sqrt((fl["u_ins_mean"]**2).mean("set"))
        v_rms = np.sqrt((fl["v_ins_mean"]**2).mean("set"))
        w_rms = np.sqrt((fl["w_ins_mean"]**2).mean("set"))
        
        stats = xr.Dataset({
            'u_ins_u_ins': (fl['u_ins_mean']**2).mean('set'),
            'v_ins_v_ins': (fl['v_ins_mean']**2).mean('set'),
            'w_ins_w_ins': (fl['w_ins_mean']**2).mean('set'),
            'u_ins_v_ins': (fl['u_ins_mean']*fl['v_ins_mean']).mean('set'),
            'u_ins_w_ins': (fl['u_ins_mean']*fl['w_ins_mean']).mean('set'),
            'v_ins_w_ins': (fl['v_ins_mean']*fl['w_ins_mean']).mean('set'),
            'u_rms': u_rms,
            'v_rms': v_rms,
            'w_rms': w_rms,
        })
        
        print("[postptv-combine] Computing hemodynamic derived fields (MKE, TKE, VEL)...")
        derived = post_analysis_xr.derived_fields(avg, stats, fields=['MKE', 'TKE', 'VEL'])
        final_ds = xr.merge([avg, stats, derived])

    nc_out = res_dir / "post_analysis.nc" if res_dir.exists() else data_path / "post_analysis.nc"
    print(f"[postptv-combine] Saving NetCDF output -> {nc_out}")
    post_analysis_xr.save_netcdf(final_ds, nc_out)
    
    print(f"[postptv-combine] Exporting VTK phase snapshots -> {vtk_dir}")
    post_analysis_xr.export_vtk(final_ds, vtk_dir)
    print(f"[postptv-combine] Complete! Saved {nc_out.name}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    run_combine(target)


if __name__ == "__main__":
    main()
