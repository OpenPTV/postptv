#!/usr/bin/env python3
"""
PostPTV Post-Processing & Post-Analysis Pipeline for TT13_aorta
----------------------------------------------------------------
1. Converts openptv2 raw PTV output (wp1/res/ptv_is.* and wp2/res/ptv_is.*) to HDF5 trajectory databases (trajectories.h5).
2. Computes 3D Eulerian gridded velocity fields using flowtracks.
3. Computes ensemble phase-averaging across realizations (wp1, wp2).
4. Computes turbulent velocity fluctuations, Reynolds stress tensor components, TKE, and MKE.
5. Exports NetCDF dataset (post_analysis.nc) and ParaView 3D VTK visualization files (.vtu, .pvd).
"""

import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr

import flowtracks
from flowtracks.io import trajectories_ptvis, save_particles_table, Scene
from flowtracks import eulerian as post_analysis_xr
from flowtracks.phase_average import phase_average, fluctuations

def process_folder_trajectories(folder_path: Path, first_frame: int = 1, last_frame: int = 847, frate: float = 1.0):
    res_dir = folder_path / "res"
    if not res_dir.exists():
        print(f"[postptv] Skipping {folder_path.name}: res/ directory not found.")
        return None
        
    ptvis_pattern = str(res_dir / "ptv_is.%d")
    out_h5 = res_dir / "trajectories.h5"
    
    print(f"[postptv] Converting raw ptv_is files in {folder_path.name} -> {out_h5.name}...")
    trajects = trajectories_ptvis(ptvis_pattern, first_frame, last_frame, frate=frate)
    if not trajects:
        print(f"[postptv] WARNING: No trajectories found in {folder_path.name} for frames {first_frame}..{last_frame}")
        return None
        
    save_particles_table(str(out_h5), trajects)
    print(f"[postptv] Saved {len(trajects)} Lagrangian trajectories to {out_h5}")
    return out_h5

def run_postptv_analysis(exp_root: Path):
    print("============================================================")
    print("  PostPTV Stage 4 Post-Processing & Turbulence Analysis")
    print("============================================================")
    
    wp1_dir = exp_root / "wp1"
    wp2_dir = exp_root / "wp2"
    
    h5_files = []
    for wp in [wp1_dir, wp2_dir]:
        if wp.exists():
            h5_p = process_folder_trajectories(wp, first_frame=1, last_frame=847)
            if h5_p and h5_p.exists():
                h5_files.append((wp.name, h5_p))
                
    if not h5_files:
        print("[postptv] ERROR: No valid trajectory HDF5 files generated. Cannot proceed with Eulerian & Turbulence analysis.")
        sys.exit(1)
        
    grid_params = {
        'min_x': -0.021, 'max_x': 0.070, 'stepx': 45,
        'min_y': -0.071, 'max_y': 0.016, 'stepy': 43,
        'min_z': -0.061, 'max_z': 0.001, 'stepz': 31,
    }
    
    ds_sets = {}
    set_names = []
    
    for set_name, h5_path in h5_files:
        print(f"\n[postptv] Computing 3D Eulerian velocity grid for realization '{set_name}'...")
        scene = Scene(str(h5_path))
        ds = post_analysis_xr.eulerian_grid(
            scene, grid_params, first=1, last=847, cycletime=847, deltat=10, base_time=1, min_count=1
        )
        ds_sets[set_name] = ds
        set_names.append(set_name)
        
    if len(ds_sets) == 1:
        final_ds = list(ds_sets.values())[0]
    else:
        print(f"\n[postptv] Stacking realizations along 'set' dimension: {set_names}")
        combined = xr.concat(list(ds_sets.values()), dim=xr.DataArray(set_names, dims="set", name="set"))
        
        print("[postptv] Computing ensemble phase-averaging across realizations...")
        avg = phase_average(combined, weights=combined["par_ave2"])
        
        print("[postptv] Computing turbulent velocity fluctuations and Reynolds stress tensor...")
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
        
        print("[postptv] Computing hemodynamic derived fields (MKE, TKE, VEL)...")
        derived = post_analysis_xr.derived_fields(avg, stats, fields=['MKE', 'TKE', 'VEL'])
        final_ds = xr.merge([avg, stats, derived])
        
    output_res = exp_root / "res"
    output_res.mkdir(parents=True, exist_ok=True)
    
    nc_out = output_res / "TT13_aorta_post_analysis.nc"
    print(f"\n[postptv] Saving NetCDF output -> {nc_out}")
    post_analysis_xr.save_netcdf(final_ds, nc_out)
    
    vtk_dir = output_res / "paraview_vtk"
    print(f"[postptv] Exporting ParaView 3D VTK snapshots -> {vtk_dir}")
    post_analysis_xr.export_vtk(final_ds, vtk_dir)
    
    print("\n============================================================")
    print(f"  PostPTV Analysis Complete! Results written to {output_res}")
    print("============================================================")

if __name__ == "__main__":
    exp_dir = Path(__file__).resolve().parent
    run_postptv_analysis(exp_dir)
