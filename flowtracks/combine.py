# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""flowtracks.combine - CLI entrypoint for postptv-combine

Vectorized 3D Eulerian binning, ensemble phase-averaging across realizations/subfolders,
and Zarr + VTK (ParaView) export.

Usage:
    postptv-combine [DATA_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from flowtracks import eulerian as post_analysis_xr
from flowtracks.io import Scene
from flowtracks.phase_average import fluctuations, phase_average
from flowtracks.writers import write_eulerian_series


def run_combine(data_dir: str | Path = ".", method: str = "per_run",
                grid_params=None, first=1, last=5005, cycletime=5005,
                deltat=50, base_time=1, min_count=2):
    """Combine realizations into Eulerian/phase-averaged outputs.

    method:
      - "per_run" (default, current): bin each realization to a per-run
        phase grid, then count-weighted ensemble/phase averaging across runs.
        Per-run ``min_count`` masking is applied before combining.
      - "pooled": collect all filtered samples from all runs together (keeping
        run index and per-run counts per bin/phase), then bin/average on the
        pooled total. ``min_count`` applies to pooled counts, so a bin sparse
        in one run but sufficient across runs can remain valid.
    """
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

    if grid_params is None:
        grid_params = {
            'min_x': -0.021, 'max_x': 0.070, 'stepx': 45,
            'min_y': -0.071, 'max_y': 0.016, 'stepy': 43,
            'min_z': -0.061, 'max_z': 0.001, 'stepz': 31,
        }

    run_scenes = []
    for traj_file in traj_files:
        set_name = traj_file.parent.name if traj_file.parent.name != "res" else traj_file.parent.parent.name
        if set_name == data_path.name:
            set_name = "run1"
        print(f"[postptv-combine] Loading realization '{set_name}' ({traj_file})...")
        scene = Scene(str(traj_file)) if str(traj_file).endswith(".h5") else str(traj_file)
        run_scenes.append((set_name, scene))

    pooled_result = None
    if method == "pooled":
        print("[postptv-combine] Pooling ALL runs together (keeping run index/counts per bin/phase)...")
        pooled_result = post_analysis_xr.eulerian_grid_pooled(
            run_scenes, grid_params, first=first, last=last,
            cycletime=cycletime, deltat=deltat, base_time=base_time,
            min_count=min_count,
        )
        final_ds = pooled_result.dataset
        # attach per-run counts for bookkeeping/comparison
        final_ds = final_ds.assign(counts_per_run=pooled_result.counts_per_run)

        # pooled turbulent statistics + derived fields (functional parity with
        # per_run path, computed from collected particle samples)
        print("[postptv-combine] Computing pooled turbulent statistics from collected samples...")
        pooled_stats = post_analysis_xr.pooled_turbulent_statistics(pooled_result)
        print("[postptv-combine] Computing derived fields (MKE, TKE, VEL)...")
        derived = post_analysis_xr.derived_fields(
            final_ds, pooled_stats, fields=["MKE", "TKE", "VEL"],
            counts=final_ds[post_analysis_xr.COUNT_VAR],
            fluid_min=2,
        )
        final_ds = xr.merge([final_ds, pooled_stats, derived])
    else:
        ds_sets = {}
        set_names = []
        for set_name, scene in run_scenes:
            print(f"[postptv-combine] Processing realization '{set_name}' (per-run binning)...")
            ds = post_analysis_xr.eulerian_grid(
                scene, grid_params, first=first, last=last,
                cycletime=cycletime, deltat=deltat, base_time=base_time,
                min_count=min_count,
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

            print("[postptv-combine] Computing derived fields (MKE, TKE, VEL)...")
            derived = post_analysis_xr.derived_fields(avg, stats, fields=['MKE', 'TKE', 'VEL'])
            final_ds = xr.merge([avg, stats, derived])

    out_dir = res_dir if res_dir.exists() else data_path
    zarr_out = out_dir / ("post_analysis_pooled.zarr" if method == "pooled" else "post_analysis.zarr")
    print(f"[postptv-combine] Saving Zarr output -> {zarr_out}")
    post_analysis_xr.save_zarr(final_ds, zarr_out)

    if pooled_result is not None:
        samples_npz = out_dir / "post_analysis_pooled_samples.npz"
        print(f"[postptv-combine] Saving pooled sample table -> {samples_npz}")
        np.savez_compressed(
            samples_npz,
            run_names=np.array(pooled_result.run_names, dtype=object),
            **pooled_result.samples,
        )

    print(f"[postptv-combine] Exporting ParaView series -> {vtk_dir}")
    write_eulerian_series(final_ds, vtk_dir)
    print(f"[postptv-combine] Complete! Saved {zarr_out.name}")
    return {"dataset": final_ds, "pooled_result": pooled_result, "output": zarr_out}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combine 3D-PTV realizations (postptv-combine)")
    parser.add_argument("data_dir", nargs="?", default=".", help="Data directory to search")
    parser.add_argument("--method", choices=["per_run", "pooled"], default="per_run",
                        help="Averaging method: per_run (default, current: bin per run then "
                             "combine) or pooled (collect all runs together, pooled min_count)")
    args = parser.parse_args()
    run_combine(args.data_dir, method=args.method)


if __name__ == "__main__":
    main()
