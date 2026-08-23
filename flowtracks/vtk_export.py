#!/usr/bin/env python3
"""
Python translation of Sample_vtkcode.m
- Loads masked_fluct__exp6.mat and phaseaver_exp6.mat
- Performs time shifting, calculates MKE, TKE, VSS, RSS, Mean Loss, Scalar Shear, TurbLoss, etc.
- Handles NaNs as in Matlab
- Writes VTK files using vtk package
"""
import numpy as np
import h5py
import yaml
from pathlib import Path
try:
    import vtk
except ImportError:
    vtk = None

from flowtracks.eulerian import shift_fields


def write_structured_grid_vtk(filename, x, y, z, fields):
    """
    Write a VTK structured grid file with given fields.
    fields: dict of {name: (array, type)}, type is 'scalar' or 'vector'
    """
    nx, ny, nz = x.shape
    points = vtk.vtkPoints()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                points.InsertNextPoint(float(x[i, j, k]), float(y[i, j, k]), float(z[i, j, k]))
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, nz)
    grid.SetPoints(points)
    for name, (arr, kind) in fields.items():
        arr = np.ascontiguousarray(arr)
        if kind == 'vector':
            arr_vtk = vtk.vtkFloatArray()
            arr_vtk.SetNumberOfComponents(3)
            arr_vtk.SetName(name)
            arr_vtk.SetNumberOfTuples(arr.size // 3)
            arr_vtk.SetVoidArray(arr.astype(np.float32).reshape(-1, 3), arr.size, 1)
            grid.GetPointData().AddArray(arr_vtk)
        elif kind == 'scalar':
            arr_vtk = vtk.vtkFloatArray()
            arr_vtk.SetName(name)
            arr_vtk.SetNumberOfComponents(1)
            arr_vtk.SetNumberOfTuples(arr.size)
            arr_vtk.SetVoidArray(arr.astype(np.float32).reshape(-1), arr.size, 1)
            grid.GetPointData().AddArray(arr_vtk)
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()


def max_eigenvalue_spread(p1, p2, p3, p4, p5, p6, p7, p8, p9):
    """Maximum minus minimum eigenvalue (0.5 * spread) of the symmetric
    3x3 tensor [[p1, p2, p3], [p4, p5, p6], [p7, p8, p9]] for every cell.

    Computed in a single vectorised ``np.linalg.eigvalsh`` call over the
    whole (batched) array at once, instead of a Python per-voxel loop.
    Inputs are broadcastable arrays of the same shape (e.g. a single
    time slice, shape ``(nx, ny, nz)``).
    """
    comps = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
    for arr in comps:
        np.nan_to_num(arr, copy=False)
    shape = comps[0].shape
    # Build the (..., 3, 3) tensor: row-major ordering of the 9 components.
    tensor = np.stack(comps, axis=-1).reshape(*shape, 3, 3)
    evals = np.linalg.eigvalsh(tensor)
    return 0.5 * (evals.max(axis=-1) - evals.min(axis=-1))


def main():
    # --- Load config and grid ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('grid.yaml', 'r') as f:
        grid = yaml.safe_load(f)

    full_h5_path = Path(config['data_path']) / Path(config['h5_path'])
    traj_min_length = config['traj_min_length']
    first = config['first']
    last = config['last']
    frate = config['frate']
    rho = config['rho']
    mu = config['mu']

    h5files = [full_h5_path / f"{set_name}_traj{traj_min_length}.h5" for set_name in config['set_names']]
    grid_files = [full_h5_path / f"{set_name}_grid.h5" for set_name in config['set_names']]
    phase_averaged_file = full_h5_path / "phase_averaged.h5"
    turb_stats_file = full_h5_path / 'turbulent_statistics_phase_averaged.h5'

    shift = config.get('shift', 20)

    # --- Time shift phase-averaged means and grid files (idempotent) ---
    shift_fields(phase_averaged_file,
                 ['u_phase_averaged', 'v_phase_averaged', 'w_phase_averaged'],
                 shift)
    for set_h5_file in grid_files:
        shift_fields(set_h5_file,
                     ['u_ins_mean', 'v_ins_mean', 'w_ins_mean',
                      'u_fluct', 'v_fluct', 'w_fluct'],
                     shift)

    # --- Read data, compute derived fields via post_analysis_xr, export VTK ---
    import xarray as xr
    from flowtracks.eulerian import ALL_DERIVED, derived_fields

    DIMS = ('x', 'y', 'z', 'phase')
    with h5py.File(grid_files[0], 'r') as f:
        coords = {'x': f['x_vals'][()], 'y': f['y_vals'][()],
                  'z': f['z_vals'][()]}
    with h5py.File(phase_averaged_file, 'r') as f:
        avg = xr.Dataset(
            {f'{a}_ins_mean': (DIMS, f[f'{a}_phase_averaged'][()])
             for a in 'uvw'})
    with h5py.File(turb_stats_file, 'r') as f:
        stats = xr.Dataset(
            {name: (DIMS, f[name][()])
             for name in ('u_ins_u_ins', 'v_ins_v_ins', 'w_ins_w_ins',
                          'u_ins_v_ins', 'u_ins_w_ins', 'v_ins_w_ins')})
    coords['phase'] = np.arange(avg['u_ins_mean'].shape[3])
    avg = avg.assign_coords(coords)
    stats = stats.assign_coords(coords)

    derived = derived_fields(avg, stats, rho=rho, mu=mu, fields=ALL_DERIVED)

    x3, y3, z3 = np.meshgrid(coords['x'], coords['y'], coords['z'],
                             indexing='ij')
    full_vtk_path = Path(config['data_path']) / Path('vtk_output')
    full_vtk_path.mkdir(parents=True, exist_ok=True)

    legacy_names = {'uu': 'u_ins_u_ins', 'vv': 'v_ins_v_ins',
                    'ww': 'w_ins_w_ins', 'uv': 'u_ins_v_ins',
                    'uw': 'u_ins_w_ins', 'vw': 'v_ins_w_ins'}
    for time_id in range(avg['u_ins_mean'].shape[3]):
        print(f'time_id={time_id+1}')
        a = avg.isel(phase=time_id)
        s = stats.isel(phase=time_id)
        d = derived.isel(phase=time_id)
        vel = np.stack([np.nan_to_num(a[f'{c}_ins_mean'].values)
                        for c in 'uvw'], axis=-1)
        fields = {'velocity': (vel.reshape(-1, 3), 'vector')}
        for short, name in legacy_names.items():
            fields[short] = (np.nan_to_num(s[name].values).flatten(), 'scalar')
        for short, name in [('ScalarShear', 'ScalarShear'), ('H1', 'H1'),
                            ('H2', 'H2'), ('H3', 'H3'), ('H4', 'H4'),
                            ('mke', 'MKE'), ('tke', 'TKE'), ('VSS', 'VSS'),
                            ('RSS', 'RSS'), ('ML', 'ML'), ('TL', 'TL')]:
            fields[short] = (np.nan_to_num(d[name].values).flatten(), 'scalar')

        name_root = f'Exp6_hemodynamics_t{time_id+1}.vtk'
        write_structured_grid_vtk(full_vtk_path / name_root, x3, y3, z3, fields)


if __name__ == "__main__":
    main()
