"""Flowtracks: Complete 3D PTV Lagrangian & Eulerian Post-Processing Toolkit."""

__version__ = '1.4.0'

from flowtracks.eulerian import clean_field, derived_fields, eulerian_grid, eulerian_windowed, export_vtk, finite_difference_velocity, fluid_mask, qc_mask, save_dataset, save_netcdf, save_zarr
from flowtracks.phase_average import fluctuations, phase_average
from flowtracks.smoothing import savitzky_golay
from flowtracks.stitching import stitch_trajectories
from flowtracks.writers import (
    export_run_to_paraview,
    write_eulerian_series,
    write_pvd,
    write_trajectories_vtp,
)

__all__ = [
    "stitch_trajectories",
    "savitzky_golay",
    "eulerian_grid",
    "eulerian_windowed",
    "clean_field",
    "fluid_mask",
    "qc_mask",
    "finite_difference_velocity",
    "phase_average",
    "fluctuations",
    "derived_fields",
    "export_vtk",
    "save_zarr",
    "save_dataset",
    "save_netcdf",
    "write_pvd",
    "write_trajectories_vtp",
    "write_eulerian_series",
    "export_run_to_paraview",
]
