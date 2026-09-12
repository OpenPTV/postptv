"""Flowtracks: Complete 3D PTV Lagrangian & Eulerian Post-Processing Toolkit."""

__version__ = "1.2.1"

from flowtracks.eulerian import derived_fields, eulerian_grid, export_vtk, save_netcdf
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
    "phase_average",
    "fluctuations",
    "derived_fields",
    "export_vtk",
    "save_netcdf",
    "write_pvd",
    "write_trajectories_vtp",
    "write_eulerian_series",
    "export_run_to_paraview",
]
