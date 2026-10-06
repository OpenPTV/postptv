# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Yosef Meller, Alex Liberzon
"""Flowtracks: Complete 3D PTV Lagrangian & Eulerian Post-Processing Toolkit."""

# flowtracks/_version.py is written by setuptools-scm at build/install time
try:
    from flowtracks._version import __version__
except ImportError:  # a bare source tree, never installed
    __version__ = "0+unknown"

from flowtracks.eulerian import (
    clean_field,
    derived_fields,
    eulerian_grid,
    eulerian_windowed,
    finite_difference_velocity,
    fluid_mask,
    qc_mask,
    save_dataset,
    save_netcdf,
    save_zarr,
)
from flowtracks.phase_average import fluctuations, phase_average
from flowtracks.phasing import (
    assign_bins,
    bin_of_frame,
    phase_of_frame,
    validate_bins,
)
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
    "validate_bins",
    "assign_bins",
    "phase_of_frame",
    "bin_of_frame",
    "derived_fields",
    "save_zarr",
    "save_dataset",
    "save_netcdf",
    "write_pvd",
    "write_trajectories_vtp",
    "write_eulerian_series",
    "export_run_to_paraview",
]
