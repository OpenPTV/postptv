# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""Browser-based 3D viewer for flowtracks Zarr stores.

Native formats in, interactive browser view out: pyvista scene + trame
WebSocket view. ``flowtracks-viz`` / ``python -m flowtracks.visualization``
runs the full app; ``from flowtracks.visualization import launch`` starts it
from a marimo notebook or script without blocking.
"""
from .launch import launch
from .scene import (
    TrajectoryScene,
    build_polydata,
    filter_through_frame,
    load_mesh,
    load_scene,
    load_volume,
)

__all__ = [
    "TrajectoryScene",
    "build_polydata",
    "filter_through_frame",
    "launch",
    "load_mesh",
    "load_scene",
    "load_volume",
]
