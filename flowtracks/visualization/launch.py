# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""Spawn the browser viewer as a detached subprocess (notebook-friendly)."""
from __future__ import annotations

import subprocess
import sys


def launch(
    zarr_path,
    *,
    mesh=None,
    volume=None,
    time_range=None,
    max_trajectories=2000,
    color_by="speed",
    port=0,
):
    """Start the browser viewer in a background subprocess.

    Returns the ``subprocess.Popen`` handle so callers can keep a reference
    and terminate the viewer with ``proc.terminate()``. The viewer opens a
    browser tab automatically (see ``python -m flowtracks.visualization
    --help`` for the full interface).
    """

    cmd = [
        sys.executable,
        "-m",
        "flowtracks.visualization",
        str(zarr_path),
        "--port",
        str(port),
        "--max-trajectories",
        str(max_trajectories),
        "--color-by",
        str(color_by),
    ]
    if mesh is not None:
        cmd += ["--mesh", str(mesh)]
    if volume is not None:
        cmd += ["--volume", str(volume)]
    if time_range is not None:
        t0, t1 = time_range
        spec = f"{t0 if t0 is not None else ''},{t1 if t1 is not None else ''}"
        cmd += ["--time-range", spec]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc
