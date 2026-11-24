#!/usr/bin/env python
"""Build a mean-speed volume from the demo Zarr for the browser viewer demo.

A small synthetic Eulerian-style overlay: the mean per-point speed
(|vel|) averaged by voxel over the full time window, written as a .vti
that the browser viewer can overlay on trajectories.

Not the repo's phase-averaged Eulerian pipeline (that is flowtracks/eulerian.py
and lives on phase-resolved grids); this is a quick, honest demo overlay.
Empty voxels are left NaN.

Usage:
    uv run python scripts/make_demo_volume.py
"""
from pathlib import Path

import numpy as np
import pyvista as pv

from flowtracks.visualization.scene import load_scene

OUT = Path("data/tracers_mean_speed.vti")
GRID = 24  # points per axis (cells = 23 per axis)


def main():
    data = load_scene("data/tracers.zarr", max_trajectories=0)
    mins, maxs = data.pos.min(axis=0), data.pos.max(axis=0)
    span = maxs - mins
    spacing = (span[0] / (GRID - 1), span[1] / (GRID - 1), span[2] / (GRID - 1))
    points_per_axis = [np.linspace(mins[i], maxs[i], GRID) for i in range(3)]
    shape = (GRID - 1, GRID - 1, GRID - 1)

    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    for x, y, z, speed in zip(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2], data.speed):
        ix = np.clip(np.searchsorted(points_per_axis[0], x, side="right") - 1, 0, shape[0] - 1)
        iy = np.clip(np.searchsorted(points_per_axis[1], y, side="right") - 1, 0, shape[1] - 1)
        iz = np.clip(np.searchsorted(points_per_axis[2], z, side="right") - 1, 0, shape[2] - 1)
        sums[ix, iy, iz] += speed
        counts[ix, iy, iz] += 1

    speed_grid = np.full(shape, np.nan, dtype=np.float64)
    nonempty = counts > 0
    speed_grid[nonempty] = sums[nonempty] / counts[nonempty]

    volume = pv.ImageData(dimensions=(GRID, GRID, GRID), origin=mins, spacing=spacing)
    volume.cell_data["speed"] = speed_grid.ravel(order="C")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    volume.save(OUT)
    n_filled = int(nonempty.sum())
    print(f"wrote {OUT}: {n_filled}/{nonempty.size} voxels, mean speed over filled voxels = {np.nanmean(speed_grid):.6f} m/s")


if __name__ == "__main__":
    main()
