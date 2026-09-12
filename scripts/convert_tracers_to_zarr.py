#!/usr/bin/env python
"""One-shot conversion of the demo dataset from HDF5 to Zarr.

data/tracers.h5 (PyTables, the original two-phase-experiment format) ->
data/tracers.zarr (a ZarrScene-shaped `trajectories/` group: pos/vel/time/
trajid), via the same flowtracks.io.save_zarr_trajectories() any Scene ->
Zarr conversion uses. Run once; data/tracers.zarr is then a first-class demo
dataset alongside data/tracers.h5, readable by ZarrScene or by
flowtracks.writers directly (no Scene/pytables needed downstream).

Usage: uv run --extra legacy-io python scripts/convert_tracers_to_zarr.py
"""
from pathlib import Path

from flowtracks.io import save_zarr_trajectories
from flowtracks.scene import Scene

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    scene = Scene(str(DATA_DIR / "tracers.h5"))
    trajs = list(scene.iter_trajectories())
    zarr_path = DATA_DIR / "tracers.zarr"
    save_zarr_trajectories(trajs, zarr_path)
    print(f"Converted {len(trajs)} trajectories: {DATA_DIR / 'tracers.h5'} -> {zarr_path}")


if __name__ == "__main__":
    main()
