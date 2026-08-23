#!/usr/bin/env python
"""Inspect Flowtracks Zarr trajectory stores."""

import sys
from pathlib import Path
import numpy as np
import zarr


def peek_zarr(zarr_path: str):
    """Print a human-readable summary of a Zarr trajectory dataset."""
    path = Path(zarr_path)
    if not path.exists():
        print(f"Error: Path '{zarr_path}' does not exist.")
        return

    root = zarr.open_group(str(path), mode="r")
    print("=" * 65)
    print(f"[PEEK] Flowtracks Zarr Dataset: {path.resolve()}")
    print("=" * 65)

    if "correspondences" in root:
        corr_grp = root["correspondences"]
        f_keys = sorted([k for k in corr_grp.keys() if k.startswith("frame_")])
        if f_keys:
            f_min = f_keys[0].split("_")[1]
            f_max = f_keys[-1].split("_")[1]
            sample_frame = np.asarray(corr_grp[f_keys[0]])
            print(f"[Correspondences]: {len(f_keys)} frames ({f_min} .. {f_max}), ~{len(sample_frame)} points/frame")

    if "trajectories" in root:
        traj_grp = root["trajectories"]
        if "trajid" in traj_grp:
            trids = np.asarray(traj_grp["trajid"])
            times = np.asarray(traj_grp["time"])
            pos = np.asarray(traj_grp["pos"])
            u_ids = len(np.unique(trids))
            print(f"[Trajectories]: {u_ids} unique trajectories, {len(pos)} total points (Time: {times.min()} .. {times.max()})")
            if "vel" in traj_grp:
                vel = np.asarray(traj_grp["vel"])
                v_mag = np.linalg.norm(vel, axis=1)
                print(f"   - Velocity stats: min={v_mag.min():.4f}, max={v_mag.max():.4f}, mean={v_mag.mean():.4f} m/s")
        else:
            print(f"[Trajectories]: Subgroup present ({list(traj_grp.keys())})")

    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python peek_zarr.py <path_to_run.zarr>")
        sys.exit(1)
    peek_zarr(sys.argv[1])
