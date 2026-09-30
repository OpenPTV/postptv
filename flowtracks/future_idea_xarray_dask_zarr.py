# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
#!/usr/bin/env python3
"""
Boilerplate for 3D-PTV trajectory post-processing
using xarray + dask + zarr

Features:
- Ragged array encoding for trajectories of different lengths
- Vector-style storage (obs, component)
- Spline smoothing and derivative calculation (velocity, acceleration)
- Resampling onto a uniform time base
- Streaming mode: append processed trajectories to a Zarr store
"""

import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline

# ---------------------------------------------------------------------
# 1. Example: Ragged array encoding for variable-length trajectories
# ---------------------------------------------------------------------

def build_ragged_example():
    traj1_t = np.array([0, 1, 2])
    traj1_x = np.array([0, 1, 2])
    traj1_y = np.array([0, 1, 4])
    traj1_z = np.array([0, 0, 0])

    traj2_t = np.array([0, 2, 4, 6, 8])
    traj2_x = np.array([0, 2, 4, 6, 8])
    traj2_y = np.array([0, -1, -2, -3, -4])
    traj2_z = np.array([0, 1, 0, -1, 0])

    # Concatenate into ragged structure
    times = np.concatenate([traj1_t, traj2_t])
    positions = np.vstack([
        np.stack([traj1_x, traj1_y, traj1_z], axis=-1),
        np.stack([traj2_x, traj2_y, traj2_z], axis=-1)
    ])
    trajectory_id = np.concatenate([
        np.full(traj1_t.shape, 0),
        np.full(traj2_t.shape, 1)
    ])

    ds = xr.Dataset(
        {
            "t": ("obs", times),
            "pos": (("obs", "component"), positions),
            "trajectory": ("obs", trajectory_id),
        },
        coords={"component": ["x", "y", "z"], "obs": np.arange(len(times))}
    )

    return ds


# ---------------------------------------------------------------------
# 2. Compute derivatives (velocity, acceleration) in ragged array
# ---------------------------------------------------------------------

def compute_derivatives_ragged(ds):
    def _derivs(sub):
        dt = np.gradient(sub.t.values)
        dpos = np.gradient(sub.pos.values, axis=0)
        vel = dpos / dt[:, None]
        acc = np.gradient(vel, axis=0) / dt[:, None]
        return xr.Dataset({
            "vel": (("obs", "component"), vel),
            "acc": (("obs", "component"), acc)
        })

    derivs = ds.groupby("trajectory").map(_derivs)
    return xr.merge([ds, derivs])


# ---------------------------------------------------------------------
# 3. Spline smoothing + resampling on uniform time base
# ---------------------------------------------------------------------

def smooth_and_resample(t, pos, t_uniform, s=0.0):
    """Smooth trajectory with spline, resample to uniform time base.
    Returns position, velocity, acceleration arrays of shape (len(t_uniform), 3).
    """
    comps = []
    vels = []
    accs = []
    for d in range(pos.shape[1]):  # loop over x,y,z
        spline = UnivariateSpline(t, pos[:, d], s=s)
        p = spline(t_uniform)
        v = spline.derivative(1)(t_uniform)
        a = spline.derivative(2)(t_uniform)
        comps.append(p)
        vels.append(v)
        accs.append(a)

    pos_u = np.stack(comps, axis=-1)
    vel_u = np.stack(vels, axis=-1)
    acc_u = np.stack(accs, axis=-1)
    return pos_u, vel_u, acc_u


# ---------------------------------------------------------------------
# 4. Streaming mode: append processed trajectories to Zarr
# ---------------------------------------------------------------------

def init_zarr_store(store_path, t_uniform):
    components = ["x", "y", "z"]

    ds = xr.Dataset(
        data_vars={
            "position": (("trajectory", "time", "component"),
                         np.empty((0, len(t_uniform), len(components)))),
            "velocity": (("trajectory", "time", "component"),
                         np.empty((0, len(t_uniform), len(components)))),
            "acceleration": (("trajectory", "time", "component"),
                             np.empty((0, len(t_uniform), len(components))))
        },
        coords={
            "time": t_uniform,
            "component": components,
            "trajectory": []
        }
    )
    ds.to_zarr(store_path, mode="w")


def append_to_zarr(store_path, traj_id, t, pos, t_uniform, s=0.0):
    pos_u, vel_u, acc_u = smooth_and_resample(t, pos, t_uniform, s=s)

    new = xr.Dataset(
        {
            "position": (("trajectory", "time", "component"), pos_u[np.newaxis, ...]),
            "velocity": (("trajectory", "time", "component"), vel_u[np.newaxis, ...]),
            "acceleration": (("trajectory", "time", "component"), acc_u[np.newaxis, ...]),
        },
        coords={
            "trajectory": [traj_id],
            "time": t_uniform,
            "component": ["x", "y", "z"]
        }
    )
    new.to_zarr(store_path, mode="a", append_dim="trajectory")


# ---------------------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: Build ragged example
    ds_ragged = build_ragged_example()
    print("Ragged dataset:")
    print(ds_ragged)

    # Step 2: Compute derivatives in ragged array
    ds_with_derivs = compute_derivatives_ragged(ds_ragged)
    print("\nRagged dataset with velocity and acceleration:")
    print(ds_with_derivs)

    # Step 3+4: Streaming to Zarr
    t_uniform = np.linspace(0, 8, 81)  # uniform time base (0.1s step)
    store = "trajectories.zarr"
    init_zarr_store(store, t_uniform)

    # Add first trajectory
    obs0 = ds_ragged.where(ds_ragged.trajectory == 0, drop=True)
    append_to_zarr(store, traj_id=0, t=obs0.t.values, pos=obs0.pos.values, t_uniform=t_uniform, s=0.1)

    # Add second trajectory
    obs1 = ds_ragged.where(ds_ragged.trajectory == 1, drop=True)
    append_to_zarr(store, traj_id=1, t=obs1.t.values, pos=obs1.pos.values, t_uniform=t_uniform, s=0.1)

    # Open the final store lazily with xarray+dask
    ds_zarr = xr.open_zarr(store)
    print("\nZarr dataset (streamed trajectories):")
    print(ds_zarr)
