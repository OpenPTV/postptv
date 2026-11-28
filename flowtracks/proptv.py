# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""proPTV track import: read proPTV tracks as flowtracks Trajectories.

proPTV (https://github.com/RobinBarta/proPTV) saves every pipeline stage as
a float64 HDF5 file at ``data/{case}/output/tracks/tracks*.hdf5`` (for example
``tracks.hdf5``, ``tracks_initial.hdf5`` or ``tracks_broken{t}.hdf5``). Each
track is a dataset named ``track_{t}_{key}`` — ``t`` is the track's last
frame number and ``key`` is the track index assigned by proPTV — with columns::

    [frame, x, y, z, vx, vy, vz, ax, ay, az, <cx, cy> * n_cameras]

(10 + 2*n_cameras columns). Positions appear in millimetres (track volumes in
proPTV's configs, e.g. ``[-2500, 2500]`` per axis, confirm the convention;
the source file records no units of its own). Frame numbers are integers
stored as floats. The per-camera 2D projection columns are ignored — only
the 3D Lagrangian data enters the Zarr store.

``read_proptv_hdf5`` converts the file into flowtracks Trajectory objects.
``convert_proptv_to_zarr`` additionally writes a ``trajectories`` store
(the same layout ``openptv2``'s RunStore and ``save_zarr_trajectories``
write) so the results can be consumed by flowtracks directly — the
interop bridge between the proPTV and flowtracks (openptv/openptv-cloud)
pipelines.
"""

from pathlib import Path

import numpy as np

from flowtracks.io import save_zarr_trajectories
from flowtracks.trajectory import Trajectory


def _track_id(key, fallback):
    """Trajid for a track dataset: the numeric suffix in ``track_{t}_{key}``."""
    parts = str(key).split("_")
    if parts and parts[0] == "track" and len(parts) >= 3 and parts[-1].isdigit():
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return int(fallback)


def read_proptv_hdf5(path, sort=True):
    """
    Read a proPTV ``tracks*.hdf5`` file into a list of Trajectories.

    Arguments:
    path - path to the tracks HDF5 file.
    sort - if True (default), return trajectories sorted by (trajid, frame).

    Returns:
    trajects - a list of :class:`~flowtracks.trajectory.Trajectory` objects,
        each with ``pos``, ``velocity``, ``accel`` and ``time``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"proPTV tracks file not found: {path}")

    import h5py

    trajects = []
    with h5py.File(str(path), "r") as f:
        keys = sorted(f.keys(), key=lambda k: (int(k.split("_")[1]) if k.split("_")[-2].isdigit() else 0, k))
        for i, key in enumerate(keys):
            arr = np.asarray(f[key])
            if arr.ndim != 2 or arr.shape[1] < 10:
                # Not a track dataset (or not a proPTV tracks file)
                continue
            trajid = _track_id(key, i)
            time = np.round(arr[:, 0]).astype(np.int64)
            pos = np.ascontiguousarray(arr[:, 1:4], dtype=np.float64)
            vel = np.ascontiguousarray(arr[:, 4:7], dtype=np.float64)
            accel = np.ascontiguousarray(arr[:, 7:10], dtype=np.float64)
            trajects.append(Trajectory(pos, vel, time, trajid, accel=accel))

    if sort:
        trajects.sort(key=lambda t: (t.trajid(), t.time()[0] if len(t.time()) else 0))

    return trajects


def convert_proptv_to_zarr(path, zarr_path, group="trajectories", overwrite=True,
                           units=None, sort=True):
    """
    Convert a proPTV ``tracks*.hdf5`` file into a Zarr store.

    The store holds a ``trajectories`` group with ``pos``, ``vel``, ``time``,
    ``trajid`` and ``accel`` arrays — the layout :class:`flowtracks.zarr_scene.ZarrScene`
    reads natively.

    Arguments:
    path - path to the proPTV tracks HDF5 file.
    zarr_path - target ``.zarr`` directory.
    group - sub-group name inside the store (default ``"trajectories"``).
    overwrite - if True, overwrite existing arrays in the group.
    units - units dict recorded on the group. Defaults to ``{"pos": "mm",
        "vel": "mm frame-1", "time": "frame", "accel": "mm frame-2"}`` —
        per-frame units with a conventional millimetre positional unit,
        NOT recorded by proPTV itself (mm is the conventional scale of
        proPTV's volume/camera configs; override with ``units=`` to state
        the source's actual units).
    sort - if True (default), trajectories are sorted by (trajid, frame).

    Returns:
    trajects - the list of :class:`~flowtracks.trajectory.Trajectory` objects
        that were written.
    """
    path = Path(path)
    trajects = read_proptv_hdf5(path, sort=sort)

    has_accel = (len(trajects) > 0 and hasattr(trajects[0], "accel")
                 and trajects[0].accel() is not None)
    if units is None:
        units = {"pos": "mm", "vel": "mm frame-1", "time": "frame"}
        if has_accel:
            units["accel"] = "mm frame-2"

    save_zarr_trajectories(
        trajects, zarr_path, group=group, overwrite=overwrite, units=units
    )

    # Record provenance on the group so the store is self-describing.
    import zarr

    root = zarr.open_group(str(zarr_path), mode="a")
    target = root[group] if group in root else root
    target.attrs.update({
        "source_format": "proptv",
        "source_file": path.name,
        "source_path": str(path),
    })

    return trajects
