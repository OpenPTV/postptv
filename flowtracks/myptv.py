# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""MyPTV trajectory import: read MyPTV TSV output as flowtracks Trajectories.

MyPTV writes each pipeline stage as a plain-text, tab-separated file with
no header row. The two forms this module understands are:

- smoothed / stitched trajectories (11 columns): ``[trajid, x, y, z, vx, vy, vz, ax, ay, az, frame]`` — the raw output of ``myptv.tracking_mod.save_results`` stage produces the tracked form, while ``myptv.traj_smoothing_mod.smooth_trajectories.save_results`` and ``myptv.traj_stitching_mod.traj_stitching.save_results`` produce the smoothed/stitched form (the final ``stitching`` stage file is the canonical one).
- tracked particles (5 or 9 columns): ``[trajid, x, y, z, <per-camera blob index columns>, <per-camera error>, frame]`` — the last column is always the frame number. Velocity is not stored by MyPTV in this form; the translator re-derives it by finite differences over the frame column, exactly as ``traj_stitching_mod`` does for thin trajectories. Acceleration is unavailable and is not guessed.

Rows with trajectory id -1 are unlinked samples (particles the tracker could
not link). They are emitted as one-sample Trajectory objects so no data is
dropped.
"""

from pathlib import Path

import numpy as np

from flowtracks.io import save_zarr_trajectories
from flowtracks.trajectory import Trajectory

_SCHEMA_SMOOTHED = 11
_SCHEMA_TRACKED = (5, 9)


def _finite_difference_velocity(pos, time):
    """Forward-difference velocity from position over the frame column.

    Matches the convention used by ``traj_stitching_mod`` for the thin
    trajectories it encounters: velocity in metres per frame.
    """
    if len(pos) < 2:
        return np.zeros_like(pos)
    dt = np.diff(time.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.diff(pos, axis=0) / dt[:, None]
    v = np.where(np.isfinite(v), v, 0.0)
    out = np.zeros_like(pos)
    out[:-1] = v
    out[-1] = v[-1] if len(v) else 0.0
    return out


def read_myptv_trajectories(path, sort=True):
    """
    Read a MyPTV trajectory file into a list of :class:`Trajectory` objects.

    The schema is detected from the column count (5, 9 or 11 columns).
    The last column is always treated as the frame number and the first
    three columns (after the id) as x, y, z — a rule that holds for every
    MyPTV tracker regardless of the number of cameras.

    Arguments:
    path - path to the MyPTV TSV file.
    sort - if True (default), trajectories are returned sorted by (trajid, frame).

    Returns:
    trajects - a list of :class:`~flowtracks.trajectory.Trajectory` objects.
        Unlinked samples (trajid == -1 in the source) each become a one-sample
        trajectory.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MyPTV trajectory file not found: {path}")
    if path.stat().st_size == 0:
        return []
    raw = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        return []

    ncols = raw.shape[1]
    if ncols == _SCHEMA_SMOOTHED:
        trajid = raw[:, 0].astype(np.int64)
        pos = raw[:, 1:4]
        vel = raw[:, 4:7]
        accel = raw[:, 7:10]
        frame = raw[:, 10].astype(np.int64)
    elif ncols in _SCHEMA_TRACKED:
        trajid = raw[:, 0].astype(np.int64)
        pos = raw[:, 1:4]
        vel = None
        accel = None
        frame = raw[:, -1].astype(np.int64)
    else:
        raise ValueError(
            f"Unsupported MyPTV trajectory schema: {ncols} columns in {path}. "
            "Expected 5 or 9 columns (raw tracked) or 11 columns (smoothed/stitched)."
        )

    # Unlinked samples: keep as one-sample trajectories so no data is dropped.
    unlinked = trajid == -1
    unlinked_pos, unlinked_vel, unlinked_accel, unlinked_frame = (
        (pos[unlinked], (vel[unlinked] if vel is not None else None),
         (accel[unlinked] if accel is not None else None), frame[unlinked])
    )

    linked = ~unlinked
    trajid_l, pos_l, frame_l = trajid[linked], pos[linked], frame[linked]
    vel_l = vel[linked] if vel is not None else None
    accel_l = accel[linked] if accel is not None else None

    # Group the linked rows by trajid; sort the work we key on.
    order = np.lexsort((frame_l, trajid_l))
    trajid_l, pos_l, frame_l = trajid_l[order], pos_l[order], frame_l[order]
    if vel_l is not None:
        vel_l = vel_l[order]
    if accel_l is not None:
        accel_l = accel_l[order]

    bounds = np.flatnonzero(np.diff(trajid_l)) + 1
    groups = np.split(np.arange(len(trajid_l)), bounds) if len(trajid_l) else [np.array([], dtype=int)]

    trajects = []
    for idx in groups:
        if len(idx) == 0:
            continue
        trid = int(trajid_l[idx[0]])
        tr_pos, tr_frame = pos_l[idx], frame_l[idx]
        tr_vel = (vel_l[idx] if vel_l is not None
                  else _finite_difference_velocity(tr_pos, tr_frame))
        if accel_l is not None:
            trajects.append(Trajectory(tr_pos, tr_vel, tr_frame, trid, accel=accel_l[idx]))
        else:
            trajects.append(Trajectory(tr_pos, tr_vel, tr_frame, trid))

    # One-sample trajectories for the unlinked samples, preserving any stored
    # velocity/acceleration present in the source.
    for i in range(len(unlinked_frame)):
        tr_pos = unlinked_pos[i].reshape(1, 3)
        tr_frame = unlinked_frame[i].reshape(1)
        if unlinked_vel is not None:
            tr_vel = unlinked_vel[i].reshape(1, 3)
        else:
            tr_vel = np.zeros((1, 3))
        if unlinked_accel is not None:
            trajects.append(
                Trajectory(tr_pos, tr_vel, tr_frame, -1, accel=unlinked_accel[i].reshape(1, 3))
            )
        else:
            trajects.append(Trajectory(tr_pos, tr_vel, tr_frame, -1))

    if sort:
        trajects.sort(key=lambda t: (t.trajid(), t.time()[0]))

    return trajects


def convert_myptv_to_zarr(path, zarr_path, group="trajectories", overwrite=True,
                         units=None, sort=True):
    """
    Convert a MyPTV trajectory TSV file into a Zarr store.

    The store holds a ``trajectories`` group with ``pos``, ``vel``, ``time``,
    ``trajid`` (and ``accel`` when present) arrays — the layout
    :class:`flowtracks.zarr_scene.ZarrScene` reads natively, and the same
    layout produced by ``openptv2``'s RunStore. It can therefore be fed to
    flowtracks' Eulerian/phase-averaging pipeline without any conversion.

    Arguments:
    path - path to the MyPTV TSV file (tracked, smoothed or stitched).
    zarr_path - target ``.zarr`` directory.
    group - sub-group name inside the store (default ``"trajectories"``).
    overwrite - if True, overwrite existing arrays in the group.
    units - units dict recorded on the group, e.g. ``{"pos": "m",
        "vel": "m frame-1", "time": "frame"}``. Defaults to per-frame per-metre
        units honest for a file with no stated frame rate; pass your own to
        state absolute units.
    sort - if True (default), trajectories are sorted by (trajid, frame).

    Returns:
    trajects - the list of :class:`~flowtracks.trajectory.Trajectory` objects
        that were written.
    """
    path = Path(path)
    trajects = read_myptv_trajectories(path, sort=sort)

    has_accel = any(hasattr(t, "accel") for t in trajects)

    if units is None:
        units = {"pos": "m", "vel": "m frame-1", "time": "frame"}
        if has_accel:
            units["accel"] = "m frame-2"

    save_zarr_trajectories(
        trajects, zarr_path, group=group, overwrite=overwrite, units=units
    )

    # Record provenance on the group so the store is self-describing.
    import zarr

    root = zarr.open_group(str(zarr_path), mode="a")
    target = root[group] if group in root else root
    target.attrs.update({
        "source_format": "myptv",
        "source_file": path.name,
        "source_path": str(path),
    })

    return trajects
