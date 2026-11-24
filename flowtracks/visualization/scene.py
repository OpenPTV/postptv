# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""Data loading and scene-geometry assembly for the browser 3D viewer.

Pure numpy / pyvista (no trame, no server) -- import this module from tests,
notebooks, or scripts without opening a browser.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ZARR_READ_KEYS = ("pos", "velocity", "time", "trajid")


@dataclass(frozen=True)
class TrajectoryScene:
    """One decimated, time-filtered view of a trajectory store."""

    pos: np.ndarray  # (N, 3) float64
    vel: np.ndarray  # (N, 3) float64
    time: np.ndarray  # (N,) float64 -- frame numbers as recorded (e.g. 10001.0)
    trajid: np.ndarray  # (N,) int64
    speed: np.ndarray  # (N,) float64

    @property
    def n_points(self) -> int:
        return len(self.time)

    @property
    def n_trajectories(self) -> int:
        return int(len(np.unique(self.trajid)))

    @property
    def frame_min(self) -> float:
        return float(self.time.min()) if self.n_points else float("nan")

    @property
    def frame_max(self) -> float:
        return float(self.time.max()) if self.n_points else float("nan")


def load_scene(
    zarr_path,
    *,
    time_range=None,
    max_trajectories=None,
):
    """Load trajectories from a flowtracks/openptv2 Zarr store.

    Arguments:
        zarr_path - path to the ``.zarr`` store (a ``trajectories/`` group
            with ``pos``/``vel``/``time``/``trajid``).
        time_range - optional ``(t0, t1)``; only points with
            ``t0 <= t <= t1`` are kept (either bound may be ``None``).
        max_trajectories - keep at most this many trajectories; an evenly
            spaced subset (by ``trajid``) keeps the scene light. Deterministic
            -- same inputs always yield the same subset.

    Returns:
        A :class:`TrajectoryScene` sorted by (trajid, time).
    """
    from flowtracks.zarr_scene import ZarrScene

    scene = ZarrScene(zarr_path)
    pos, vel, time, trajid = scene.collect(list(ZARR_READ_KEYS))
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    trajid = np.asarray(trajid)

    # Drop any rows carrying non-finite values so a stray NaN can never blank
    # the scene or poison the speed colormap.
    finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1) & np.isfinite(time)
    if not finite.all():
        pos, vel, time, trajid = pos[finite], vel[finite], time[finite], trajid[finite]

    if time_range is not None:
        t0, t1 = time_range
        mask = np.ones(len(time), dtype=bool)
        if t0 is not None:
            mask &= time >= t0
        if t1 is not None:
            mask &= time <= t1
        pos, vel, time, trajid = pos[mask], vel[mask], time[mask], trajid[mask]

    if max_trajectories is not None and max_trajectories > 0:
        uniq = np.unique(trajid)
        if len(uniq) > max_trajectories:
            keep_ids = uniq[
                np.floor(
                    np.linspace(0, len(uniq) - 1, max_trajectories)
                ).astype(np.int64)
            ]
            keep = np.isin(trajid, keep_ids)
            pos, vel, time, trajid = pos[keep], vel[keep], time[keep], trajid[keep]

    order = np.lexsort((time, trajid))
    pos, vel, time, trajid = pos[order], vel[order], time[order], trajid[order]

    return TrajectoryScene(
        pos=pos,
        vel=vel,
        time=time,
        trajid=trajid,
        speed=np.linalg.norm(vel, axis=1),
    )


def filter_through_frame(scene, frame):
    """Sub-scene with points at or before ``frame`` (time slider)."""
    idx = np.flatnonzero(scene.time <= frame)
    return TrajectoryScene(
        pos=scene.pos[idx],
        vel=scene.vel[idx],
        time=scene.time[idx],
        trajid=scene.trajid[idx],
        speed=scene.speed[idx],
    )


def build_polydata(scene):
    """One PolyData with points + one polyline per trajid.

    Point data: ``trajid``, ``time``, ``velocity``, ``speed`` -- mirrors
    :func:`flowtracks.writers.trajectory_polydata` so the in-browser scenes
    and the ParaView .vtp export share a data contract.
    """
    import pyvista as pv

    if scene.n_points == 0:
        return pv.PolyData()

    poly = pv.PolyData()
    poly.points = scene.pos

    starts = np.flatnonzero(np.r_[True, scene.trajid[1:] != scene.trajid[:-1]])
    ends = np.r_[starts[1:], len(scene.trajid)]
    segments = [(s, e) for s, e in zip(starts, ends) if e - s > 1]
    if segments:
        poly.lines = np.concatenate(
            [np.r_[e - s, np.arange(s, e, dtype=np.int64)] for s, e in segments]
        ).astype(np.int64)

    poly.point_data["trajid"] = scene.trajid
    poly.point_data["time"] = scene.time
    poly.point_data["velocity"] = scene.vel
    poly.point_data["speed"] = scene.speed
    return poly


def load_mesh(path):
    """Load an optional vessel/geometry mesh file."""
    import pyvista as pv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mesh file not found: {path}")
    return pv.read(str(path))


def load_volume(path):
    """Load an optional Eulerian volume file (.vti/.vtr).

    Empty voxels are commonly stored as NaN; these are replaced with 0.0 so
    the volume renderer never receives NaNs (which would blank the scene).
    The original scalar range is preserved for the colormap where possible.
    """
    import numpy as np
    import pyvista as pv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"volume file not found: {path}")
    volume = pv.read(str(path))
    for name in list(volume.point_data.keys()) + list(volume.cell_data.keys()):
        arr = volume.point_data[name] if name in volume.point_data else volume.cell_data[name]
        if np.issubdtype(arr.dtype, np.floating):
            bad = ~np.isfinite(arr)
            if bad.any():
                arr = np.where(bad, 0.0, arr)
                if name in volume.point_data:
                    volume.point_data[name] = arr
                else:
                    volume.cell_data[name] = arr
    return volume
