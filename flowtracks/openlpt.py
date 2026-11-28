# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""OpenLPT track import: read ``ConvergeTrack`` CSVs as Trajectories.

Covers both OpenLPT_GUI (the GUI and C++ core, PyPI package ``openlpt``)
and pyOpenLPT — a packaging fork of the same software with identical code
and identical output; :func:`read_pyopenlpt_tracks` and
:func:`convert_pyopenlpt_to_zarr` are aliases with ``source_format="pyopenlpt"``
provenance recorded.

OpenLPT_GUI (https://github.com/JHU-NI-LAB/OpenLPT_GUI) writes each
tracking stage as a family of CSV files under
``Results/ConvergeTrack/`` — named ``{Outcome}_{frame}.csv`` where the
outcomes are ``LongTrackActive``, ``ShortTrackActive``, ``LongTrackInactive``
and ``ExitTrack``. Each file carries a header row and one row per track
sample:

- Tracer: ``[TrackID, FrameID, WorldX, WorldY, WorldZ, <camX, camY> * n_cam]``
- Bubble: ``[TrackID, FrameID, WorldX, WorldY, WorldZ, R3D, <camX, camY, camRpx> * n_cam]``

(one 2D projection column per camera, ``camX/camY`` for tracers and
``camX/camY/camRpx`` for bubbles). Positions are millimetres (OpenLPT's 3D
error tolerance and volume scale are in mm). Frame IDs are integer frame
numbers. Velocity and acceleration are NOT stored in these files; the
translator re-derives velocity as finite differences over the frame column
(mm per frame) — the same convention used by MyPTV's smoothing and
stitching stages. Acceleration is not guessed.

``read_openlpt_tracks`` parses the directory into flowtracks Trajectory
objects (with a ``radius`` property when the object type is Bubble). The
file-to-track id assignment mirrors the GUI's own post-processing: a running
offset is added to each file's TrackID so ids are unique across the whole
directory. ``convert_openlpt_to_zarr`` additionally writes a ``trajectories``
store (the same layout ``openptv2``'s RunStore and ``save_zarr_trajectories``
write) with ``source_format="openlpt"`` provenance.
"""

from pathlib import Path

import numpy as np

from flowtracks.io import save_zarr_trajectories
from flowtracks.trajectory import Trajectory

_OUTCOMES = ("LongTrackActive", "ShortTrackActive", "LongTrackInactive", "ExitTrack")


def _natsort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", s)]


def _finite_difference_velocity(pos, time):
    """Forward-difference velocity from position over the frame column."""
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


def _detect_object_type(header_first_col_row):
    """Return ('Bubble'|'Tracer', n_cam) from the header row."""
    if header_first_col_row is None:
        return "Tracer", 0
    if "R3D" in header_first_col_row:
        n_cam = max(0, (header_first_col_row.count("cam") // 3))
        return "Bubble", n_cam
    n_cam = max(0, (header_first_col_row.count("cam") // 2))
    return "Tracer", n_cam


def _read_track_csv(path, obj_type):
    """Parse one CSV file into (orig_id, rows) pairs; obj_type known or None.

    rows: list of (frame, x, y, z, radius) — radius is float or None.
    """
    rows_out = []
    with open(path, "r", newline="") as f:
        header = f.readline().strip()
        if obj_type is None:
            obj_type, _ = _detect_object_type(header)
        is_bubble = obj_type == "Bubble"

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                orig_id = int(parts[0])
                frame = int(float(parts[1]))
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue
            radius = np.nan
            if is_bubble:
                try:
                    radius = float(parts[5]) if len(parts) > 5 and parts[5] else np.nan
                except ValueError:
                    radius = np.nan
            rows_out.append((orig_id, frame, x, y, z, radius))
    return obj_type, rows_out


def read_openlpt_tracks(path, outcomes=None, sort=True):
    """
    Read an OpenLPT_GUI ``ConvergeTrack`` directory into Trajectory objects.

    Arguments:
    path - path to a ``ConvergeTrack`` directory (or a single track CSV file —
        in that case only that file is read).
    outcomes - sequence of file-name prefixes to read (default: all four).
    sort - if True (default), trajectories are returned sorted by (trajid, frame).

    Returns:
    trajects - a list of :class:`~flowtracks.trajectory.Trajectory` objects.
        ``velocity`` is re-derived by finite differences (mm per frame);
        ``radius`` is an extra property for Bubble files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OpenLPT ConvergeTrack path not found: {path}")

    outcomes = outcomes or _OUTCOMES
    files = []
    if path.is_dir():
        files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".csv"
                 and any(p.name.startswith(o) for o in outcomes)]
    else:
        files = [path]
    files = sorted(files, key=lambda p: _natsort_key(p.name))

    # Assign ids like the GUI: a running offset keeps ids unique across
    # the whole directory.
    max_id_overall = -1
    by_track = {}
    for f in files:
        obj_type, rows = _read_track_csv(f, None)
        local_max = -1
        offset = max_id_overall + 1
        for orig_id, frame, x, y, z, radius in rows:
            cum_id = orig_id + offset
            if orig_id > local_max:
                local_max = orig_id
            if cum_id not in by_track:
                by_track[cum_id] = {"pos": [], "time": [], "radius": []}
            by_track[cum_id]["pos"].append([x, y, z])
            by_track[cum_id]["time"].append(frame)
            by_track[cum_id]["radius"].append(radius)
        if local_max != -1:
            max_id_overall += (local_max + 1)

    trajects = []
    for trajid, data in by_track.items():
        pos = np.asarray(data["pos"], dtype=np.float64)
        time = np.asarray(data["time"], dtype=np.int64)
        radius = np.asanyarray(data["radius"], dtype=np.float64)
        order = np.argsort(time, kind="stable")
        pos, time, radius = pos[order], time[order], radius[order]

        vel = _finite_difference_velocity(pos, time)
        has_radius = radius.size > 0 and not np.isnan(radius).all()

        # Radius property only added when it is present (Bubble tracks).
        if has_radius:
            trajects.append(Trajectory(pos, vel, time, trajid, radius=radius))
        else:
            trajects.append(Trajectory(pos, vel, time, trajid))

    if sort:
        trajects.sort(key=lambda t: (t.trajid(), t.time()[0] if len(t.time()) else 0))

    return trajects


def convert_openlpt_to_zarr(path, zarr_path, group="trajectories", overwrite=True,
                            units=None, sort=True, outcomes=None):
    """
    Convert an OpenLPT_GUI ``ConvergeTrack`` directory into a Zarr store.

    The store holds a ``trajectories`` group with ``pos``, ``vel``, ``time``
    and ``trajid`` arrays — the same layout ``openptv2``'s RunStore and
    ``save_zarr_trajectories`` write, so it is consumed by flowtracks'
    Eulerian/phase-averaging pipeline directly.

    Arguments:
    path - path to the ``ConvergeTrack`` directory (or a single track CSV file).
    zarr_path - target ``.zarr`` directory.
    group - sub-group name inside the store (default ``"trajectories"``).
    overwrite - if True, overwrite existing arrays in the group.
    units - units dict recorded on the group. Defaults to per-frame
        millimetre units (``{"pos": "mm", "vel": "mm frame-1",
        "time": "frame"}``) — the source file records neither, and OpenLPT
        volumes are conventionally in millimetres. Pass ``units=`` to state
        absolute units explicitly.
    sort - if True (default), trajectories are sorted by (trajid, frame).
    outcomes - file-name prefixes to read (default: all four).

    Returns:
    trajects - the list of :class:`~flowtracks.trajectory.Trajectory` objects
        that were written.
    """
    path = Path(path)
    trajects = read_openlpt_tracks(path, outcomes=outcomes, sort=sort)

    if units is None:
        units = {"pos": "mm", "vel": "mm frame-1", "time": "frame"}

    save_zarr_trajectories(
        trajects, zarr_path, group=group, overwrite=overwrite, units=units
    )

    # Record provenance on the group so the store is self-describing.
    import zarr

    root = zarr.open_group(str(zarr_path), mode="a")
    target = root[group] if group in root else root
    target.attrs.update({
        "source_format": "openlpt",
        "source_file": path.name,
        "source_path": str(path),
    })

    return trajects


def read_pyopenlpt_tracks(path, outcomes=None, sort=True):
    """Alias for :func:`read_openlpt_tracks`.

    pyOpenLPT (https://github.com/alexlib/pyOpenLPT) is a packaging fork of
    OpenLPT_GUI — identical C++ core and identical ``ConvergeTrack/*.csv``
    output, so the same reader covers both. ``openlpt`` is the installed
    package name of that fork.
    """
    return read_openlpt_tracks(path, outcomes=outcomes, sort=sort)


def convert_pyopenlpt_to_zarr(path, zarr_path, group="trajectories", overwrite=True,
                              units=None, sort=True, outcomes=None):
    """Convert a pyOpenLPT ``ConvergeTrack`` directory into a Zarr store.

    Same code path as :func:`convert_openlpt_to_zarr` (same file format, so
    identical store) — records ``source_format="pyopenlpt"`` on the group so
    the store is self-describing about where its data came from.
    """
    path = Path(path)
    trajects = read_openlpt_tracks(path, outcomes=outcomes, sort=sort)

    if units is None:
        units = {"pos": "mm", "vel": "mm frame-1", "time": "frame"}

    save_zarr_trajectories(
        trajects, zarr_path, group=group, overwrite=overwrite, units=units
    )

    import zarr

    root = zarr.open_group(str(zarr_path), mode="a")
    target = root[group] if group in root else root
    target.attrs.update({
        "source_format": "pyopenlpt",
        "source_file": path.name,
        "source_path": str(path),
    })

    return trajects
