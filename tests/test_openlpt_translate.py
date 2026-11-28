"""Tests for the OpenLPT_GUI CSV import (flowtracks.openlpt)."""

import numpy as np
import pytest

from flowtracks.io import read_zarr_trajectories
from flowtracks.openlpt import convert_openlpt_to_zarr, read_openlpt_tracks
from flowtracks.zarr_scene import ZarrScene


def _write_track_csv(path, obj_type, ids_rows):
    """ids_rows: list of (orig_id, [(frame, x, y, z, radius_or_None), ...])."""
    if obj_type == "Tracer":
        n_cam = 2
        header = "TrackID,FrameID,WorldX,WorldY,WorldZ,cam0_x(col),cam0_y(row),cam1_x(col),cam1_y(row)"
    else:
        n_cam = 2
        header = "TrackID,FrameID,WorldX,WorldY,WorldZ,R3D,cam0_x(col),cam0_y(row),cam0_rpx,cam1_x(col),cam1_y(row),cam1_rpx"
    lines = [header]
    for orig_id, rows in ids_rows:
        for frame, x, y, z, radius in rows:
            base = f"{orig_id},{frame},{x:.3f},{y:.3f},{z:.3f}"
            if obj_type == "Bubble":
                base += f",{radius:.3f}" if radius is not None else ","
            for i in range(n_cam):
                base += ",100.0,400.0"
                if obj_type == "Bubble":
                    base += ",5.0"
            lines.append(base)
    path.write_text("\n".join(lines) + "\n")


def _make_dir(tmp_path, obj_type, files):
    """files: list of (prefix, [(orig_id, samples), ...])."""
    directory = tmp_path / "ConvergeTrack"
    directory.mkdir()
    for prefix, ids_rows in files:
        _write_track_csv(directory / f"{prefix}_00042.csv", obj_type, ids_rows)
    return directory


def test_read_tracer_csv(tmp_path):
    directory = _make_dir(tmp_path, "Tracer", [(
        "LongTrackActive",
        [(0, [(0, 0.0, 0.0, 0.0, None), (1, 1.0, 1.0, 1.0, None)])],
    )])
    trajs = read_openlpt_tracks(directory)
    assert len(trajs) == 1
    tr = trajs[0]
    assert tr.trajid() == 0
    np.testing.assert_allclose(tr.pos(), [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(tr.time(), [0, 1])
    # finite-difference velocity over frame 1: (1.0 - 0.0)/1
    assert tr.velocity()[0, 0] == pytest.approx(1.0)
    assert not hasattr(tr, "radius")


def test_read_bubble_csv_keeps_radius(tmp_path):
    directory = _make_dir(tmp_path, "Bubble", [(
        "LongTrackActive",
        [(0, [(0, 0.0, 0.0, 0.0, 4.5), (1, 1.0, 1.0, 1.0, 4.6)])],
    )])
    trajs = read_openlpt_tracks(directory)
    assert len(trajs) == 1
    tr = trajs[0]
    np.testing.assert_allclose(tr.radius(), [4.5, 4.6])


def test_ids_are_unique_across_files(tmp_path):
    directory = _make_dir(tmp_path, "Tracer", [
        ("LongTrackActive", [(0, [(0, 0.0, 0.0, 0.0, None)]), (1, [(0, 1.0, 1.0, 1.0, None)])]),
        ("ExitTrack", [(0, [(0, 9.0, 9.0, 9.0, None)])]),
    ])
    trajs = read_openlpt_tracks(directory)
    ids = [t.trajid() for t in trajs]
    assert len(set(ids)) == len(ids) == 3


def test_unparseable_rows_skipped(tmp_path):
    directory = _make_dir(tmp_path, "Tracer", [
        ("LongTrackActive", [(0, [(0, 0.0, 0.0, 0.0, None), (1, 1.0, 1.0, 1.0, None)])]),
    ])
    csv = directory / "LongTrackActive_00042.csv"
    with open(csv, "a") as f:
        f.write("notanumber,42,0.0,0.0,0.0,100.0,400.0,100.0,400.0\n")
    trajs = read_openlpt_tracks(directory)
    assert len(trajs) == 1
    assert len(trajs[0]) == 2


def test_convert_to_zarr_roundtrip(tmp_path):
    directory = _make_dir(tmp_path, "Tracer", [(
        "LongTrackActive",
        [(0, [(0, 0.0, 0.0, 0.0, None), (1, 1.0, 1.0, 1.0, None)])],
    )])
    zarr_path = tmp_path / "out.zarr"
    convert_openlpt_to_zarr(directory, zarr_path)
    assert zarr_path.exists()

    scene = ZarrScene(zarr_path)
    assert len(scene.trajectory_ids()) == 1
    read_back = read_zarr_trajectories(zarr_path)
    assert len(read_back) == 1
    np.testing.assert_allclose(read_back[0].time(), [0, 1])


def test_unit_defaults_and_provenance(tmp_path):
    directory = _make_dir(tmp_path, "Tracer", [(
        "LongTrackActive", [(0, [(0, 0.0, 0.0, 0.0, None)])],
    )])
    zarr_path = tmp_path / "out.zarr"
    convert_openlpt_to_zarr(directory, zarr_path)
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(root["trajectories"].attrs)
    assert attrs.get("vel_units") == "mm frame-1"
    assert attrs.get("source_format") == "openlpt"


def test_empty_directory_returns_empty(tmp_path):
    directory = tmp_path / "ConvergeTrack"
    directory.mkdir()
    assert read_openlpt_tracks(directory) == []


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_openlpt_tracks(tmp_path / "does_not_exist")


def test_pyopenlpt_alias_records_pyopenlpt_provenance(tmp_path):
    """The pyOpenLPT alias writes the same store but with source_format=pyopenlpt."""
    from flowtracks.openlpt import convert_pyopenlpt_to_zarr

    directory = _make_dir(tmp_path, "Tracer", [(
        "LongTrackActive", [(0, [(0, 0.0, 0.0, 0.0, None)])],
    )])
    zarr_path = tmp_path / "out.zarr"
    convert_pyopenlpt_to_zarr(directory, zarr_path)

    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(root["trajectories"].attrs)
    assert attrs.get("source_format") == "pyopenlpt"
    assert attrs.get("source_file") == directory.name
