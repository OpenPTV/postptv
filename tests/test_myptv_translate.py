"""Tests for the MyPTV TSV -> flowtracks translator (flowtracks.myptv)."""

import numpy as np
import pytest

from flowtracks.io import read_zarr_trajectories
from flowtracks.myptv import convert_myptv_to_zarr, read_myptv_trajectories
from flowtracks.trajectory import Trajectory
from flowtracks.zarr_scene import ZarrScene


def _write_smoothed_tsv(path, rows):
    """rows: iterable of (id, x, y, z, vx, vy, vz, ax, ay, az, frame)."""
    arr = np.array(rows, dtype=np.float64)
    np.savetxt(path, arr, delimiter="\t", fmt="%.6f")


def _row(trajid, x, y, z, vx, vy, vz, ax, ay, az, frame):
    return (trajid, x, y, z, vx, vy, vz, ax, ay, az, frame)


def test_read_smoothed_file_11col(tmp_path):
    f = tmp_path / "smoothed.tsv"
    _write_smoothed_tsv(f, [
        _row(1, 1.0, 2.0, 3.0, 0.10, 0.20, 0.30, 0.01, 0.02, 0.03, 5),
        _row(1, 2.0, 3.0, 4.0, 0.10, 0.20, 0.30, 0.01, 0.02, 0.03, 6),
        _row(2, 5.0, 5.0, 5.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 6),
    ])
    trajs = read_myptv_trajectories(f)
    assert len(trajs) == 2
    by_id = {t.trajid(): t for t in trajs}
    assert len(by_id[1]) == 2
    np.testing.assert_allclose(by_id[1].pos(), [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    np.testing.assert_allclose(by_id[1].time(), [5, 6])
    np.testing.assert_allclose(by_id[1].velocity(), [[0.10, 0.20, 0.30]] * 2)
    np.testing.assert_allclose(by_id[1].accel(), [[0.01, 0.02, 0.03]] * 2)
    assert len(by_id[2]) == 1


def test_read_raw_tracked_9col_finite_diff_velocity(tmp_path):
    # 9 columns: [trajid, x, y, z, blob_cam1, blob_cam2, blob_cam3, err, frame]
    f = tmp_path / "tracked.tsv"
    arr = np.array([
        # trajid, x, y, z, blob1, blob2, blob3, err, frame
        [1, 0.0, 0.0, 0.0, 3, 4, 5, 0.01, 0],
        [1, 1.0, 1.0, 1.0, 3, 4, 5, 0.01, 1],
        [1, 3.0, 3.0, 3.0, 3, 4, 5, 0.01, 2],   # jump to frame 2
        [2, 9.0, 9.0, 9.0, -1, -1, -1, 0.01, 2],
    ], dtype=np.float64)
    np.savetxt(f, arr, delimiter="\t", fmt="%.6f")
    trajs = read_myptv_trajectories(f)
    by_id = {t.trajid(): t for t in trajs}
    assert set(by_id) == {1, 2}
    tr1 = by_id[1]
    assert len(tr1) == 3
    np.testing.assert_allclose(tr1.time(), [0, 1, 2])
    # finite-difference velocity over frames: (1.0/1) then (2.0/1)
    assert tr1.velocity()[0, 0] == pytest.approx(1.0)
    assert tr1.velocity()[1, 0] == pytest.approx(2.0)
    # last sample carries the previous velocity
    assert tr1.velocity()[2, 0] == pytest.approx(2.0)
    assert not hasattr(tr1, "accel")


def test_unlinked_minus_one_samples_kept(tmp_path):
    f = tmp_path / "stitched.tsv"
    arr = np.array([
        [1, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 5],
        [-1, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5],
        [-1, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6],
    ], dtype=np.float64)
    np.savetxt(f, arr, delimiter="\t", fmt="%.6f")
    trajs = read_myptv_trajectories(f)
    # one trajectory for id 1, two one-sample trajectories for id -1
    assert len(trajs) == 3
    ones = [t for t in trajs if t.trajid() == -1]
    assert len(ones) == 2
    assert all(len(t) == 1 for t in ones)
    # no samples lost
    assert sum(len(t) for t in trajs) == 3


def test_unsupported_schema_raises(tmp_path):
    f = tmp_path / "bad.tsv"
    np.savetxt(f, np.zeros((2, 7)), delimiter="\t", fmt="%.3f")
    with pytest.raises(ValueError, match="Unsupported MyPTV trajectory schema"):
        read_myptv_trajectories(f)


def test_empty_file_returns_empty(tmp_path):
    f = tmp_path / "empty.tsv"
    f.write_text("")
    assert read_myptv_trajectories(f) == []


def test_convert_to_zarr_roundtrip(tmp_path):
    f = tmp_path / "smoothed.tsv"
    _write_smoothed_tsv(f, [
        _row(1, 1.0, 2.0, 3.0, 0.10, 0.20, 0.30, 0.01, 0.02, 0.03, 5),
        _row(1, 2.0, 3.0, 4.0, 0.10, 0.20, 0.30, 0.01, 0.02, 0.03, 6),
        _row(2, 5.0, 5.0, 5.0, 0.50, 0.50, 0.50, 0.05, 0.05, 0.05, 6),
    ])
    zarr_path = tmp_path / "out.zarr"
    convert_myptv_to_zarr(f, zarr_path)

    # ZarrScene (the flowtracks reader) opens it with no conversion
    scene = ZarrScene(zarr_path)
    assert len(scene.trajectory_ids()) == 2
    scene_pos = scene.collect(["pos"])[0]
    assert len(scene_pos) == 3

    # round trip back through the flowtracks reader
    read_back = read_zarr_trajectories(zarr_path)
    assert len(read_back) == 2
    by_id = {t.trajid(): t for t in read_back}
    np.testing.assert_allclose(by_id[1].pos(), [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    np.testing.assert_allclose(by_id[1].time(), [5, 6])
    np.testing.assert_allclose(by_id[1].velocity(), [[0.10, 0.20, 0.30]] * 2)
    np.testing.assert_allclose(by_id[1].accel(), [[0.01, 0.02, 0.03]] * 2)


def test_convert_to_zarr_tracked_file_uses_finite_difference_velocity(tmp_path):
    f = tmp_path / "tracked.tsv"
    arr = np.array([
        [1, 0.0, 0.0, 0.0, -1, -1, -1, 0.01, 0],
        [1, 1.0, 0.0, 0.0, -1, -1, -1, 0.01, 1],
    ], dtype=np.float64)
    np.savetxt(f, arr, delimiter="\t", fmt="%.6f")
    zarr_path = tmp_path / "out.zarr"
    convert_myptv_to_zarr(f, zarr_path)
    read_back = read_zarr_trajectories(zarr_path)
    assert len(read_back) == 1
    assert read_back[0].velocity()[0, 0] == pytest.approx(1.0)


def test_frame_rate_units_not_claimed(tmp_path):
    f = tmp_path / "smoothed.tsv"
    _write_smoothed_tsv(f, [_row(1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0)])
    zarr_path = tmp_path / "out.zarr"
    convert_myptv_to_zarr(f, zarr_path)
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(root["trajectories"].attrs)
    # per-frame units, not per-second units
    assert attrs.get("vel_units") == "m frame-1"
    assert attrs.get("time_units") == "frame"
    # accel is present in this schema
    assert attrs.get("accel_units") == "m frame-2"
