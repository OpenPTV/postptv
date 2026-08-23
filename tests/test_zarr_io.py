import numpy as np
import pytest
import zarr
from pathlib import Path

from flowtracks.io import read_zarr_trajectories, save_zarr_trajectories, trajectories, infer_format
from flowtracks.trajectory import Trajectory


def test_infer_format_zarr():
    assert infer_format("path/to/run.zarr") == "zarr"
    assert infer_format("run.zarr/") == "zarr"


def test_zarr_trajectories_roundtrip(tmp_path):
    zarr_path = tmp_path / "test_run.zarr"

    # Create dummy trajectories
    pos1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    vel1 = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    time1 = np.array([10000, 10001, 10002])

    pos2 = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    vel2 = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
    time2 = np.array([10000, 10001])

    tr1 = Trajectory(pos1, vel1, time1, 1)
    tr2 = Trajectory(pos2, vel2, time2, 2)

    trajects = [tr1, tr2]

    # Save to Zarr
    save_zarr_trajectories(trajects, zarr_path, group="trajectories")

    # Read back using read_zarr_trajectories
    loaded = read_zarr_trajectories(zarr_path, group="trajectories")
    assert len(loaded) == 2

    # Verify auto-detection via trajectories(path)
    loaded_auto = trajectories(str(zarr_path))
    assert len(loaded_auto) == 2

    tr1_loaded = [t for t in loaded_auto if t.trajid() == 1][0]
    np.testing.assert_allclose(tr1_loaded.pos(), pos1)
    np.testing.assert_allclose(tr1_loaded.velocity(), vel1)
    np.testing.assert_array_equal(tr1_loaded.time(), time1)


def test_zarr_read_correspondences(tmp_path):
    zarr_path = tmp_path / "openptv_output.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    corr_group = root.create_group("correspondences")

    # Mock openptv2 correspondences for frames 10000, 10001
    frame_10000 = np.array([
        [1.0, 2.0, 3.0, 101],
        [4.0, 5.0, 6.0, 102],
    ])
    frame_10001 = np.array([
        [1.1, 2.1, 3.1, 101],
        [4.1, 5.1, 6.1, 102],
    ])

    corr_group.create_array("frame_10000", data=frame_10000)
    corr_group.create_array("frame_10001", data=frame_10001)

    loaded = read_zarr_trajectories(zarr_path, group="correspondences")
    assert len(loaded) == 2
    ids = {t.trajid() for t in loaded}
    assert ids == {101, 102}
