"""Tests for the proPTV tracks import (flowtracks.proptv)."""

import numpy as np
import pytest

from flowtracks.io import read_zarr_trajectories
from flowtracks.proptv import convert_proptv_to_zarr, read_proptv_hdf5
from flowtracks.zarr_scene import ZarrScene


def _write_proptv_hdf5(path, tracks):
    """Write a synthetic proPTV tracks file: {key: (n, 10 + 2*n_cam) array}.

    tracks: iterable of (t, key, samples) where samples is (N, 10 + 2*n_cam).
    """
    import h5py

    with h5py.File(path, "w") as f:
        for t, key, samples in tracks:
            f.create_dataset(f"track_{t}_{key}", data=np.asarray(samples, dtype=np.float64))


def _sample(frame=50000, x=0.0, y=0.0, z=0.0, dx=0.0):
    arr = np.zeros((1, 14))  # 2-camera track: 10 + 2*2 = 14
    arr[0, 0] = float(frame)
    arr[0, 1:4] = [x, y, z]
    arr[0, 4:7] = [dx, dx, dx]
    arr[0, 7:10] = [0.01, 0.02, 0.03]
    arr[0, 10:] = [100.0, 400.0, 200.0, 500.0]
    return arr


def test_read_tracks_hdf5(tmp_path):
    f = tmp_path / "tracks.hdf5"
    samples_a = np.concatenate([_sample(50000, 0.0, 0.0, 0.0), _sample(50001, 1.0, 1.0, 1.0)], axis=0)
    samples_b = np.concatenate([_sample(50000, 9.0, 9.0, 9.0), _sample(50001, 8.0, 8.0, 8.0)], axis=0)
    _write_proptv_hdf5(f, [(50000, 0, samples_a), (50000, 1, samples_b)])

    trajs = read_proptv_hdf5(f)
    assert len(trajs) == 2
    by_id = {t.trajid(): t for t in trajs}
    assert set(by_id) == {0, 1}
    np.testing.assert_allclose(by_id[0].pos(), [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(by_id[0].time(), [50000, 50001])
    np.testing.assert_allclose(by_id[1].velocity(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert hasattr(by_id[0], "accel")
    np.testing.assert_allclose(by_id[0].accel(), [[0.01, 0.02, 0.03]] * 2)


def test_read_tracks_hdf5_empty(tmp_path):
    f = tmp_path / "empty.hdf5"
    import h5py

    with h5py.File(f, "w"):
        pass
    assert read_proptv_hdf5(f) == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_proptv_hdf5(tmp_path / "does_not_exist.hdf5")


def test_convert_to_zarr_roundtrip(tmp_path):
    f = tmp_path / "tracks.hdf5"
    _write_proptv_hdf5(f, [(50000, 0, _sample(50000, 0.0, 0.0, 0.0))])
    zarr_path = tmp_path / "out.zarr"
    convert_proptv_to_zarr(f, zarr_path)

    assert zarr_path.exists()
    scene = ZarrScene(zarr_path)
    assert len(scene.trajectory_ids()) == 1

    read_back = read_zarr_trajectories(zarr_path)
    assert len(read_back) == 1
    np.testing.assert_allclose(read_back[0].pos(), [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(read_back[0].time(), [50000])


def test_unit_defaults_are_conventional_mm_not_claimed(tmp_path):
    f = tmp_path / "tracks.hdf5"
    _write_proptv_hdf5(f, [(50000, 0, _sample())])
    zarr_path = tmp_path / "out.zarr"
    convert_proptv_to_zarr(f, zarr_path)
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(root["trajectories"].attrs)
    assert attrs.get("pos_units") == "mm"
    assert attrs.get("vel_units") == "mm frame-1"
    assert attrs.get("accel_units") == "mm frame-2"
    assert attrs.get("time_units") == "frame"
    assert attrs.get("source_format") == "proptv"
