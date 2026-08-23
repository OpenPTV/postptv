"""ZarrScene must behave identically to the PyTables-backed Scene for the
same data (cross-validated by writing the same trajectories to both an
HDF5 file and a Zarr store), and must also read an openptv2 RunStore's
Zarr layout directly (traj/ + trajectories/ groups, no /bounds table)."""

import numpy as np
import pytest

from flowtracks.io import save_particles_table, save_zarr_trajectories
from flowtracks.scene import Scene, open_scene
from flowtracks.trajectory import Trajectory
from flowtracks.zarr_scene import ZarrScene


def _make_trajectories():
    tr1 = Trajectory(
        pos=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        velocity=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]),
        time=np.array([10, 11, 12]),
        trajid=1,
        accel=np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03]]),
    )
    tr2 = Trajectory(
        pos=np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0]]),
        velocity=np.array(
            [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]
        ),
        time=np.array([10, 11, 12, 13]),
        trajid=2,
        accel=np.array(
            [[0.05] * 3, [0.06] * 3, [0.07] * 3, [0.08] * 3]
        ),
    )
    tr3 = Trajectory(
        pos=np.array([[9.0, 9.0, 9.0], [9.5, 9.5, 9.5]]),
        velocity=np.array([[0.9, 0.9, 0.9], [0.95, 0.95, 0.95]]),
        time=np.array([11, 12]),
        trajid=3,
        accel=np.array([[0.09] * 3, [0.095] * 3]),
    )
    return [tr1, tr2, tr3]


@pytest.fixture
def scenes(tmp_path):
    trajects = _make_trajectories()
    h5_path = tmp_path / "ref.h5"
    zarr_path = tmp_path / "ref.zarr"
    save_particles_table(str(h5_path), trajects)
    save_zarr_trajectories(trajects, zarr_path)
    return Scene(str(h5_path)), ZarrScene(zarr_path)


def _sorted_rows(arr):
    return arr[np.lexsort(arr.T)]


def test_keys_and_shapes(scenes):
    hdf, zarr_scene = scenes
    assert set(zarr_scene.keys()) == set(hdf.keys())
    assert dict(zip(zarr_scene.keys(), zarr_scene.shapes())) == dict(
        zip(hdf.keys(), hdf.shapes())
    )


def test_trajectory_ids_and_tags(scenes):
    hdf, zarr_scene = scenes
    assert sorted(zarr_scene.trajectory_ids().tolist()) == sorted(
        hdf.trajectory_ids().tolist()
    )
    assert np.array_equal(
        _sorted_rows(zarr_scene.trajectory_tags().astype(float)),
        _sorted_rows(hdf.trajectory_tags().astype(float)),
    )


def test_trajectory_by_id_matches(scenes):
    hdf, zarr_scene = scenes
    for trid in [1, 2, 3]:
        h = hdf.trajectory_by_id(trid)
        z = zarr_scene.trajectory_by_id(trid)
        np.testing.assert_allclose(z.pos(), h.pos())
        np.testing.assert_allclose(z.velocity(), h.velocity())
        np.testing.assert_array_equal(z.time(), h.time())
        assert z.trajid() == h.trajid()


def test_iter_trajectories_matches(scenes):
    hdf, zarr_scene = scenes
    h_by_id = {t.trajid(): t for t in hdf.iter_trajectories()}
    z_by_id = {t.trajid(): t for t in zarr_scene.iter_trajectories()}
    assert set(h_by_id) == set(z_by_id)
    for trid in h_by_id:
        np.testing.assert_allclose(z_by_id[trid].pos(), h_by_id[trid].pos())
        np.testing.assert_array_equal(z_by_id[trid].time(), h_by_id[trid].time())


def test_iter_frames_matches(scenes):
    hdf, zarr_scene = scenes
    h_frames = {f.time(): f for f in hdf.iter_frames()}
    z_frames = {f.time(): f for f in zarr_scene.iter_frames()}
    assert set(h_frames) == set(z_frames)
    for t in h_frames:
        assert sorted(h_frames[t].trajid().tolist()) == sorted(z_frames[t].trajid().tolist())
        np.testing.assert_allclose(
            _sorted_rows(h_frames[t].pos()), _sorted_rows(z_frames[t].pos())
        )


def test_frame_by_time_matches(scenes):
    hdf, zarr_scene = scenes
    h = hdf.frame_by_time(11)
    z = zarr_scene.frame_by_time(11)
    assert sorted(h.trajid().tolist()) == sorted(z.trajid().tolist())


def test_iter_segments_matches(scenes):
    hdf, zarr_scene = scenes
    h_segs = list(hdf.iter_segments())
    z_segs = list(zarr_scene.iter_segments())
    assert len(h_segs) == len(z_segs)
    for (h_a, h_b), (z_a, z_b) in zip(h_segs, z_segs):
        assert h_a.time() == z_a.time()
        assert sorted(h_a.trajid().tolist()) == sorted(z_a.trajid().tolist())
        assert sorted(h_b.trajid().tolist()) == sorted(z_b.trajid().tolist())


def test_collect_matches(scenes):
    hdf, zarr_scene = scenes
    h_pos, h_time = hdf.collect(["pos", "time"])
    z_pos, z_time = zarr_scene.collect(["pos", "time"])
    np.testing.assert_allclose(
        _sorted_rows(np.column_stack([h_pos, h_time])),
        _sorted_rows(np.column_stack([z_pos, z_time])),
    )


def test_collect_with_where_matches(scenes):
    hdf, zarr_scene = scenes
    where = {"time": (11, 13, False)}
    h_pos = hdf.collect(["pos"], where=where)[0]
    z_pos = zarr_scene.collect(["pos"], where=where)[0]
    np.testing.assert_allclose(_sorted_rows(h_pos), _sorted_rows(z_pos))


def test_bounding_box_matches(scenes):
    hdf, zarr_scene = scenes
    h_min, h_max = hdf.bounding_box()
    z_min, z_max = zarr_scene.bounding_box()
    np.testing.assert_allclose(z_min, h_min)
    np.testing.assert_allclose(z_max, h_max)


def test_frame_range_filters(scenes):
    hdf, zarr_scene = scenes
    zarr_scene.set_frame_range((11, 13))
    hdf.set_frame_range((11, 13))
    assert zarr_scene.frame_range() == hdf.frame_range()
    z_pos = zarr_scene.collect(["time"])[0]
    assert z_pos.min() >= 11 and z_pos.max() < 13


def test_open_scene_dispatches_by_format(tmp_path):
    trajects = _make_trajectories()
    h5_path = tmp_path / "d.h5"
    zarr_path = tmp_path / "d.zarr"
    save_particles_table(str(h5_path), trajects)
    save_zarr_trajectories(trajects, zarr_path)

    assert isinstance(open_scene(str(zarr_path)), ZarrScene)
    assert isinstance(open_scene(str(h5_path)), Scene)


def test_reads_openptv2_run_store_layout(tmp_path):
    """openptv2's RunStore.seal() writes traj/{trajid,first,last,length} and
    trajectories/{pos,vel,accel,time,trajid} -- no /bounds table, just the
    traj/ index. ZarrScene must read this directly, not just flowtracks'
    own save_zarr_trajectories output."""
    import zarr

    store_path = tmp_path / "run.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    traj_grp = root.create_group("traj")
    traj_grp.create_array("trajid", data=np.array([7, 8], dtype=np.int32))
    traj_grp.create_array("first", data=np.array([10, 10], dtype=np.int32))
    traj_grp.create_array("last", data=np.array([11, 12], dtype=np.int32))
    traj_grp.create_array("length", data=np.array([2, 3], dtype=np.int32))

    trajectories_grp = root.create_group("trajectories")
    trajectories_grp.create_array(
        "pos",
        data=np.array(
            [[0.0, 0, 0], [0.1, 0, 0], [1.0, 1, 1], [1.1, 1, 1], [1.2, 1, 1]]
        ),
    )
    trajectories_grp.create_array(
        "vel", data=np.zeros((5, 3))
    )
    trajectories_grp.create_array("time", data=np.array([10, 11, 10, 11, 12]))
    trajectories_grp.create_array("trajid", data=np.array([7, 7, 8, 8, 8]))

    scene = ZarrScene(store_path)
    assert sorted(scene.trajectory_ids().tolist()) == [7, 8]
    tr7 = scene.trajectory_by_id(7)
    assert len(tr7) == 2
    tr8 = scene.trajectory_by_id(8)
    assert len(tr8) == 3
    assert "accel" not in scene.keys()
