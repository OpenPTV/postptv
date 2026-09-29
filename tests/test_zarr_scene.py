"""ZarrScene behaviour checks against Zarr stores written by
:func:`flowtracks.io.save_zarr_trajectories` and against an openptv2
RunStore-style Zarr layout (traj/ + trajectories/ groups)."""

import numpy as np
import pytest

from flowtracks.io import save_zarr_trajectories
from flowtracks.scene import open_scene
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
def zarr_scene(tmp_path):
    trajects = _make_trajectories()
    zarr_path = tmp_path / "ref.zarr"
    save_zarr_trajectories(trajects, zarr_path)
    return ZarrScene(zarr_path), {t.trajid(): t for t in trajects}


def _sorted_rows(arr):
    return arr[np.lexsort(arr.T)]


def test_keys_and_shapes(zarr_scene):
    scene, _ = zarr_scene
    assert set(scene.keys()) == {"pos", "velocity", "accel"}
    assert dict(zip(scene.keys(), scene.shapes())) == {
        "pos": 3, "velocity": 3, "accel": 3,
    }


def test_trajectory_ids_and_tags(zarr_scene):
    scene, ref = zarr_scene
    assert sorted(scene.trajectory_ids().tolist()) == [1, 2, 3]
    tags = {int(r[0]): (int(r[1]), int(r[2])) for r in scene.trajectory_tags()}
    assert tags == {1: (10, 12), 2: (10, 13), 3: (11, 12)}


def test_trajectory_by_id_matches(zarr_scene):
    scene, ref = zarr_scene
    for trid in [1, 2, 3]:
        z = scene.trajectory_by_id(trid)
        h = ref[trid]
        np.testing.assert_allclose(z.pos(), h.pos())
        np.testing.assert_allclose(z.velocity(), h.velocity())
        np.testing.assert_array_equal(z.time(), h.time())
        assert z.trajid() == h.trajid()


def test_iter_trajectories_matches(zarr_scene):
    scene, ref = zarr_scene
    z_by_id = {t.trajid(): t for t in scene.iter_trajectories()}
    assert set(z_by_id) == set(ref)
    for trid in ref:
        np.testing.assert_allclose(z_by_id[trid].pos(), ref[trid].pos())
        np.testing.assert_array_equal(z_by_id[trid].time(), ref[trid].time())


def test_iter_frames_matches(zarr_scene):
    scene, ref = zarr_scene
    z_frames = {f.time(): f for f in scene.iter_frames()}
    assert set(z_frames) == {10, 11, 12, 13}
    assert sorted(z_frames[10].trajid().tolist()) == [1, 2]
    assert sorted(z_frames[11].trajid().tolist()) == [1, 2, 3]
    np.testing.assert_allclose(
        _sorted_rows(z_frames[10].pos()),
        _sorted_rows(np.array([[0.0, 0, 0], [5.0, 5, 5]])),
    )


def test_frame_by_time_matches(zarr_scene):
    scene, _ = zarr_scene
    z = scene.frame_by_time(11)
    assert sorted(z.trajid().tolist()) == [1, 2, 3]


def test_iter_segments_matches(zarr_scene):
    scene, _ = zarr_scene
    z_segs = list(scene.iter_segments())
    # frames 10..13 -> 3 consecutive pairs
    assert len(z_segs) == 3
    (z_a, z_b) = z_segs[0]
    assert z_a.time() == 10 and z_b.time() == 11
    assert sorted(z_a.trajid().tolist()) == [1, 2]
    assert sorted(z_b.trajid().tolist()) == [1, 2]


def test_collect_matches(zarr_scene):
    scene, _ = zarr_scene
    z_pos, z_time = scene.collect(["pos", "time"])
    assert z_pos.shape == (9, 3)
    assert z_time.shape == (9,)
    assert set(np.unique(z_time).tolist()) == {10, 11, 12, 13}


def test_collect_with_where_matches(zarr_scene):
    scene, _ = zarr_scene
    where = {"time": (11, 13, False)}
    z_pos = scene.collect(["pos"], where=where)[0]
    assert len(z_pos) == 6  # frames 11, 12 across tr1 (2) + tr2 (2) + tr3 (2)


def test_bounding_box_matches(zarr_scene):
    scene, _ = zarr_scene
    z_min, z_max = scene.bounding_box()
    np.testing.assert_allclose(z_min, [0.0, 0, 0])
    np.testing.assert_allclose(z_max, [9.5, 9.5, 9.5])


def test_frame_range_filters(zarr_scene):
    scene, _ = zarr_scene
    scene.set_frame_range((11, 13))
    assert scene.frame_range() == (11, 13)
    z_pos = scene.collect(["time"])[0]
    assert z_pos.min() >= 11 and z_pos.max() < 13


def test_open_scene_dispatches_zarr(tmp_path):
    trajects = _make_trajectories()
    zarr_path = tmp_path / "d.zarr"
    save_zarr_trajectories(trajects, zarr_path)

    assert isinstance(open_scene(str(zarr_path)), ZarrScene)


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
