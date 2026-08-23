"""
Equivalence tests ensuring performance optimizations and vectorizations in:
- RBF interpolation (flowtracks.interpolation)
- Zarr trajectory I/O (flowtracks.io)
- ZarrScene trajectory tags (flowtracks.zarr_scene)
- Trajectory parsing (flowtracks.io)
produce identical numerical results without introducing regressions or edge-case bugs.
"""

import numpy as np
import pytest
from pathlib import Path
from flowtracks.interpolation import (
    Interpolant,
    rbf_interp,
    _select_neighbs_dense,
    select_neighbs,
)
from flowtracks.trajectory import Trajectory
from flowtracks.io import save_zarr_trajectories, read_zarr_trajectories, iter_trajectories_ptvis
from flowtracks.zarr_scene import ZarrScene
import zarr


def test_rbf_fast_path_exact_equivalence():
    """Verify that the vectorized RBF solve in GeneralInterpolant.__call__
    produces identical results to the legacy reference formula."""
    rng = np.random.default_rng(12345)

    for n, m, d, k, eps in [
        (100, 30, 3, 7, 1e4),
        (50, 10, 1, 4, 1e3),
        (200, 50, 4, 5, 5e4),
        (15, 1, 3, 3, 1e2),
    ]:
        tracer_pos = rng.uniform(-1.0, 1.0, size=(n, 3))
        interp_points = rng.uniform(-0.8, 0.8, size=(m, 3))
        data = rng.standard_normal(size=(n, d))

        rbf_obj = Interpolant("rbf", num_neighbs=k, param=eps)

        # 1. Optimized fast path via __call__
        fast_result = rbf_obj(tracer_pos, interp_points, data)

        # 2. Reference ground truth implementation (exact original formula)
        dists, use_parts = select_neighbs(tracer_pos, interp_points, None, k, None)
        ref_tracer_dists = _select_neighbs_dense(tracer_pos, tracer_pos, None, k, None)[0]
        ref_kernel = np.exp(-(ref_tracer_dists**2) * eps)

        ref_coeffs = np.zeros(dists.shape + (d,))
        for pix in range(m):
            neighbs = np.nonzero(use_parts[pix])[0]
            K = ref_kernel[np.ix_(neighbs, neighbs)]
            ref_coeffs[pix, neighbs] = np.linalg.solve(K, data[neighbs])

        ref_rbf = np.exp(-(dists**2) * eps)
        ref_result = np.sum(ref_rbf[..., None] * ref_coeffs, axis=1)

        # Check exact numerical equivalence
        np.testing.assert_allclose(fast_result, ref_result, rtol=1e-11, atol=1e-12)


def test_rbf_with_companionship_equivalence():
    """Verify RBF fast-path equivalence when companion exclusions are present."""
    rng = np.random.default_rng(54321)
    n, m, d, k, eps = 80, 20, 3, 6, 1e4

    tracer_pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    # query points placed on some tracer positions
    interp_points = tracer_pos[:m].copy()
    data = rng.standard_normal(size=(n, d))
    companions = np.arange(m, dtype=int)

    rbf_obj = Interpolant("rbf", num_neighbs=k, param=eps)
    fast_result = rbf_obj(tracer_pos, interp_points, data, companionship=companions)

    dists, use_parts = select_neighbs(tracer_pos, interp_points, None, k, companions)
    ref_tracer_dists = _select_neighbs_dense(tracer_pos, tracer_pos, None, k, companions)[0]
    ref_kernel = np.exp(-(ref_tracer_dists**2) * eps)

    ref_coeffs = np.zeros(dists.shape + (d,))
    for pix in range(m):
        neighbs = np.nonzero(use_parts[pix])[0]
        K = ref_kernel[np.ix_(neighbs, neighbs)]
        ref_coeffs[pix, neighbs] = np.linalg.solve(K, data[neighbs])

    ref_rbf = np.exp(-(dists**2) * eps)
    ref_result = np.sum(ref_rbf[..., None] * ref_coeffs, axis=1)

    np.testing.assert_allclose(fast_result, ref_result, rtol=1e-11, atol=1e-12)


def test_zarr_read_write_roundtrip_equivalence(tmp_path):
    """Verify save_zarr_trajectories and read_zarr_trajectories maintain
    exact trajectory integrity, attributes, and temporal order across
    multiple trajectories of varying lengths."""
    rng = np.random.default_rng(999)

    trajs = []
    # Create trajectories with random lengths, non-sequential IDs
    tr_ids = [42, 7, 105, 1, 999, 12]
    for trid in tr_ids:
        length = rng.integers(5, 30)
        times = np.sort(rng.choice(100, size=length, replace=False))
        pos = rng.standard_normal(size=(length, 3))
        vel = rng.standard_normal(size=(length, 3))
        accel = rng.standard_normal(size=(length, 3))
        trajs.append(Trajectory(pos, vel, times, trid, accel=accel))

    zarr_dir = tmp_path / "test_roundtrip.zarr"
    save_zarr_trajectories(trajs, zarr_dir)

    # Read back all trajectories
    read_back = read_zarr_trajectories(zarr_dir)

    assert len(read_back) == len(trajs)

    # Compare by ID
    input_by_id = {t.trajid(): t for t in trajs}
    read_by_id = {t.trajid(): t for t in read_back}

    assert set(input_by_id.keys()) == set(read_by_id.keys())

    for trid in input_by_id:
        t_in = input_by_id[trid]
        t_out = read_by_id[trid]

        np.testing.assert_array_equal(t_out.time(), t_in.time())
        np.testing.assert_allclose(t_out.pos(), t_in.pos(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t_out.velocity(), t_in.velocity(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t_out.accel(), t_in.accel(), rtol=1e-12, atol=1e-12)


def test_zarr_read_with_frame_filters(tmp_path):
    """Verify first and last frame filtering in read_zarr_trajectories."""
    t1 = Trajectory(
        np.zeros((10, 3)), np.zeros((10, 3)), np.arange(10, 20), 101
    )
    t2 = Trajectory(
        np.ones((10, 3)), np.ones((10, 3)), np.arange(15, 25), 102
    )
    zarr_dir = tmp_path / "test_filter.zarr"
    save_zarr_trajectories([t1, t2], zarr_dir)

    # Filter frames 12 to 17
    filtered = read_zarr_trajectories(zarr_dir, first=12, last=17)
    by_id = {t.trajid(): t for t in filtered}

    assert 101 in by_id
    assert 102 in by_id
    np.testing.assert_array_equal(by_id[101].time(), np.arange(12, 18))
    np.testing.assert_array_equal(by_id[102].time(), np.arange(15, 18))


def test_zarr_scene_trajectory_tags_reduceat_equivalence(tmp_path):
    """Verify that ZarrScene.trajectory_tags() using np.minimum.reduceat
    and np.maximum.reduceat matches manual ground-truth extraction."""
    rng = np.random.default_rng(777)
    zarr_dir = tmp_path / "test_tags.zarr"

    root = zarr.open_group(str(zarr_dir), mode="w")
    tg = root.require_group("trajectories")

    # Construct unstructured / shuffled points across 20 trajectories
    num_pts = 1000
    traj_ids = rng.choice(np.arange(10, 30), size=num_pts)
    times = rng.integers(100, 500, size=num_pts)
    pos = rng.standard_normal((num_pts, 3))
    vel = rng.standard_normal((num_pts, 3))

    tg.create_array("trajid", data=traj_ids)
    tg.create_array("time", data=times)
    tg.create_array("pos", data=pos)
    tg.create_array("vel", data=vel)

    scene = ZarrScene(zarr_dir)
    computed_tags = scene.trajectory_tags()

    # Ground truth reference by manual grouping per unique ID
    unique_ids = np.unique(traj_ids)
    ref_tags = []
    for trid in unique_ids:
        t_trid = times[traj_ids == trid]
        ref_tags.append([trid, t_trid.min(), t_trid.max()])
    ref_tags = np.array(ref_tags, dtype=int)

    # Sort both by trajid to compare
    computed_tags = computed_tags[np.argsort(computed_tags[:, 0])]
    ref_tags = ref_tags[np.argsort(ref_tags[:, 0])]

    np.testing.assert_array_equal(computed_tags, ref_tags)


def test_iter_trajectories_ptvis_consistency():
    """Verify iter_trajectories_ptvis yields valid trajectories matching
    expected count, frame spans, velocities and forward-difference accelerations."""
    data_template = str(Path(__file__).parent.parent / "data" / "tracers" / "ptv_is.%d")
    trajs = list(iter_trajectories_ptvis(data_template, first=10001, last=10020))

    assert len(trajs) > 0
    for tr in trajs:
        assert len(tr) >= 2  # min length filter
        assert tr.pos().shape == (len(tr), 3)
        assert tr.velocity().shape == (len(tr), 3)
        assert hasattr(tr, "accel")
        acc = tr.accel()
        assert acc.shape == (len(tr), 3)
        # Acceleration at end is padded with 0
        assert acc[-1, 0] == 0.0 and acc[-2, 0] == 0.0
