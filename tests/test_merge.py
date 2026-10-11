# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
import numpy as np
import pytest

from flowtracks.merge import (
    apply_rigid,
    best_partners,
    duplicates,
    estimate_rigid_map,
    overlap_counts,
)

YAW180 = np.diag([-1.0, 1.0, -1.0])
ROTATIONS = [("yaw 0", np.eye(3)), ("yaw 180", YAW180)]


def _two_systems(rng, offset, n_frames=40, n=200, noise=0.5):
    """Same particles seen by A and by B; B's frame is A's turned 180 deg about
    y and shifted: p_B = R^T (p_A + offset)."""
    pairs = []
    for _ in range(n_frames):
        pa = rng.uniform(-1000, 1000, (n, 3))
        pb = (pa + offset) @ YAW180 + rng.normal(0, noise, (n, 3))
        extra_a = rng.uniform(-1000, 1000, (n // 2, 3))  # seen by A only
        pairs.append((np.r_[pa, extra_a], pb))
    return pairs


def test_estimate_rigid_map_finds_rotation_and_offset():
    rng = np.random.default_rng(1)
    offset = np.array([-1.1, 0.6, 8.9])
    got = estimate_rigid_map(_two_systems(rng, offset), ROTATIONS, log=None)
    assert got["label"] == "yaw 180"
    assert np.allclose(got["matrix"], YAW180)
    assert np.allclose(got["offset"], offset, atol=0.1)
    assert got["n_pairs"] > 1000 and np.all(got["spread"] < 1.0)
    pa, pb = _two_systems(rng, offset, n_frames=1)[0]
    back = apply_rigid(pb, got["matrix"], got["offset"])  # B's points in A's frame
    assert np.abs(back - pa[: len(pb)]).max() < 4.0  # within the 0.5 noise


def test_estimate_rigid_map_refuses_unrelated_systems():
    rng = np.random.default_rng(2)
    pairs = [(rng.uniform(-1000, 1000, (100, 3)), rng.uniform(-1000, 1000, (100, 3)))
             for _ in range(20)]
    with pytest.raises(ValueError, match="no common"):
        estimate_rigid_map(pairs, ROTATIONS, log=None)


def _track(t, x):
    t = np.asarray(t)
    return {"time": t, "pos": np.column_stack([np.full(len(t), x), t * 1.0, np.zeros(len(t))])}


def test_overlaps_partners_and_duplicates():
    tracks_a = {1: _track(range(0, 10), 0.0), 2: _track(range(0, 3), 100.0)}
    tracks_b = {10: _track(range(5, 9), 0.5),     # shares 4 points with a1
                11: _track(range(0, 6), 100.2),   # shares 3 with a2, longer than a2
                12: _track(range(0, 5), 500.0)}   # alone
    shared = overlap_counts(tracks_a, tracks_b, tol=1.0)
    assert shared == {(1, 10): 4, (2, 11): 3}
    pa, pb = best_partners(shared)
    assert pa == {1: (10, 4), 2: (11, 3)} and pb == {10: (1, 4), 11: (2, 3)}
    lens_a = {k: len(v["time"]) for k, v in tracks_a.items()}
    lens_b = {k: len(v["time"]) for k, v in tracks_b.items()}
    assert duplicates(shared, lens_a, lens_b) == {10}  # 11 is the longer copy
    assert overlap_counts(tracks_a, tracks_b, tol=0.4) == {(2, 11): 3}
