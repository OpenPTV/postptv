"""read_zarr_trajectories must trust the tracker linkage over derived caches."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from flowtracks.io import read_zarr_trajectories  # noqa: E402


def _write_linkage(root, frame, prev, pos):
    g = root.require_group("linkage/ptv_is/frame_%05d" % frame)
    g.create_array("prev", data=np.asarray(prev, dtype=np.int32))
    g.create_array("next", data=np.full(len(prev), -1, dtype=np.int32))
    g.create_array("pos", data=np.asarray(pos, dtype=np.float64))


def test_stale_trajectories_group_does_not_shadow_linkage(tmp_path):
    """A post-processing cache must not be replayed instead of fresh linkage."""
    root = zarr.open_group(str(tmp_path / "run.zarr"), mode="a")
    # Fresh tracker output: one particle moving 1 unit/frame over 3 frames.
    for i, frame in enumerate((1, 2, 3)):
        _write_linkage(root, frame, [-1 if i == 0 else 0], [[i * 1.0, 0.0, 0.0]])
    # Stale cache from an older, buggy run: a 50-unit jump.
    grp = root.require_group("trajectories")
    grp.create_array("pos", data=np.array([[0.0, 0, 0], [50.0, 0, 0]]))
    grp.create_array("vel", data=np.zeros((2, 3)))
    grp.create_array("time", data=np.array([1, 2]))
    grp.create_array("trajid", data=np.array([1, 1]))

    trajs = read_zarr_trajectories(str(tmp_path / "run.zarr"))
    assert len(trajs) == 1
    steps = np.linalg.norm(np.diff(np.asarray(trajs[0].pos()), axis=0), axis=1)
    assert steps.max() == pytest.approx(1.0)  # linkage won, not the 50-unit cache


def test_frame_gap_breaks_the_chain(tmp_path):
    """`prev` indexes the immediately previous frame; a gap makes it meaningless."""
    root = zarr.open_group(str(tmp_path / "run.zarr"), mode="a")
    _write_linkage(root, 1, [-1], [[0.0, 0, 0]])
    _write_linkage(root, 5, [0], [[900.0, 0, 0]])  # would fake a 900-unit link

    assert read_zarr_trajectories(str(tmp_path / "run.zarr")) == []
