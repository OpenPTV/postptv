"""
HDF5 / pytables vs. Zarr reader benchmarks (SPEEDUP_PLAN Phase 0 / Phase 2).

Run with::

    uv run pytest benchmarks/ --benchmark-save=<phase>

Reads the real ``data/tracers/ptv_is.*`` corpus, then exercises the HDF5
readers both as text-ingest and as batched ``Scene`` iteration, plus the
Zarr readers on the same trajectories for a direct format comparison.
"""

import numpy as np
import pytest
from flowtracks import io
from flowtracks.scene import Scene

PTVIS = 'data/tracers/ptv_is.%d'
FIRST, LAST = 10001, 10115


@pytest.fixture(scope='module')
def trajectories():
    return list(io.iter_trajectories_ptvis(PTVIS, first=FIRST, last=LAST))


@pytest.fixture(scope='module')
def h5_path(trajectories, tmp_path_factory):
    d = tmp_path_factory.mktemp('bench')
    fname = str(d / 'trajs.h5')
    io.save_particles_table(fname, trajectories)
    return fname


@pytest.fixture(scope='module')
def zarr_path(trajectories, tmp_path_factory):
    d = tmp_path_factory.mktemp('bench_zarr')
    path = d / 'trajs.zarr'
    io.save_zarr_trajectories(trajectories, path)
    return path


def test_iter_trajectories_ptvis(benchmark):
    benchmark(lambda: list(
        io.iter_trajectories_ptvis(PTVIS, first=FIRST, last=LAST)))


def test_trajectories_table(benchmark, h5_path):
    benchmark(lambda: io.trajectories_table(h5_path))


def test_scene_iter_trajectories(benchmark, h5_path):
    scene = Scene(h5_path)
    benchmark(lambda: list(scene.iter_trajectories()))


def test_scene_iter_frames(benchmark, h5_path):
    scene = Scene(h5_path)
    benchmark(lambda: list(scene.iter_frames()))


def test_save_zarr_trajectories(benchmark, trajectories, tmp_path_factory):
    def _save():
        d = tmp_path_factory.mktemp('bench_zarr_save')
        io.save_zarr_trajectories(trajectories, d / 'trajs.zarr')

    benchmark(_save)


def test_read_zarr_trajectories(benchmark, zarr_path):
    benchmark(lambda: io.read_zarr_trajectories(zarr_path))


def test_zarr_matches_hdf5(trajectories, h5_path, zarr_path):
    """Not a benchmark: confirms the two formats round-trip to the same data."""
    from_h5 = sorted(io.trajectories_table(h5_path), key=lambda t: t.trajid())
    from_zarr = sorted(io.read_zarr_trajectories(zarr_path), key=lambda t: t.trajid())

    assert len(from_h5) == len(from_zarr) == len(trajectories)
    for t_h5, t_zarr in zip(from_h5, from_zarr):
        assert t_h5.trajid() == t_zarr.trajid()
        np.testing.assert_allclose(t_h5.pos(), t_zarr.pos())
        np.testing.assert_allclose(t_h5.velocity(), t_zarr.velocity())
        np.testing.assert_allclose(t_h5.time(), t_zarr.time())
