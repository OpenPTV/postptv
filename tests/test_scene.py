"""
Test the input/output routines in flowtracks: read/write files and verify
correct results (Zarr backend).
"""

import tempfile
import unittest

import numpy as np
import numpy.testing as nptest

from flowtracks.io import save_zarr_trajectories
from flowtracks.scene import gen_query_string
from flowtracks.trajectory import Trajectory, take_snapshot
from flowtracks.zarr_scene import ZarrScene


class TestScene(unittest.TestCase):
    def setUp(self):
        """
        There are 3 trajectories of 4 frames each, each going along a separate
        axis.
        """
        # ptv_is files are in [mm], [mm/s], etc.
        correct_pos = np.r_[0.1, 0.2, 0.3, 0.5] / 1000.
        correct_vel = np.r_[0.1, 0.1, 0.2, 0.] / 1000.
        correct_accel = np.r_[0., 0.1, 0., 0.] / 1000.
        t = np.r_[1:5] + 10000

        self.correct = []
        for axis in [0, 1, 2]:
            pos = np.zeros((4, 3))
            pos[:, axis] = correct_pos

            vel = np.zeros((4, 3))
            vel[:, axis] = correct_vel

            accel = np.zeros((4, 3))
            accel[:, axis] = correct_accel

            self.correct.append(Trajectory(pos, vel, t, len(self.correct),
                accel=accel))

        self._tmpdir = tempfile.TemporaryDirectory()
        self._zarr_path = self._tmpdir.name + "/three_trajects_simple.zarr"
        save_zarr_trajectories(self.correct, self._zarr_path)
        self.scene = ZarrScene(self._zarr_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_keys(self):
        """Reading known available keys with the needed exclusions"""
        self.assertEqual(set(self.scene.keys()), {'velocity', 'pos', 'accel'})

    def test_iter_trajectories(self):
        """Iterating trajectories and getting correct Trajectory objects"""
        trjs = [tr for tr in self.scene.iter_trajectories()]
        self.assertEqual(len(trjs), len(self.correct))

        for trj, correct in zip(
            sorted(trjs, key=lambda t: t.trajid()),
            sorted(self.correct, key=lambda t: t.trajid()),
        ):
            nptest.assert_array_almost_equal(trj.pos(), correct.pos())
            nptest.assert_array_almost_equal(trj.velocity(), correct.velocity())
            nptest.assert_array_almost_equal(trj.accel(), correct.accel())
            nptest.assert_array_almost_equal(trj.time(), correct.time())
            self.assertEqual(trj.trajid(), correct.trajid())

    def test_iter_trajectories_subrange(self):
        """Iterating trajectories in part of the frame range."""
        self.scene.set_frame_range((10002, 10004))
        trjs = [tr for tr in self.scene.iter_trajectories()]
        self.assertEqual(len(trjs), len(self.correct))

        for trj, correct in zip(
            sorted(trjs, key=lambda t: t.trajid()),
            sorted(self.correct, key=lambda t: t.trajid()),
        ):
            nptest.assert_array_almost_equal(trj.pos(), correct.pos()[1:-1])
            nptest.assert_array_almost_equal(trj.velocity(), correct.velocity()[1:-1])
            nptest.assert_array_almost_equal(trj.accel(), correct.accel()[1:-1])
            nptest.assert_array_almost_equal(trj.time(), correct.time()[1:-1])
            self.assertEqual(trj.trajid(), correct.trajid())

    def test_iter_frames(self):
        """Iterating the store by frames, getting correct ParticleSnapshot objects"""
        schm = self.correct[0].schema()
        correct_frames = [take_snapshot(self.correct, frm, schm)
            for frm in range(10001, 10005)]

        frames = [frm for frm in self.scene.iter_frames()]
        self.assertEqual(len(frames), len(correct_frames))

        for frm, correct in zip(frames, correct_frames):
            nptest.assert_array_almost_equal(
                frm.pos()[np.argsort(frm.trajid())],
                correct.pos()[np.argsort(correct.trajid())])
            nptest.assert_array_almost_equal(
                frm.velocity()[np.argsort(frm.trajid())],
                correct.velocity()[np.argsort(correct.trajid())])
            nptest.assert_array_almost_equal(
                frm.accel()[np.argsort(frm.trajid())],
                correct.accel()[np.argsort(correct.trajid())])
            nptest.assert_array_equal(
                np.sort(frm.trajid()), np.sort(correct.trajid()))
            self.assertEqual(frm.time(), correct.time())

    def test_iter_frames_subrange(self):
        """Iterating frames subrange"""
        self.scene.set_frame_range((10002, 10004))

        schm = self.correct[0].schema()
        correct_frames = [take_snapshot(self.correct, frm, schm)
            for frm in range(10002, 10004)]

        frames = [frm for frm in self.scene.iter_frames()]
        self.assertEqual(len(frames), len(correct_frames))

        for frm, correct in zip(frames, correct_frames):
            nptest.assert_array_almost_equal(
                frm.pos()[np.argsort(frm.trajid())],
                correct.pos()[np.argsort(correct.trajid())])
            nptest.assert_array_almost_equal(
                frm.velocity()[np.argsort(frm.trajid())],
                correct.velocity()[np.argsort(correct.trajid())])
            nptest.assert_array_almost_equal(
                frm.accel()[np.argsort(frm.trajid())],
                correct.accel()[np.argsort(correct.trajid())])
            nptest.assert_array_equal(
                np.sort(frm.trajid()), np.sort(correct.trajid()))
            self.assertEqual(frm.time(), correct.time())

    def test_collect(self):
        """Slicing the store by keys or expressions"""
        v = self.scene.collect(['velocity'])[0]
        self.assertEqual(v.shape, (12, 3))

        by_id = {}
        for trj in self.scene.iter_trajectories():
            by_id[trj.trajid()] = trj.velocity()
        v_stacked = np.stack(
            [by_id[i] for i in sorted(by_id)], axis=0)
        nptest.assert_array_almost_equal(v_stacked[0, :, 0], v_stacked[1, :, 1])
        nptest.assert_array_almost_equal(v_stacked[0, :, 0], v_stacked[2, :, 2])

class TestUtils(unittest.TestCase):
    def test_query_string(self):
        self.assertEqual(gen_query_string('example', (-1, 1, False)),
            "((example >= -1) & (example < 1))")
        self.assertEqual(gen_query_string('example', (-1, 1, True)),
            '((example < -1) | (example >= 1))')
