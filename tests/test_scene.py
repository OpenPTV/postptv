"""
Test the input/output routines in flowtracks: read/write files and verify 
correct results.
"""

import unittest
import numpy as np, numpy.testing as nptest

from flowtracks.scene import Scene
from flowtracks.trajectory import Trajectory, take_snapshot

class TestScene(unittest.TestCase):
    def setUp(self):
        """
        There are 3 trajectories of 4 frames each, each going along a separate
        axis.
        """
        tmpl = "tests/testing_fodder/three_trajects_simple.h5"
        self.scene = Scene(tmpl)
        
        # ptv_is files are in [mm], [mm/s], etc.
        correct_pos = np.r_[0.1, 0.2, 0.3, 0.5] / 1000.
        correct_vel = np.r_[0.1, 0.1, 0.2, 0.] / 1000.
        correct_accel = np.r_[0., 0.1, 0., 0.] / 1000.
        t = np.r_[1:5] + 10000
        
        self.correct = []
        for axis in [0, 1, 2]:
            pos = np.zeros((4, 3))
            pos[:,axis] = correct_pos
            
            vel = np.zeros((4, 3))
            vel[:,axis] = correct_vel
            
            accel = np.zeros((4, 3))
            accel[:,axis] = correct_accel
            
            self.correct.append(Trajectory(pos, vel, t, len(self.correct),
                accel=accel))
    
    def test_keys(self):
        """Reading known available keys with the needed exclusions"""
        correct_keys = ['velocity', 'pos', 'accel']
        self.failUnlessEqual(self.scene.keys(), correct_keys)
        
    def test_iter_trajectories(self):
        """Iterating trajectories and getting correct Trajectory objects"""
        trjs = [tr for tr in self.scene.iter_trajectories()]
        self.failUnlessEqual(len(trjs), len(self.correct))
        
        for trj, correct in zip(trjs, self.correct):
            nptest.assert_array_almost_equal(trj.pos(), correct.pos())
            nptest.assert_array_almost_equal(trj.velocity(), correct.velocity())
            nptest.assert_array_almost_equal(trj.accel(), correct.accel())
            nptest.assert_array_almost_equal(trj.time(), correct.time())
            self.failUnlessEqual(trj.trajid(), correct.trajid())
    
    def test_iter_trajectories_subrange(self):
        """Iterating trajectories in part of the frame range."""
        self.scene.set_frame_range((10002, 10004))
        trjs = [tr for tr in self.scene.iter_trajectories()]
        self.failUnlessEqual(len(trjs), len(self.correct))
        
        for trj, correct in zip(trjs, self.correct):
            nptest.assert_array_almost_equal(trj.pos(), correct.pos()[1:-1])
            nptest.assert_array_almost_equal(trj.velocity(), correct.velocity()[1:-1])
            nptest.assert_array_almost_equal(trj.accel(), correct.accel()[1:-1])
            nptest.assert_array_almost_equal(trj.time(), correct.time()[1:-1])
            self.failUnlessEqual(trj.trajid(), correct.trajid())
     
    def test_iter_frames(self):
        """Iterating the HDF files by frames, and getting correct ParticleSnapshot objects"""
        schm = self.correct[0].schema()
        correct_frames = [take_snapshot(self.correct, frm, schm) \
            for frm in xrange(10001, 10005)]
        
        frames = [frm for frm in self.scene.iter_frames()]
        self.failUnlessEqual(len(frames), len(correct_frames))
        
        for frm, correct in zip(frames, correct_frames):
            nptest.assert_array_almost_equal(frm.pos(), correct.pos())
            nptest.assert_array_almost_equal(frm.velocity(), correct.velocity())
            nptest.assert_array_almost_equal(frm.accel(), correct.accel())
            nptest.assert_array_almost_equal(frm.trajid(), correct.trajid())
            self.failUnlessEqual(frm.time(), correct.time())
    
    def test_iter_frames_subrange(self):
        """Iterating frames subrange"""
        self.scene.set_frame_range((10002, 10004))
        
        schm = self.correct[0].schema()
        correct_frames = [take_snapshot(self.correct, frm, schm) \
            for frm in xrange(10002, 10004)]
        
        frames = [frm for frm in self.scene.iter_frames()]
        self.failUnlessEqual(len(frames), len(correct_frames))
        
        for frm, correct in zip(frames, correct_frames):
            nptest.assert_array_almost_equal(frm.pos(), correct.pos())
            nptest.assert_array_almost_equal(frm.velocity(), correct.velocity())
            nptest.assert_array_almost_equal(frm.accel(), correct.accel())
            nptest.assert_array_almost_equal(frm.trajid(), correct.trajid())
            self.failUnlessEqual(frm.time(), correct.time())

