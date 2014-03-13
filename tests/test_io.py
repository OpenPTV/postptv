"""
Test the input/output routimes in flowtracks: read/write files and verify 
correct results.
"""

import unittest
import numpy as np, numpy.testing as nptest

from flowtracks import io
from flowtracks.trajectory import Trajectory

class TestPtvis(unittest.TestCase):
    def setUp(self):
        """
        There are 3 trajectories of 4 frames each, each going along a separate
        axis.
        """
        self.tmpl = "testing_fodder/ptvis/ptv_is.%d"
        self.first = 10001
        self.last = 10004
        
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
    
    def test_trajectories_ptvis(self):
        trjs = io.trajectories_ptvis(self.tmpl, self.first, self.last)
        self.failUnlessEqual(len(trjs), len(self.correct))
        
        for trj, correct in zip(trjs, self.correct):
            nptest.assert_array_almost_equal(trj.pos(), correct.pos())
            nptest.assert_array_almost_equal(trj.velocity(), correct.velocity())
            nptest.assert_array_almost_equal(trj.accel(), correct.accel())
            nptest.assert_array_almost_equal(trj.time(), correct.time())
            self.failUnlessEqual(trj.trajid(), correct.trajid())
        
