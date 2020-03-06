"""
Test the input/output routines in flowtracks: read/write files and verify 
correct results.
"""

import unittest, os
import numpy as np, numpy.testing as nptest

from flowtracks import io
from flowtracks.trajectory import Trajectory

class TestPtvis(unittest.TestCase):
    def setUp(self):
        """
        There are 3 trajectories of 4 frames each, each going along a separate
        axis.
        """
        self.tmpl = "tests/testing_fodder/ptvis/ptv_is.%d"
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
        self.assertEqual(len(trjs), len(self.correct))
        self.compare_trajectories(trjs, self.correct)
        
    def compare_trajectories(self, list1, list2):
        """
        Compare two lists of trajectories, fail if not the same.
        """
        for trj, correct in zip(list1, list2):
            nptest.assert_array_almost_equal(trj.pos(), correct.pos())
            nptest.assert_array_almost_equal(trj.velocity(), correct.velocity())
            nptest.assert_array_almost_equal(trj.accel(), correct.accel())
            nptest.assert_array_almost_equal(trj.time(), correct.time())
            self.assertEqual(trj.trajid(), correct.trajid())
            
    def test_iter_trajectories_ptvis_xuap(self):
    	inName = './data/particles/xuap.%d'
    	trjs = io.trajectories_ptvis(inName, xuap = True, traj_min_len = 2)
    	self.assertEqual(len(trjs), 332)
    	
    	# here it fails due to the same problem with traj_min_len
    	trjs = io.trajectories_ptvis(inName, xuap = True, traj_min_len = None)
    	self.assertEqual(len(trjs), 332)
    	
    def test_trajectories_hdf(self):
        """HDF reading works"""
        outfile = 'tests/testing_fodder/table.h5'
        io.save_particles_table(outfile, self.correct)
        loaded = io.trajectories_table(outfile)
        self.compare_trajectories(loaded, self.correct)
        os.remove(outfile)
    
    def test_trim_hdf(self):
        """HDF trajectory trimming"""
        correct = io.trajectories_ptvis(self.tmpl, self.first+1, self.last-1)
        
        outfile = 'tests/testing_fodder/table.h5'
        io.save_particles_table(outfile, self.correct, trim=1)
        
        loaded = io.trajectories_table(outfile)
        self.compare_trajectories(loaded, correct)
        
        os.remove(outfile)

