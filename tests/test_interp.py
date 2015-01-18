# -*- coding: utf-8 -*-
"""
Some unit-testing for the interpolation module. Far from full coverage.

Created on Tue Feb  4 11:52:38 2014

@author: yosef
"""

import unittest, os, ConfigParser, numpy as np
from flowtracks import interpolation

class TestReadWrite(unittest.TestCase):
    def test_read_sequence(self):
        """The interpolant read from testing_fodder/ has the right values"""
        fdir = os.path.dirname(__file__)
        fname = os.path.join(fdir, 'testing_fodder/interpolant.cfg')
        interp = interpolation.read_interpolant(fname)
        
        self.failUnlessEqual(interp.num_neighbs(), 4)
        self.failUnlessEqual(interp._method, "inv")
        self.failUnlessEqual(interp._par, 0.1)
    
    def test_write_sequence(self):
        """A test interpolant is faithfully reproduced from rewritten sequence"""
        fdir = os.path.dirname(__file__)
        fname = os.path.join(fdir, 'testing_fodder/interpolant.cfg')
        interp = interpolation.read_interpolant(fname)
                
        cfg = ConfigParser.SafeConfigParser()
        interp.save_config(cfg)
        nfname = os.path.join(fdir, 'testing_fodder/analysis.cfg')
        with open(nfname, 'w') as fobj:
            cfg.write(fobj)
        
        # Round-trip check:
        ninterp = interpolation.read_interpolant(fname)
        self.failUnlessEqual(interp.num_neighbs(), ninterp.num_neighbs())
        self.failUnlessEqual(interp._method, ninterp._method)
        self.failUnlessEqual(interp._par, ninterp._par)
        
        os.remove(nfname)

class TestRepeatedInterp(unittest.TestCase):
    def setUp(self):
        # Tracers are radially placed around one poor particle.
        r = np.r_[0.001, 0.002, 0.003]
        theta = np.r_[:360:45]*np.pi/180
        tracer_pos = np.array((
            r[:,None]*np.cos(theta), r[:,None]*np.sin(theta), 
            np.zeros((len(r), len(theta))) )).transpose().reshape(-1,3)
        self.num_tracers = tracer_pos.shape[0]
        
        interp_points = np.zeros((1,3))
        self.data = np.random.rand(tracer_pos.shape[0], 3)
        
        self.interp = interpolation.Interpolant('inv', 4, 1.5)
        self.interp.set_scene(tracer_pos, interp_points, self.data)
        
    def test_set_scene(self):
        """Scene data recorded and dists/use_parts selected"""
        
        # Truth: use_parts selects the first 4 closest particles.
        # The test_case has 8 almost equally spaced closest neighbs,
        # so the final 4 are selected based on floating-point jitter. Don't
        # fret about it.
        use_parts = self.interp.which_neighbours()
        correct_use_parts = np.zeros((1,self.num_tracers))
        correct_use_parts[0,[0,3,-3,-6]] = True
        np.testing.assert_array_equal(use_parts, correct_use_parts)
    
    def test_interp_once(self):
        """Interpolating a recorded scene"""
        interped = self.interp.interpolate()
        
        # Since all are equally spaced, 
        use_parts = self.interp.which_neighbours()
        correct_interped = self.data[use_parts[0]].mean(axis=0)
        
        np.testing.assert_array_almost_equal(interped[0], correct_interped)
    
    def test_interp_subset(self):
        """Interpolate using a temporary neighbour selection."""
        use_parts = self.interp.which_neighbours()
        use_parts[:,::3] = ~use_parts[:,::3]
        interped = self.interp.interpolate(use_parts)
        
        correct_interped = self.data[use_parts[0]].mean(axis=0)
        np.testing.assert_array_almost_equal(interped[0], correct_interped)
