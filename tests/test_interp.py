# -*- coding: utf-8 -*-
"""
Some unit-testing for the interpolation module. Far from full coverage.

Created on Tue Feb  4 11:52:38 2014

@author: yosef
"""

import unittest, os, ConfigParser
import numpy as np
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
    
class TestJacobian(unittest.TestCase):
    def test_inv(self):
        pos = np.array([[0.,0.,0.]])
        tracer_pos = np.array([
            [ 0.001, 0, 0],
            [-0.001, 0, 0],
            [0,  0.001, 0],
            [0, -0.001, 0],
            [0, 0,  0.001],
            [0, 0, -0.001]
        ])
        # Basically we interpolate something based on the average position
        # change, because it's easy for me to visualize.
        interp_data = tracer_pos*2
        
        interp = interpolation.Interpolant('inv', 6, 3)
        local = interp(tracer_pos, pos, interp_data)
        np.testing.assert_array_equal(local, np.zeros((1,3)))
        
        jac = interp.eulerian_jacobian(tracer_pos, pos, interp_data, local)
        self.failUnless(np.all(jac[:, [0,1,2], [0,1,2]] != 0))
        jac[:, [0,1,2], [0,1,2]] = 0
        self.failUnless(np.all(jac == 0))
        