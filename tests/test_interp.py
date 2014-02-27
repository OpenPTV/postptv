# -*- coding: utf-8 -*-
"""
Some unit-testing for the interpolation module. Far from full coverage.

Created on Tue Feb  4 11:52:38 2014

@author: yosef
"""

import unittest, os, ConfigParser
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
