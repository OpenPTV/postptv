# -*- coding: utf-8 -*-
"""
Some unit-testing for the sequence module. Far far away from full coverage.

Created on Tue Feb  4 11:52:38 2014

@author: yosef
"""

import unittest, os

try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
    
from flowtracks import sequence

class TestReadWrite(unittest.TestCase):
    def test_read_sequence(self):
        """The sequence read from testing_fodder/ has the right values"""
        fdir = os.path.dirname(__file__)
        fname = os.path.join(fdir, 'testing_fodder/sequence.cfg')
        seq = sequence.read_sequence(fname)
        
        self.assertEqual(seq.frate, 500)
        self.assertEqual(seq.range(), (10000, 10201))
        self.assertEqual(seq.part_fname(), '../data/particles/xuap.%d')
    
    def test_write_sequence(self):
        """A test sequence is faithfully reproduced from rewritten sequence"""
        fdir = os.path.dirname(__file__)
        fname = os.path.join(fdir, 'testing_fodder/sequence.cfg')
        seq = sequence.read_sequence(fname)
        
        cfg = ConfigParser()
        seq.save_config(cfg)
        nfname = os.path.join(fdir, 'testing_fodder/analysis.cfg')
        with open(nfname, 'w') as fobj:
            cfg.write(fobj)
        
        # Round-trip check:
        nseq = sequence.read_sequence(nfname)
        self.assertEqual(seq.frate, nseq.frate)
        self.assertEqual(seq.range(), nseq.range())
        self.assertEqual(seq.part_fname(), nseq.part_fname())
        
        os.remove(nfname)
