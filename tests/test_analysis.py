"""
Unit tests for a small part of the analysis code.
"""

import unittest, numpy as np
from flowtracks.analysis import FluidVelocitiesAnalyser
from flowtracks.trajectory import ParticleSnapshot, Frame
from flowtracks.interpolation import Interpolant

class TestFluidVels(unittest.TestCase):
    def setUp(self):
        tracer_pos = np.array([
            [ 0.001, 0, 0],
            [-0.001, 0, 0],
            [0,  0.001, 0],
            [0, -0.001, 0],
            [0, 0,  0.001],
            [0, 0, -0.001]
        ])
        tracer_vel = np.array([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 2., 0.],
            [0., 2., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
        ])
        tracers = ParticleSnapshot(
            tracer_pos, tracer_vel, trajid=np.arange(6), time=1.)
        
        self.pos = np.array([
            [0,  0.0009, 0],
            [0, -0.0009, 0],
        ])
        self.companions = np.r_[2, 3]
        
        self.frm = Frame()
        self.frm.tracers = tracers
        
        self.analyser = FluidVelocitiesAnalyser(Interpolant('inv'))
        
    def test_vel_interp(self):
        """Interpolating fluid velocity"""
        particles =  ParticleSnapshot(
            self.pos, np.zeros((2,3)), trajid=np.arange(2), time=1.)
        self.frm.particles = particles 
        
        fvel, _ = self.analyser.analyse(self.frm, None)
        correct_fvel = np.array([
            [ 0., 1.81766935, 0.],
            [ 0., 1.81766935, 0.]
        ])
        np.testing.assert_array_almost_equal(fvel, correct_fvel)
        
    def test_comp_interp(self):
        """Interpolating fluid velocity, excluding companion."""
        particles =  ParticleSnapshot(
            self.pos, np.zeros((2,3)), trajid=np.arange(2), time=1.,
            companion=self.companions)
        self.frm.particles = particles
        
        fvel, _ = self.analyser.analyse(self.frm, None)
        correct_fvel = np.array([
            [ 0., 1., 0.],
            [ 0., 1., 0.]
        ])
        np.testing.assert_array_almost_equal(fvel, correct_fvel)
