# -*- coding: utf-8 -*-
"""
Some unit-testing for the interpolation module. Far from full coverage.

Created on Tue Feb  4 11:52:38 2014

@author: yosef
"""

import unittest
import os
import configparser
import numpy as np
from flowtracks import interpolation


class TestReadWrite(unittest.TestCase):
    def test_read_sequence(self):
        """The interpolant read from testing_fodder/ has the right values"""
        fdir = os.path.dirname(__file__)
        fname = os.path.join(fdir, 'testing_fodder/interpolant.cfg')
        interp = interpolation.read_interpolant(fname)
        
        self.assertEqual(interp.num_neighbs(), 4)
        self.assertEqual(interp._method, "inv")
        self.assertEqual(interp._par, 0.1)
    
    def test_write_sequence(self):
        """A test interpolant is faithfully reproduced from rewritten 
        sequence"""
        fdir = os.path.dirname(__file__)
        fname = os.path.join(fdir, 'testing_fodder/interpolant.cfg')
        interp = interpolation.read_interpolant(fname)
                
        cfg = configparser.ConfigParser()
        interp.save_config(cfg)
        nfname = os.path.join(fdir, 'testing_fodder/analysis.cfg')
        with open(nfname, 'w') as fobj:
            cfg.write(fobj)
        
        # Round-trip check:
        ninterp = interpolation.read_interpolant(fname)
        self.assertEqual(interp.num_neighbs(), ninterp.num_neighbs())
        self.assertEqual(interp._method, ninterp._method)
        self.assertEqual(interp._par, ninterp._par)

        os.remove(nfname)


class TestRepeatedInterp(unittest.TestCase):
    def setUp(self):
        # Tracers are radially placed around one poor particle.
        r = np.r_[0.001, 0.002, 0.003]
        theta = np.r_[:360:45]*np.pi/180
        tracer_pos = np.array((
            r[:, None]*np.cos(theta), r[:, None]*np.sin(theta), 
            np.zeros((len(r), len(theta))))).transpose().reshape(-1, 3)
        self.num_tracers = tracer_pos.shape[0]
        
        interp_points = np.zeros((1, 3))
        self.data = np.random.rand(tracer_pos.shape[0], 3)
        
        self.interp = interpolation.interpolant('inv', 4, param=1.5)
        self.interp.set_scene(tracer_pos, interp_points, self.data)
            
    def test_set_scene(self):
        """Scene data recorded and dists/use_parts selected"""
        
        # Truth: use_parts selects the first 4 closest particles.
        # The test_case has 8 almost equally spaced closest neighbs,
        # so the final 4 are selected based on floating-point jitter. Don't
        # fret about it.
        use_parts = self.interp.current_active_neighbs()
        correct_use_parts = np.array([[0, 3, 6, 9, 12, 15, 18, 21]])
        used_in_correct = use_parts[:, None, :] == correct_use_parts[:, :, 
                                                                     None]
        self.assertEqual(
            used_in_correct.any(axis=2).sum(axis=1), self.interp.num_neighbs())
    
    def test_interp_once(self):
        """Interpolating a recorded scene"""
        interped = self.interp.interpolate()
        
        # Since all are equally spaced, 
        use_parts = self.interp.current_active_neighbs()
        correct_interped = self.data[use_parts[0]].mean(axis=0)
        
        np.testing.assert_array_almost_equal(interped[0], correct_interped)
    
    def test_interp_subset(self):
        """Interpolate using a temporary neighbour selection."""
        use_parts = self.interp.current_active_neighbs().copy()
        use_parts[:, ::3] = ~use_parts[:, ::3]
        interped = self.interp.interpolate(use_parts)
        
        correct_interped = self.data[use_parts[0]].mean(axis=0)
        np.testing.assert_array_almost_equal(interped[0], correct_interped)
    
    def test_trim_scene(self):
        """Dropping particles from the interpolated scene"""
        self.interp.trim_points(np.r_[True])
        # Now the scene is empty, so we expect empty arrays
        self.assertEqual(self.interp.interpolate().shape[0], 0.)


class MethodInterp(unittest.TestCase):
    def test_interp_rbf(self):
        """Interpolating with rbf method finishes"""
        r = np.r_[0.001, 0.002, 0.003]
        theta = np.r_[:360:45]*np.pi/180
        tracer_pos = np.array((
            r[:, None]*np.cos(theta), r[:, None]*np.sin(theta), 
            np.zeros((len(r), len(theta))))).transpose().reshape(-1, 3)
        
        interp_points = np.zeros((1, 3))
        data = np.random.rand(tracer_pos.shape[0], 3)
        
        interp = interpolation.interpolant('rbf', 4, param=1e5)
        interp.set_scene(tracer_pos, interp_points, data)
        
        # interped = interp.interpolate()
        # use_parts = interp.current_active_neighbs()
        # If we reached this line, we tested what we wanted.

       
class RadiusInterp(unittest.TestCase):
    def test_radius(self):
        """finding neighbours by radius"""
        r = np.r_[0.001, 0.002, 0.003]
        theta = np.r_[:360:45]*np.pi/180
        tracer_pos = np.array((
            r[:, None]*np.cos(theta), r[:, None]*np.sin(theta), 
            np.zeros((len(r), len(theta))))).transpose().reshape(-1, 3)
        
        interp_points = np.zeros((1, 3))
        data = np.random.rand(tracer_pos.shape[0], 3)
        
        interp = interpolation.interpolant(
            'inv', num_neighbs=8, radius=0.0015, param=1.5)
        interp.set_scene(tracer_pos, interp_points, data)
        
        interped = interp.interpolate()
        correct_interped = data[::3].mean(axis=0)
        
        np.testing.assert_array_almost_equal(interped[0], correct_interped)


class TestJacobian(unittest.TestCase):
    def test_inv(self):
        pos = np.array([[0., 0., 0.]])
        tracer_pos = np.array([
            [0.001, 0, 0],
            [-0.001, 0, 0],
            [0,  0.001, 0],
            [0, -0.001, 0],
            [0, 0,  0.001],
            [0, 0, -0.001]
        ])
        # Basically we interpolate something based on the average position
        # change, because it's easy for me to visualize.
        interp_data = tracer_pos*2

        interp = interpolation.interpolant('inv', 6, 3)
        interp.set_scene(tracer_pos, pos, interp_data)

        local = interp.interpolate()
        np.testing.assert_array_equal(local, np.zeros((1, 3)))

        jac = interp.eulerian_jacobian()
        self.assertTrue(np.all(jac[:, [0, 1, 2], [0, 1, 2]] != 0))

        # Above test is symmetric. This would catch derivation direction
        # bugs:
        np.testing.assert_array_equal(np.sign(jac[:, [0, 1, 2], [0, 1, 2]]),
                                      np.ones((1, 3)))
        
        # Check compared to numeric:
        numeric = interpolation.GeneralInterpolant.eulerian_jacobian(
            interp, eps=1e-6)
        np.testing.assert_array_almost_equal(jac, numeric)

        # Non-diagonal elements:
        jac[:, [0, 1, 2], [0, 1, 2]] = 0
        self.assertTrue(np.all(jac == 0))


class TestCompanion(unittest.TestCase):
    def test_select(self):
        """Selecting neighbours excludes correct companions"""
        tracer_pos = np.array([
            [0.001, 0, 0],
            [-0.001, 0, 0],
            [0,  0.001, 0],
            [0, -0.001, 0],
            [0, 0,  0.001],
            [0, 0, -0.001]
        ])
        pos = np.array([
            [0,  0.0009, 0],
            [0, -0.0009, 0],
        ])
        companions = np.r_[2, 3]
        
        dist, use_parts = interpolation.select_neighbs(
            tracer_pos, pos, num_neighbs=4, companionship=companions)
        
        np.testing.assert_array_equal(use_parts, np.array([
            [True, True, False, False, True, True],
            [True, True, False, False, True, True]
        ]))
