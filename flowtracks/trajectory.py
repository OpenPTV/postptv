# -*- coding: utf-8 -*-

import numpy as np
import scipy.interpolate as interp

class Trajectory(object):
    def __init__(self, pos, velocity, time, trajid):
        """
        Arguments:
        pos - a (t,3) array, the position of one particle in t time-points,
            [m].
        velocity - (t,3) array, corresponding velocity, [m/s].
        time - (t,) array, the clock ticks. No specific units needed.
        trajid - the unique identifier of this trajectory in the set of
            trajectories that belong to the same sequence.
        """
        self._pos = pos
        self._vel = velocity
        self._t = time
        self._id = trajid
    
    def pos(self):
        return self._pos
    
    def velocity(self):
        return self._vel
    
    def time(self):
        return self._t
    
    def trajid(self):
        return self._id
    
    def __len__(self):
        return len(self._t)
    
    def __getitem__(self, selector):
        """
        Gets the data for time points selected as a table of shape (n,8),
        concatenating position, velocity, time, broadcasted trajid.
        
        Arguments:
        selector - any 1d indexing expression known to numpy.
        """
        return np.hstack((self._pos[selector], self._vel[selector],
            self._t[selector][...,None], 
            np.ones(self._pos[selector].shape[:-1] + (1,))*self._id))
    
    def smoothed(self, smoothness=3.0):
        """
        Creates a trajectory generated from this trajectory using cubic 
        B-spline interpolation.
        
        Arguments:
        smoothness - strength of smoothing, larger is smoother. See 
            scipy.interpolate.splprep()'s s parameter.
        
        Returns:
        a new Trajectory object with the interpolated positions and 
        velocities. If the length of the trajectory < 4, returns self.
        """
        if len(self.time()) < 4: return self
        
        spline, _ = interp.splprep(list(self.pos().T), s=smoothness)
        eval_prms = np.linspace(0, 1, len(self))
        
        new_pos = np.array(interp.splev(eval_prms, spline)).T
        new_vel = np.array(interp.splev(eval_prms, spline, der=1)).T
        
        return Trajectory(new_pos, new_vel, self.time(), self.trajid())
