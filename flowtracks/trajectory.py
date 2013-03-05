# -*- coding: utf-8 -*-

import numpy as np

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
