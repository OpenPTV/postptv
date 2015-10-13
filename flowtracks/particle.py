# -*- coding: utf-8 -*-

"""
This class is needed for modeling the dynamics of a particle in a flow scene.
"""

import numpy as np

class Particle(object):
    """
    A class to hold particle properties.
    """
    def __init__(self, diameter, density):
        """
        Arguments:
        diameter - particle diameter, [m]
        density - particle density, [kg/m^3]
        """
        self.diam = diameter
        self.density = density
    
    def volume(self):
        return np.pi * self.diam**3 / 6.
    
    def mass(self):
        return self.density * self.volume()

