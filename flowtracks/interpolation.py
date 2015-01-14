# -*- coding: utf-8 -*-
"""
Interpolation routines.

Created on Tue May 28 10:27:15 2013

@author: yosef

References:
[1] Lüthi, Beat. Some Aspects of Strain, Vorticity and Material Element 
    Dynamics as Measured with 3D Particle Tracking Velocimetry in a Turbulent 
    Flow. PhD Thesis, ETH-Zürich (2002).

"""

import numpy as np, warnings
from ConfigParser import SafeConfigParser

def select_neighbs(tracer_pos, interp_points, radius=None, num_neighbs=None):
    """
    For each of m interpolation points, find its distance to all tracers. Use
    result to decide which tracers are the neighbours of each interpolation
    point, based on either a fixed radius or the closest num_neighbs.
    
    Arguments:
    tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row, [m]
    interp_points - (m,3) array, coordinates of points where interpolation will
        be done.
    radius - of the search area for neighbours, [m]. If None, select closest
        num_neighbs.
    num_neighbs - number of closest neighbours to interpolate from. If None.
        uses all neighbours in a given radius. ``radius`` has precedence.
    
    Returns:
    dists - (m,n) array, the distance from each interpolation point to each
        tracer.
    use_parts - (m,n) boolean array, True where tracer j=1...n is a neighbour
        of interpolation point i=1...m.
    """
    dists =  np.linalg.norm(tracer_pos[None,:,:] - interp_points[:,None,:],
        axis=2)
    
    dists[dists <= 0] = np.inf # Only for selection phase,later changed back.
    
    if radius is None:
        if num_neighbs is None:
            raise ValueError("Either radius or num_neighbs must be given.")
        
        dist_sort = np.argsort(dists, axis=1)
        use_parts = np.zeros(dists.shape, dtype=np.bool)
        
        eff_num_neighbs = min(num_neighbs, tracer_pos.shape[0])
        use_parts[
            np.repeat(np.arange(interp_points.shape[0]), eff_num_neighbs),
            dist_sort[:,:num_neighbs].flatten()] = True
    
    else:
        use_parts = dists < radius
    
    dists[np.isinf(dists)] = 0.
    return dists, use_parts
    
def inv_dist_interp(dists, use_parts, velocity, p=1):
    """
    For each of n particle, generate the velocity interpolated to its 
    position from all neighbours as selected by caller. Interpolation method is
    inverse-distance weighting, [1]
    
    Arguments:
    dists - (m,n) array, the distance of interpolation_point i=1...m from 
        tracer j=1...n, for (row,col) (i,j) [m] 
    use_parts - (m,n) boolean array, whether tracer j is a neighbour of 
        particle i, same indexing as ``dists``.
    velocity - (n,3) array, the u,v,w velocity components for each of n
        tracers, [m/s]
    p - the power of inverse distance weight, w = r^(-p). default 1. Use 0 for
        simple averaging.
    
    Returns:
    vel_avg - an (m,3) array with the interpolated velocity at each 
        interpolation point, [m/s].
    """
    weights = np.zeros_like(dists)
    weights[use_parts] = 1./dists[use_parts]**p
    
    vel_avg = (weights[...,None] * velocity[None,...]).sum(axis=1) / \
        weights.sum(axis=1)[:,None]

    return vel_avg

def corrfun_interp(dists, use_parts, data, corrs_hist, corrs_bins):
    """
    For each of n particle, generate the velocity interpolated to its 
    position from all neighbours as selected by caller. The weighting of 
    neighbours is by the correlation function, e.g. if the distance at 
    neighbor i is r_i, then it adds \rho(r_i)*v_i to the interpolated velocity.
    This is done for each component separately.
    
    Arguemnts:
    dists - (m,n) array, the distance of interpolation_point i=1...m from 
        tracer j=1...n, for (row,col) (i,j) [m] 
    use_parts - (m,n) boolean array, whether tracer j is a neighbour of 
        particle i, same indexing as ``dists``.
    data - (n,d) array, the d components of the data that is interpolated from,
        for each of n tracers.
    corrs_hist - the correlation function histogram, an array of b bins.
    corrs_bins - same size array, the bin start point for each bin.
        
    Returns:
    vel_avg - an (m,3) array with the interpolated velocity at each 
        interpolation point, [units of ``data``].
    """
    weights = np.zeros(dists.shape + (data.shape[-1],))
    weights[use_parts] = corrs_hist[
        np.digitize(dists[use_parts].flatten(), corrs_bins) - 1]
    
    vel_avg = (weights * data[None,...]).sum(axis=1) / \
        weights.sum(axis=1)

    return vel_avg

def rbf_interp(tracer_dists, dists, use_parts, data, epsilon=1e-2):
    """
    Radial-basis interpolation [3] for each particle, from all neighbours 
    selected by caller. The difference from inv_dist_interp is that the 
    weights are independent of interpolation point, among other differences.
    
    Arguments:
    tracer_dists - (n,n) array, the distance of tracer i=1...n from tracer 
        j=1...n, for (row,col) (i,j) [m]
    dists - (m,n) array, the distance from interpolation point i=1...m to
        tracer j. [m]
    use_parts - (m,n) boolean array, True where tracer j=1...n is a neighbour
        of interpolation point i=1...m.
    data - (n,d) array, the d components of the data for each of n tracers.
    
    Returns:
    vel_interp - an (m,3) array with the interpolated velocity at the position
        of each particle, [m/s].
    """
    kernel = np.exp(-tracer_dists**2 * epsilon)
    
    # Determine the set of coefficients for each particle:
    coeffs = np.zeros(dists.shape + (data.shape[-1],))
    for pix in xrange(dists.shape[0]):
        neighbs = np.nonzero(use_parts[pix])[0]
        K = kernel[np.ix_(neighbs, neighbs)]
        
        coeffs[pix, neighbs] = np.linalg.solve(K, data[neighbs])
    
    rbf = np.exp(-dists**2 * epsilon)
    vel_interp = np.sum(rbf[...,None] * coeffs, axis=1)
    return vel_interp

class Interpolant(object):
    """
    Holds all parameters necessary for performing an interpolation. Use is as
    a callable object after initialization, see __call__().
    """
    def __init__(self, method, num_neighbs=None, param=None):
        """
        Arguments:
        method - interpolation method. Either 'inv' for inverse-distance 
            weighting, 'rbf' for gaussian-kernel Radial Basis Function
            method, or 'corrfun' for using a correlation function.
        neighbs - number of closest neighbours to interpolate from. If None.
            uses 4 neighbours for 'inv' method, and 7 for 'rbf'.
        param - the parameter adjusting the interpolation method. For IDW it is
            the inverse power (default 1), for rbf it is epsilon (default 1e5).
        """        
        if method == 'inv':
            if num_neighbs is None:
                num_neighbs = 4
            if param is None: 
                param = 1
        
        elif method == 'rbf':
            if num_neighbs is None:
                num_neighbs = 7
            if param is None:
                param = 1e5
        
        elif method == 'corrfun':
            if num_neighbs is None:
                num_neighbs = 4
            if param is None: 
                raise ValueError("'corrfun' method requires param to be "\
                    "an NPZ file name containing the corrs and bins arrays.")
            c = np.load(param)
            self._corrs = c['corrs']
            self._bins = c['bins']
        
        else:
            raise NotImplementedError("Interpolation method %s not supported" \
                % method)
            
        self._method = method
        self._neighbs = num_neighbs
        self._par = param
    
    def num_neighbs(self):
        return self._neighbs
    
    def set_scene(self, tracer_pos, interp_points, data):
        """
        Records scene data for future interpolation using the same scene.
        
        Arguments:
        tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row, 
            in [m]
        interp_points - (m,3) array, coordinates of points where interpolation 
            will be done.
        data - (n,d) array, the for the d-dimensional data for tracer n. For 
            example, in velocity interpolation this would be (n,3), each tracer
            having 3 components of velocity.
        """
        self.__tracers = tracer_pos
        self.__interp_pts = interp_points
        self.__data = data
        
        # empty the neighbours cache:
        self.__dists = None
        self.__active_neighbs = None
    
    def which_neighbours(self):
        """
        Finds the neighbours that would be selected for use at each 
        interpolation point, given the current scene as set by set_scene().
        """
        if self.__active_neighbs is None:
            self.__dists, self.__active_neighbs = select_neighbs(
                self.__tracers, self.__interp_pts, None, self._neighbs)
        return self.__active_neighbs
        
    def __call__(self, tracer_pos, interp_points, data):
        """
        Sets up the necessary parameters, and performs the interpolation.
        Does not change the scene set by set_scene if any, so may be used
        for any off-scene interpolation.
        
        Arguments:
        tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row, 
            in [m]
        interp_points - (m,3) array, coordinates of points where interpolation 
            will be done.
        data - (n,d) array, the for the d-dimensional data for tracer n. For 
            example, in velocity interpolation this would be (n,3), each tracer
            having 3 components of velocity.
        
        Returns:
        vel_interp - an (m,3) array with the interpolated value at the position
            of each particle, [m/s].
        """
        # If for some reason tracking failed for a whole frame, interpolation 
        # is impossible at that frame. This checks for frame tracking failure.
        if len(tracer_pos) == 0:
            # Temporary measure until I can safely discard frames.
            warnings.warn("No tracers im frame, interpolation returned zeros.")
            ret_shape = data.shape[-1] if data.ndim > 1 else 1
            return np.zeros((interp_points.shape[0], ret_shape))
            
        dists, use_parts = select_neighbs(tracer_pos, interp_points, 
            None, self._neighbs)
        
        if self._method == 'inv':
            return inv_dist_interp(dists, use_parts, data, self._par)
            
        elif self._method == 'rbf':
            tracer_dists = select_neighbs(tracer_pos, tracer_pos, 
                None, self._neighbs)[0]
            return rbf_interp(tracer_dists, dists, use_parts, data, self._par)
        
        elif self._method == 'corrfun':
            return corrfun_interp(dists, use_parts, data,
                self._corrs, self._bins)
        
        else:
            # This isn't supposed to ever happen. The constructor should fail.
            raise NotImplementedError("Interpolation method %s not supported" \
                % self._method)
            
    
    def neighb_dists(self, tracer_pos, interp_points):
        """
        The distance from each interpolation point to each data point of those
        used for interpolation. Assumes, for now, a constant number of
        neighbours.
        
        Arguments:
        tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row, 
            in [m]
        interp_points - (m,3) array, coordinates of points where interpolation 
            will be done.
        
        Returns:
        ndists - an (m,c) array, for c closest neighbours as defined during
            object construction.
        """
        dists, use_parts = select_neighbs(tracer_pos, interp_points, 
            None, self._neighbs)
        ndists = np.zeros((interp_points.shape[0], self._neighbs))
        
        for pt in xrange(interp_points.shape[0]):
            # allow assignment of less than the desired number of neighbours.
            ndists[pt] = dists[pt, use_parts[pt]]
        
        return ndists
    
    def save_config(self, cfg):
        """
        Adds the keys necessary for recreating this interpolant into a 
        configuration object. It is the caller's responsibility to do a 
        writeback to file.
        
        Arguments:
        cfg - a ConfigParser object.
        """
        if not cfg.has_section("Interpolant"):
            cfg.add_section("Interpolant")
        cfg.set('Interpolant', 'num_neighbs', str(self.num_neighbs()))
        cfg.set('Interpolant', 'param', str(self._par))
        cfg.set('Interpolant', 'method', self._method)

def read_interpolant(conf_fname):
    """
    Builds an Interpolant object based on values in an INI-formatted file.
    
    Arguments:
    conf_fname - path to configuration file.
    
    Returns:
    an Interpolant object constructed from values in the configuration file.
    """
    parser = SafeConfigParser()
    parser.read(conf_fname)
    
    # Optional arguments:
    kwds = {}
    if parser.has_option('Interpolant', 'num_neighbs'):
        kwds['num_neighbs'] = parser.getint('Interpolant', 'num_neighbs')
    if parser.has_option('Interpolant', 'param'):
        kwds['param'] = parser.getfloat('Interpolant', 'param')
    
    return Interpolant(parser.get('Interpolant', 'method'), **kwds)
