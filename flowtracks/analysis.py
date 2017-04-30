# -*- coding: utf-8 -*-
# Created on Mon Aug 11 15:14:21 2014
"""
Infrastructure for running a frame-by-frame analysis on a DualScene object.
The main point of interest here is :func:`analysis`, which performs a segment
iteration over a :class:`~flowtracks.scene.DualScene` and applies to each 
a user-selected list of analyzers. Analysers are instances of a 
:class:`GeneralAnalyser` subclass which implements the necessary methods, 
as described in the base class documentation.

There is one base class supplied here, :class:`FluidVelocitiesAnalyser`, 
which ties in the :mod:`flowtracks.interpolation` module for analysing the 
fluid velocity around a particle from its surrounding tracers.
"""

import numpy as np, tables

def companion_indices(trids, companions):
    """
    Return an array giving for each companion its respective index in the 
    trajectory ID array, or a negative number if not found.
    """
    comp = np.full_like(companions, -1)
    idx = np.nonzero(trids[:,None] == companions)
    comp[idx[1]] = idx[0]
    return comp
                
class GeneralAnalyser(object):
    """
    This is the parent class for all analysers to be used by :func:`analysis`.
    It does not do anything but define and document the methods that must be
    implenmented by the child class (in other words, this class is abstract).
    Attempting to use its methods will result in a ``NotImplementedError``.
    """
    def descr(self):
        """
        Need to return a list of tuples, each of the form 
        (name, data type, row length), e.g. ('trajid', int, 1) 
        """
        raise NotImplementedError
    
    def analyse(self, frame, next_frame):
        """
        Arguments:
        frame, next_frame - the Frame object for the currently-analysed frame
            and the one after it, respectively.
        
        Returns:
        a list of arrays, each of shape (f,d) where f is the number of 
        particles in the current frame, and d is the row length of the
        corresponding item returned by self.descr(). Each array's dtype also
        corresponds to the dtype given to it by self.descr().
        """
        raise NotImplementedError

class FluidVelocitiesAnalyser(GeneralAnalyser):
    """
    Finds, for each particle in the ``particles`` set of a frame, the 
    so-called *undisturbed* fluid velocity at the particle's position, by
    interpolating from nearby particles in the ``tracers`` set.
    """
    def __init__(self, interp):
        """
        Arguments:
        interp - the Interpolant object to use for finding velocities.
        """
        self._interp = interp
        
    def descr(self):
        """
        Return a list of two tuples, each of the form 
        (name, data type, row length), describing the arrays returned by 
        analyse() for fluid velocity and relative velocity.
        """
        return [('fluid_vel', float, 3), ('rel_vel', float, 3)]
    
    def analyse(self, frame, next_frame):
        """
        Arguments:
        frame, next_frame - the Frame object for the currently-analysed frame
            and the one after it, respectively.
        
        Returns:
        a list of two arrays, each of shape (f,3) where f is the number of 
        particles in the current frame. 1st array - fluid velocity. 2nd array
        - relative velocity.
        """
        if frame.particles.has_property('companion'):
            comp = companion_indices(
                frame.tracers.trajid(), frame.particles.companion())
        else:
            comp = None
        self._interp.set_scene(frame.tracers.pos(), frame.particles.pos(),
            frame.tracers.velocity(), comp)
        vel_interp = self._interp.interpolate()
        rel_vel = frame.particles.velocity() - vel_interp
        
        return [vel_interp, rel_vel]
    
def analysis(scene, analysis_file, conf_file, analysers, frame_range=-1):
    """
    Generate the analysis table for a given scene with separate data for 
    inertial particles and tracers.
    
    Arguments:
    scene - a DualScene object representing an experiment with coordinated 
        particles and tracers data streams.
    analysis_file - path to the file where analysis should be saved. If the 
        file exists, it will be cloberred.
    conf_file - name of config file used for creating the analysis.
    analysers - a list of GeneralAnalyser subclasses that do the actual
        analysis work and know all that is needed about output shape.
    frame_range - if -1 no adjustment is necessary, otherwise see 
        :meth:`DualScene.iter_segments() <flowtracks.scene.DualScene.iter_segments>`
    """
    # Structure the output file:
    descr = [('trajid', int, 1), ('time', int, 1)]
    for analyser in analysers:
        descr.extend(analyser.descr())
    descr = np.dtype(descr)
    
    outfile = tables.open_file(analysis_file, "w", title="Analysis results.")
    table = outfile.create_table('/', 'analysis', descr)
    table.attrs['config'] = conf_file
    table.attrs['trajects'] = scene.get_particles_path()
    
    for frame, next_frame in scene.iter_segments(frame_range):
        length = len(frame.particles)
        arr = np.empty(length, dtype=descr)
        arr['trajid'] = frame.particles.trajid()
        arr['time'] = frame.particles.time()
        
        for analyser in analysers:
            analysis = analyser.analyse(frame, next_frame)
            this_descr = analyser.descr()
            
            for res, desc in zip(analysis, this_descr):
                arr[desc[0]] = res
        
        table.append(arr)
    
    # Wrap up and close.
    table.cols.trajid.create_index()
    table.cols.time.create_index()
    outfile.flush()
    outfile.close()
