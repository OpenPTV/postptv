# -*- coding: utf-8 -*-
"""
Infrastructure for running a frame-by-frame analysis on a DualScene object.

Created on Mon Aug 11 15:14:21 2014

@author: yosef
"""

import numpy as np, tables

class GeneralAnalyser(object):
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
        vel_interp = self._interp(frame.tracers.pos(), frame.particles.pos(),
            frame.tracers.velocity())
        rel_vel = frame.particles.velocity() - vel_interp
        
        return [vel_interp, rel_vel]
    
def analysis(scene, analysis_file, conf_file, analysers, frame_range=-1):
    """
    Simplified version of ``generate_reconstructions()`` with all the crap 
    removed and using HDF5 files only.
    
    Generate the analysis table for a given scene with separate data for 
    inertial particles and tracers.
    
    Arguments:
    scene - a dualScene object representing an experiment with coordinated 
        particles and tracers data streams.
    analysis_file - path to the file where analysis should be saved. If the 
        file exists, it will be cloberred.
    conf_file - name of config file used for creating the analysis.
    analysers - a list of GeneralAnalyser subclasses that do the actual
        analysis work and know all that is needed about output shape.
    frame_range - if -1 no adjustment is necessary, otherwise see 
        Scene.dual_scene_iterator()
    """
    # Structure the output file:
    descr = [('trajid', int, 1), ('time', int, 1)]
    for analyser in analysers:
        descr.extend(analyser.descr())
    descr = np.dtype(descr)
    
    outfile = tables.openFile(analysis_file, "w", title="Analysis results.")
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
