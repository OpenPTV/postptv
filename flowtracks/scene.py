# -*- coding: utf-8 -*-
"""
A class for manipulating PTV analyses saved as HDF5 files in the flowtracks 
format. Allows reading the file by iterating over frames or over trajectories.

Main design goals: 
1. Keep as little as possible in memory.
2. Minimize separate file accesses by allowing reading by frames instead of 
   only by trajectories as in the old code.

Created on Sun Aug 10 11:28:42 2014

@author: yosef
"""

import itertools, tables, numpy as np
from ConfigParser import SafeConfigParser

from .trajectory import Trajectory, ParticleSnapshot
from .particle import Particle

class Frame(object):
    pass

class Scene(object):
    def __init__(self, file_name, frame_range=None):
        self._file = tables.open_file(file_name)
        self._table = self._file.get_node('/particles')
        self._trids = np.unique(self._table.col('trajid'))
        self.set_frame_range(frame_range)
        
    def set_frame_range(self, frame_range):
        """
        Prepare a query part that limits the frame numbers is needed.
        
        Arguments:
        frame_range - a tuple (first, last) frame number, with the usual 
            pythonic convention that first <= i < last. Any element may be 
            None, in which case no limit is generated for it, and for no limits
            at all, passing none instead of a tuple is acceptable.
        """
        self._frame_limit = ""
        
        if frame_range is None:
            t = self._table.col('time')
            self._first = int(t.min())
            self._last = int(t.max()) + 1
            return
        
        first, last = frame_range
        if first is None:
            t = self._table.col('time')
            self._first = int(t.min())
        else:
            self._first = first
            self._frame_limit += " & (time >= %d)" % first
        
        # Working on the assumptions that usually not both will be None 
        # (then you can simply pass None), to simplify the code.
        if last is None:
            t = self._table.col('time')
            self._last = int(t.max()) + 1
        else:
            self._last = last
            self._frame_limit += " & (time < %d)" % last
    
    def __del__(self):
        self._file.close()
    
    def iter_trajectories(self):
        """
        Iterator over trajectories. Generates a Trajectory object for each 
        trajectory in the file (in no particular order, but the same order 
        every time on the same PyTables version) and yields it.
        """
        query_string = '(trajid == trid)' +  self._frame_limit
        
        for trid in self._trids:
            arr = self._table.read_where(query_string)
            kwds = dict((field, arr[field]) for field in arr.dtype.fields \
                if field != 'trajid')
            kwds['trajid'] = trid
            yield Trajectory(**kwds)
        
    def iter_frames(self):
        """
        Iterator over frames. Generates a ParticleSnapshot object for each
        frame, in the file, ordered by frame number, and yields it.
        """
        query_string = '(time == t)'
        
        for t in xrange(self._first, self._last):
            arr = self._table.read_where(query_string)
            kwds = dict((field, arr[field]) for field in arr.dtype.fields \
                if field != 'time')
            kwds['time'] = t
            yield ParticleSnapshot(**kwds)
        

class DualScene(object):
    """
    Holds a scene orresponding to the dual-PTV systems, which shoot separate
    but coordinated streams for the tracers data and inertial particles data.
    """
    def __init__(self, tracers_path, particles_path, frate, particle,
        frame_range=None):
        """
        Arguments:
        tracers_path, particles_path - respectively the path to the tracers 
            and particles HDF files.
        frate - frame rate at which the scene was shot, [1/s].
        particle - a Particle object describing the inertial particles'
            diameter and density.
        frame_range - a uniform frame range to set to both of them. The
            default is None, meaning to use all frames (assuming 
            equal-length data streams)
        """
        self.frate = frate
        self.part = particle
        
        self._paths = (tracers_path, particles_path)
        self._tracers = Scene(tracers_path, frame_range)
        self._particles = Scene(particles_path, frame_range)
        self._rng = frame_range # for restoring it after iteration.
    
    def get_particles_path(self):
        return self._paths[1]
    
    def iter_frames(self, frame_range=-1):
        """
        Iterates over a scene represented by two HDF files (one for inertial 
        particles, one for tracers), and returns a Frame object whose two
        attributes (.tracers, .particles) hold a corresponding
        ParticleSnapshot object.
        
        Arguments:
        particle_scene, tracer_scene - each a Scene object for the 
            corresponding file.
        frame_range - tuple (first, last) sets the frame range of both scenes
            to an identical frame range. Argument format as in 
            Scene.set_frame_range(). Default is (-1) meaning to skip this. 
            Then the object's initialization range is used, so initialize
            to a coordinated range if you use the default.
        
        Yields:
        the Frame object for each frame in turn.
        """
        if frame_range != -1:
            self._particles.set_frame_range(frame_range)
            self._tracers.set_frame_range(frame_range)
    
        for particles, tracers in itertools.izip(
            self._particles.iter_frames(), self._tracers.iter_frames()):
            frame = Frame()
            frame.tracers = tracers
            frame.particles = particles
            yield frame
        
        # restore original frame range.
        if frame_range != -1:
            self._particles.set_frame_range(self._frame_range)
            self._tracers.set_frame_range(self._frame_range)

def read_dual_scene(conf_fname):
    """
    Read dual-scene parameters, such as unchanging particle properties and
    frame range. Values are stored in an INI-format file.
    
    Arguments:
    conf_fname - name of the config file
    
    Returns:
    a DualScene object initialized with the configuration values found.
    """
    parser = SafeConfigParser()
    parser.read(conf_fname)

    particle = Particle(
        parser.getfloat("Particle", "diameter"),
        parser.getfloat("Particle", "density"))
    
    frate = parser.getfloat("Scene", "frame rate")
    tracer_file = parser.get("Scene", "tracers file")
    part_file = parser.get("Scene", "particles file")
    frange = (parser.getint("Scene", "first frame"),
        parser.getint("Scene", "last frame") + 1)
    
    return DualScene(tracer_file, part_file, frate, particle, frange)
