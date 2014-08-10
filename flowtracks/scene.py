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
from .trajectory import Trajectory, ParticleSnapshot

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
            self._frame_limit += " & (time >= %d)" % first
        
        # Working on the assumptions that usually not both will be None 
        # (then you can simply pass None), to simplify the code.
        if last is None:
            t = self._table.col('time')
            self._last = int(t.max()) + 1
        else:
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
        

def dual_scene_iterator(particle_scene, tracer_scene, frame_range=-1):
    """
    Iterates over a scene represented by two HDF files (one for inertial 
    particles, one for tracers), and returns a Frame object whose two
    attributes (.tracers, .particles) hold a corresponding ParticleSnapshot
    object.
    
    Arguments:
    particle_scene, tracer_scene - each a Scene object for the corresponding
        file.
    frame_range - tuple (first, last) sets the frame range of both scenes to
        an identical frame range. Argument format as in 
        Scene.set_frame_range(). Default is (-1) meaning to skip this. Only
        use that if you know your scenes to be coordinated.
    
    Yields:
    the Frame object for each frame in turn.
    """
    if frame_range is not None:
        particle_scene.set_frame_range(frame_range)
        tracer_scene.set_frame_range(frame_range)

    for tracers, particles in itertools.izip(
        particle_scene.iter_frames(), tracer_scene.iter_frames()):
        frame = Frame()
        frame.tracers = tracers
        frame.particles = particles
        yield frame
