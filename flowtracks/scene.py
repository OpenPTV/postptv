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

References:
[1] https://docs.python.org/2/library/itertools.html
"""

import itertools as it, tables, numpy as np
from ConfigParser import SafeConfigParser

from .trajectory import Trajectory, ParticleSnapshot
from .particle import Particle

class Frame(object):
    pass

def pairwise(iterable):
    """
    copied from itertools documentation, [1]
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = it.tee(iterable)
    next(b, None)
    return it.izip(a, b)

class Scene(object):
    def __init__(self, file_name, frame_range=None):
        self._file = tables.open_file(file_name)
        self._table = self._file.get_node('/particles')
        self._trids = np.unique(self._table.col('trajid'))
        self.set_frame_range(frame_range)
        
        # Cache data on user-visible columsn:
        filt = ('trajid', 'time')
        self._keys = []
        self._shapes = []
        desc = self._table.coldescrs
        
        for name in self._table.colnames:
            if name in filt:
                continue
            self._keys.append(name)
            shape = desc[name].shape
            self._shapes.append(1 if len(shape) == 0 else shape[0])
    
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
    
    def keys(self):
        """
        Return all the possible trajectory properties that may be queried as
        a data series (i.e. not the scalar property trajid), as a list of
        strings.
        """
        return self._keys
        
    def shapes(self):
        """
        Return the number of components per item of each key in the order 
        returned by ``keys()``.
        """
        return self._shapes
        
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
        for t, arr in self._iter_frame_arrays():
            kwds = dict((field, arr[field]) for field in arr.dtype.fields \
                if field != 'time')
            kwds['time'] = t
            yield ParticleSnapshot(**kwds)
    
    def _iter_frame_arrays(self, cond=None):
        """
        Private. Like iter_frames but does not create a ParticleSnapshot
        object, leaving the raw array. Also allows heavier filtering.
        
        Arguments:
        cond - an optional PyTables condition string to apply to each frame.
        """
        query_string = '(time == t)'
        if cond is not None:
            query_string = '&'.join(query_string, cond)
            
        for t in xrange(self._first, self._last):
            yield t, self._table.read_where(query_string)
        
    def iter_segments(self):
        """
        Iterates over frames, taking out only the particles whose trajectory 
        continues in the next frame.
        
        Yields:
        frame - a ParticleSnapshot object representing the current frame with
            the particles that have continuing trajectories.
        next_frame - same object, for the same particles in the next frame
            (the time attribute is obviously +1 from ``frame``).
        """
        for arr, next_arr in pairwise(self._iter_frame_arrays()):
            t, arr = arr
            tn, next_arr = next_arr
            
            # find continuing trajectories:
            arr_trids = arr['trajid']
            next_arr_trids = next_arr['trajid']
            trajids = set(arr_trids) & set(next_arr_trids)
            
            # select only those from the two frames:
            in_arr = np.array([True if tr in trajids else False \
                for tr in arr_trids])
            in_next_arr = np.array([True if tr in trajids else False \
                for tr in next_arr_trids])
            
            if len(in_arr) > 0:
                arr = arr[in_arr]
            if len(in_next_arr) > 0:
                next_arr = next_arr[in_next_arr]
            
            # format as ParticleSnapshot.
            kwds = dict((field, arr[field]) for field in arr.dtype.fields \
                if field != 'time')
            kwds['time'] = t
            frame = ParticleSnapshot(**kwds)
            
            kwds = dict((field, next_arr[field]) for field in arr.dtype.fields \
                if field != 'time')
            kwds['time'] = tn
            next_frame = ParticleSnapshot(**kwds)
            
            yield frame, next_frame
    

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
    
    def get_particles(self):
        return self._particles
    
    def get_range(self):
        return self._rng
    
    def iter_frames(self, frame_range=-1):
        """
        Iterates over a scene represented by two HDF files (one for inertial 
        particles, one for tracers), and returns a Frame object whose two
        attributes (.tracers, .particles) hold a corresponding
        ParticleSnapshot object.
        
        Arguments:
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
    
        for particles, tracers in it.izip(
            self._particles.iter_frames(), self._tracers.iter_frames()):
            frame = Frame()
            frame.tracers = tracers
            frame.particles = particles
            yield frame
        
        # restore original frame range.
        if frame_range != -1:
            self._particles.set_frame_range(self._rng)
            self._tracers.set_frame_range(self._rng)
    
    def iter_segments(self, frame_range=-1):
        """
        Like iter_frames, but returns two consecutive frames, both having the
        same trajids set (in other words, both contain only particles from 
        the first frame whose trajectory continues to the next frame).
        
        Arguments:
        frame_range - tuple (first, last) sets the frame range of both scenes
            to an identical frame range. Argument format as in 
            Scene.set_frame_range(). Default is (-1) meaning to skip this. 
            Then the object's initialization range is used, so initialize
            to a coordinated range if you use the default.
        
        Yields:
        two Frame objects, representing the consecutive selective frames.
        """
        if frame_range != -1:
            self._particles.set_frame_range(frame_range)
            self._tracers.set_frame_range(frame_range)
    
        for part_frames, tracer_frames in it.izip(
            self._particles.iter_segments(), self._tracers.iter_segments()):
            frame = Frame()
            frame.tracers = tracer_frames[0]
            frame.particles = part_frames[0]
            
            next_frame = Frame()
            next_frame.tracers = tracer_frames[1]
            next_frame.particles = part_frames[1]
            
            yield frame, next_frame
        
        # restore original frame range.
        if frame_range != -1:
            self._particles.set_frame_range(self._rng)
            self._tracers.set_frame_range(self._rng)

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
