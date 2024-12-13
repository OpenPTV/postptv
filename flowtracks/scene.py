# -*- coding: utf-8 -*-
#Created on Sun Aug 10 11:28:42 2014
#
# Private references:
# [1] https://docs.python.org/2/library/itertools.html
"""
A module for manipulating PTV analyses saved as HDF5 files in the flowtracks 
format. Allows reading the data by iterating over frames or over trajectories.

Main design goals: 

1. Keep as little as possible in memory.
2. Minimize separate file accesses by allowing reading by frames instead of \
   only by trajectories as in the old code.

"""

import itertools as it, tables, numpy as np
from configparser import ConfigParser

from .trajectory import Trajectory, ParticleSnapshot
from .particle import Particle
# from past.builtins import range

class Frame(object):
    pass

def pairwise(iterable):
    """
    copied from itertools documentation, [1]
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def gen_query_string(key, range_spec):
    """
    A small utility to create query string suitable for PyTables' 
    ``read_where()`` from a range specification.
    
    Arguments:
    key - name of search field.
    range_spec - a tuple (min, max, invert). If ``invert`` is false, the search 
        range is between min and max. Otherwise it is anywhere except that.
        In regular ranges, the max boundary is excluded as usual in Python. In
        inverted range, consequentlt, it is the min boundary that's excluded.
    
    Returns:
    A string representing all boolean conditions necessary for representing the
    given range.
    
    Example:
    >>> gen_query_string('example', (-1, 1, False))
    '((example >= -1) & (example < 1))'
    
    >>> gen_query_string('example', (-1, 1, True))
    '((example < -1) | (example >= 1))'
    """
    smin, smax, invert = range_spec
    cop1, cop2, lop = ('<','>=','|') if invert else ('>=','<','&')
    cond_string = "((%s %s %g) %s (%s %s %g))" % \
        (key, cop1, smin, lop, key, cop2, smax)
    
    return cond_string
    
class Scene(object):
    """
    This class is the programmer's interface to an HDF files containing 
    particle trajectory data. It manages access by frames or trajectories,
    as well as by segments. 
    """
    def __init__(self, file_name, frame_range=None):
        """
        Arguments:
        file_name - path to the HDF file hilding the data.
        frame_range - use only frames in this range for iterating the data.
            the default is None, meaning to use all present frams.
        """
        self._file = tables.open_file(file_name)
        self._table = self._file.get_node('/particles')
        try:
            traj_tags = self._file.get_node('/bounds')
            self._trids = traj_tags.col('trajid')
        except:
            self._trids = np.unique(self._table.col('trajid'))
        
        self.set_frame_range(frame_range)
        
        # Cache data on user-visible columns:
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
    
    def trajectory_tags(self):
        tags = self._file.get_node('/bounds')
        return np.hstack([tags.col(name)[:,None] for name in ['trajid', 'first', 'last']])
            
        return np.array([np.int64(row[:]) for row in self._file.get_node('/bounds').read()])
    
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
        rng_exprs = []
        if first is None:
            t = self._table.col('time')
            self._first = int(t.min())
        else:
            self._first = first
            rng_exprs.append("(time >= %d)" % first)
        
        if last is None:
            t = self._table.col('time')
            self._last = int(t.max()) + 1
        else:
            self._last = last
            rng_exprs.append("(time < %d)" % last)
        
        self._frame_limit = ' & '.join(rng_exprs)
    
    def frame_range(self):
        return self._first, self._last
    
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
    
    def trajectory_ids(self):
        """
        Returns all trajectory IDs in the scene as an array.
        """
        return self._trids
    
    def trajectory_by_id(self, trid):
        """
        Get trajectory data by trajectory ID.
        
        Arguments:
        trid - trajectory ID, a unique number assigned to each trajectory when
            the scene file was written.
        
        Returns:
        a Trajectory object.
        """
        # I just repeat most of the code in iter_trajectories(). It is not the
        # pretties thing but trying to abstract these 5 lines would be uglier.
        query_string = '(trajid == trid)'
        if self._frame_limit != '':
            query_string += ' & ' + self._frame_limit
        
        arr = self._table.read_where(query_string)
        kwds = dict((field, arr[field]) for field in arr.dtype.fields \
            if field != 'trajid')
        kwds['trajid'] = trid
        return Trajectory(**kwds)
        
    def iter_trajectories(self):
        """
        Iterator over trajectories. Generates a Trajectory object for each 
        trajectory in the file (in no particular order, but the same order 
        every time on the same PyTables version) and yields it.
        """
        query_string = '(trajid == trid)'
        if self._frame_limit != '':
            query_string += ' & ' + self._frame_limit
        
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
            
        for t in range(self._first, self._last):
            yield t, self._table.read_where(query_string)
    
    def frame_by_time(self, t):
        """
        Get a Frame object for data occuring at time t. Assumes that the time 
        exists in the data, and does not check range.
        
        Arguments:
        t - the frame count at the requested frame.
        
        Returns:
        a ParticleSnapshot object.
        """
        query_string = '(time == t)'
        arr = self._table.read_where(query_string)
        
        kwds = dict((field, arr[field]) for field in arr.dtype.fields \
            if field != 'time')
        kwds['time'] = t
        return ParticleSnapshot(**kwds)
        
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
    
    def collect(self, keys, where=None):
        """
        Get values of given keys, either all of them or the ones corresponding
        to a selection given by 'where'.
        
        Arguments:
        keys - a list of keys to take from the data
        where - a dictionary of particle property names, with a tuple 
            (min,max,invert) as values. If ``invert`` is false, the search 
            range is between min and max. Otherwise it is anywhere except that.
        
        Returns:
        a list of arrays, in the order of ``keys``.
        """
        # Compose query to PyTables engine:
        conds = [self._frame_limit]
        if where is not None:
            for key, rng in where.items():
                conds.append(gen_query_string(key, rng))
        cond_string = ' & '.join(conds)
        
        # No frame range or user-defined conditions:
        if cond_string == '':
            return [self._table.col(k) for k in keys]
        
        # Single key is natively handled in PyTables.
        if len(keys) == 1:
            return [self._table.read_where(cond_string, field=keys[0])]
        
        # Otherwise do the extraction manually.
        ret = []
        raw = self._table.read_where(cond_string)
        for k in keys:
            ret.append(raw[k])
        
        return ret
    
    def bounding_box(self):
        """
        Gets the min and max positions in the data - either from file 
        attributes if present, or from a brute-force collect().
        
        Returns:
        min_pos, max_pos - each a (3,) array.
        """
        if 'min_pos' in self._table._v_attrs._f_list():
            min_pos = self._table._v_attrs.min_pos
            max_pos = self._table._v_attrs.max_pos
        else:
            poses = self.collect(['pos'])[0]
            min_pos, max_pos = poses.min(axis=0), poses.max(axis=0)
        return min_pos, max_pos
            
        
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
        """
        Returns the path to the HDF file holding inertial particle data
        """
        return self._paths[1]
    
    def get_particles(self):
        """
        Returns the :class:`Scene` that manages inertial particles' data. 
        """
        return self._particles
        
    def get_tracers(self):
        """
        Returns the :class:`Scene` that manages tracer data. 
        """
        return self._tracers
    
    def get_range(self):
        """
        Returns the frame renge set for the dual scene.
        """
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
    
        for particles, tracers in zip(
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
    
        for part_frames, tracer_frames in zip(
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
    parser = ConfigParser()
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
