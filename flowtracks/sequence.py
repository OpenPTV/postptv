# -*- coding: utf-8 -*-
import numpy as np

from .io import trajectories, infer_format
from .particle import Particle
from .trajectory import take_snapshot, trajectories_in_frame
from ConfigParser import SafeConfigParser

class Sequence(object):
    def __init__(self, frange, frate, particle, part_tmpl, tracer_tmpl,
                 smooth_tracers=False):
        """
        Arguments:
        frange - tuple, (first frame #, after last frame #)
        frate - the frame rate at which the scene was shot.
        particle - a Particle object representing the suspended particles'
            properties.
        part_tmpl, tracer_tmpl - the templates for filenames of particles and
            tracers respectively. Should contain one %d for the frame number.
        smoothe_tracers - if True, uses trajectory smoothing on the tracer
            trajectories when iterating over frames.
        """
        self.part = particle
        self.frate = frate
        
        self._rng = frange
        self._ptmpl = part_tmpl
        self._trtmpl = tracer_tmpl
        self._smooth = smooth_tracers
        
        # No-op particle selector, can be changed by the setter below later.
        self.set_particle_selector(lambda traj: traj)
        
        # This is just a chache. Else trajectories() would check every time.
        self._pfmt = infer_format(part_tmpl)
        self._trfmt = infer_format(tracer_tmpl)
    
    def part_fname(self):
        return self._ptmpl
    
    def part_format(self):
        return self._pfmt
    
    def range(self):
        return self._rng
        
    def subrange(self):
        """
        Returns the earliest and latest time points covered by the subset of
        trajectories that the particle selector selects, bounded by the range
        restricting the overall sequence.
        """
        trs = self.particle_trajectories()
        mins = np.empty(len(trs))
        maxs = np.empty(len(trs))
        
        for trn, tr in enumerate(trs):
            t = tr.time()
            mins[trn] = t.min()
            maxs[trn] = t.max()
        
        return max(mins.min(), self._rng[0]), min(maxs.max(), self._rng[1])
    
    def set_particle_selector(self, selector):
        """
        Sets a filter on the particle trajectories used in sequencing.
        
        Arguments:
        selector - a function which receives a list of Trajectory objects and
            returns a sublit thereof.
        """
        self._psel = selector
        
        # Clear the trajectory cache, so __iter__ reads trajectories.
        self.__ttraj = None
        self.__ptraj = None
    
    def particle_trajectories(self):
        """
        Return (and possibly generate and cache) the list of Trajectory objects
        as selected by the particle selector.
        """
        if (self.__ptraj is not None):
            return self.__ptraj
        self.__ptraj = self._psel(trajectories(self._ptmpl, self._rng[0], 
            self._rng[1], self.frate, self._pfmt))
        return self.__ptraj
        
    def __iter__(self):
        """
        Iterate over frames. For each frame return the data for the tracers and
        particles in it.
        """
        if not hasattr(self, '_act_rng'):
            self._act_rng = self._rng
        
        self._frame = self._act_rng[0]
        
        # Make sure the particle trajectory cache is populated.
        self.particle_trajectories()
        self.__schema = self.__ptraj[0].schema()
        if not (self.__ttraj is None):
            return self
        
        # Make sure the tracers cache is populated:
        trac_trajects = trajectories(self._trtmpl, self._rng[0], self._rng[1],
                self.frate, self._trfmt)
        
        if self._smooth:
            self.__ttraj = [tr.smoothed() for tr in trac_trajects]
        else:
            self.__ttraj = trac_trajects
        
        self.__tschem = self.__ttraj[0].schema()
        
        return self
    
    def iter_subrange(self, first, last):
        self._act_rng = (first, last)
        return self
    
    def next(self):
        if self._frame == self._act_rng[1]:
            # The real problem is in reconstruction_frame() where
            # somethong fails after frame 5
            del self._act_rng
            raise StopIteration
        
        frame = Frame()
        tracer_ixs = trajectories_in_frame(self.__ttraj, self._frame, segs=True)
        tracer_trjs = [self.__ttraj[t] for t in tracer_ixs]
        frame.tracers = take_snapshot(tracer_trjs, self._frame, self.__tschem)
        part_ixs = trajectories_in_frame(self.__ptraj, self._frame, segs=True)
        part_trjs = [self.__ptraj[t] for t in part_ixs]
        frame.particles = take_snapshot(part_trjs, self._frame, self.__schema)

        self._frame += 1
        
        next_frame = Frame()
        next_frame.tracers = take_snapshot(tracer_trjs, self._frame,
            self.__tschem)
        next_frame.particles = take_snapshot(part_trjs, self._frame,
            self.__schema)
        self._next_frame = next_frame
        
        return frame, next_frame
    
    def map_trajectories(self, func, subrange=None, history=False, args=()):
        """
        Iterate over frames, for each frame call a function that generates a
        per-trajectory result and add the results up in a per-trajectory
        time-series.
        
        Arguments:
        func - the function to call. Returns a dictionary keyed by trajid.
            receives as arguments (self, particles, tracers) where particles,
            tracers are the sequence iteration results as given by __iter__.
        subrange - tuple (first, last). Iterate over a subrange of the sequence
            delimited by these frame numbers.
        history - true if the result of one frame depends on earlier results.
            If true, func receives a 4th argument, the accumulated results so 
            far as a dictionary of time-series lists.
        args - a tuple of extra positional arguments to pass to the function
            after the usual arguments and the possible history argument.
        
        Returns:
        a dictionary keyed by trajid, where for each trajectory a time series 
        of results obtained during the trajectory's lifetime is the value.
        """
        if subrange is None:
            subrange = self._rng
        trajects = self.particle_trajectories()
        
        # Allocate result space:
        res = dict((tr.trajid(), [None]*(len(tr) - 1)) for tr in trajects)
        frame_counters = dict((tr.trajid(), 0) for tr in trajects)
        
        for frame, next_frame in self.iter_subrange(*subrange):
            if history:
                fargs = (self, frame, next_frame, res) + args
            else:
                fargs = (self, frame, next_frame) + args
            frm_res = func(*fargs)
            
            for k, v in frm_res.iteritems():
                res[k][frame_counters[k]] = v
                frame_counters[k] += 1
        
        for k in res.keys():
            res[k] = np.array(res[k])
        return res

def read_sequence(conf_fname, smooth=False):
    """
    Read sequence-wide parameters, such as unchanging particle properties and
    frame range. Values are stored in an INI-format file.
    
    Arguments:
    conf_fname - name of the config file
    smooth - whether the sequence shoud use tracers trajectory-smoothing.
    
    Returns:
    a Sequence object initialized with the configuration values found.
    """
    parser = SafeConfigParser()
    parser.read(conf_fname)

    particle = Particle(
        parser.getfloat("Particle", "diameter"),
        parser.getfloat("Particle", "density"))
    
    frate = parser.getfloat("Scene", "frame rate")
    tracer_tmpl = parser.get("Scene", "tracers file")
    part_tmpl = parser.get("Scene", "particles file")
    frange = (parser.getint("Scene", "first frame"),
        parser.getint("Scene", "last frame") + 1)
    
    return Sequence(frange, frate, particle, part_tmpl, tracer_tmpl, smooth)

class Frame(object):
    pass
