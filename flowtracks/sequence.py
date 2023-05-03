# -*- coding: utf-8 -*-


from .io import trajectories, infer_format
from .particle import Particle
from .trajectory import take_snapshot, trajectories_in_frame, Frame
from configparser import ConfigParser
import numpy as np

class Sequence(object):
    """
    Tracks a dual particles database (for both inertial particles and 
    tracers), allowing a number of underlying formats. Provides segment
    iteration and trajectory-mapping.
    """
    def __init__(self, frange, frate, particle, part_tmpl, tracer_tmpl,
                 smooth_tracers=False, traj_min_len=0.):
        """
        Arguments:
        frange - tuple, (first frame #, after last frame #)
        frate - the frame rate at which the scene was shot.
        particle - a Particle object representing the suspended particles'
            properties.
        part_tmpl, tracer_tmpl - the filenames for particle- and tracer-
            databases respectively. Names must be as understood by
            :func:`flowtracks.io.trajectories'.
        smooth_tracers - if True, uses trajectory smoothing on the tracer
            trajectories when iterating over frames. Possibly out of date.
        traj_min_len - when reading trajectories (tracers and particles) 
            discard trajectories shorter than this many frames.
        """
        self.part = particle
        self.frate = frate
        
        self._rng = frange
        self._ptmpl = part_tmpl
        self._trtmpl = tracer_tmpl
        self._smooth = smooth_tracers
        self._minlen = traj_min_len
        
        # No-op particle selector, can be changed by the setter below later.
        identity = lambda traj: traj
        self.set_particle_selector(identity)
        self.set_tracer_selector(identity)
        
        # This is just a cache. Else trajectories() would check every time.
        self._pfmt = infer_format(part_tmpl)
        self._trfmt = infer_format(tracer_tmpl)
    
    def part_fname(self):
        """
        Returns the file name used for reading inertial particles database.
        """
        return self._ptmpl
    
    def part_format(self):
        """
        Returns the format inferred for the inertial particles database.
        """
        return self._pfmt
    
    def range(self):
        """
        Returns the frame number range set for the object, as a tuple (first,
        last).
        """
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
        selector - a function which receives a list of 
            :class:`~flowtracks.trajectory.Trajectory` objects and returns 
            a sublist thereof.
        """
        self._psel = selector
        self.__ptraj = None

    def set_tracer_selector(self, selector):
        """
        Sets a filter on the tracer trajectories used in sequencing.
        
        Arguments:
        selector - a function which receives a list of 
            :class:`~flowtracks.trajectory.Trajectory` objects and returns 
            a sublist thereof.
        """
        self._tsel = selector
        self.__ttraj = None
            
    def particle_trajectories(self):
        """
        Return (and possibly generate and cache) the list of 
        :class:`~flowtracks.trajectory.Trajectory` objects as selected by 
        the particle selector.
        """
        if (self.__ptraj is not None):
            return self.__ptraj
        
        self.__ptraj = self._psel(trajectories(self._ptmpl, self._rng[0], 
            self._rng[1], self.frate, self._pfmt, self._minlen))
        
        # Also caches the starts and ends of trajectories, so that accessing
        # only the ones relevant to a specific frame is easier.
        start_end = [(tr.time()[0], tr.time()[-1]) for tr in self.__ptraj]
        self.__pstarts, self.__pends =  map(np.array, zip(*start_end))
        
        return self.__ptraj
    
    def tracer_trajectories(self):
        """
        Return (and possibly generate and cache) the list of 
        :class:`~flowtracks.trajectory.Trajectory` objects corresponding to 
        tracers.
        """
        if (self.__ttraj is not None):
            return self.__ttraj
        
        ttraj = self._tsel(trajectories(self._trtmpl, self._rng[0], 
            self._rng[1], self.frate, self._trfmt, self._minlen))
        
        if self._smooth:
            self.__ttraj = [tr.smoothed() for tr in ttraj]
        else:
            self.__ttraj = ttraj
        
        # Also caches the starts and ends of trajectories, so that accessing
        # only the ones relevant to a specific frame is easier.
        start_end = [(tr.time()[0], tr.time()[-1]) for tr in self.__ttraj]
        self.__tstarts, self.__tends =  map(np.array, zip(*start_end))
            
        return self.__ttraj
        
    def __iter__(self):
        """
        Iterate over frames. For each frame return the data for the tracers and
        particles in it, as a tuple containing two 
        :class:`~flowtracks.trajectory.ParticleSnapshot` objects 
        corresponding to the current frame data and the next frame's.
        
        Returns:
        A Python iteraor.
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
        self.tracer_trajectories()
        
        self.__tschem = self.__ttraj[0].schema()
        
        return self
    
    def iter_subrange(self, first, last):
        """
        The same as :meth:`__iter__`, except it changes the frame range for 
        the duration of the iteration.
        
        Arguments:
        first, last - frame numbers of the first and last frames in the 
        acting range of frames from the sequence.
        
        Returns:
        A Python iteraor.
        """
        self._act_rng = (first, last)
        return self
    
    def next(self):
        if self._frame == self._act_rng[1]:
            del self._act_rng
            raise StopIteration
        
        frame = Frame()
        tracer_ixs = trajectories_in_frame(self.__ttraj, self._frame, 
            self.__tstarts, self.__tends, segs=True)
        tracer_trjs = [self.__ttraj[t] for t in tracer_ixs]
        frame.tracers = take_snapshot(tracer_trjs, self._frame, self.__tschem)
        
        part_ixs = trajectories_in_frame(self.__ptraj, self._frame,
            self.__pstarts, self.__pends, segs=True)
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
        res = {}
        frame_counters = {}
        
        # Initialize result buffer and frame counter per trajectory.
        for tr in trajects:
            trid = tr.trajid()
            t = tr.time()[:-1]
            
            # This handles trajectories partly out of subrange bounds.
            in_range = (t >= subrange[0]) & (t < subrange[1])
            len_in_range = in_range.sum()
            
            if len_in_range < 1:
                continue
            
            res[trid] = [None]*len_in_range
            frame_counters[trid] = 0
        
        for frame, next_frame in self.iter_subrange(*subrange):
            if history:
                fargs = (self, frame, next_frame, res) + args
            else:
                fargs = (self, frame, next_frame) + args
            frm_res = func(*fargs)
            
            for k, v in frm_res.items():
                res[k][frame_counters[k]] = v
                frame_counters[k] += 1
        
        for k in res.keys():
            res[k] = np.array(res[k])
        return res
    
    def save_config(self, cfg):
        """
        Adds the keys necessary for recreating this sequence into a 
        configuration object. It is the caller's responsibility to do a 
        writeback to file.
        
        Arguments:
        cfg - a ConfigParser object.
        """
        if not cfg.has_section("Particle"):
            cfg.add_section("Particle")
        cfg.set("Particle", "diameter", str(self.part.diam))
        cfg.set("Particle", "density", str(self.part.density))
        
        if not cfg.has_section("Scene"):
            cfg.add_section("Scene")
        cfg.set("Scene", "frame rate", str(self.frate))
        cfg.set("Scene", "first frame", str(self._rng[0]))
        cfg.set("Scene", "last frame", str(self._rng[1] - 1))
        cfg.set("Scene", "apply smoothing", "yes" if self._smooth else "no")
        cfg.set("Scene", "trajectory minimal length", str(self._minlen))
        
        # Need to escape these because of ConfigParser's 'magic variables'.
        cfg.set("Scene", "tracers file", self._trtmpl.replace('%', '%%', 1))
        cfg.set("Scene", "particles file", self._ptmpl.replace('%', '%%', 1))

def read_sequence(conf_fname, smooth=None, traj_min_len=None):
    """
    Read sequence-wide parameters, such as unchanging particle properties and
    frame range. Values are stored in an INI-format file.
    
    Arguments:
    conf_fname - name of the config file
    smooth - whether the sequence shoud use tracers trajectory-smoothing. Used
        to override the config value if present, and supply it if missing. If 
        None and missing, default is False.
    traj_min_len - tells the sequence to ignore trajectories shorter than this
        many frames. Overrides file. If None and file has no value, default is
        0.
    
    Returns:
    a Sequence object initialized with the configuration values found.
    """
    parser = ConfigParser()
    parser.read(conf_fname)

    particle = Particle(
        parser.getfloat("Particle", "diameter"),
        parser.getfloat("Particle", "density"))
    
    frate = parser.getfloat("Scene", "frame rate")
    tracer_tmpl = parser.get("Scene", "tracers file")
    part_tmpl = parser.get("Scene", "particles file")
    frange = (parser.getint("Scene", "first frame"),
        parser.getint("Scene", "last frame") + 1)
    
    # The smoothing option is subject to default/override rules.
    if parser.has_option("Scene", "apply smoothing"):
        if smooth is None:
            smooth = parser.getboolean("Scene", "apply smoothing")
    else:
        if smooth is None:
            smooth = False
    
    # Same goes for traj_min_len
    if parser.has_option("Scene", "trajectory minimal length"):
        if traj_min_len is None:
            traj_min_len = parser.getint("Scene", "trajectory minimal length")
    else:
        if traj_min_len is None:
            traj_min_len = 0
    
    return Sequence(frange, frate, particle, part_tmpl, tracer_tmpl, smooth,
        traj_min_len)
