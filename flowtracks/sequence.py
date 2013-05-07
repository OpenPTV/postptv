# -*- coding: utf-8 -*-
import numpy as np

from .io import collect_particles_generic, trajectories, infer_format
from .particle import Particle
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
        if not (self.__ttraj is None):
            return self
        
        # Make sure the tracers cache is populated:
        trac_trajects = trajectories(self._trtmpl, self._rng[0], self._rng[1],
                self.frate, self._trfmt)
        
        if self._smooth:
            self.__ttraj = [tr.smoothed() for tr in trac_trajects]
        else:
            self.__ttraj = trac_trajects
        
        return self
    
    def iter_subrange(self, first, last):
        self._act_rng = (first, last)
        return self
    
    def next(self):
        if self._frame == self._act_rng[1]:
            del self._act_rng
            raise StopIteration
        
        parts = collect_particles_generic(self.__ptraj, self._frame, True)
        tracers = collect_particles_generic(self.__ttraj, self._frame, True)
        
        self._frame += 1
        return parts, tracers

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
