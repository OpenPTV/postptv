# -*- coding: utf-8 -*-
from .io import collect_particles_generic, trajectories, infer_format
from .particle import Particle
from ConfigParser import SafeConfigParser

class Sequence(object):
    def __init__(self, frange, frate, particle, part_tmpl, tracer_tmpl):
        """
        Arguments:
        frange - tuple, (first frame #, after last frame #)
        frate - the frame rate at which the scene was shot.
        particle - a Particle object representing the suspended particles'
            properties.
        part_tmpl, tracer_tmpl - the templates for filenames of particles and
            tracers respectively. Should contain one %d for the frame number.
        """
        self.part = particle
        self.frate = frate
        
        self._rng = frange
        self._ptmpl = part_tmpl
        self._trtmpl = tracer_tmpl
        
        # This is just a chache. Else trajectories() would check every time.
        self._pfmt = infer_format(part_tmpl)
        self._trfmt = infer_format(tracer_tmpl)
    
    def part_fname(self):
        return self._ptmpl
    
    def part_format(self):
        return self._pfmt
    
    def range(self):
        return self._rng
    
    def __iter__(self):
        """
        Iterate over frames. For each frame return the data for the tracers and
        particles in it.
        """
        self._frame = self._rng[0]
        
        traj = [None]*2
        fnames = [self._ptmpl, self._trtmpl]
        formats = [self._pfmt, self._trfmt]
        
        for dix, fname in enumerate(fnames):
            traj[dix] = trajectories(fname, self._rng[0], self._rng[1],
                self.frate, formats[dix])

        self.__ptraj, self.__ttraj = tuple(traj)
        return self
    
    def next(self):
        if self._frame == self._rng[1]:
            del self.__ptraj, self.__ttraj
            raise StopIteration
        
        parts = collect_particles_generic(self.__ptraj, self._frame, True)
        tracers = collect_particles_generic(self.__ttraj, self._frame, True)
        
        self._frame += 1
        return parts, tracers

def read_sequence(conf_fname):
    """
    Read sequence-wide parameters, such as unchanging particle properties and
    frame range. Values are stored in an INI-format file.
    
    Arguments:
    conf_fname - name of the config file
    
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
    
    return Sequence(frange, frate, particle, part_tmpl, tracer_tmpl)
