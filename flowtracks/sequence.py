# -*- coding: utf-8 -*-
from flowtracks.io import collect_particles, collect_particles_mat
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
        
        self._pfmt = 'mat' if part_tmpl.endswith('.mat') else 'acc'
        self._trfmt = 'mat' if tracer_tmpl.endswith('.mat') else 'acc'
    
    def part_fname(self):
        return self._ptmpl
    
    def __iter__(self):
        """
        Iterate over frames. For each frame return the data for the tracers and
        particles in it.
        """
        self._frame = self._rng[0]
        return self
    
    def next(self):
        if self._frame == self._rng[1]:
            raise StopIteration
        
        data = [None]*2
        fnames = [self._ptmpl, self._trtmpl]
        formats = [self._pfmt, self._trfmt]
        
        for dix, fname in enumerate(fnames):
            if formats[dix] == 'acc':
                data[dix] = collect_particles(fname, self._frame,
                    path_seg=True)
            elif formats[dix] == 'mat':
                data[dix] = collect_particles_mat(fname, self._frame,
                    path_seg=True)
        
        self._frame += 1
        return tuple(data)

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
