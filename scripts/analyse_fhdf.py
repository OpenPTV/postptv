# -*- coding: utf-8 -*-
"""
Runs an analysis of a tracer/inertial scene encoded by two FHDF files.

Created on Sun Aug 10 17:22:04 2014

@author: yosef

References:
[1] https://docs.python.org/2/library/itertools.html
"""

import itertools as it, numpy as np, tables
from flowtracks.scene import read_dual_scene

def pairwise(iterable):
    """
    copied from itertools documentation, [1]
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = it.tee(iterable)
    next(b, None)
    return it.izip(a, b)
    
def analysis(scene, analysis_file, conf_file, interp, frame_range=-1):
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
    interp - the Interpolant object to use for fluid velocity etc.
    frame_range - if -1 no adjustment is necessary, otherwise see 
        Scene.dual_scene_iterator()
    """
    # Structure the output file:
    descr = np.dtype([('trajid', int, 1), ('fluid_vel', float, 3), 
                      ('rel_vel', float, 3)])
    outfile = tables.openFile(analysis_file, "w", title="Analysis results.")
    table = outfile.create_table('/', 'analysis', descr)
    table.attrs['config'] = conf_file
    table.attrs['trajects'] = scene.get_particles_path()
    
    for frame, next_frame in pairwise(scene.iter_frames(frame_range)):
        vel_interp = interp(frame.tracers.pos(), frame.particles.pos(),
            frame.tracers.velocity())
        
        # Add velocities to frame for continued analysis:
        
        # Save current analysis frame.
        # Note that since the rames are iterated in time order, we don't need
        # to duplicate the time column from the trajectory, and it will be 
        # read in the right order.
        length = len(vel_interp) # assume all analyses are the size of frame
        arr = np.empty(length, dtype=descr)
        arr['trajid'] = frame.particles.trajid()
        
        arr['fluid_vel'] = vel_interp
        arr['rel_vel'] = frame.particles.velocity() - vel_interp
        
        table.append(arr)
    
    # Wrap up and close.
    table.cols.trajid.createIndex()
    outfile.flush()
    outfile.close()
    
if __name__ == "__main__":
    from flowtracks.interpolation import Interpolant
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="file containing configuration values.")
    parser.add_argument('output', help="A directory where the generated data "\
        "should be saved as one file per trajectory.")
    parser.add_argument("--method", choices=['inv', 'rbf', 'corrfun'], 
        default='inv', help="Interpolation method (*inv*, rbf, corrfun)")
    parser.add_argument("--neighbs", type=int, help="Number of closest "
        "neighbours from which to interpolate.", default=4)
    parser.add_argument('--param', '-p', help="Interpolation adjustment "
        "parameter. Inverse power for inv, epsilon for RBF, filename for corrfun")
    args = parser.parse_args()
    
    interp = Interpolant(args.method, args.neighbs, args.param)
    scene = read_dual_scene(args.config)
    analysis(scene, args.output, args.config, interp)
    