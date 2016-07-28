#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Runs an analysis of a tracer/inertial scene encoded by two FHDF files.

Created on Sun Aug 10 17:22:04 2014

@author: yosef
"""

from flowtracks.scene import read_dual_scene
from flowtracks.analysis import analysis, FluidVelocitiesAnalyser
    
if __name__ == "__main__":
    from flowtracks.interpolation import interpolant
    
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
    
    interp = interpolant(args.method, args.neighbs, args.param)
    scene = read_dual_scene(args.config)
    analysers = [ FluidVelocitiesAnalyser(interp) ]
    analysis(scene, args.output, args.config, analysers)
    
