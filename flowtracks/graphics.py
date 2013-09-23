# -*- coding: utf-8 -*-
"""
Various specialized graphing routines collected from scripts that do the same 
thing for different data.

Created on Sun Sep 22 16:11:34 2013

@author: yosef
"""

import numpy as np, matplotlib.pyplot as pl

def pdf_graph(data, num_bins, log=False):
    """
    Draw a normalized PDF of the given data, according to the visual custom of
    the fluid dynamics community, and possibly with logarithmioc bins.
    
    Arguments:
    data - the samples to histogram.
    bins - the number of bins in the histogram.
    log - if True, the bin edges are equally spaced on the log scale, otherwise
        they are linearly spaced (a normal histogram). If True, ``data`` should
        not contain zeros.
    """
    if log:
        minv = np.min(data)
        bins = np.logspace(np.log10(minv), np.log10(data.max()), num_bins + 1)
        plt = pl.semilogx
    else:
        bins = num_bins
        plt = pl.plot
        
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt(bin_edges[:-1], hist, '-o')
    pl.ylabel("Probability density [-]")
