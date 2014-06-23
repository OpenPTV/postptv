# -*- coding: utf-8 -*-
"""
Various specialized graphing routines collected from scripts that do the same 
thing for different data.

Created on Sun Sep 22 16:11:34 2013

@author: yosef
"""

import numpy as np, matplotlib.pyplot as pl

def pdf_bins(data, num_bins, log_bins=False):
    """
    Generate a PDF of the given data possibly with logarithmic bins, ready for
    using in a histogram plot.
    
    Arguments:
    data - the samples to histogram.
    bins - the number of bins in the histogram.
    log_bins - if True, the bin edges are equally spaced on the log scale, 
        otherwise they are linearly spaced (a normal histogram). If True, 
        ``data`` should not contain zeros.
    
    Returns:
    hist - num_bins-lenght array of density values for each bin.
    bin_edges - array of size num_bins + 1 with the edges of the bins including
        the ending limit of the bins.
    """
    if log_bins:
        data = data[data > 0] 
        minv = np.min(data)
        bins = np.logspace(np.log10(minv), np.log10(data.max()), num_bins + 1)
    else:
        bins = num_bins
    
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    return hist, bin_edges
    
def pdf_graph(data, num_bins, log=False, log_density=False, marker='o'):
    """
    Draw a PDF of the given data, according to the visual custom of
    the fluid dynamics community, and possibly with logarithmic bins.
    
    Arguments:
    data - the samples to histogram.
    bins - the number of bins in the histogram.
    log - if True, the bin edges are equally spaced on the log scale, otherwise
        they are linearly spaced (a normal histogram). If True, ``data`` should
        not contain zeros.
    log_density - Show the log of the probability density value. Only if log 
        is False.
    marker - override the circle marker with any string acceptable to 
        matplotlib.
    """
    if log:
        data = data[data > 0] 
        minv = np.min(data)
        bins = np.logspace(np.log10(minv), np.log10(data.max()), num_bins + 1)
        plt = pl.semilogx
    else:
        bins = num_bins
        plt = pl.semilogy if log_density else pl.plot 
    
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    plt(bin_edges[:-1], hist, '-' + marker)
    pl.ylabel("Probability density [-]")

def plot_vectors(vecs, indep, xlabel, fig=None, marker='-', 
    ytick_dens=None, yticks_format=None, unit_str=""):
    """
    Plot 3D vectors as 3 subplots sharing the same independent axis.
    
    Arguments:
    vecs - an (n,3) array, with n vectors to plot against the independent
        variable.
    indep - the corresponding n values of the independent variable.
    xlabel - label for the independent axis.
    fig - an optional figure object to use. If None, one will be created.
    ytick_dens - if not None, place this many yticks on each subplot, instead
        of the automatic tick marks.
    yticks_format - a pyplot formatter object.
    unit_str - a string to add to the Y labels representing the vector's units.
    
    Returns:
    fig - the figure object used for plotting.
    """
    fig = pl.figure(None if fig is None else fig.number)
    
    labels = ("X " + unit_str, "Y" + unit_str, "Z" + unit_str)
    for subplt in xrange(3):
        pl.subplot(3,1,subplt + 1)
        pl.plot(indep, vecs[:,subplt], marker)
        pl.gca().get_xaxis().set_visible(False)
        pl.grid()
        pl.ylabel(labels[subplt])
        
        if yticks_format is not None:
             pl.gca().get_yaxis().set_major_formatter(yticks_format)
        
        if ytick_dens is not None:
            loc, _ = pl.yticks()
            pl.yticks(np.linspace(vecs[:,subplt].min(), vecs[:,subplt].max(), 
                    ytick_dens))
    
    pl.gca().get_xaxis().set_visible(True)
    pl.xlabel(xlabel)
    return fig
