# -*- coding: utf-8 -*-
# Created on Sun Sep 22 16:11:34 2013

"""
Various specialized graphing routines. The Probability Density Function
graphing is best accessed by calling :func:`pdf_graph` on the raw data, but
you can generate the PDF from the data separately (e.g. using
:func:`pdf_bins`) and calling :func:`generalized_histogram_disp` on the
result.

The other facility here is a function to plot a time-dependent 3D vector as
3 component subplots, which is another customary presentation in fluid
dynamics circles. See :func:`plot_vectors`.

3D trajectories: :func:`select_trajectories` picks the longest or fastest
tracks of a store, :func:`plot_trajectories_3d` draws them interactively with
PyVista (VTK, GPU-rendered polylines -- the ``vtk`` extra), from the same
PolyData the ``.vtp`` ParaView writer saves.
"""

import matplotlib.pyplot as pl
import numpy as np


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

def generalized_histogram_disp(hist, bin_edges, log_bins=False,
    log_density=False, marker='o'):
    """
    Draws a given histogram  according to the visual custom of the fluid
    dynamics community.

    Arguments:
    hist - an array containing the number of values (or density) for each bin.
    bin_edges - the start value of each bin, same length as ``hist``.
    log_bins - indicates that the bin edges are log-spaced.
    log_densify - Show the log of the probability density value. May cause
        problems if ``log_bins`` is True.
    marker - marker style for matplotlib.

    Returns:
    the list of lines drawn, Matplotlib objects.
    """
    if log_bins:
        plt = pl.loglog if log_density else pl.semilogx
    else:
        plt = pl.semilogy if log_density else pl.plot

    lines = plt(bin_edges, hist, marker)
    pl.ylabel("Probability density [-]")

    return lines

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
    hist, bin_edges = pdf_bins(data, num_bins, log)
    generalized_histogram_disp(hist, bin_edges[:-1], log, log_density,
        marker='-' + marker)

def plot_vectors(vecs, indep, xlabel, fig=None, marker='-',
    ytick_dens=None, yticks_format=None, unit_str="", common_scale=None,
    arrows=None, arrow_color=None):
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
    arrows - an (n,3) array of values to represent as vertical arrows attached
        to each trajectory point.
    arrow_color - a matplotlib color spec for the arrow bodies.

    Returns:
    fig - the figure object used for plotting.
    """
    fig = pl.figure(None if fig is None else fig.number)
    u = np.zeros(vecs.shape[0])

    labels = ("X " + unit_str, "Y" + unit_str, "Z" + unit_str)
    for subplt in range(3):
        pl.subplot(3,1,subplt + 1)
        pl.plot(indep, vecs[:,subplt], marker)
        pl.gca().get_xaxis().set_visible(False)
        pl.grid()
        pl.ylabel(labels[subplt])

        if yticks_format is not None:
             pl.gca().get_yaxis().set_major_formatter(yticks_format)

        if common_scale is not None:
            pl.ylim(np.r_[-common_scale, common_scale] + vecs[:,subplt].mean())
        if ytick_dens is not None:
            loc, _ = pl.yticks()
            pl.yticks(np.linspace(vecs[:,subplt].min(), vecs[:,subplt].max(),
                    ytick_dens))

        if arrows is not None:
            pl.quiver(indep, vecs[:,subplt], u, arrows[:,subplt],
                scale=30, width=1e-3, color=arrow_color)

    pl.gca().get_xaxis().set_visible(True)
    pl.xlabel(xlabel)
    return fig


def select_trajectories(source, by="length", n=10):
    """Trajectory ids of the ``n`` longest (``by="length"``, points) or fastest
    (``by="speed"``, max point speed) trajectories, best first.

    ``source``: a zarr path with a ``trajectories/`` group or a ``Scene``/
    ``ZarrScene`` (anything :func:`flowtracks.writers.trajectory_polydata`
    accepts).
    """
    from flowtracks.writers import _trajectory_arrays

    _, vel, _, trajid = _trajectory_arrays(source)
    ids, inverse, counts = np.unique(trajid, return_inverse=True, return_counts=True)
    if by == "length":
        score = counts
    elif by == "speed":
        score = np.full(len(ids), -np.inf)
        np.maximum.at(score, inverse, np.linalg.norm(vel, axis=1))
    else:
        raise ValueError(f"by must be 'length' or 'speed', got {by!r}")
    return ids[np.argsort(-score, kind="stable")[:n]]


def plot_trajectories_3d(source, trajids=None, scalars="speed", context=0,
                         plotter=None, cmap="plasma", clim=None, line_width=3,
                         point_size=6, title=None, bar_title=None, show=True):
    """Interactive 3D trajectory view (PyVista): polylines coloured by a point
    array (``speed``, ``time``, ``trajid``), points drawn as spheres so short
    tracks stay visible.

    trajids - draw only these (e.g. from :func:`select_trajectories`);
        None draws every trajectory.
    context - also draw this many randomly sampled points of ALL
        trajectories, faint grey, to show where the selection sits.
    plotter - draw into an existing ``pyvista.Plotter`` (or a subplot of
        one); a new one is created otherwise.
    bar_title - colour-bar title (default: ``scalars``). PyVista merges bars
        with the same title into one shared colour range, so give each
        subplot its own title to keep its own range.
    show - call ``plotter.show()``; pass False to compose or screenshot.

    Returns the plotter.
    """
    import pyvista as pv

    from flowtracks.writers import _trajectory_arrays, trajectory_polydata

    poly = trajectory_polydata(source, trajids)
    if plotter is None:
        plotter = pv.Plotter()
    if context:
        pos = _trajectory_arrays(source)[0]
        pick = np.random.default_rng(0).choice(len(pos), min(context, len(pos)), replace=False)
        plotter.add_points(pos[pick], color="grey", opacity=0.15, point_size=2,
                           name="context")
    if poly.n_points:
        bar = {"title": bar_title or scalars}
        plotter.add_mesh(poly, scalars=scalars, cmap=cmap, clim=clim, line_width=line_width,
                         render_lines_as_tubes=True, scalar_bar_args=bar, name="tracks")
        plotter.add_mesh(poly.extract_points(np.arange(poly.n_points), adjacent_cells=False),
                         scalars=scalars, cmap=cmap, clim=clim, point_size=point_size,
                         render_points_as_spheres=True, style="points",
                         show_scalar_bar=False, name="points")
    plotter.show_grid(xtitle="x", ytitle="y", ztitle="z", fmt="%.3g",
                      n_xlabels=3, n_ylabels=3, n_zlabels=3)
    plotter.add_axes()
    if title:
        plotter.add_text(title, font_size=10)
    if show:
        plotter.show()
    return plotter
