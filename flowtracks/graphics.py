# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Yosef Meller, Alex Liberzon
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

from pathlib import Path

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


def trajectory_summary(source):
    """Per-trajectory table: ids, points, mean position, mean velocity,
    spatial extent (bounding-box diagonal), path length (sum of step lengths:
    distance actually travelled, back-and-forth motion included) and largest
    jump -- the largest
    one-frame spike: |step_k - (step_(k-1) + step_(k+1)) / 2|, a step that
    disagrees with its two neighbours (a smooth speed-up is not a jump), in
    position units per time unit of ``time`` (frames), so a step across a
    bridged gap is not mistaken for one. Rows sorted by trajid."""
    from flowtracks.writers import _trajectory_arrays

    pos, vel, time, trajid = _trajectory_arrays(source)
    order = np.lexsort((time, trajid))
    pos, vel, time, trajid = pos[order], vel[order], time[order], trajid[order]
    starts = np.flatnonzero(np.r_[True, trajid[1:] != trajid[:-1]])
    n = np.diff(np.r_[starts, len(trajid)])
    mean_pos = np.add.reduceat(pos, starts, axis=0) / n[:, None]
    mean_vel = np.add.reduceat(vel, starts, axis=0) / n[:, None]
    extent = np.linalg.norm(np.maximum.reduceat(pos, starts, axis=0)
                            - np.minimum.reduceat(pos, starts, axis=0), axis=1)
    jump = np.zeros(len(starts))
    step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    step[starts] = 0.0                      # no step across two trajectories
    path = np.add.reduceat(step, starts)
    for k, (a, b) in enumerate(zip(starts, np.r_[starts[1:], len(trajid)])):
        if b - a > 3:
            d = np.diff(pos[a:b], axis=0) / np.diff(time[a:b]).astype(float)[:, None]
            jump[k] = np.linalg.norm(d[1:-1] - 0.5 * (d[:-2] + d[2:]), axis=1).max()
    speed = np.linalg.norm(vel, axis=1)
    return {"trajid": trajid[starts], "n": n, "mean_pos": mean_pos, "mean_vel": mean_vel,
            "extent": extent, "path": path, "jump": jump, "max_speed": np.maximum.reduceat(speed, starts)}


def select_trajectories(source, by="length", n=10, min_points=20, min_extent=0.0, seed=0):
    """Trajectory ids of ``n`` trajectories chosen ``by``:

    "length"   - longest (points), best first.
    "speed"    - fastest (max point speed), best first.
    "path"     - longest path travelled (sum of step lengths), best first:
                 long AND moving -- back-and-forth motion counts in full,
                 tracks that sit still (wall points) rank low.
    "typical"  - one per main flow pattern: k-means (k = n) of the tracks'
                 standardized mean position + mean velocity; from each cluster
                 the track nearest its centre, largest cluster first -- the
                 "most common" tracks in position and velocity.
    "coverage" - fill the flow: k-means (k = n) of the tracks' mean
                 positions splits the volume into n regions; the longest track
                 of each region.
    "jumps"    - largest one-frame spike (see :func:`trajectory_summary`),
                 largest first.

    typical / coverage / jumps only consider tracks with at least
    ``min_points`` points and a spatial extent of at least ``min_extent``
    (position units) -- e.g. to leave out points on a solid, reflecting wall
    that never move. ``source``: anything :func:`trajectory_summary` reads.
    """
    tab = trajectory_summary(source)
    ids = tab["trajid"]
    if by == "length":
        return ids[np.argsort(-tab["n"], kind="stable")[:n]]
    if by == "speed":
        return ids[np.argsort(-tab["max_speed"], kind="stable")[:n]]
    if by == "path":
        return ids[np.argsort(-tab["path"], kind="stable")[:n]]
    if by not in ("typical", "coverage", "jumps"):
        raise ValueError(f"by must be length/speed/path/typical/coverage/jumps, got {by!r}")
    keep = np.flatnonzero((tab["n"] >= min_points) & (tab["extent"] >= min_extent))
    if len(keep) == 0:
        return ids[:0]
    if by == "jumps":
        return ids[keep[np.argsort(-tab["jump"][keep], kind="stable")[:n]]]
    from scipy.cluster.vq import kmeans2

    def standardize(x):
        sd = x.std(0)
        return (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)   # a flat axis stays 0, not NaN

    mp = tab["mean_pos"][keep]
    k = min(n, len(keep))
    if by == "coverage":
        _, label = kmeans2(standardize(mp), k, minit="++", seed=seed)
        lengths = tab["n"][keep]
        picks = [np.flatnonzero(label == c)[np.argmax(lengths[label == c])]
                 for c in range(k) if np.any(label == c)]
        return ids[keep[picks]]
    feats = standardize(np.column_stack([mp, tab["mean_vel"][keep]]))
    centres, label = kmeans2(feats, k, minit="++", seed=seed)
    sizes = np.bincount(label, minlength=k)
    picks = []
    for c in np.argsort(-sizes, kind="stable"):
        members = np.flatnonzero(label == c)
        if len(members):
            picks.append(members[np.argmin(np.linalg.norm(feats[members] - centres[c], axis=1))])
    return ids[keep[picks]]


def plot_trajectories_3d(source, trajids=None, scalars="speed", context=0,
                         plotter=None, cmap="plasma", clim=None, line_width=3,
                         point_size=6, title=None, bar_title=None, show=True,
                         units=None):
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
    units - ``{"pos": "m", "vel": "m s-1", "time": "frame"}`` for axis and
        colour-bar titles (``x [m]``, ``speed [m s-1]``). When omitted and
        ``source`` is a Zarr path, whatever the writer recorded is used;
        otherwise titles stay bare rather than guessed.

    Returns the plotter.
    """
    import pyvista as pv

    from flowtracks.writers import (
        _trajectory_arrays,
        trajectory_polydata,
        trajectory_units,
    )

    if units is None and isinstance(source, (str, Path)):
        units = trajectory_units(source)
    units = units or {}
    pos_u = f" [{units['pos']}]" if units.get("pos") else ""
    scalar_u = f" [{units['vel']}]" if scalars in ("speed", "velocity") and units.get("vel") else ""
    if not scalar_u and scalars == "time" and units.get("time"):
        scalar_u = f" [{units['time']}]"

    poly = trajectory_polydata(source, trajids, units=units or None)
    if plotter is None:
        plotter = pv.Plotter()
    if context:
        pos = _trajectory_arrays(source)[0]
        pick = np.random.default_rng(0).choice(len(pos), min(context, len(pos)), replace=False)
        plotter.add_points(pos[pick], color="grey", opacity=0.15, point_size=2,
                           name="context")
    if poly.n_points:
        bar = {"title": (bar_title or scalars) + scalar_u}
        plotter.add_mesh(poly, scalars=scalars, cmap=cmap, clim=clim, line_width=line_width,
                         render_lines_as_tubes=True, scalar_bar_args=bar, name="tracks")
        plotter.add_mesh(poly.extract_points(np.arange(poly.n_points), adjacent_cells=False),
                         scalars=scalars, cmap=cmap, clim=clim, point_size=point_size,
                         render_points_as_spheres=True, style="points",
                         show_scalar_bar=False, name="points")
    plotter.show_grid(xtitle=f"x{pos_u}", ytitle=f"y{pos_u}", ztitle=f"z{pos_u}",
                       fmt="%.3g",
                      n_xlabels=3, n_ylabels=3, n_zlabels=3)
    plotter.add_axes()
    if title:
        plotter.add_text(title, font_size=10)
    if show:
        plotter.show()
    return plotter


def _far_walls(bounds, toward_camera, step):
    """Grey back walls (the three box faces away from the camera) and their
    grid lines every ``step``, as two PolyData."""
    import pyvista as pv

    lo, hi = np.asarray(bounds[0::2], float), np.asarray(bounds[1::2], float)
    walls, lines = [], []
    for ax in range(3):
        level = lo[ax] if toward_camera[ax] > 0 else hi[ax]
        u, v = [a for a in range(3) if a != ax]
        corners = np.zeros((4, 3))
        corners[:, ax] = level
        corners[:, u] = [lo[u], hi[u], hi[u], lo[u]]
        corners[:, v] = [lo[v], lo[v], hi[v], hi[v]]
        walls.append(pv.PolyData(corners, faces=[4, 0, 1, 2, 3]))
        # grid lines lifted a hair off the wall so they never z-fight it
        lift = 1e-3 * (hi[ax] - lo[ax]) * (1 if level == lo[ax] else -1)
        for a, b in ((u, v), (v, u)):
            ticks = np.arange(np.ceil(lo[a] / step) * step, hi[a] + 1e-9, step)
            for t in ticks:
                seg = np.zeros((2, 3))
                seg[:, ax] = level + lift
                seg[:, a] = t
                seg[:, b] = [lo[b], hi[b]]
                lines.append(pv.Line(seg[0], seg[1]))
    return pv.merge(walls), pv.merge(lines)


def _box_axis_labels(pl, bounds, toward_camera, step, titles, font_size):
    """Tick labels every ``step`` and axis titles along the box edges nearest
    the camera on the floor (x, z) and on the far vertical edge (y), as
    ParaView draws them. (VTK's cube-axes actor caps the label count at its
    own tick spacing, so it cannot label every ``step``.)"""
    lo, hi = np.asarray(bounds[0::2], float), np.asarray(bounds[1::2], float)
    near = np.where(np.asarray(toward_camera) > 0, hi, lo)
    far = np.where(np.asarray(toward_camera) > 0, lo, hi)
    out = np.where(np.asarray(toward_camera) > 0, 1.0, -1.0)  # outward, camera side
    off = 0.035 * np.max(hi - lo)
    edges = (  # axis, anchor of the edge, outward offset of its labels
        (0, np.array([0.0, lo[1], near[2]]), np.array([0.0, -off, out[2] * off])),
        (2, np.array([near[0], lo[1], 0.0]), np.array([out[0] * off, -off, 0.0])),
        (1, np.array([far[0], 0.0, near[2]]), np.array([-out[0] * off, 0.0, out[2] * off])),
    )
    for ax, anchor, shift in edges:
        ticks = np.arange(np.ceil(lo[ax] / step) * step, hi[ax] + 1e-9, step)
        pts = np.tile(anchor + shift, (len(ticks), 1))
        pts[:, ax] = ticks
        pl.add_point_labels(pts, [f"{t:.0f}" for t in ticks], font_size=font_size,
                            text_color="black", shape=None, show_points=False,
                            always_visible=True, name=f"ticks{ax}")
        mid = anchor + 2.6 * shift
        mid[ax] = (lo[ax] + hi[ax]) / 2
        pl.add_point_labels([mid], [titles[ax]], font_size=int(font_size * 1.3),
                            text_color="black", shape=None, show_points=False,
                            always_visible=True, name=f"title{ax}")


def _use_font_file(pl, font_file):
    """Point every text of the plotter at a TrueType file. (VTK's built-in
    sans font draws '[' and ']' as parentheses.)"""
    props = []
    for actor in pl.actors.values():
        if hasattr(actor, "GetTextProperty"):
            props.append(actor.GetTextProperty())
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        if mapper is not None and hasattr(mapper, "GetLabelTextProperty"):
            props.append(mapper.GetLabelTextProperty())
        # add_point_labels: the style sits on the label hierarchy feeding the mapper
        source = mapper.GetInputAlgorithm() if hasattr(mapper, "GetInputAlgorithm") else None
        if source is not None and hasattr(source, "GetTextProperty"):
            props.append(source.GetTextProperty())
    for bar in pl.scalar_bars.values():
        props += [bar.GetTitleTextProperty(), bar.GetLabelTextProperty()]
    for prop in props:
        prop.SetFontFamily(4)  # VTK_FONT_FILE
        prop.SetFontFile(str(font_file))


def animate_trajectories_3d(source, out_dir, frames=None, *, component=1, tail=20,
                            pos_scale=1.0, bounds=None, cmap="coolwarm",
                            clim=(-0.5, 0.5), bar_title=None, bar_labels=21,
                            bar_fmt="%.2f", view_direction=(-1.0, 1.0, 1.0),
                            view_up=(0.0, 1.0, 0.0), zoom=1.0, window_center=(0.0, 0.0),
                            window_size=(1920, 1080), point_size=6.0,
                            grid_step=500.0, axis_titles=("x", "y", "z"),
                            wall_color=(0.54, 0.54, 0.54), background="white",
                            font_size=20, font_file=None, outlines=(),
                            outline_color="black", outline_width=3.0, movie=None,
                            fps=10, label=None):
    """Off-screen movie frames of trajectories as trailing beads (PyVista).

    Every frame shows each particle's last ``tail`` positions as shaded
    spheres coloured by one velocity component (``component``: 0, 1, 2), in a
    box with grey back walls, a ``grid_step`` grid and labelled axes, seen
    with a fixed parallel-projection camera -- the usual ParaView look.

    source - zarr path / ZarrScene / Scene, as :func:`plot_trajectories_3d`.
    out_dir - PNG frames are written as ``frame_0000.png`` ...
    frames - frame numbers to render (default: every frame of the source).
    pos_scale - multiplies positions (e.g. 1000 for m -> mm axes).
    bounds - (x0, x1, y0, y1, z0, z1) of the box, in scaled units (default:
        the data's 0.5-99.5 percentile box).
    view_direction - from the scene towards the camera; ``view_up`` the
        screen's up; ``zoom`` > 1 enlarges; ``window_center`` shifts the
        projection centre in normalized window units (VTK SetWindowCenter).
    movie - if given, the frames are also encoded to this .mp4 with ffmpeg.
    label - optional ``str.format`` text drawn bottom-left, given ``frame``.
    font_file - TrueType file for all text (VTK's built-in font has no
        square brackets).
    outlines - (N, 3) point arrays in scaled units, each drawn as a closed
        line (e.g. the top and bottom rims of a cylindrical vessel).

    Returns the list of PNG paths.
    """
    import shutil
    import subprocess

    import pyvista as pv

    from flowtracks.writers import _trajectory_arrays

    pos, vel, time, _ = _trajectory_arrays(source)
    pos = pos * pos_scale
    value = vel[:, component]
    order = np.argsort(time, kind="stable")
    pos, value, time = pos[order], value[order], time[order]
    if frames is None:
        frames = np.unique(time)
    if bounds is None:
        lo, hi = np.percentile(pos, 0.5, axis=0), np.percentile(pos, 99.5, axis=0)
        bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direction = np.asarray(view_direction, float)
    direction /= np.linalg.norm(direction)
    walls, grid = _far_walls(bounds, direction, grid_step)

    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background(background)
    pl.add_mesh(walls, color=wall_color, lighting=False, name="walls")
    pl.add_mesh(grid, color="black", line_width=1.5, name="grid")
    for k, ring in enumerate(outlines):
        ring = np.asarray(ring, float)
        line = pv.lines_from_points(np.vstack([ring, ring[:1]]))
        pl.add_mesh(line, color=outline_color, line_width=outline_width,
                    render_lines_as_tubes=True, name=f"outline{k}")
    _box_axis_labels(pl, bounds, direction, grid_step, axis_titles, font_size)
    pl.add_axes(viewport=(0.0, 0.82, 0.12, 1.0))
    centre = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2,
                       (bounds[4] + bounds[5]) / 2])
    size = np.linalg.norm(np.subtract(bounds[1::2], bounds[0::2]))
    pl.enable_parallel_projection()
    pl.camera.focal_point = centre
    pl.camera.position = centre + direction * 4 * size
    pl.camera.up = view_up
    pl.reset_camera(bounds=list(bounds))
    pl.camera.parallel_scale /= zoom
    pl.camera.SetWindowCenter(*window_center)

    bar = dict(title=bar_title or "", vertical=True, position_x=0.93, position_y=0.02,
               height=0.96, width=0.012, n_labels=bar_labels, fmt=bar_fmt,
               color="black", title_font_size=int(font_size * 1.2),
               label_font_size=font_size)
    paths = []
    starts = np.searchsorted(time, np.asarray(frames) - tail + 1, side="left")
    ends = np.searchsorted(time, np.asarray(frames), side="right")
    for i, (frame, s, e) in enumerate(zip(frames, starts, ends)):
        beads = pv.PolyData(pos[s:e] if e > s else np.zeros((0, 3)))
        if e > s:
            beads.point_data["value"] = value[s:e]
            pl.add_mesh(beads, scalars="value", cmap=cmap, clim=clim,
                        point_size=point_size, render_points_as_spheres=True,
                        style="points", scalar_bar_args=bar, name="beads")
        if label:
            pl.add_text(label.format(frame=frame), position="lower_left",
                        font_size=font_size, color="black", name="label")
        if font_file:
            _use_font_file(pl, font_file)
        path = out_dir / f"frame_{i:04d}.png"
        pl.screenshot(str(path))
        paths.append(path)
    pl.close()

    if movie is not None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found; the PNG frames are in " + str(out_dir))
        subprocess.run([ffmpeg, "-v", "error", "-y", "-framerate", str(fps),
                        "-i", str(out_dir / "frame_%04d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", str(movie)], check=True)
    return paths
