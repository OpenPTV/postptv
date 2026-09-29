#!/usr/bin/env python
"""Interactive 3D view of representative trajectories of one or more runs.

One row per run store, one column per selection, all
views linked (rotate one, all follow). Tracks are coloured by speed along
their length over a faint sample of every particle in the run. Selections
(flowtracks.graphics.select_trajectories):

  typical   one track per main flow pattern (k-means of mean position +
            mean velocity) -- the "most common" tracks
  coverage  fill the flow: the longest track in each of n regions of the volume
  jumps     the largest one-frame spikes (a step disagreeing with both neighbours)
            (title shows the largest, in mm) -- are there any?
  path      the longest distance travelled (sum of steps; back-and-forth counts)
  length / speed   the longest / fastest

typical / coverage / jumps skip tracks that never leave --min-extent-mm (points
on a solid, reflecting wall, e.g. a silicone phantom) and those shorter than
--min-points.

Built on flowtracks.graphics.select_trajectories / plot_trajectories_3d
(PyVista: GPU-rendered, interactive even with every track of a run shown).

Usage:
    uv run --extra vtk python scripts/view_trajectories_3d.py RUN.zarr [RUN.zarr ...]
        [-n 12] [--by typical coverage jumps] [--min-extent-mm 1] [--min-points 20]
        [--scalars speed] [--context 20000] [--screenshot out.png]

RUN.zarr is any store with a trajectories/ group in metres (openptv-cloud
res/run.zarr, a flowtracks Zarr export). --screenshot renders off-screen to a
PNG instead of opening a window.
"""
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from flowtracks.graphics import (
    plot_trajectories_3d,
    select_trajectories,
    trajectory_summary,
)

TITLES = {"typical": "typical", "coverage": "filling the flow", "jumps": "largest jumps",
          "length": "longest", "speed": "fastest", "path": "longest path travelled"}


def run_label(store: Path) -> str:
    """Short name for a store: .../wp1/test/res/run.zarr -> wp1/test."""
    parts = [p for p in store.resolve().parts if p not in ("res", store.name)]
    return "/".join(parts[-2:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stores", nargs="+", type=Path, help="run.zarr stores (one row each)")
    ap.add_argument("-n", type=int, default=12, help="tracks per panel (default 12)")
    ap.add_argument("--by", nargs="+", choices=list(TITLES), default=["typical", "coverage", "jumps"],
                    help="selections, one column each (default: typical coverage jumps)")
    ap.add_argument("--min-extent-mm", type=float, default=1.0,
                    help="typical/coverage/jumps ignore tracks that stay inside this (wall points)")
    ap.add_argument("--min-points", type=int, default=20, help="... and tracks shorter than this")
    ap.add_argument("--scalars", default="speed", choices=["speed", "time", "trajid"],
                    help="colour field (default speed)")
    ap.add_argument("--context", type=int, default=20000,
                    help="all-particle sample points drawn faint grey (0 = none)")
    ap.add_argument("--screenshot", type=Path, help="save a PNG off-screen instead of a window")
    args = ap.parse_args(argv)

    rows, cols = len(args.stores), len(args.by)
    pl = pv.Plotter(shape=(rows, cols), window_size=(800 * cols, 550 * rows),
                    off_screen=args.screenshot is not None,
                    title="Trajectories: " + ", ".join(TITLES[b] for b in args.by))
    for r, store in enumerate(args.stores):
        for c, by in enumerate(args.by):
            pl.subplot(r, c)
            ids = select_trajectories(store, by, args.n, min_points=args.min_points,
                                      min_extent=args.min_extent_mm / 1000.0)
            title = f"{run_label(store)}: {len(ids)} {TITLES[by]} (colour = {args.scalars})"
            if by == "path" and len(ids):
                tab = trajectory_summary(store)
                sel = np.isin(tab["trajid"], ids)
                title += (f"\npath {1000 * tab['path'][sel].min():.0f}-{1000 * tab['path'][sel].max():.0f} mm, "
                          f"{tab['n'][sel].min()}-{tab['n'][sel].max()} frames")
            if by == "jumps" and len(ids):
                tab = trajectory_summary(store)
                worst = 1000.0 * tab["jump"][np.isin(tab["trajid"], ids)].max()
                title += f"\nlargest one-frame spike {worst:.3f} mm"

            plot_trajectories_3d(store, trajids=ids, scalars=args.scalars,
                                 context=args.context, plotter=pl, title=title,
                                 bar_title=f"{args.scalars} {run_label(store)} {TITLES[by]}",
                                 show=False)
    pl.link_views()
    if args.screenshot:
        pl.screenshot(str(args.screenshot))
        print(f"saved {args.screenshot}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
