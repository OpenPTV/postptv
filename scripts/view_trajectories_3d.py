#!/usr/bin/env python
"""Interactive 3D view of the longest and fastest trajectories of one or more runs.

One row per run store, one column per selection (longest / fastest), all
views linked (rotate one, all follow). Tracks are coloured by speed along
their length over a faint sample of every particle in the run, so it is easy
to see whether the extremes are real flow features or tracking artefacts
(e.g. 2-6 point tracks jumping several mm per frame).

Built on flowtracks.graphics.select_trajectories / plot_trajectories_3d
(PyVista: GPU-rendered, interactive even with every track of a run shown).

Usage:
    uv run --extra vtk python scripts/view_trajectories_3d.py RUN.zarr [RUN.zarr ...]
        [-n 20] [--by length speed] [--scalars speed] [--context 20000]
        [--screenshot out.png]

RUN.zarr is any store with a trajectories/ group (openptv-cloud res/run.zarr,
a flowtracks Zarr export). --screenshot renders off-screen to a PNG instead
of opening a window.
"""
import argparse
from pathlib import Path

import pyvista as pv

from flowtracks.graphics import plot_trajectories_3d, select_trajectories

TITLES = {"length": "longest", "speed": "fastest"}


def run_label(store: Path) -> str:
    """Short name for a store: .../wp1/test/res/run.zarr -> wp1/test."""
    parts = [p for p in store.resolve().parts if p not in ("res", store.name)]
    return "/".join(parts[-2:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stores", nargs="+", type=Path, help="run.zarr stores (one row each)")
    ap.add_argument("-n", type=int, default=20, help="tracks per panel (default 20)")
    ap.add_argument("--by", nargs="+", choices=list(TITLES), default=list(TITLES),
                    help="selections, one column each (default: length speed)")
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
            ids = select_trajectories(store, by, args.n)
            title = f"{run_label(store)}: {len(ids)} {TITLES[by]} (colour = {args.scalars})"
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
