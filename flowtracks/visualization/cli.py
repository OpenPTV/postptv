# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""Command-line entry point for the browser 3D viewer.

Examples:
    uv run python -m flowtracks.visualization data/tracers.zarr
    flowtracks-viz data/tracers.zarr --mesh vessel.vtp --color-by time
    flowtracks-viz data/tracers.zarr --screenshot shot.png --max-trajectories 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .scene import build_polydata, load_mesh, load_scene, load_volume


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Open a browser-based 3D viewer of flowtracks trajectory data "
            "(pyvista scene + trame WebSocket view)."
        ),
        epilog="The viewer renders client-side with VTK.wasm when vtk >= 9.7 "
        "is available; otherwise it falls back to server-side rendering.",
    )
    p.add_argument("zarr", help="Path to a .zarr store with trajectories/ "
                              "(flowtracks export or openptv2 RunStore).")
    p.add_argument("--mesh", metavar="VTP", default=None,
                   help="Optional vessel/geometry mesh to overlay "
                   "(.vtp/.vtr/.ply, whatever pyvista can read).")
    p.add_argument("--volume", metavar="VTI", default=None,
                   help="Optional Eulerian voxel volume to overlay "
                   "(.vti/.vtr).")
    p.add_argument("--volume-only", action="store_true",
                   help="Render only the volume (no trajectories) -- a "
                   "dedicated Eulerian view.")
    p.add_argument("--time-range", metavar="T0,T1", default=None,
                   help="Optional time window to keep (either bound may be "
                   "open, e.g. '10001,' or ',10150').")
    p.add_argument("--max-trajectories", type=int, default=2000, metavar="N",
                   help="Keep at most N trajectories (evenly spaced). "
                   "0 means keep everything.")
    p.add_argument("--color-by", choices=["speed", "time"], default="speed",
                   help="Point-data colormap by 'speed' (default) or 'time'.")
    p.add_argument("--port", type=int, default=0,
                   help="Port to serve on (0 picks a free one, default).")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not open the browser tab automatically.")
    p.add_argument("--screenshot", metavar="PNG", default=None,
                   help="Render off-screen and exit instead of opening "
                   "the browser (no window will open).")
    return p


def parse_time_range(text):
    """Parse a 'T0,T1' spec; either side may be left open."""
    if text is None:
        return None
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"--time-range must look like 'T0,T1', got {text!r}"
        )

    def _num(s):
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"could not parse time bound {s!r} in {text!r}")

    return _num(parts[0]), _num(parts[1])


def render_screenshot(data, *, mesh_path=None, volume_path=None,
                      color_by="speed", volume_only=False, out_path, size=(1280, 900)):
    """Render the scene off-screen and save a PNG. Returns the path."""
    import pyvista as pv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=size)
    if not volume_only:
        plotter.add_mesh(
            build_polydata(data),
            scalars=color_by,
            cmap="viridis",
            line_width=1.5,
            show_scalar_bar=True,
            point_size=4,
        )
    if volume_path is not None:
        plotter.add_volume(load_volume(volume_path), opacity=0.4, cmap="viridis",
                           show_scalar_bar=True)
    if mesh_path is not None:
        plotter.add_mesh(
            load_mesh(mesh_path),
            opacity=0.32,
            color="#ded2c0",
            show_edges=True,
        )
    plotter.reset_camera()
    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        time_range = parse_time_range(args.time_range)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    zarr_path = Path(args.zarr)
    if not zarr_path.exists():
        print(f"error: Zarr store not found: {zarr_path}", file=sys.stderr)
        return 2

    data = load_scene(
        zarr_path,
        time_range=time_range,
        max_trajectories=args.max_trajectories,
    )

    if args.screenshot is not None:
        out_path = render_screenshot(
            data,
            mesh_path=args.mesh,
            volume_path=args.volume,
            color_by=args.color_by,
            volume_only=args.volume_only,
            out_path=args.screenshot,
        )
        print(f"wrote {out_path} ({data.n_points:,} points, "
              f"{data.n_trajectories:,} trajectories)")
        return 0

    from .app import create_app

    app = create_app(
        data,
        mesh_path=args.mesh,
        volume_path=args.volume,
        color_by=args.color_by,
        volume_only=args.volume_only,
    )
    app.server.start(port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
