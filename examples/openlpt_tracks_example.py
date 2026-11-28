#!/usr/bin/env python3
"""Example: convert OpenLPT / pyOpenLPT tracks into a Zarr store.

This example is SELF-CONTAINED: it generates synthetic ``ConvergeTrack``
data in a temporary directory (no OpenLPT install needed) and runs the
conversion steps end-to-end. To use it with a REAL OpenLPT/pyOpenLPT project,
replace the ``converge_track`` path with the ``Results/ConvergeTrack``
directory from that project — the conversion code is identical.

    python examples/openlpt_tracks_example.py
"""

import tempfile
from pathlib import Path

import numpy as np

from flowtracks.openlpt import convert_openlpt_to_zarr


def write_synthetic_converge_track(path, n_frames=20, seed=0):
    """Write a synthetic tracer ConvergeTrack directory.

    Layout mirrors the real one: one CSV per frame (``LongTrackActive_*.csv``).
    Columns (Tracer): TrackID,FrameID,WorldX,WorldY,WorldZ, camX,camY (x2 cams).
    This is synthetic data — nothing here was produced by the real OpenLPT
    pipeline — but the format is exactly the one OpenLPT writes.
    """
    rng = np.random.default_rng(seed)
    # 3 synthetic tracks with independent drift
    tracks = []
    for trajid in (0, 1, 2):
        x0, y0, z0 = rng.uniform(0, 10, 3)     # mm
        vx, vy, vz = rng.uniform(-0.3, 0.3, 3)  # mm/frame
        tracks.append((trajid, x0, y0, z0, vx, vy, vz))

    # One CSV per outcome prefix, each holding full history rows per track
    # (the real format keeps the live tracks in the current frame's file;
    # the reader does not care about the frame-file structure).
    header = "TrackID,FrameID,WorldX,WorldY,WorldZ,cam0_x(col),cam0_y(row),cam1_x(col),cam1_y(row)"
    lines = [header]
    for trajid, x0, y0, z0, vx, vy, vz in tracks:
        for t in range(n_frames):
            x = x0 + vx * t + 1.0 * np.sin(t * 0.4)
            y = y0 + vy * t + 1.0 * np.cos(t * 0.3)
            z = z0 + vz * t
            lines.append(f"{trajid},{t},{x:.4f},{y:.4f},{z:.4f},{x + 100:.1f},{y + 400:.1f},{x + 200:.1f},{y + 500:.1f}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "LongTrackActive_00001.csv").write_text("\n".join(lines) + "\n")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="openlpt_example_"))
    converge_track = tmp / "ConvergeTrack"
    write_synthetic_converge_track(converge_track)
    print(f"Synthetic ConvergeTrack directory: {converge_track}")
    print()

    # --- convert to flowtracks -----------------------------------------
    zarr_path = tmp / "run.zarr"
    trajects = convert_openlpt_to_zarr(converge_track, zarr_path, overwrite=True)
    print(f"Converted {len(trajects)} trajectories "
          f"({sum(len(t) for t in trajects)} samples)")
    print(f"Zarr store written to {zarr_path}")
    print()

    # --- inspect in flowtracks -------------------------------------------
    from flowtracks.zarr_scene import ZarrScene

    scene = ZarrScene(zarr_path)
    print("ZarrScene:")
    print(f"  trajectory ids: {list(scene.trajectory_ids())}")
    print(f"  keys: {scene.keys()}")
    print(f"  pos shape: {scene.collect(['pos'])[0].shape}")

    # optional: quick 3D view through the graphics helpers (needs pyvista)
    try:
        import pyvista as pv

        from flowtracks.graphics import plot_trajectories_3d

        plotter = pv.Plotter(off_screen=True)
        plot_trajectories_3d(zarr_path, plotter=plotter, show=False)
        plot_path = tmp / "openlpt_example.png"
        plotter.screenshot(str(plot_path), window_size=(900, 600))
        print(f"  3D view saved to {plot_path}")
    except Exception as e:  # demo nicety only — keep the conversion useful
        print(f"  (plot skipped: {e})")

    return 0


if __name__ == "__main__":
    main()
