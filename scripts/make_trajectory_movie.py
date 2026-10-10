#!/usr/bin/env python3
"""Trajectory movie from postptv/openptv2 `res/run.zarr/trajectories` stores.

Trailing-tail 3D segments colored by vertical velocity (coolwarm, +-0.5 m/s),
fixed camera angle, wireframe barrel rings, horizontal colorbar — in the
style of the Ilmenau `movie_up.avi` reference.

Defaults below are the Ilmenau 8-camera bubble run (pos in m with PTV
columns X, Y=height, Z=depth); point BASE/RIGS/OUT at any dataset with the
same `trajectories/{pos,vel,time,trajid}` schema.

Usage:
  python make_trajectory_movie.py --test-frame 10250   # single PNG preview
  python make_trajectory_movie.py --render-all          # all PNG frames
  python make_trajectory_movie.py --render-all --mode full --elev 25 --azim -55 \
      --stride 10 --max-tracks 15000 --outdir movie_frames_iso_full  # iso pathlines
Then: ffmpeg -framerate 10 -i frame_%04d.png -c:v mpeg4 movie_up2.avi
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from mpl_toolkits.mplot3d.art3d import Line3DCollection

BASE = Path("/Users/alex/Downloads/Ilmenau")
RIGS = [BASE / "openptv_illmenau_4cam" / "res" / "run.zarr",
        BASE / "openptv_illmenau_5678" / "res" / "run.zarr"]
OUT = BASE / "Full_measurement_with_8_cameras" / "movie_frames_up2"
CYL_R = 3575.0  # mm wall radius
ELEV, AZIM = 86, -45  # fixed camera angle (movie_up style: top-down)
TAIL = 5  # trailing frames per streak (tails mode)
WMIN, WMAX = -0.5, 0.5
EVERY = 5  # frame stride over 10001..10500 -> 100 movie frames
SEGLEN_CAP = 250.0  # mm; drop longer single-step jumps (ghosts)


def load():
    xs, ys, hs, ws, ts, ds = [], [], [], [], [], []
    # NOTE: trajid namespaces of the two rigs collide: offset each rig
    # (same convention as generate_ilmenau_davis_results.py).
    for zpath, off in zip(RIGS, [0, 1000000]):
        g = zarr.open(str(zpath / "trajectories"), mode="r")
        pos = g["pos"][:] * 1000.0  # m -> mm; cols X, Yheight, Zdepth
        vel = g["vel"][:]
        tim = g["time"][:].astype(np.int64)
        tra = g["trajid"][:].astype(np.int64) + off
        order = np.lexsort((tim, tra))
        xs.append(pos[order][:, 0])
        ys.append(pos[order][:, 2])  # depth on plot-Y
        hs.append(pos[order][:, 1])  # height on plot-Z
        ws.append(vel[order][:, 1])  # vertical velocity
        ts.append(tim[order])
        ds.append(tra[order])
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(hs),
            np.concatenate(ws), np.concatenate(ts), np.concatenate(ds))


def draw_frame(frm, X, Y, H, W, T, out_path, dpi=120):
    sel = (T >= frm - TAIL) & (T <= frm)
    # consecutive-point segments within the window
    idx = np.nonzero(sel)[0]
    if len(idx) < 2:
        return False
    brk = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    ok = ends > starts
    starts, ends = starts[ok], ends[ok]
    segs = np.stack(
        [np.column_stack([X[starts], Y[starts], H[starts]]),
         np.column_stack([X[ends], Y[ends], H[ends]])],
        axis=1,
    )
    # drop ghost streaks: single steps above the speed cap
    seglen = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    keep = seglen <= SEGLEN_CAP
    segs, starts, ends = segs[keep], starts[keep], ends[keep]
    if len(segs) < 1:
        return False
    cols = np.clip((W[starts] + W[ends]) / 2, WMIN, WMAX)

    render_segments(segs, cols, out_path, dpi=dpi)
    return True


def build_full_trails(X, Y, H, W, T, D, max_tracks):
    """Precompute full-length trail segments for the longest trajectories.

    Returns (segs (M,2,3), colors (M,), end_times (M,)) with gap jumps
    (dt > 2 frames) and ghost steps removed. Render loop then only masks
    end_times <= frame. Rig trajid namespaces are pre-offset in load().
    """
    order = np.lexsort((T, D))
    Xs, Ys, Hs, Ws, Ts, Ds = X[order], Y[order], H[order], W[order], T[order], D[order]
    _, inv, cnt = np.unique(Ds, return_counts=True, return_inverse=True)
    thresh = np.sort(cnt)[-min(max_tracks, len(cnt))]
    m = cnt[inv] >= thresh
    Xs, Ys, Hs, Ws, Ts, Ds = Xs[m], Ys[m], Hs[m], Ws[m], Ts[m], Ds[m]
    link = (Ds[1:] == Ds[:-1]) & (Ts[1:] - Ts[:-1] >= 1) & (Ts[1:] - Ts[:-1] <= 2)
    i0 = np.nonzero(link)[0]
    segs = np.stack(
        [np.column_stack([Xs[i0], Ys[i0], Hs[i0]]),
         np.column_stack([Xs[i0 + 1], Ys[i0 + 1], Hs[i0 + 1]])],
        axis=1,
    )
    seglen = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    good = seglen <= SEGLEN_CAP
    return (segs[good],
            np.clip((Ws[i0[good]] + Ws[i0[good] + 1]) / 2, WMIN, WMAX),
            Ts[i0[good] + 1])


def render_segments(segs, cols, out_path, dpi=120):
    fig = plt.figure(figsize=(12, 12), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    th = np.linspace(0, 2 * np.pi, 200)
    for z in (HZMIN, HZMAX):
        ax.plot(CYL_R * np.cos(th), CYL_R * np.sin(th),
                np.full_like(th, z), color="black", lw=1.5)
    for ang in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        ax.plot([CYL_R * np.cos(ang)] * 2, [CYL_R * np.sin(ang)] * 2,
                [HZMIN, HZMAX], color="black", lw=1.0, alpha=0.6)
    if len(segs) > 0:
        lc = Line3DCollection(segs, cmap="coolwarm",
                              norm=matplotlib.colors.Normalize(vmin=WMIN, vmax=WMAX),
                              linewidths=0.6, alpha=0.55)
        lc.set_array(cols)
        ax.add_collection3d(lc)
    ax.set_xlim(-4000, 4000)
    ax.set_ylim(-4000, 4000)
    ax.set_zlim(HZMIN - 200, HZMAX + 200)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.view_init(elev=ELEV, azim=AZIM)
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=matplotlib.colors.Normalize(vmin=WMIN, vmax=WMAX))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04, orientation="horizontal")
    cb.set_label(r"$w$ [m/s]")
    cb.set_ticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frame", type=int, default=None)
    ap.add_argument("--render-all", action="store_true")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--mode", choices=["tails", "full"], default="tails",
                    help="tails: TAIL-frame streaks; full: whole trajectory up to frame")
    ap.add_argument("--elev", type=float, default=None)
    ap.add_argument("--azim", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-tracks", type=int, default=15000,
                    help="longest-first cap (full mode only)")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    if args.elev is not None:
        ELEV = args.elev
    if args.azim is not None:
        AZIM = args.azim
    if args.stride is not None:
        EVERY = args.stride
    if args.outdir is not None:
        OUT = Path(args.outdir)
        if not OUT.is_absolute():
            OUT = BASE / OUT

    print("loading trajectories...", flush=True)
    X, Y, H, W, T, D = load()
    HZMIN, HZMAX = float(np.percentile(H, 1)), float(np.percentile(H, 99))
    print(f"points={len(X)} H range 1/99pct=({HZMIN:.0f},{HZMAX:.0f})", flush=True)
    OUT.mkdir(exist_ok=True)

    if args.mode == "full":
        print("building full-length trails...", flush=True)
        SEGS, COLS, ET = build_full_trails(X, Y, H, W, T, D, args.max_tracks)
        print(f"trail segments: {len(SEGS)}", flush=True)
        frames = list(range(10001, 10501, EVERY))
        for i, f in enumerate(frames):
            m = ET <= f
            if m.sum() < 1:
                continue
            render_segments(SEGS[m], COLS[m], OUT / f"frame_{i:04d}.png", dpi=args.dpi)
            if i % 10 == 0:
                print(f"{i}/{len(frames)} nseg={m.sum()}", flush=True)
        print("done", flush=True)
        sys.exit(0)

    if args.test_frame is not None:
        ok = draw_frame(args.test_frame, X, Y, H, W, T,
                        OUT / f"test_{args.test_frame}.png", dpi=args.dpi)
        print("test frame written:", ok, flush=True)
    elif args.render_all:
        frames = list(range(10001, 10501, EVERY))
        for i, f in enumerate(frames):
            draw_frame(f, X, Y, H, W, T, OUT / f"frame_{i:04d}.png", dpi=args.dpi)
            if i % 10 == 0:
                print(f"{i}/{len(frames)}", flush=True)
        print("done", flush=True)
    else:
        ap.print_help()
        sys.exit(1)
