# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "zarr",
#     "plotly",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import zarr
    import plotly.graph_objects as go
    from pathlib import Path

    return Path, go, mo, np, zarr


@app.cell
def _(mo):
    mo.md("""
    # Dual-Rig Cylindrical Test Section — 3D Lagrangian Trajectories
    Two 4-camera rigs (cams 1-4 and cams 5-8) on opposite walls of a
    cylindrical test section. Trajectories are
    pooled from both `rig_a` and `rig_b`
    result stores; only the longest tracks in the selected frame window
    are drawn, to keep the plot interactive.
    """)
    return


@app.cell
def _(Path):
    base = Path(r"path/to/your/dataset")
    stores = [
        base / "rig_a" / "res" / "run.zarr",
        base / "rig_b" / "res" / "run.zarr",
    ]
    ori_files = [
        base / "rig_a" / "cal" / "cam1.tif.ori",
        base / "rig_a" / "cal" / "cam2.tif.ori",
        base / "rig_a" / "cal" / "cam3.tif.ori",
        base / "rig_a" / "cal" / "cam4.tif.ori",
        base / "rig_b" / "cal" / "cam5.tif.ori",
        base / "rig_b" / "cal" / "cam6.tif.ori",
        base / "rig_b" / "cal" / "cam7.tif.ori",
        base / "rig_b" / "cal" / "cam8.tif.ori",
    ]
    # Cylinder test section: 7 m diameter, 3.5 m high, centered on X=Z=0,
    # base at Y=0 (matches the camera .ori heights, ~0.1-2.3 m off the floor).
    CYL_RADIUS = 3.5
    CYL_HEIGHT = 3.5
    return CYL_HEIGHT, CYL_RADIUS, ori_files, stores


@app.cell
def _(mo, stores):
    frame_range = mo.ui.range_slider(
        start=1901, stop=2401, value=(1901, 2401), step=1,
        label="Frame range", show_value=True, full_width=True,
    )
    n_traj = mo.ui.slider(
        start=10, stop=1500, step=10, value=600,
        label="Number of longest trajectories", show_value=True,
    )
    mo.vstack([mo.md(f"Data: `{stores[0].parent.parent.name}` + `{stores[1].parent.parent.name}`"),
               frame_range, n_traj])
    return frame_range, n_traj


@app.cell
def _(np):
    def read_ori(path):
        """Parse an OpenPTV .ori file: camera position (mm) + 3x3 rotation."""
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        pos_mm = np.array(lines[0].split(), dtype=float)
        rot = np.array([row.split() for row in lines[2:5]], dtype=float)
        return pos_mm / 1000.0, rot  # position in metres

    return (read_ori,)


@app.cell
def _(np, ori_files, read_ori):
    cam_pos = np.array([read_ori(p)[0] for p in ori_files])
    cam_labels = [p.stem.replace(".tif", "") for p in ori_files]
    return cam_labels, cam_pos


@app.cell
def _(np, zarr):
    def load_arrays(store_path):
        g = zarr.open(str(store_path), mode="r")
        return (
            g["trajectories/trajid"][:],
            g["trajectories/time"][:],
            g["trajectories/pos"][:],
        )

    def longest_in_window(trajid, time, pos, fmin, fmax):
        """Vectorized: trajid/time/pos of points inside [fmin, fmax], plus
        per-trajectory point counts, restricted to that window."""
        mask = (time >= fmin) & (time <= fmax)
        wt, wtime, wpos = trajid[mask], time[mask], pos[mask]
        counts = np.bincount(wt)
        return wt, wtime, wpos, counts

    return load_arrays, longest_in_window


@app.cell
def _(frame_range, load_arrays, longest_in_window, n_traj, np, stores):
    fmin, fmax = frame_range.value

    windows = []
    candidates = []  # (store_idx, trajid, count)
    for store_idx, store_path in enumerate(stores):
        trajid, time, pos = load_arrays(store_path)
        wt, wtime, wpos, counts = longest_in_window(trajid, time, pos, fmin, fmax)
        windows.append((wt, wtime, wpos))
        top_local = np.argsort(counts)[::-1][: n_traj.value]
        for tid in top_local:
            if counts[tid] >= 2:
                candidates.append((store_idx, int(tid), int(counts[tid])))

    candidates.sort(key=lambda c: c[2], reverse=True)
    selected = candidates[: n_traj.value]

    trajectories = []
    for store_idx, tid, _count in selected:
        wt, wtime, wpos = windows[store_idx]
        m = wt == tid
        _pts, _times = wpos[m], wtime[m]
        order = np.argsort(_times)
        _pts, _times = _pts[order], _times[order]
        _speed = np.zeros(len(_pts))
        if len(_pts) > 1:
            dt = np.diff(_times)
            dt[dt == 0] = 1
            _speed[1:] = np.linalg.norm(np.diff(_pts, axis=0), axis=1) / dt
            _speed[0] = _speed[1]
        trajectories.append((_pts, _speed))
    return (trajectories,)


@app.cell
def _(CYL_HEIGHT, CYL_RADIUS, np):
    def cylinder_wireframe(n=60):
        theta = np.linspace(0, 2 * np.pi, n)
        x, z = CYL_RADIUS * np.cos(theta), CYL_RADIUS * np.sin(theta)
        bottom = np.stack([x, np.zeros(n), z], axis=1)
        top = np.stack([x, np.full(n, CYL_HEIGHT), z], axis=1)
        return bottom, top

    return (cylinder_wireframe,)


@app.cell
def _(cam_labels, cam_pos, cylinder_wireframe, go, trajectories):
    fig = go.Figure()

    bottom, top = cylinder_wireframe()
    for ring in (bottom, top):
        fig.add_trace(go.Scatter3d(
            x=ring[:, 0], y=ring[:, 1], z=ring[:, 2],
            mode="lines", line=dict(color="lightgray", width=3),
            showlegend=False, hoverinfo="skip",
        ))
    # a few vertical struts
    for i in range(0, len(bottom), len(bottom) // 8):
        fig.add_trace(go.Scatter3d(
            x=[bottom[i, 0], top[i, 0]], y=[bottom[i, 1], top[i, 1]], z=[bottom[i, 2], top[i, 2]],
            mode="lines", line=dict(color="lightgray", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
        mode="markers+text", text=cam_labels, textposition="top center",
        marker=dict(size=6, color="red", symbol="diamond"),
        name="cameras",
    ))

    for _pts, _speed in trajectories:
        fig.add_trace(go.Scatter3d(
            x=_pts[:, 0], y=_pts[:, 1], z=_pts[:, 2],
            mode="lines",
            line=dict(color=_speed, colorscale="Viridis", width=3),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X [m]", yaxis_title="Y [m] (up)", zaxis_title="Z [m]",
            aspectmode="data",
        ),
        title=f"{len(trajectories)} longest trajectories — cylinder rig (cams 1-4 & 5-8)",
        height=800, margin=dict(l=0, r=0, t=40, b=0),
    )
    fig
    return


if __name__ == "__main__":
    app.run()
