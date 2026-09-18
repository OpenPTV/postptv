import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔍 Previewing `flowtracks.writers` output (pyvista)

    `flowtracks.writers` (see `WRITERS_PLAN.md`) writes ParaView-ready files in
    four formats:

    - **`.vtp`** — Lagrangian trajectories (PolyData: points + one polyline per
      `trajid`).
    - **`.vti`** — Eulerian grid, uniform axis spacing (ImageData).
    - **`.vtr`** — Eulerian grid, non-uniform axis spacing (RectilinearGrid).
    - **`.pvd`** — a Collection indexing either grid series as a ParaView time
      series.

    This notebook writes small synthetic datasets with the real writer
    functions, reads them back with **pyvista** exactly as ParaView would, and
    renders a quick preview of each — without opening ParaView. Every read
    here uses the same reader ParaView itself uses under the hood
    (`pv.read` / `pv.get_reader` wrap VTK's own XML readers), so a preview
    that looks right here is a real correctness check on the writer, not just
    a plot.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import pyvista as pv
    import xarray as xr

    from flowtracks.zarr_scene import ZarrScene
    from flowtracks.writers import write_eulerian_series, write_trajectories_vtp

    return (
        Path,
        ZarrScene,
        go,
        mo,
        np,
        pv,
        tempfile,
        write_eulerian_series,
        write_trajectories_vtp,
        xr,
    )


@app.cell
def _(Path, tempfile):
    out_dir = Path(tempfile.mkdtemp(prefix="flowtracks_writers_preview_"))
    return (out_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Lagrangian trajectories, Zarr → `.vtp`

    Real data, not synthetic: `data/tracers.zarr` -- a Zarr conversion of the
    repo's own demo dataset (first 200 frames of a homogeneous-turbulence
    tracer run, Guala et al., *"Experimental study on clustering of large
    particles in homogeneous turbulent flow"*, J. Turbulence; originally
    `data/tracers.h5`, see `data/readme.txt`), produced once via
    `scripts/convert_tracers_to_zarr.py` (`Scene` reads the legacy HDF5,
    `flowtracks.io.save_zarr_trajectories` writes the `trajectories/`
    Zarr group `write_trajectories_vtp` reads directly -- no HDF5/pytables
    anywhere downstream). 14,042 trajectories, lengths 2-133 frames -- the
    sliders below pick a readable subset via `ZarrScene`.
    """)
    return


@app.cell
def _(Path, ZarrScene):
    data_dir = Path(__file__).parent.parent / "data"
    tracer_scene = ZarrScene(str(data_dir / "tracers.zarr"))
    all_trajs = sorted(tracer_scene.iter_trajectories(), key=lambda t: t.trajid())
    return all_trajs, data_dir


@app.cell(hide_code=True)
def _(mo):
    n_traj_ui = mo.ui.slider(start=3, stop=40, step=1, value=12, label="Trajectories to show")
    min_length_ui = mo.ui.slider(start=5, stop=100, step=5, value=30, label="Minimum trajectory length (frames)")
    seed_ui = mo.ui.slider(start=0, stop=99, step=1, value=0, label="Sample seed")
    mo.hstack([n_traj_ui, min_length_ui, seed_ui], gap=2)
    return min_length_ui, n_traj_ui, seed_ui


@app.cell
def _(all_trajs, min_length_ui, mo, n_traj_ui, np, seed_ui):
    long_enough = [t for t in all_trajs if len(t) >= min_length_ui.value]
    mo.stop(
        not long_enough,
        mo.md(f"No trajectories reach length {min_length_ui.value} -- lower the minimum length slider."),
    )
    rng = np.random.default_rng(seed_ui.value)
    chosen = rng.choice(len(long_enough), size=min(n_traj_ui.value, len(long_enough)), replace=False)
    picked_trajs = [long_enough[i] for i in chosen]
    return (picked_trajs,)


@app.cell
def _(np, picked_trajs):
    class TrajectorySubset:
        """Duck-typed ZarrScene/Scene stand-in wrapping a chosen list of
        flowtracks Trajectory objects into the flat arrays write_trajectories_vtp
        expects from `.collect(["pos", "velocity", "time", "trajid"])`."""

        def __init__(self, trajs):
            self.pos = np.concatenate([t.pos() for t in trajs])
            self.vel = np.concatenate([t.velocity() for t in trajs])
            self.time = np.concatenate([t.time() for t in trajs])
            self.trajid = np.concatenate([np.full(len(t), t.trajid()) for t in trajs])

        def collect(self, keys):
            cols = {"pos": self.pos, "velocity": self.vel, "time": self.time, "trajid": self.trajid}
            return [cols[k] for k in keys]

    scene = TrajectorySubset(picked_trajs)
    return (scene,)


@app.cell
def _(out_dir, scene, write_trajectories_vtp):
    vtp_path = write_trajectories_vtp(scene, out_dir / "trajectories.vtp")
    vtp_path
    return (vtp_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The subset above goes through `ZarrScene` (so the slider-picked
    trajectories can be assembled first). `write_trajectories_vtp` also
    accepts a Zarr store path **directly** -- the true zarr-native entry
    point from `WRITERS_PLAN.md`, no Scene/ZarrScene wrapper at all. Proving
    it on the *whole* 14,042-trajectory store (too many lines to usefully
    plot, so just the counts):
    """)
    return


@app.cell
def _(data_dir, out_dir, pv, write_trajectories_vtp):
    full_vtp_path = write_trajectories_vtp(data_dir / "tracers.zarr", out_dir / "all_trajectories.vtp")
    full_poly = pv.read(full_vtp_path)
    f"{full_poly.n_points:,} points, {full_poly.n_lines:,} polylines, written straight from data/tracers.zarr"
    return


@app.cell
def _(mo, pv, vtp_path):
    trajectories_poly = pv.read(vtp_path)
    mo.md(
        f"Read back with `pyvista.read`: **{trajectories_poly.n_points}** points, "
        f"**{trajectories_poly.n_lines}** polylines, "
        f"point_data arrays: `{list(trajectories_poly.point_data.keys())}`."
    )
    return (trajectories_poly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Splitting pyvista's flat `.lines` cell array back into one run of point
    indices per trajectory (this is exactly what a ParaView `Tube`/`Cell Data
    to Point Data` filter does internally):
    """)
    return


@app.cell
def _(np, trajectories_poly):
    def split_polylines(poly):
        """VTK's flat cell-connectivity array -> list of point-index arrays,
        one per polyline: [n0, i0, i1, ..., n1, j0, j1, ...] -> [[i0,i1,...], [j0,j1,...]]."""
        cells = poly.lines
        runs, offset = [], 0
        while offset < len(cells):
            n = cells[offset]
            runs.append(cells[offset + 1: offset + 1 + n])
            offset += 1 + n
        return runs

    trajectory_runs = split_polylines(trajectories_poly)
    return (trajectory_runs,)


@app.cell
def _(go, trajectories_poly, trajectory_runs):
    import plotly.colors as pcolors

    points = trajectories_poly.points
    palette = pcolors.qualitative.Dark24

    traj_fig = go.Figure()
    for _i, _run in enumerate(trajectory_runs):
        _color = palette[_i % len(palette)]
        traj_fig.add_trace(go.Scatter3d(
            x=points[_run, 0], y=points[_run, 1], z=points[_run, 2],
            mode="lines+markers",
            line=dict(color=_color, width=6),
            marker=dict(size=2, color=_color),
            name=f"trajid {_i}",
        ))
        traj_fig.add_trace(go.Scatter3d(  # start (circle) / end (diamond) markers show direction
            x=points[_run[[0, -1]], 0], y=points[_run[[0, -1]], 1], z=points[_run[[0, -1]], 2],
            mode="markers", marker=dict(size=5, color=_color, symbol=["circle", "diamond"]),
            showlegend=False, hoverinfo="skip",
        ))
    traj_fig.update_layout(
        title="Lagrangian trajectories, read back from .vtp (one polyline per trajid; ● = start, ♦ = end)",
        scene=dict(aspectmode="data"), height=600, margin=dict(l=0, r=0, t=40, b=0),
    )
    traj_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Eulerian grid → `.vti` / `.vtr` + `.pvd` time series
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    grid_shape_ui = mo.ui.slider(start=3, stop=12, step=1, value=6, label="Grid points per axis (x, y, z)")
    n_phase_ui = mo.ui.slider(start=2, stop=8, step=1, value=4, label="Phases (time steps)")
    uniform_ui = mo.ui.checkbox(value=True, label="Uniform axis spacing (writer picks .vti; unchecked -> .vtr)")
    mo.hstack([grid_shape_ui, n_phase_ui, uniform_ui], gap=2)
    return grid_shape_ui, n_phase_ui, uniform_ui


@app.cell
def _(grid_shape_ui, n_phase_ui, np, uniform_ui, xr):
    _n = grid_shape_ui.value
    if uniform_ui.value:
        x_coord = np.linspace(0.0, 1.0, _n)
    else:
        # geometric spacing: definitely not uniform, forces the .vtr path
        x_coord = np.geomspace(0.05, 1.0, _n)

    _shape = (_n, _n, _n, n_phase_ui.value)
    _rng = np.random.default_rng(0)
    _phase = np.arange(n_phase_ui.value)
    # a simple rotating-vortex-ish field so the mid-plane preview shows structure
    _yy, _xx = np.meshgrid(np.linspace(-1, 1, _n), np.linspace(-1, 1, _n), indexing="ij")
    eulerian_ds = xr.Dataset(
        {
            "u_ins_mean": (("x", "y", "z", "phase"),
                          np.stack([-_yy * np.cos(p) + 0.05 * _rng.standard_normal((_n, _n))
                                   for p in _phase], axis=-1)[:, :, None, :].repeat(_n, axis=2)),
            "v_ins_mean": (("x", "y", "z", "phase"),
                          np.stack([_xx * np.cos(p) + 0.05 * _rng.standard_normal((_n, _n))
                                   for p in _phase], axis=-1)[:, :, None, :].repeat(_n, axis=2)),
            "w_ins_mean": (("x", "y", "z", "phase"), 0.1 * _rng.standard_normal(_shape)),
            "par_ave2": (("x", "y", "z", "phase"), _rng.integers(1, 100, _shape)),
        },
        coords={"x": x_coord, "y": np.linspace(0, 1, _n), "z": np.linspace(0, 1, _n), "phase": _phase},
    )
    return (eulerian_ds,)


@app.cell
def _(eulerian_ds, out_dir, write_eulerian_series):
    eulerian_pvd = write_eulerian_series(eulerian_ds, out_dir / "eulerian", prefix="phase", dt=1.0)
    written_format = sorted(eulerian_pvd.parent.glob("phase_0000.*"))[0].suffix
    eulerian_pvd, written_format
    return (eulerian_pvd,)


@app.cell(hide_code=True)
def _(eulerian_pvd, mo, pv):
    eulerian_reader = pv.get_reader(str(eulerian_pvd))
    mo.md(
        f"`pyvista.get_reader` on the `.pvd`: time_values = "
        f"`{eulerian_reader.time_values}` (standard PVD Collection, same reader "
        "ParaView's own File > Open uses)."
    )
    return (eulerian_reader,)


@app.cell(hide_code=True)
def _(eulerian_reader, mo):
    phase_ui = mo.ui.slider(
        start=0, stop=len(eulerian_reader.time_values) - 1, step=1, value=0,
        label="Phase (time step) to preview",
    )
    phase_ui
    return (phase_ui,)


@app.cell
def _(eulerian_reader, phase_ui):
    eulerian_reader.set_active_time_value(eulerian_reader.time_values[phase_ui.value])
    active_block = eulerian_reader.read()[0]
    active_block
    return (active_block,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Mid-`z` plane velocity preview, plain matplotlib on arrays pulled
    straight out of the pyvista mesh (`.point_data`, `.dimensions`) -- no
    ParaView, no GPU render window needed for a quick sanity check:
    """)
    return


@app.cell
def _(active_block, np):
    import matplotlib.pyplot as plt

    nx, ny, nz = active_block.dimensions
    u = active_block.point_data["velocity"][:, 0].reshape((nx, ny, nz), order="F")
    v = active_block.point_data["velocity"][:, 1].reshape((nx, ny, nz), order="F")
    x_1d = np.unique(active_block.points[:, 0])
    y_1d = np.unique(active_block.points[:, 1])
    mid_z = nz // 2

    fig, ax = plt.subplots(figsize=(5, 5))
    speed_slice = np.hypot(u[:, :, mid_z], v[:, :, mid_z])
    mesh_x, mesh_y = np.meshgrid(x_1d, y_1d, indexing="ij")
    pc = ax.pcolormesh(mesh_x, mesh_y, speed_slice, shading="auto", cmap="viridis")
    ax.quiver(mesh_x, mesh_y, u[:, :, mid_z], v[:, :, mid_z], color="white", alpha=0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{type(active_block).__name__}, mid-z plane (z index {mid_z})")
    fig.colorbar(pc, ax=ax, label="|velocity| (x,y)")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Full 3D views of the same mesh -- a matplotlib 3D quiver (as in
    `examples/marimo_flowtracks.py`'s `mpl_toolkits.mplot3d` convention)
    and a real off-screen pyvista render (velocity glyphs + a scalar slice),
    the same primitives ParaView itself would show:
    """)
    return


@app.cell
def _(active_block, np):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    import matplotlib.pyplot as plt3d
    import matplotlib.cm as cm

    pts = active_block.points
    vel = active_block.point_data["velocity"]
    speed_3d = np.linalg.norm(vel, axis=1)

    # subsample so arrows stay legible on a busy grid
    stride = max(1, len(pts) // 400)
    idx = np.arange(0, len(pts), stride)
    colors = cm.viridis(speed_3d[idx] / speed_3d.max())

    fig3d = plt3d.figure(figsize=(6, 6))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.quiver(
        pts[idx, 0], pts[idx, 1], pts[idx, 2],
        vel[idx, 0], vel[idx, 1], vel[idx, 2],
        length=0.15, normalize=True, color=colors,
    )
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"{type(active_block).__name__}: 3D velocity quiver (colored by |velocity|)")
    fig3d.tight_layout()
    fig3d
    return


@app.cell
def _(active_block, mo, np, pv):
    scalar_mesh = active_block.copy()
    scalar_mesh["speed"] = np.linalg.norm(scalar_mesh.point_data["velocity"], axis=1)
    glyphs = scalar_mesh.glyph(orient="velocity", scale="speed", factor=0.08, tolerance=0.08)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(600, 600))
    plotter.add_mesh(scalar_mesh.slice(normal="z"), scalars="speed", cmap="viridis", opacity=0.85)
    plotter.add_mesh(glyphs, scalars="speed", cmap="viridis", show_scalar_bar=False)
    plotter.add_mesh(scalar_mesh.outline(), color="black")
    plotter.camera_position = "iso"
    screenshot = plotter.screenshot(return_img=True)
    plotter.close()

    mo.image(screenshot, caption="pyvista off-screen render: velocity glyphs + mid-z scalar slice, colored by speed")
    return


if __name__ == "__main__":
    app.run()
