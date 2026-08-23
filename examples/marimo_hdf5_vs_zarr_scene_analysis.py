import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🆚 HDF5 vs. Zarr: same `data/particles.h5` scene, two backends

    Companion to `hdf5_scene_analysis.ipynb`. That notebook reads
    `data/particles.h5` through `flowtracks.scene.Scene` (pytables/HDF5).
    Here we take the **same trajectories**, convert them to a Zarr store with
    `flowtracks.io.save_zarr_trajectories`, read them back with
    `read_zarr_trajectories`, and compare every trajectory's positions,
    velocities, and accelerations between the two backends to confirm Zarr
    is a faithful drop-in for this data.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from flowtracks.scene import Scene
    from flowtracks.io import read_zarr_trajectories, save_zarr_trajectories

    return Path, Scene, mo, np, plt, read_zarr_trajectories, save_zarr_trajectories


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 1 — load `data/particles.h5` via `Scene` (the original path)""")
    return


@app.cell
def _(Path, Scene, mo):
    data_dir = Path(__file__).parent.parent / "data"
    h5_path = data_dir / "particles.h5"

    scene = Scene(str(h5_path))
    h5_trajs = list(scene.iter_trajectories())
    h5_trajs.sort(key=lambda t: t.trajid())

    mo.md(f"Loaded **{len(h5_trajs)} trajectories** from `{h5_path}`.")
    return h5_path, h5_trajs, scene


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 2 — convert to Zarr and read it back""")
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path as _Path

    tmp_zarr_dir = _Path(tempfile.mkdtemp(prefix="postptv_zarr_demo_"))
    return (tmp_zarr_dir,)


@app.cell
def _(h5_trajs, mo, read_zarr_trajectories, save_zarr_trajectories, tmp_zarr_dir):
    zarr_path = tmp_zarr_dir / "particles.zarr"
    save_zarr_trajectories(h5_trajs, zarr_path)

    zarr_trajs = read_zarr_trajectories(zarr_path)
    zarr_trajs.sort(key=lambda t: t.trajid())

    mo.md(f"Round-tripped **{len(zarr_trajs)} trajectories** through `{zarr_path}`.")
    return zarr_path, zarr_trajs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 3 — compare, trajectory by trajectory""")
    return


@app.cell
def _(h5_trajs, mo, np, zarr_trajs):
    assert len(h5_trajs) == len(zarr_trajs), "trajectory count differs between backends!"

    max_pos_err, max_vel_err, max_accel_err = 0.0, 0.0, 0.0
    mismatched = []
    for t_h5, t_zarr in zip(h5_trajs, zarr_trajs):
        if t_h5.trajid() != t_zarr.trajid():
            mismatched.append(t_h5.trajid())
            continue
        max_pos_err = max(max_pos_err, float(np.abs(t_h5.pos() - t_zarr.pos()).max()))
        max_vel_err = max(max_vel_err, float(np.abs(t_h5.velocity() - t_zarr.velocity()).max()))
        if t_h5.has_property("accel") and t_zarr.has_property("accel"):
            max_accel_err = max(max_accel_err, float(np.abs(t_h5.accel() - t_zarr.accel()).max()))

    mo.md(f"""
    | check | result |
    |---|---|
    | trajectories compared | {len(h5_trajs)} |
    | trajid mismatches | {len(mismatched)} |
    | max \\|Δpos\\| [m] | {max_pos_err:.3e} |
    | max \\|Δvelocity\\| [m/s] | {max_vel_err:.3e} |
    | max \\|Δaccel\\| [m/s²] | {max_accel_err:.3e} |

    {"✅ Zarr matches HDF5 exactly." if max_pos_err == 0 and max_vel_err == 0 and not mismatched else "⚠️ differences detected — see above."}
    """)
    return max_accel_err, max_pos_err, max_vel_err, mismatched


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 4 — visual check: same trajectories, both backends overlaid""")
    return


@app.cell
def _(h5_trajs, mo, plt, zarr_trajs):
    n_show = min(40, len(h5_trajs))

    fig, (ax_h5, ax_zarr) = plt.subplots(1, 2, figsize=(12, 5), dpi=100, sharex=True, sharey=True)
    for t in h5_trajs[:n_show]:
        p = t.pos()
        ax_h5.plot(p[:, 0], p[:, 1], ".-", markersize=2, linewidth=0.8)
    for t in zarr_trajs[:n_show]:
        p = t.pos()
        ax_zarr.plot(p[:, 0], p[:, 1], ".-", markersize=2, linewidth=0.8)

    ax_h5.set_title(f"HDF5 / Scene ({n_show} trajectories)")
    ax_zarr.set_title(f"Zarr round-trip ({n_show} trajectories)")
    for ax in (ax_h5, ax_zarr):
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    mo.hstack([fig])
    return ax_h5, ax_zarr, fig, n_show


if __name__ == "__main__":
    app.run()
