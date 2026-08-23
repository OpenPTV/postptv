import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🆚 Trajectory Linking: HDF5 vs. Zarr-backed input

    Companion to `linking_trajectories.ipynb`, which links (welds) short
    trajectories from `data/particles.h5` into longer ones using a
    greedy nearest-predicted-endpoint heuristic. Here we run the **exact
    same linking algorithm** twice — once on trajectories loaded from
    HDF5 via `Scene`, once on the same trajectories round-tripped through
    Zarr — and compare the welded output to confirm the backend makes no
    difference to the result.
    """)
    return


@app.cell
def _():
    import itertools as it
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from flowtracks.io import read_zarr_trajectories, save_zarr_trajectories
    from flowtracks.scene import Scene
    from flowtracks.trajectory import Trajectory

    return (
        Path,
        Scene,
        Trajectory,
        it,
        mo,
        np,
        plt,
        read_zarr_trajectories,
        save_zarr_trajectories,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Controls""")
    return


@app.cell
def _(mo):
    frate_ui = mo.ui.number(label="Frame rate [fps]", value=100.0, step=10.0)
    dist_thresh_ui = mo.ui.slider(start=0.0005, stop=0.01, step=0.0005, value=0.0025, label="Max link distance [m]")
    max_dt_ui = mo.ui.slider(start=1, stop=5, step=1, value=1, label="Max gap [frames]")
    min_len_ui = mo.ui.slider(start=5, stop=60, step=5, value=25, label="Min trajectory length [frames]")

    mo.hstack([frate_ui, dist_thresh_ui, max_dt_ui, min_len_ui], gap=2)
    return dist_thresh_ui, frate_ui, max_dt_ui, min_len_ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 1 — load the same particles two ways""")
    return


@app.cell
def _(Path, Scene, mo):
    data_dir = Path(__file__).parent.parent / "data"
    h5_path = data_dir / "particles.h5"

    h5_trajs = list(Scene(str(h5_path)).iter_trajectories())
    mo.md(f"HDF5 (`Scene`): **{len(h5_trajs)} trajectories** from `{h5_path}`.")
    return h5_path, h5_trajs


@app.cell
def _(h5_trajs, mo, read_zarr_trajectories, save_zarr_trajectories, tempfile, Path):
    tmp_dir = Path(tempfile.mkdtemp(prefix="postptv_zarr_link_demo_"))
    zarr_path = tmp_dir / "particles.zarr"
    save_zarr_trajectories(h5_trajs, zarr_path)
    zarr_trajs = read_zarr_trajectories(zarr_path)

    mo.md(f"Zarr round-trip: **{len(zarr_trajs)} trajectories** from `{zarr_path}`.")
    return tmp_dir, zarr_path, zarr_trajs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — the linking algorithm (unchanged from `linking_trajectories.ipynb`)

    Greedy nearest-neighbor welding: for each pair of trajectories whose time
    gap is within `max_dt`, predict trj1's next position forward and trj2's
    previous position backward; if the average prediction error is below
    `dist_thresh` and better than any earlier candidate, register the link.
    """)
    return


@app.cell
def _(Trajectory, it, np):
    def link_trajectories(trajs, frate, dist_thresh, max_dt, min_len):
        long_trajects = [t for t in trajs if len(t) > min_len]

        links, back_links = {}, {}
        for trj1, trj2 in it.combinations(long_trajects, 2):
            dt = (trj2.time(0) - trj1.time(-1)) / frate
            if not (0 < dt <= max_dt):
                continue

            master_id, slave_id = trj1.trajid(), trj2.trajid()
            links.setdefault(master_id, (None, dist_thresh))
            back_links.setdefault(slave_id, (None, dist_thresh))
            min_dist = min(links[master_id][1], back_links[slave_id][1])

            predicted_forward = trj1.pos(-1) + dt * trj1.velocity(-1)
            predicted_backward = trj2.pos(0) - dt * trj2.velocity(0)
            dist_forward = np.linalg.norm(predicted_forward - trj2.pos(0))
            dist_backward = np.linalg.norm(predicted_backward - trj1.pos(-1))
            avg_dist = (dist_forward + dist_backward) / 2.0

            if avg_dist < min_dist:
                old_link = back_links[slave_id][0]
                if old_link is not None:
                    links[old_link] = (None, dist_thresh)
                links[master_id] = (slave_id, avg_dist)
                back_links[slave_id] = (master_id, avg_dist)

        out_trajects = []
        used_trids = set()
        for trid, cand in links.items():
            if trid in used_trids:
                continue
            trj_weld = next(t for t in long_trajects if t.trajid() == trid)
            while cand[0] is not None:
                used_trids.add(cand[0])
                trj2 = next(t for t in long_trajects if t.trajid() == cand[0])
                trj_weld = Trajectory(
                    np.vstack((trj_weld.pos(), trj2.pos())),
                    np.vstack((trj_weld.velocity(), trj2.velocity())),
                    trajid=trj_weld.trajid(),
                    time=np.hstack((trj_weld.time(), trj2.time())),
                    accel=np.vstack((trj_weld.accel(), trj2.accel())),
                )
                if cand[0] not in links:
                    break
                cand = links[cand[0]]
            out_trajects.append(trj_weld)

        return long_trajects, out_trajects

    return (link_trajectories,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 3 — run it on both backends and compare""")
    return


@app.cell
def _(
    dist_thresh_ui,
    frate_ui,
    h5_trajs,
    link_trajectories,
    max_dt_ui,
    min_len_ui,
    zarr_trajs,
):
    h5_long, h5_welded = link_trajectories(
        h5_trajs, frate_ui.value, dist_thresh_ui.value, max_dt_ui.value, min_len_ui.value)
    zarr_long, zarr_welded = link_trajectories(
        zarr_trajs, frate_ui.value, dist_thresh_ui.value, max_dt_ui.value, min_len_ui.value)
    return h5_long, h5_welded, zarr_long, zarr_welded


@app.cell
def _(h5_welded, mo, np, zarr_welded):
    h5_by_id = {t.trajid(): t for t in h5_welded}
    zarr_by_id = {t.trajid(): t for t in zarr_welded}

    same_ids = set(h5_by_id) == set(zarr_by_id)
    max_len_diff = max((abs(len(h5_by_id[i]) - len(zarr_by_id[i])) for i in h5_by_id if i in zarr_by_id), default=None)
    max_pos_diff = max(
        (float(np.abs(h5_by_id[i].pos() - zarr_by_id[i].pos()).max())
         for i in h5_by_id if i in zarr_by_id and len(h5_by_id[i]) == len(zarr_by_id[i])),
        default=None,
    )

    mo.md(f"""
    | check | HDF5 | Zarr |
    |---|---|---|
    | welded trajectories | {len(h5_welded)} | {len(zarr_welded)} |
    | same welded trajid set | {same_ids} | |
    | max length difference (matching ids) | {max_len_diff} | |
    | max \\|Δpos\\| [m] (matching-length ids) | {max_pos_diff} | |

    {"✅ Zarr-backed input produces an identical welding result." if same_ids and max_len_diff == 0 and (max_pos_diff or 0) == 0 else "⚠️ differences detected — see above."}
    """)
    return h5_by_id, max_len_diff, max_pos_diff, same_ids, zarr_by_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 4 — visual overlay""")
    return


@app.cell
def _(h5_long, h5_welded, mo, plt, zarr_long, zarr_welded):
    fig, (ax_h5, ax_zarr) = plt.subplots(1, 2, figsize=(12, 6), dpi=100, sharex=True, sharey=True)

    for trj in h5_long:
        p = trj.pos()
        ax_h5.plot(p[:, 0], p[:, 1], "-", alpha=0.4)
    for trj in h5_welded:
        p = trj.pos()
        ax_h5.plot(p[:, 0], p[:, 1], "--", linewidth=2)
    ax_h5.set_title(f"HDF5: {len(h5_long)} raw -> {len(h5_welded)} welded")

    for trj in zarr_long:
        p = trj.pos()
        ax_zarr.plot(p[:, 0], p[:, 1], "-", alpha=0.4)
    for trj in zarr_welded:
        p = trj.pos()
        ax_zarr.plot(p[:, 0], p[:, 1], "--", linewidth=2)
    ax_zarr.set_title(f"Zarr: {len(zarr_long)} raw -> {len(zarr_welded)} welded")

    for ax in (ax_h5, ax_zarr):
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    mo.hstack([fig])
    return ax_h5, ax_zarr, fig


if __name__ == "__main__":
    app.run()
