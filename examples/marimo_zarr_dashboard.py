import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import xarray as xr
    import zarr

    from flowtracks.io import read_zarr_trajectories, save_zarr_trajectories
    from flowtracks.smoothing import savitzky_golay
    from flowtracks.stitching import stitch_trajectories

    return (
        Path,
        go,
        mo,
        np,
        os,
        plt,
        read_zarr_trajectories,
        save_zarr_trajectories,
        savitzky_golay,
        stitch_trajectories,
        xr,
        zarr,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # ⚡ 3D-PTV Interactive Zarr Dashboard & Data Explorer

        Cross-cut, filter, visualize 3D Lagrangian trajectories and Eulerian turbulence fields directly from **Zarr datasets** (`run.zarr`).
        """
    )
    return


@app.cell
def _(mo):
    path_input = mo.ui.text(
        value="run.zarr",
        label="📁 Path to Zarr Store (.zarr):",
        placeholder="e.g. C:/Users/alex/projects/openptv2/test_data/run.zarr",
    )
    path_input
    return (path_input,)


@app.cell
def _(Path, mo, path_input, zarr):
    z_path = Path(path_input.value)
    if not z_path.exists():
        status_md = f"⚠️ **Store path not found**: `{z_path.resolve()}`. Please specify a valid `.zarr` store."
        root = None
    else:
        try:
            root = zarr.open_group(str(z_path), mode="r")
            groups = list(root.keys())
            status_md = f"✅ **Zarr Store Loaded**: `{z_path.resolve()}` | Subgroups: `{', '.join(groups)}`"
        except Exception as e:
            status_md = f"❌ **Error opening Zarr store**: {e}"
            root = None

    mo.md(status_md)
    return root, status_md, z_path


@app.cell
def _(mo, root):
    if root is None:
        mo.stop()

    tabs = mo.ui.tabs({
        "📊 Dataset Summary": None,
        "🌊 Lagrangian Trajectories": None,
        "🧩 3D Correspondences": None,
        "📈 Eulerian Field Slicer": None,
        "🛠️ Processing & Export": None,
    })
    tabs
    return (tabs,)


@app.cell
def _(mo, np, root, z_path):
    summary_lines = []
    summary_lines.append(f"### 📂 Dataset Overview: `{z_path.name}`")

    if "targets" in root:
        cams = sorted([k for k in root["targets"].keys() if k.startswith("cam_")])
        summary_lines.append(f"- **Targets**: {len(cams)} cameras (`{', '.join(cams)}`)")
    if "correspondences" in root:
        f_keys = sorted([k for k in root["correspondences"].keys() if k.startswith("frame_")])
        summary_lines.append(f"- **Correspondences**: {len(f_keys)} frames")
    if "trajectories" in root:
        tr_grp = root["trajectories"]
        if "trajid" in tr_grp:
            n_pt = len(tr_grp["pos"])
            n_tr = len(np.unique(tr_grp["trajid"]))
            summary_lines.append(f"- **Trajectories**: {n_tr} trajectories ({n_pt} points)")
    if "eulerian" in root:
        vars_f = list(root["eulerian"].keys())
        summary_lines.append(f"- **Eulerian Fields**: {len(vars_f)} variables (`{', '.join(vars_f)}`)")

    mo.md("\n".join(summary_lines))
    return (summary_lines,)


@app.cell
def _(go, mo, np, read_zarr_trajectories, root, z_path):
    if root is None or "trajectories" not in root:
        mo.md("ℹ️ No trajectory group found in this Zarr store.")
        mo.stop()

    trajs = read_zarr_trajectories(z_path, group="trajectories")
    if not trajs:
        mo.md("ℹ️ Trajectories group is empty.")
        mo.stop()

    n_trajs = len(trajs)

    num_display = mo.ui.slider(
        start=1,
        stop=min(200, n_trajs),
        value=min(30, n_trajs),
        label="Max Trajectories to Render:",
    )
    color_by = mo.ui.dropdown(
        options=["Velocity Magnitude", "Z-Height", "Trajectory ID"],
        value="Velocity Magnitude",
        label="Color Trajectories By:",
    )

    mo.hstack([num_display, color_by])
    return color_by, n_trajs, num_display, trajs


@app.cell
def _(color_by, go, mo, np, num_display, trajs):
    N_show = int(num_display.value)
    selected_trajs = trajs[:N_show]

    fig = go.Figure()

    for tr in selected_trajs:
        p = tr.pos()
        v = tr.velocity()
        v_mag = np.linalg.norm(v, axis=1) if v is not None else np.zeros(len(p))

        if color_by.value == "Velocity Magnitude":
            colors = v_mag
            colorscale = "Viridis"
            c_title = "Speed (m/s)"
        elif color_by.value == "Z-Height":
            colors = p[:, 2]
            colorscale = "Plasma"
            c_title = "Z (m)"
        else:
            colors = np.full(len(p), tr.trajid())
            colorscale = "Rainbow"
            c_title = "Traj ID"

        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="lines+markers",
                marker=dict(size=3, color=colors, colorscale=colorscale, showscale=False),
                line=dict(width=4, color=colors, colorscale=colorscale),
                name=f"Traj {tr.trajid()}",
                text=[f"Traj {tr.trajid()} | Speed: {s:.3f} m/s" for s in v_mag],
                hoverinfo="text+x+y+z",
            )
        )

    fig.update_layout(
        title=f"3D Trajectories View ({N_show} Trajectories)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
    )

    mo.hstack([fig])
    return N_show, c_title, colors, colorscale, fig, selected_trajs


@app.cell
def _(mo, np, root, xr, z_path):
    if root is None or "eulerian" not in root:
        mo.md("ℹ️ No eulerian fields found in this Zarr store.")
        mo.stop()

    try:
        ds_eul = xr.open_zarr(z_path, group="eulerian")
    except Exception:
        try:
            ds_eul = xr.open_zarr(z_path)
        except Exception as e:
            mo.md(f"Error reading Eulerian group: {e}")
            mo.stop()

    var_options = list(ds_eul.data_vars.keys())
    if not var_options:
        mo.md("No Eulerian data variables present.")
        mo.stop()

    var_select = mo.ui.dropdown(
        options=var_options,
        value=var_options[0],
        label="Select Eulerian Variable:",
    )

    phase_slider = (
        mo.ui.slider(
            start=0,
            stop=len(ds_eul.phase) - 1,
            value=0,
            label="Phase / Frame Index:",
        )
        if "phase" in ds_eul.dims
        else None
    )

    mo.hstack([var_select, phase_slider] if phase_slider else [var_select])
    return ds_eul, phase_slider, var_options, var_select


@app.cell
def _(ds_eul, go, mo, np, phase_slider, var_select):
    v_name = var_select.value
    da = ds_eul[v_name]

    if "phase" in da.dims and phase_slider is not None:
        da_slice = da.isel(phase=int(phase_slider.value))
    else:
        da_slice = da

    if "z" in da_slice.dims:
        slice_2d = da_slice.mean(dim="z").values
    else:
        slice_2d = da_slice.values

    fig_slice = go.Figure(
        data=go.Heatmap(
            z=slice_2d.T,
            x=ds_eul.x.values if "x" in ds_eul.coords else None,
            y=ds_eul.y.values if "y" in ds_eul.coords else None,
            colorscale="Thermal",
        )
    )
    fig_slice.update_layout(
        title=f"2D XY Mean Cross-Section: {v_name}",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=500,
    )

    mo.hstack([fig_slice])
    return da, da_slice, fig_slice, slice_2d, v_name


@app.cell
def _(mo, trajs, z_path):
    mo.md(
        r"""
        ### 🛠️ Processing & Export Tools
        Apply trajectory stitching or export filtered Zarr subsets.
        """
    )

    export_btn = mo.ui.button(
        label="📥 Export Filtered Trajectories to CSV", value=False
    )
    export_btn
    return (export_btn,)


@app.cell
def _(export_btn, mo, np, trajs):
    if export_btn.value:
        rows = []
        for tr in trajs:
            p = tr.pos()
            v = tr.velocity()
            t = tr.time()
            for i in range(len(p)):
                rows.append([
                    tr.trajid(),
                    t[i],
                    p[i, 0],
                    p[i, 1],
                    p[i, 2],
                    v[i, 0] if v is not None else 0.0,
                    v[i, 1] if v is not None else 0.0,
                    v[i, 2] if v is not None else 0.0,
                ])
        header = "trajid,time,x,y,z,u,v,w\n"
        csv_text = header + "\n".join(
            ",".join(f"{val:.6f}" if isinstance(val, float) else str(val) for val in r)
            for r in rows[:1000]
        )
        download_ui = mo.download(
            data=csv_text.encode("utf-8"),
            filename="trajectories_export.csv",
            label="💾 Download CSV Sample (First 1,000 Points)",
        )
        mo.hstack([download_ui])
    return csv_text, download_ui, header, rows


if __name__ == "__main__":
    app.run()
