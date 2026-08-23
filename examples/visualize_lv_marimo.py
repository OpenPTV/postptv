import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🫀 3D-PTV Flow Field Interactive Visualization Dashboard

    Interactive **Marimo** dashboard for exploring 3D Particle Tracking Velocimetry post-processing results.
    Compare **NetCDF (.nc)**, **Zarr (.zarr)**, and **VTK (.vtk)** outputs, inspect 3D vector fields,
    multi-plane orthogonal slices, turbulent fields (TKE, VSS, Helicity), and cardiac phase animation.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import xarray as xr
    import vtk

    return Path, mo, np, px, go, make_subplots, vtk, xr


@app.cell
def _(Path, mo):
    preset_picker = mo.ui.dropdown(
        options={
            "LV Dataset (Real Experiment)": r"C:\Users\alex\Downloads\hidimaging_test\LV",
            "PTV Output Sample (Repo Test Data)": str((Path(__file__).parent.parent / "ptv_output").resolve()),
        },
        value="LV Dataset (Real Experiment)",
        label="📁 Quick Folder Presets",
    )
    preset_picker
    return (preset_picker,)


@app.cell
def _(mo, preset_picker):
    folder_input = mo.ui.text(
        value=preset_picker.value,
        label="Target Results Directory Path",
        full_width=True,
    )
    format_radio = mo.ui.radio(
        options=["NetCDF (.nc)", "Zarr (.zarr)", "VTK (.vtk)"],
        value="NetCDF (.nc)",
        label="📦 Data Source Format",
    )
    mo.vstack([folder_input, format_radio])
    return folder_input, format_radio


@app.cell
def _(Path, folder_input, format_radio, mo, xr):
    target_dir = Path(folder_input.value)
    fmt = format_radio.value

    ds = None
    file_info = {}

    nc_file = target_dir / "post_analysis.nc"
    zarr_dir = target_dir / "post_analysis.zarr"

    if fmt == "NetCDF (.nc)" and nc_file.exists():
        with xr.open_dataset(nc_file) as _ds:
            ds = _ds.load()
        file_info["size_mb"] = nc_file.stat().st_size / (1024 * 1024)
        file_info["path"] = str(nc_file)
    elif fmt == "Zarr (.zarr)" and zarr_dir.exists():
        with xr.open_zarr(zarr_dir) as _ds:
            ds = _ds.load()
        z_bytes = sum(f.stat().st_size for f in zarr_dir.glob("**/*") if f.is_file())
        file_info["size_mb"] = z_bytes / (1024 * 1024)
        file_info["path"] = str(zarr_dir)
    elif nc_file.exists():
        with xr.open_dataset(nc_file) as _ds:
            ds = _ds.load()
        file_info["size_mb"] = nc_file.stat().st_size / (1024 * 1024)
        file_info["path"] = str(nc_file)

    if ds is not None:
        msg = f"✅ **Dataset Loaded Successfully** | Source: `{file_info.get('path', target_dir)}` | Size: {file_info.get('size_mb', 0):.2f} MB | Variables: {len(ds.data_vars)}"
    else:
        msg = f"⚠️ **Dataset Not Found** in `{target_dir}`. Please run the pipeline or check the path."

    mo.md(msg)
    return ds, file_info, fmt, nc_file, target_dir, zarr_dir


@app.cell
def _(ds, mo):
    if ds is None:
        vis_mode = None
    else:
        vis_mode = mo.ui.radio(
            options=[
                "1. 🚀 3D Flow Field Cones",
                "2. 🔲 Multi-Plane Orthogonal Slices (XY, XZ, YZ)",
                "3. 🌀 Turbulent Field Comparison (TKE, MKE, VSS, Helicity)",
                "4. 📈 Cardiac Phase Time-Series Explorer",
                "5. 🔀 3D Lagrangian Trajectory Inspector (wp4 / wp5)",
                "6. 📊 Format & Data Schema Inspector",
            ],
            value="1. 🚀 3D Flow Field Cones",
            label="🎨 Choose Visualization Mode",
        )
    vis_mode

    return (vis_mode,)


@app.cell
def _(ds, mo, vis_mode):
    if ds is None or vis_mode is None:
        num_phases, nx, ny, nz = 0, 0, 0, 0
        var_pick, phase_slider, x_slider, y_slider, z_slider, stride_slider = None, None, None, None, None, None
    else:
        num_phases = ds.sizes["phase"]
        nx, ny, nz = ds.sizes["x"], ds.sizes["y"], ds.sizes["z"]
        vars_all = sorted(list(ds.data_vars))

        var_pick = mo.ui.dropdown(options=vars_all, value="VEL" if "VEL" in ds else vars_all[0], label="Variable")
        phase_slider = mo.ui.slider(0, num_phases - 1, value=min(17, num_phases - 1), label="Phase Index")
        x_slider = mo.ui.slider(0, nx - 1, value=nx // 2, label="X Slice")
        y_slider = mo.ui.slider(0, ny - 1, value=ny // 2, label="Y Slice")
        z_slider = mo.ui.slider(0, nz - 1, value=nz // 2, label="Z Slice")
        stride_slider = mo.ui.slider(1, 4, value=2, label="3D Cone Stride")

        mode_str = vis_mode.value
        if "1." in mode_str:
            mo.hstack([phase_slider, stride_slider])
        elif "2." in mode_str:
            mo.hstack([var_pick, phase_slider, x_slider, y_slider, z_slider])
        elif "3." in mode_str:
            mo.hstack([phase_slider, z_slider])
        elif "4." in mode_str:
            mo.hstack([var_pick, phase_slider])
        else:
            mo.md("Format & Schema Inspector Selected")

    return (
        num_phases,
        nx,
        ny,
        nz,
        phase_slider,
        stride_slider,
        var_pick,
        x_slider,
        y_slider,
        z_slider,
    )


@app.cell
def _(
    ds,
    go,
    make_subplots,
    mo,
    np,
    phase_slider,
    stride_slider,
    var_pick,
    vis_mode,
    x_slider,
    y_slider,
    z_slider,
):
    if ds is None or vis_mode is None or vis_mode.value is None:
        fig_out = mo.md("No dataset loaded.")
    else:
        mode_val = vis_mode.value

        p_idx = phase_slider.value if phase_slider else 0

        if "1." in mode_val:
            st = stride_slider.value if stride_slider else 2
            snap = ds.isel(phase=p_idx)
            X3, Y3, Z3 = np.meshgrid(snap.x.values * 1000, snap.y.values * 1000, snap.z.values * 1000, indexing="ij")
            U3 = snap["u_ins_mean"].values if "u_ins_mean" in snap else np.zeros_like(X3)
            V3 = snap["v_ins_mean"].values if "v_ins_mean" in snap else np.zeros_like(X3)
            W3 = snap["w_ins_mean"].values if "w_ins_mean" in snap else np.zeros_like(X3)
            Speed3 = snap["VEL"].values if "VEL" in snap else np.sqrt(U3**2 + V3**2 + W3**2)

            sub_x, sub_y, sub_z = X3[::st, ::st, ::st].ravel(), Y3[::st, ::st, ::st].ravel(), Z3[::st, ::st, ::st].ravel()
            sub_u, sub_v, sub_w = U3[::st, ::st, ::st].ravel(), V3[::st, ::st, ::st].ravel(), W3[::st, ::st, ::st].ravel()
            sub_spd = Speed3[::st, ::st, ::st].ravel()
            mask_nz = sub_spd > 0.01

            fig_out = go.Figure(go.Cone(
                x=sub_x[mask_nz], y=sub_y[mask_nz], z=sub_z[mask_nz],
                u=sub_u[mask_nz], v=sub_v[mask_nz], w=sub_w[mask_nz],
                colorscale="Viridis", sizemode="scaled", sizeref=0.5, anchor="tail",
                colorbar=dict(title="Speed (m/s)"), hoverinfo="x+y+z+u+v+w+norm"
            ))
            fig_out.update_layout(
                title=f"3D Velocity Vector Cones (Phase {p_idx})",
                scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data"),
                width=900, height=650, margin=dict(l=0, r=0, t=40, b=0)
            )

        elif "2." in mode_val:
            v_name = var_pick.value if var_pick else "VEL"
            xi = x_slider.value if x_slider else 0
            yi = y_slider.value if y_slider else 0
            zi = z_slider.value if z_slider else 0
            snap = ds.isel(phase=p_idx)

            field3d = snap[v_name].values
            x_mm, y_mm, z_mm = snap.x.values * 1000, snap.y.values * 1000, snap.z.values * 1000
            slice_xy, slice_xz, slice_yz = field3d[:, :, zi].T, field3d[:, yi, :].T, field3d[xi, :, :].T

            cs = "Inferno" if any(k in v_name for k in ["TKE", "MKE", "VSS", "RSS"]) else "Viridis"
            fig_out = make_subplots(rows=1, cols=3, subplot_titles=[
                f"XY Plane (Z={z_mm[zi]:.1f} mm)", f"XZ Plane (Y={y_mm[yi]:.1f} mm)", f"YZ Plane (X={x_mm[xi]:.1f} mm)"
            ])
            fig_out.add_trace(go.Heatmap(x=x_mm, y=y_mm, z=slice_xy, colorscale=cs, showscale=False), row=1, col=1)
            fig_out.add_trace(go.Heatmap(x=x_mm, y=z_mm, z=slice_xz, colorscale=cs, showscale=False), row=1, col=2)
            fig_out.add_trace(go.Heatmap(x=y_mm, y=z_mm, z=slice_yz, colorscale=cs, showscale=True, colorbar=dict(title=v_name)), row=1, col=3)
            fig_out.update_layout(title=f"Orthogonal Slices for {v_name} (Phase {p_idx})", width=950, height=450)

        elif "3." in mode_val:
            zi = z_slider.value if z_slider else 0
            snap = ds.isel(phase=p_idx)
            x_mm, y_mm = snap.x.values * 1000, snap.y.values * 1000

            fig_out = make_subplots(rows=2, cols=2, subplot_titles=[
                "Speed VEL (m/s)", "TKE (J/m³)", "Viscous Shear Stress VSS (Pa)", "Helicity H1 (m/s²)"
            ])
            fig_out.add_trace(go.Heatmap(x=x_mm, y=y_mm, z=snap["VEL"].isel(z=zi).values.T, colorscale="Viridis"), row=1, col=1)
            fig_out.add_trace(go.Heatmap(x=x_mm, y=y_mm, z=snap["TKE"].isel(z=zi).values.T, colorscale="Inferno"), row=1, col=2)
            fig_out.add_trace(go.Heatmap(x=x_mm, y=y_mm, z=snap["VSS"].isel(z=zi).values.T, colorscale="Plasma"), row=2, col=1)
            fig_out.add_trace(go.Heatmap(x=x_mm, y=y_mm, z=snap["H1"].isel(z=zi).values.T, colorscale="RdBu", zmid=0), row=2, col=2)
            fig_out.update_layout(title=f"Turbulent Fields Comparison (Phase {p_idx}, Z={snap.z.values[zi]*1000:.1f} mm)", width=950, height=700)

        elif "4." in mode_val:
            v_name = var_pick.value if var_pick else "VEL"
            phases = ds.phase.values
            mean_val = ds[v_name].mean(("x", "y", "z")).values
            max_val = ds[v_name].max(("x", "y", "z")).values

            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(x=phases, y=mean_val, mode="lines+markers", name=f"Mean {v_name}", line=dict(width=3, color="royalblue")))
            fig_out.add_trace(go.Scatter(x=phases, y=max_val, mode="lines+markers", name=f"Max {v_name}", line=dict(width=2, color="crimson", dash="dash")))
            fig_out.add_vline(x=p_idx, line_dash="dash", line_color="orange", annotation_text=f"Phase {p_idx}")
            fig_out.update_layout(title=f"Cardiac Cycle Evolution: {v_name}", xaxis_title="Phase Bin", yaxis_title=v_name, width=900, height=450)

        elif "5." in mode_val:
            from flowtracks.io import Scene
            set_pick = "wp4" if (target_dir / "wp4" / "trajectories.h5").exists() else "wp1"
            h5_p = target_dir / set_pick / "trajectories.h5"
            if not h5_p.exists():
                h5_p = target_dir / f"{set_pick}_traj4.h5"

            if h5_p.exists():
                sc = Scene(str(h5_p))
                trajs = [t for t in sc.iter_trajectories() if len(t) >= 15]
                trajs.sort(key=lambda tr: len(tr), reverse=True)
                top_trajs = trajs[:50]

                fig_out = go.Figure()
                for idx, tr in enumerate(top_trajs):
                    pos = tr.pos() * 1000.0
                    vel = tr.velocity()
                    spd = np.linalg.norm(vel, axis=1)
                    fig_out.add_trace(go.Scatter3d(
                        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                        mode="lines+markers",
                        marker=dict(size=2, color=spd, colorscale="Viridis", showscale=(idx == 0)),
                        line=dict(color=spd, colorscale="Viridis", width=3),
                        opacity=0.7,
                        name=f"Track #{idx+1} (len={len(tr)})",
                        showlegend=False
                    ))
                fig_out.update_layout(
                    title=f"3D Lagrangian Particle Trajectories — Set [{set_pick}] (Top {len(top_trajs)} Longest Tracks)",
                    scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data"),
                    width=900, height=650, margin=dict(l=0, r=0, t=40, b=0)
                )
            else:
                fig_out = mo.md(f"⚠️ Trajectory HDF5 file not found at `{h5_p}`")

        else:
            var_table = []
            for var in sorted(list(ds.data_vars)):
                da = ds[var]
                var_table.append({
                    "Variable": var, "Dimensions": str(da.dims), "Shape": str(da.shape),
                    "Min": f"{float(da.min()):.4e}", "Mean": f"{float(da.mean()):.4e}", "Max": f"{float(da.max()):.4e}",
                    "Units": da.attrs.get("units", "N/A"), "Long Name": da.attrs.get("long_name", "N/A")
                })
            fig_out = mo.ui.table(var_table, label="Dataset Variable Schema & Metadata")

    fig_out
    return (fig_out,)


if __name__ == "__main__":
    app.run()

