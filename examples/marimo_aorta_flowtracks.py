import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import flowtracks
    import flowtracks.io
    from flowtracks.stitching import stitch_trajectories
    from flowtracks.smoothing import savitzky_golay

    return flowtracks, mo, np, plt, savitzky_golay, stitch_trajectories


@app.cell
def _(mo):
    mo.md("""
    # 🌊 Flowtracks Trajectory Post-Processing: TT13_aorta Dataset
    This interactive notebook loads 3D particle trajectories from OpenPTV output (`ptv_is.*` files)
    and performs:
    1. **Raw Trajectories Inspection**
    2. **Gap Stitching / Relinking** (`flowtracks.stitching.stitch_trajectories`)
    3. **Savitzky-Golay Smoothing & Kinematic Derivatives** (`flowtracks.smoothing.savitzky_golay`)
    """)
    return


@app.cell
def _(mo):
    res_dir = r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1\res"
    fname_pattern = res_dir + r"\ptv_is.%d"

    ui_fps = mo.ui.number(label="Camera FPS", value=10.0, step=1.0)
    ui_max_gap = mo.ui.slider(start=1, stop=5, step=1, value=3, label="Stitching Max Gap [frames]")
    ui_max_dist = mo.ui.slider(start=1.0, stop=30.0, step=1.0, value=15.0, label="Stitching Max Dist [mm]")
    ui_max_vel_diff = mo.ui.slider(start=10.0, stop=200.0, step=10.0, value=100.0, label="Stitching Max Vel Diff [mm/s]")

    ui_sg_window = mo.ui.dropdown(options=["5", "7", "9", "11"], value="7", label="SG Window Size")
    ui_sg_order = mo.ui.dropdown(options=["2", "3"], value="2", label="SG Polynomial Order")

    ui_num_display = mo.ui.slider(start=10, stop=200, step=10, value=50, label="Trajectories to Plot (3D)")
    ui_color_by = mo.ui.dropdown(
        options=["Trajectory ID", "Speed [mm/s]", "Acceleration [mm/s²]"],
        value="Speed [mm/s]",
        label="Color Trajectories By",
    )

    mo.vstack([
        mo.md("### ⚙️ Post-Processing Controls"),
        mo.hstack([ui_fps, ui_max_gap, ui_max_dist, ui_max_vel_diff], gap=2),
        mo.hstack([ui_sg_window, ui_sg_order, ui_num_display, ui_color_by], gap=2),
    ])
    return (
        fname_pattern,
        ui_color_by,
        ui_fps,
        ui_max_dist,
        ui_max_gap,
        ui_max_vel_diff,
        ui_num_display,
        ui_sg_order,
        ui_sg_window,
    )


@app.cell
def _(
    flowtracks,
    fname_pattern,
    savitzky_golay,
    stitch_trajectories,
    ui_fps,
    ui_max_dist,
    ui_max_gap,
    ui_max_vel_diff,
    ui_sg_order,
    ui_sg_window,
):
    fps_val = float(ui_fps.value)
    # Load raw trajectories from ptv_is files using flowtracks.io
    raw_trajs = flowtracks.io.trajectories(fname_pattern, 1, 10, frate=fps_val)

    # Convert distance from mm to meters for SI internal calculation
    dist_m = float(ui_max_dist.value) / 1000.0
    vel_diff_m = float(ui_max_vel_diff.value) / 1000.0

    # 1. Stitching
    stitched_trajs = stitch_trajectories(
        raw_trajs,
        fps=fps_val,
        max_gap=int(ui_max_gap.value),
        max_distance=dist_m,
        max_vel_diff=vel_diff_m,
    )

    # 2. Savitzky-Golay Smoothing
    w_size = int(ui_sg_window.value)
    poly_ord = int(ui_sg_order.value)
    smoothed_trajs = savitzky_golay(
        stitched_trajs,
        fps=fps_val,
        window_size=w_size,
        order=poly_ord,
    )
    return raw_trajs, smoothed_trajs, stitched_trajs


@app.cell
def _(mo, np, raw_trajs, smoothed_trajs, stitched_trajs):
    def compute_stats(t_list):
        if not t_list:
            return {"count": 0, "mean_len": 0, "max_len": 0, "mean_speed": 0, "max_acc": 0}
        lens = [len(t) for t in t_list]
        speeds = []
        accs = []
        for t in t_list:
            # velocities in mm/s
            v_mag = np.linalg.norm(t.velocity(), axis=1) * 1000.0
            speeds.extend(v_mag)
            if t.has_property("accel"):
                a_mag = np.linalg.norm(t.accel(), axis=1) * 1000.0
                accs.extend(a_mag)

        mean_spd = float(np.mean(speeds)) if speeds else 0.0
        max_a = float(np.max(accs)) if accs else 0.0
        return {
            "count": len(t_list),
            "mean_len": round(float(np.mean(lens)), 2),
            "max_len": int(np.max(lens)),
            "mean_speed": round(mean_spd, 2),
            "max_acc": round(max_a, 2),
        }

    s_raw = compute_stats(raw_trajs)
    s_stitched = compute_stats(stitched_trajs)
    s_smoothed = compute_stats(smoothed_trajs)

    table_md = f"""
    ### 📊 Pipeline Comparison Summary

    | Pipeline Stage | Trajectories Count | Mean Length (frames) | Max Length | Mean Speed [mm/s] | Max Acceleration [mm/s²] |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **1. Raw `ptv_is` Data** | {s_raw['count']} | {s_raw['mean_len']} | {s_raw['max_len']} | {s_raw['mean_speed']} | {s_raw['max_acc']} |
    | **2. After Gap Stitching** | {s_stitched['count']} | {s_stitched['mean_len']} | {s_stitched['max_len']} | {s_stitched['mean_speed']} | {s_stitched['max_acc']} |
    | **3. Stitched + SG Smoothed** | {s_smoothed['count']} | {s_smoothed['mean_len']} | {s_smoothed['max_len']} | {s_smoothed['mean_speed']} | {s_smoothed['max_acc']} |
    """
    mo.md(table_md)
    return


@app.cell
def _(mo, np, plt, raw_trajs, stitched_trajs, ui_color_by, ui_num_display):
    N_disp = int(ui_num_display.value)
    color_choice = ui_color_by.value

    fig = plt.figure(figsize=(15, 6), dpi=100)

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    def plot_trajs_on_ax(ax, t_list, title):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")

        subset = t_list[:N_disp]
        for idx, t in enumerate(subset):
            # convert positions to mm
            p_mm = t.pos() * 1000.0

            if color_choice == "Trajectory ID":
                c = plt.cm.tab20(idx % 20)
            elif color_choice == "Speed [mm/s]":
                spd = np.linalg.norm(t.velocity(), axis=1) * 1000.0
                c = plt.cm.viridis(np.clip(spd / 100.0, 0, 1))
            else:
                if t.has_property("accel"):
                    acc = np.linalg.norm(t.accel(), axis=1) * 1000.0
                    c = plt.cm.plasma(np.clip(acc / 500.0, 0, 1))
                else:
                    c = "blue"

            ax.plot(p_mm[:, 0], p_mm[:, 1], p_mm[:, 2], "o-", markersize=3, linewidth=1.5, alpha=0.8)

    plot_trajs_on_ax(ax1, raw_trajs, f"1. Raw Trajectories (First {N_disp})")
    plot_trajs_on_ax(ax2, stitched_trajs, f"2. Stitched Trajectories (First {N_disp})")

    fig.tight_layout()
    mo.hstack([fig])
    return


@app.cell
def _(mo, plt, smoothed_trajs, stitched_trajs, ui_num_display):
    N_disp2 = int(ui_num_display.value)
    fig2 = plt.figure(figsize=(15, 6), dpi=100)

    ax3 = fig2.add_subplot(121, projection="3d")
    ax4 = fig2.add_subplot(122, projection="3d")

    ax3.set_title(f"2. Stitched Trajectories (First {N_disp2})", fontsize=12, fontweight="bold")
    ax3.set_xlabel("X [mm]")
    ax3.set_ylabel("Y [mm]")
    ax3.set_zlabel("Z [mm]")

    for idx, t in enumerate(stitched_trajs[:N_disp2]):
        p_mm = t.pos() * 1000.0
        ax3.plot(p_mm[:, 0], p_mm[:, 1], p_mm[:, 2], "o-", markersize=3, linewidth=1.5, alpha=0.8, color="tab:blue")

    ax4.set_title(f"3. Stitched + SG Smoothed Trajectories (First {N_disp2})", fontsize=12, fontweight="bold")
    ax4.set_xlabel("X [mm]")
    ax4.set_ylabel("Y [mm]")
    ax4.set_zlabel("Z [mm]")

    for idx, t in enumerate(smoothed_trajs[:N_disp2]):
        p_mm = t.pos() * 1000.0
        ax4.plot(p_mm[:, 0], p_mm[:, 1], p_mm[:, 2], "s-", markersize=3, linewidth=1.5, alpha=0.8, color="tab:green")

    fig2.tight_layout()
    mo.hstack([fig2])
    return


@app.cell
def _(mo, np, plt, raw_trajs, smoothed_trajs):
    fig_kin = plt.figure(figsize=(14, 5), dpi=100)
    ax_v = fig_kin.add_subplot(121)
    ax_a = fig_kin.add_subplot(122)

    ax_v.set_title("Speed Distribution Comparison", fontsize=12, fontweight="bold")
    ax_v.set_xlabel("Speed [mm/s]")
    ax_v.set_ylabel("Density")

    raw_speeds = [np.linalg.norm(v) * 1000.0 for t in raw_trajs for v in t.velocity()]
    sm_speeds = [np.linalg.norm(v) * 1000.0 for t in smoothed_trajs for v in t.velocity()]

    if raw_speeds:
        ax_v.hist(raw_speeds, bins=30, alpha=0.5, label="Raw ptv_is", density=True, color="tab:blue")
    if sm_speeds:
        ax_v.hist(sm_speeds, bins=30, alpha=0.5, label="SG Smoothed", density=True, color="tab:green")
    ax_v.legend()

    ax_a.set_title("Acceleration Distribution Comparison", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Acceleration [mm/s²]")
    ax_a.set_ylabel("Density")

    raw_accs = [np.linalg.norm(a) * 1000.0 for t in raw_trajs if t.has_property("accel") for a in t.accel()]
    sm_accs = [np.linalg.norm(a) * 1000.0 for t in smoothed_trajs if t.has_property("accel") for a in t.accel()]

    if raw_accs:
        ax_a.hist(raw_accs, bins=30, alpha=0.5, label="Raw ptv_is", density=True, color="tab:blue")
    if sm_accs:
        ax_a.hist(sm_accs, bins=30, alpha=0.5, label="SG Smoothed", density=True, color="tab:green")
    ax_a.legend()

    fig_kin.tight_layout()
    mo.hstack([fig_kin])
    return


if __name__ == "__main__":
    app.run()
