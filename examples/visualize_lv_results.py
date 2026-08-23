"""Benchmark file readers (NetCDF, Zarr, VTK) and generate interactive + static 3D flow visualizations.
"""

import sys
import time
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import vtk
from vtk.util import numpy_support

LV_DIR = Path(r"C:\Users\alex\Downloads\hidimaging_test\LV")
ARTIFACT_DIR = Path(r"C:\Users\alex\.gemini\antigravity-cli\brain\a620070e-6ea4-424e-ac93-9510a8f07eb5")


def benchmark_readers() -> dict:
    """Benchmark file load times for NetCDF, Zarr, and VTK formats."""
    results = {}

    # 1. NetCDF Load Time
    nc_path = LV_DIR / "post_analysis.nc"
    t0 = time.perf_counter()
    with xr.open_dataset(nc_path) as ds_nc:
        ds_nc.load()
    t_nc = (time.perf_counter() - t0) * 1000.0  # ms
    results["NetCDF (.nc)"] = {
        "load_time_ms": t_nc,
        "file_size_mb": nc_path.stat().st_size / (1024 * 1024),
        "variables_count": len(ds_nc.data_vars),
        "dataset": ds_nc,
    }

    # 2. Zarr Load Time
    zarr_path = LV_DIR / "post_analysis.zarr"
    t0 = time.perf_counter()
    with xr.open_zarr(zarr_path) as ds_zarr:
        ds_zarr.load()
    t_zarr = (time.perf_counter() - t0) * 1000.0  # ms
    # Calculate total size of Zarr folder
    zarr_bytes = sum(f.stat().st_size for f in zarr_path.glob("**/*") if f.is_file())
    results["Zarr (.zarr)"] = {
        "load_time_ms": t_zarr,
        "file_size_mb": zarr_bytes / (1024 * 1024),
        "variables_count": len(ds_zarr.data_vars),
        "dataset": ds_zarr,
    }

    # 3. VTK Load Time
    vtk_path = LV_DIR / "vtk_output" / "phase_000.vtk"
    t0 = time.perf_counter()
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    grid = reader.GetOutput()
    t_vtk = (time.perf_counter() - t0) * 1000.0  # ms
    vtk_total_size = sum(f.stat().st_size for f in vtk_path.parent.glob("*.vtk"))
    results["VTK (.vtk, 24 files)"] = {
        "load_time_ms": t_vtk,
        "file_size_mb": vtk_total_size / (1024 * 1024),
        "variables_count": grid.GetPointData().GetNumberOfArrays(),
        "dataset": grid,
    }

    return results


def create_visualizations(ds: xr.Dataset) -> None:
    """Generate Matplotlib static plots and Plotly interactive 3D visualizations."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_vis_dir = LV_DIR / "visualizations"
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    # Pick phase with peak velocity magnitude
    vel_mag = ds["VEL"]
    peak_phase = int(vel_mag.mean(("x", "y", "z")).argmax("phase"))
    snap = ds.isel(phase=peak_phase)

    print(f"Generating flow visualizations for peak flow phase: {peak_phase}...")

    # --- 1. Matplotlib Static Figure (3D Quiver + Slice Heatmaps) ---
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.suptitle(f"3D-PTV Flow Field Analysis (LV Dataset, Peak Phase {peak_phase})", fontsize=16, fontweight="bold")

    X, Y, Z = np.meshgrid(snap.x.values * 1000, snap.y.values * 1000, snap.z.values * 1000, indexing="ij")
    U = snap["u_ins_mean"].values
    V = snap["v_ins_mean"].values
    W = snap["w_ins_mean"].values
    Speed = snap["VEL"].values
    TKE = snap["TKE"].values

    # Plot 1: 3D Quiver / Vector Velocity Field
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    stride = 2
    ax1.quiver(
        X[::stride, ::stride, ::stride],
        Y[::stride, ::stride, ::stride],
        Z[::stride, ::stride, ::stride],
        U[::stride, ::stride, ::stride],
        V[::stride, ::stride, ::stride],
        W[::stride, ::stride, ::stride],
        length=15.0, normalize=True, cmap="plasma", linewidth=0.8, alpha=0.8
    )
    ax1.set_title("3D Velocity Vectors (Quiver Plot)", fontsize=12)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")

    # Plot 2: Velocity Magnitude Slice (Mid Z)
    ax2 = fig.add_subplot(2, 2, 2)
    mid_z = snap.sizes["z"] // 2
    im2 = ax2.pcolormesh(
        snap.x.values * 1000, snap.y.values * 1000, Speed[:, :, mid_z].T,
        cmap="viridis", shading="auto"
    )
    fig.colorbar(im2, ax=ax2, label="Velocity Magnitude VEL (m/s)")
    ax2.set_title(f"Mean Velocity Magnitude (Mid Z-Slice = {snap.z.values[mid_z]*1000:.1f} mm)", fontsize=12)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.axis("equal")

    # Plot 3: Turbulent Kinetic Energy Slice (Mid Z)
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.pcolormesh(
        snap.x.values * 1000, snap.y.values * 1000, TKE[:, :, mid_z].T,
        cmap="inferno", shading="auto"
    )
    fig.colorbar(im3, ax=ax3, label="TKE (J/m³)")
    ax3.set_title(f"Turbulent Kinetic Energy TKE (Mid Z-Slice = {snap.z.values[mid_z]*1000:.1f} mm)", fontsize=12)
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.axis("equal")

    # Plot 4: TKE and VEL evolution across all phases
    ax4 = fig.add_subplot(2, 2, 4)
    phases = ds.phase.values
    m_vel = vel_mag.mean(("x", "y", "z")).values
    m_tke = ds["TKE"].mean(("x", "y", "z")).values
    
    color = "tab:blue"
    ax4.set_xlabel("Flow Phase")
    ax4.set_ylabel("Mean Speed (m/s)", color=color)
    ax4.plot(phases, m_vel, color=color, marker="o", linewidth=2, label="Velocity Magnitude")
    ax4.tick_params(axis="y", labelcolor=color)

    ax4_twin = ax4.twinx()
    color = "tab:red"
    ax4_twin.set_ylabel("Mean TKE (J/m³)", color=color)
    ax4_twin.plot(phases, m_tke, color=color, marker="s", linestyle="--", linewidth=2, label="TKE")
    ax4_twin.tick_params(axis="y", labelcolor=color)
    ax4.set_title("Domain-Averaged Speed & TKE over Cardiac Cycle", fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    static_png = ARTIFACT_DIR / "flow_visualization.png"
    plt.savefig(static_png, bbox_inches="tight")
    plt.savefig(out_vis_dir / "flow_visualization.png", bbox_inches="tight")
    plt.close()
    print(f"Saved static PNG artifact to {static_png}")

    # --- 2. Plotly Interactive 3D Cone Flow Field ---
    mask_nz = Speed > 0.01
    X_f, Y_f, Z_f = X[mask_nz], Y[mask_nz], Z[mask_nz]
    U_f, V_f, W_f = U[mask_nz], V[mask_nz], W[mask_nz]
    Mag_f = Speed[mask_nz]

    cone_fig = go.Figure(
        data=go.Cone(
            x=X_f, y=Y_f, z=Z_f,
            u=U_f, v=V_f, w=W_f,
            colorscale="Viridis",
            sizemode="absolute",
            sizeref=float(Mag_f.max()) * 0.5 if len(Mag_f) > 0 else 1.0,
            colorbar=dict(title="Velocity (m/s)"),
            hoverinfo="x+y+z+u+v+w+norm"
        )
    )

    cone_fig.update_layout(
        title=f"Interactive 3D Flow Field Velocity Vectors (Phase {peak_phase})",
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Z (mm)"),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        width=1000, height=700
    )

    html_out = out_vis_dir / "3d_flow_cones.html"
    cone_fig.write_html(str(html_out))
    print(f"Saved interactive 3D HTML figure to {html_out}")


def main():
    print("=" * 60)
    print("Benchmarking Readers & Generating Flow Visualizations")
    print("=" * 60)

    bench = benchmark_readers()
    print("\n--- Reader Performance Comparison ---")
    for fmt, metrics in bench.items():
        print(f"Format: {fmt:22s} | Load Time: {metrics['load_time_ms']:6.2f} ms | File Size: {metrics['file_size_mb']:6.2f} MB | Vars: {metrics['variables_count']}")

    # Use NetCDF dataset for creating visualization figures
    ds = bench["NetCDF (.nc)"]["dataset"]
    create_visualizations(ds)


if __name__ == "__main__":
    main()
