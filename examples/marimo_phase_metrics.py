import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase-Resolved Region & Wall-Shear Metrics for Pulsatile Flow

    A recipe for periodic (pump/piston-driven) flow through a
    constricted channel, on a synthetic dataset so it runs standalone:

    1. **Vortex identification** (`flowtracks.vortex.vortex_identification`) —
       Q-criterion, lambda2, enstrophy from the velocity field, vectorized
       over the whole grid and every phase at once.
    2. **Named-region time series** (`flowtracks.eulerian.region_timeseries`) —
       mean/max of any scalar field inside any boolean region, per phase.
       Works for a jet core vs. periphery, a subregion, or
       anything else a mask can express.
    3. **Wall shear from an occupancy isosurface** (`flowtracks.wall_shear`) —
       when there's no explicit wall mesh, extract one from the valid-data
       occupancy field (`occupancy_isosurface`) and estimate near-wall shear
       from the velocity gradient at a small offset (`near_wall_shear_stress`).
    4. **Cyclic wall-shear indices** — TAWSS / OSI / RRT, both the
       whole-cycle value (`tawss_osi_rrt`) and the running, phase-by-phase
       value (`running_tawss_osi_rrt`).

    Swap the synthetic generator below for a real phase-averaged Eulerian
    field (dims `x, y, z, phase`) and the rest of the notebook is unchanged.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from flowtracks.vortex import vortex_identification
    from flowtracks.eulerian import region_timeseries
    from flowtracks.wall_shear import (
        occupancy_isosurface, vertex_normals, near_wall_shear_stress,
        tawss_osi_rrt, running_tawss_osi_rrt,
    )
    import xarray as xr

    return (
        mo, np, plt, vortex_identification, region_timeseries,
        occupancy_isosurface, vertex_normals, near_wall_shear_stress,
        tawss_osi_rrt, running_tawss_osi_rrt, xr,
    )


@app.cell(hide_code=True)
def _(mo):
    n_phase_ui = mo.ui.slider(start=8, stop=32, step=1, value=16, label="Phases per cycle")
    radius_ui = mo.ui.slider(start=0.3, stop=1.0, step=0.05, value=0.6, label="Channel radius R0")
    swirl_ui = mo.ui.slider(start=0.0, stop=3.0, step=0.1, value=1.5, label="Peak swirl rate")
    offset_ui = mo.ui.slider(start=0.01, stop=0.2, step=0.01, value=0.08, label="Near-wall sampling offset")
    mu_ui = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=1.0, label="Viscosity mu")

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([n_phase_ui, radius_ui], gap=2),
        mo.hstack([swirl_ui, offset_ui, mu_ui], gap=2),
    ])
    return mu_ui, n_phase_ui, offset_ui, radius_ui, swirl_ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 0 — synthetic pulsatile, swirling flow in a constricted channel
    """)
    return


@app.cell
def _(n_phase_ui, np, radius_ui, swirl_ui):
    N_PHASE = n_phase_ui.value
    R0 = radius_ui.value
    n_grid = 24

    x = np.linspace(-1, 1, n_grid)
    y = np.linspace(-1, 1, n_grid)
    z = np.linspace(-1, 1, n_grid)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    r = np.sqrt(xx**2 + zz**2)
    valid = r < R0  # cross-section is constant along the channel axis (y)

    phase = np.arange(N_PHASE)
    axial_amp = 0.5 * (1 + np.cos(2 * np.pi * (phase - 0) / N_PHASE))       # pulsatile inflow
    swirl_amp = swirl_ui.value * np.sin(2 * np.pi * phase / N_PHASE)        # sign-reversing swirl

    profile = np.clip(1 - (r / R0) ** 2, 0, None)  # parabolic radial profile

    u = np.zeros((n_grid, n_grid, n_grid, N_PHASE))
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    for _k in range(N_PHASE):
        v[..., _k] = np.where(valid, axial_amp[_k] * profile, np.nan)
        u[..., _k] = np.where(valid, -swirl_amp[_k] * zz, np.nan)
        w[..., _k] = np.where(valid, swirl_amp[_k] * xx, np.nan)

    return N_PHASE, R0, dx, dy, dz, n_grid, r, u, v, valid, w, x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 — vortex identification (Q, lambda2, enstrophy), all phases at once
    """)
    return


@app.cell
def _(dx, dy, dz, np, u, v, vortex_identification, w):
    u_f = np.nan_to_num(u)
    v_f = np.nan_to_num(v)
    w_f = np.nan_to_num(w)
    q_field, lambda2_field, enstrophy_field = vortex_identification(u_f, v_f, w_f, dx, dy, dz)
    return enstrophy_field, lambda2_field, q_field


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — named-region time series (core vs. periphery)
    """)
    return


@app.cell
def _(enstrophy_field, mo, np, plt, q_field, r, region_timeseries, u, v, w, xr):
    dims = ("x", "y", "z", "phase")
    velmag = np.sqrt(np.nan_to_num(u) ** 2 + np.nan_to_num(v) ** 2 + np.nan_to_num(w) ** 2)

    fields = {
        "velocity_mag": xr.DataArray(velmag, dims=dims),
        "enstrophy": xr.DataArray(enstrophy_field, dims=dims),
        "Q_criterion": xr.DataArray(q_field, dims=dims),
    }

    core_mask = xr.DataArray(r < r[r > 0].max() * 0.4, dims=("x", "y", "z"))
    periphery_mask = xr.DataArray((r >= r[r > 0].max() * 0.4) & (r < r[r > 0].max() * 0.9), dims=("x", "y", "z"))

    core_ts = region_timeseries(fields, core_mask)
    periphery_ts = region_timeseries(fields, periphery_mask)

    fig_ts, axes_ts = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    for ax, name in zip(axes_ts, ["velocity_mag", "enstrophy", "Q_criterion"]):
        ax.plot(core_ts[f"{name}_mean"], "o-", label="core")
        ax.plot(periphery_ts[f"{name}_mean"], "s-", label="periphery")
        ax.set_title(f"{name} (mean)")
        ax.set_xlabel("phase")
        ax.legend(fontsize=8)
    fig_ts.tight_layout()
    mo.hstack([fig_ts])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — wall proxy from the occupancy field, and near-wall shear stress

    No explicit wall mesh here, so `occupancy_isosurface` extracts one from
    the valid-data indicator field (1 inside the channel, 0 outside). The
    same function accepts a real wall mesh's `(vertices, faces)` just as well.
    """)
    return


@app.cell
def _(
    mo, mu_ui, n_grid, near_wall_shear_stress, np, occupancy_isosurface,
    offset_ui, u, v, valid, vertex_normals, w, x, y, z,
):
    try:
        vertices, faces = occupancy_isosurface(valid.astype(float), x, y, z, level=0.5)
        have_vtk = True
    except ImportError:
        vertices, faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        have_vtk = False

    normals = vertex_normals(vertices, faces) if have_vtk and len(faces) else np.zeros((0, 3))

    n_phase_local = u.shape[-1]
    mu = mu_ui.value
    offset = offset_ui.value

    tau_series = np.full((len(vertices), n_phase_local, 3), np.nan)
    tau_mag_series = np.full((len(vertices), n_phase_local), np.nan)
    if have_vtk and len(vertices):
        for _k in range(n_phase_local):
            tau_k, mag_k = near_wall_shear_stress(
                vertices, normals, x, y, z,
                np.nan_to_num(u[..., _k]), np.nan_to_num(v[..., _k]), np.nan_to_num(w[..., _k]),
                mu, offset)
            tau_series[:, _k, :] = tau_k
            tau_mag_series[:, _k] = mag_k

    mo.md(f"Wall proxy: **{len(vertices)}** vertices, **{len(faces)}** faces "
         f"({'vtk available' if have_vtk else 'vtk NOT installed — skipping wall-shear steps'}).")
    return have_vtk, tau_mag_series, tau_series


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — TAWSS / OSI / RRT: whole-cycle and running-in-phase
    """)
    return


@app.cell
def _(have_vtk, mo, np, plt, running_tawss_osi_rrt, tawss_osi_rrt, tau_mag_series, tau_series):
    if not have_vtk:
        wss_output = mo.md("_(skipped — vtk not installed)_")
    else:
        tawss, osi, rrt = tawss_osi_rrt(tau_series, tau_mag_series)
        run_tawss, run_osi, run_rrt = running_tawss_osi_rrt(tau_series, tau_mag_series)

        fig_wss, axes_wss = plt.subplots(1, 2, figsize=(11, 4), dpi=100)
        axes_wss[0].hist(tawss, bins=20)
        axes_wss[0].set_title(f"TAWSS distribution over {len(tawss)} wall points")
        axes_wss[0].set_xlabel("TAWSS")

        axes_wss[1].plot(np.nanmean(run_tawss, axis=0), "-o", label="running TAWSS (wall-mean)")
        axes_wss[1].plot(np.nanmean(run_osi, axis=0), "-s", label="running OSI (wall-mean)")
        axes_wss[1].set_xlabel("phase")
        axes_wss[1].legend(fontsize=8)
        axes_wss[1].set_title("Running wall-shear indices converge over the cycle")
        fig_wss.tight_layout()
        wss_output = mo.hstack([fig_wss])
    wss_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Reusable pieces demoed here**: `flowtracks.vortex.vortex_identification`,
    `flowtracks.eulerian.region_timeseries`,
    `flowtracks.wall_shear.{occupancy_isosurface, vertex_normals,
    near_wall_shear_stress, tawss_osi_rrt, running_tawss_osi_rrt}`.
    None of it assumes any particular anatomy, instrument, or lab — only a
    phase-resolved velocity field on a regular grid, and optionally an
    explicit wall mesh.
    """)
    return


if __name__ == "__main__":
    app.run()
