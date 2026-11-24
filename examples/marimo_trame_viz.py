"""Browser-based 3D viewer for flowtracks data (marimo launcher).

The 3D scene does not render inside this notebook. Clicking
"Open 3D viewer" starts ``flowtracks.visualization`` as a background
process and opens a local browser tab -- the full pyvista + trame view
(orbit/pan/zoom, frame slider, Lagrangian/Eulerian/vessel switches,
screenshot button). This is the intended shape of the integration:
marimo is the shell, the browser app is the tool.

Run from the repo root — first time:
    uv sync --extra viz
    uv run marimo edit examples/marimo_trame_viz.py
"""
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from flowtracks.visualization import launch

    return launch, mo


@app.cell
def _(mo):
    mo.md(r"""
    # 🌐 Browser 3D viewer — launcher

    Native formats in (Zarr trajectories), interactive browser view out. The
    app opens a new tab on `localhost`; close the tab or `proc.terminate()`
    to stop it.
    """)
    return


@app.cell
def _(mo):
    zarr_input = mo.ui.text(
        value="data/tracers.zarr",
        label="Zarr store (repo-relative)",
        placeholder="data/tracers.zarr",
    )
    volume_input = mo.ui.text(
        value="data/tracers_mean_speed.vti",
        label="Eulerian volume (optional, .vti)",
        placeholder="data/tracers_mean_speed.vti",
    )
    button = mo.ui.button(label="Open 3D viewer ▸", kind="success")
    mo.hstack([zarr_input, volume_input, button])
    return button, volume_input, zarr_input


@app.cell
def _(button, launch, mo, volume_input, zarr_input):
    if button.value:
        proc = launch(
            zarr_input.value,
            max_trajectories=500,
            volume=volume_input.value or None,
        )
        mo.md(
            f"**Viewer started** (pid `{proc.pid}`) — the 3D scene opens in a new "
            f"browser tab. Use the side switches to toggle Lagrangian / Eulerian / "
            f"vessel layers; the frame slider shows history up to a frame. Close "
            f"the tab or terminate the process to stop it."
        )
    return
