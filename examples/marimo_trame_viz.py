"""Browser-based 3D viewer for flowtracks data (marimo launcher).

The 3D scene itself does not render inside this notebook. Clicking
"Open 3D viewer" starts ``flowtracks.visualization`` as a background
subprocess and opens a local browser tab -- the full pyvista + trame
view (pan/zoom/rotate, frame slider, vessel/volume overlays, screenshot
button). This is the intended shape of the integration: marimo is the
shell, the browser app is the tool.

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
    zarr_input
    return (zarr_input,)


@app.cell
def _(mo, launch, zarr_input):
    button = mo.ui.button(label="Open 3D viewer ▸", kind="success")
    if button.value:
        proc = launch(zarr_input.value, max_trajectories=500)
        mo.md(
            f"**Viewer started** (pid `{proc.pid}`) — the 3D scene opens in a new "
            f"browser tab. Close the tab or terminate the process to stop it."
        )
    button
    return button
