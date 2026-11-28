import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import h5py
    import numpy as np
    import tempfile
    from pathlib import Path

    import flowtracks
    from flowtracks.io import trajectories, save_zarr_trajectories
    from flowtracks.graphics import plot_trajectories_3d, select_trajectories
    from flowtracks.myptv import convert_myptv_to_zarr
    from flowtracks.proptv import convert_proptv_to_zarr
    from flowtracks.openlpt import convert_openlpt_to_zarr
    import pyvista as pv

    return (convert_myptv_to_zarr, convert_openlpt_to_zarr,
            convert_proptv_to_zarr, flowtracks, h5py, mo, np, Path, pv,
            plot_trajectories_3d, save_zarr_trajectories,
            select_trajectories, tempfile, trajectories)


@app.cell
def _(mo):
    mo.md("""
    # 🌊 Flowtracks: one post-processing pipeline, four 3D-PTV trackers

    This notebook shows how **flowtracks** reads the output of four different
    3D-PTV pipelines — **OpenPTV**, **MyPTV**, **proPTV** and **OpenLPT /
    pyOpenLPT** — converts each into the same `trajectories` Zarr layout, and
    renders them with one call. No conversion step, no per-tracker glue code.
    """)
    return


@app.cell
def _(mo):
    mo.md("## 1. OpenPTV (native format)")
    return


@app.cell
def _(Path, mo, np, pv, plot_trajectories_3d, save_zarr_trajectories,
      select_trajectories, tempfile, trajectories):
    def make1():
        trajs = trajectories("data/tracers/ptv_is.%d", first=10001, last=10020,
                              frate=500.0, traj_min_len=20)
        zp = Path(tempfile.mkdtemp()) / "openptv.zarr"
        save_zarr_trajectories(trajs, zp, overwrite=True)
        sel = select_trajectories(zp, by="length", n=8)
        plotter = pv.Plotter(off_screen=True)
        plot_trajectories_3d(zp, trajids=sel, context=2000,
                             plotter=plotter, show=False)
        img = str(Path(tempfile.mkdtemp()) / "openptv.png")
        plotter.screenshot(img, window_size=(700, 500))
        return img
    mo.image(src=make1())
    return


@app.cell
def _(mo):
    mo.md("## 2. MyPTV (TSV → Zarr)")
    return


@app.cell
def _(Path, convert_myptv_to_zarr, mo, np, pv, plot_trajectories_3d,
      select_trajectories, tempfile):
    def make2():
        d = Path(tempfile.mkdtemp()) / "myptv.tsv"
        rng = np.random.default_rng(0)
        lines = []
        for tid in range(5):
            x0, y0, z0 = rng.uniform(0, 10, 3)
            for t in range(20):
                lines.append(f"{tid}\t{x0+t*0.2:.3f}\t{y0+t*0.1:.3f}\t{z0:.3f}\t"
                             f"0.2\t0.1\t0.0\t0.0\t0.0\t0.0\t{t}")
        d.write_text("\n".join(lines) + "\n")
        zp = Path(tempfile.mkdtemp()) / "myptv.zarr"
        convert_myptv_to_zarr(d, zp)
        sel = select_trajectories(zp, by="length", n=5)
        plotter = pv.Plotter(off_screen=True)
        plot_trajectories_3d(zp, trajids=sel, context=500,
                             plotter=plotter, show=False)
        img = str(Path(tempfile.mkdtemp()) / "myptv.png")
        plotter.screenshot(img, window_size=(700, 500))
        return img
    mo.image(src=make2())
    return


@app.cell
def _(mo):
    mo.md("## 3. proPTV (HDF5 → Zarr)")
    return


@app.cell
def _(Path, convert_proptv_to_zarr, h5py, mo, np, pv, plot_trajectories_3d,
      select_trajectories, tempfile):
    def make3():
        hdf = Path(tempfile.mkdtemp()) / "tracks.hdf5"
        rng = np.random.default_rng(1)
        with h5py.File(str(hdf), "w") as f:
            for tid in range(5):
                frames = np.arange(20)
                x = rng.uniform(-10, 10, 20).cumsum() * 0.1
                y = rng.uniform(-10, 10, 20).cumsum() * 0.1
                z = rng.uniform(-10, 10, 20).cumsum() * 0.1
                arr = np.column_stack([frames, x, y, z,
                                       np.gradient(x), np.gradient(y), np.gradient(z),
                                       np.zeros(20), np.zeros(20), np.zeros(20)])
                f.create_dataset(f"track_{19}_{tid}", data=arr)
        zp = Path(tempfile.mkdtemp()) / "proptv.zarr"
        convert_proptv_to_zarr(hdf, zp)
        sel = select_trajectories(zp, by="length", n=5)
        plotter = pv.Plotter(off_screen=True)
        plot_trajectories_3d(zp, trajids=sel, context=500,
                             plotter=plotter, show=False)
        img = str(Path(tempfile.mkdtemp()) / "proptv.png")
        plotter.screenshot(img, window_size=(700, 500))
        return img
    mo.image(src=make3())
    return


@app.cell
def _(mo):
    mo.md("## 4. OpenLPT / pyOpenLPT (CSV → Zarr)")
    return


@app.cell
def _(Path, convert_openlpt_to_zarr, mo, np, pv, plot_trajectories_3d,
      tempfile):
    def make4():
        d = Path(tempfile.mkdtemp()) / "ConvergeTrack"
        d.mkdir()
        rng = np.random.default_rng(2)
        lines = ["TrackID,FrameID,WorldX,WorldY,WorldZ,cam0_x(col),cam0_y(row),cam1_x(col),cam1_y(row)"]
        for tid in range(3):
            x0, y0, z0 = rng.uniform(0, 10, 3)
            vx, vy, vz = rng.uniform(-0.3, 0.3, 3)
            for t in range(20):
                x = x0 + vx * t + 1.0 * np.sin(t * 0.4)
                y = y0 + vy * t + 1.0 * np.cos(t * 0.3)
                z = z0 + vz * t
                lines.append(f"{tid},{t},{x:.4f},{y:.4f},{z:.4f},{x+100:.1f},{y+400:.1f},{x+200:.1f},{y+500:.1f}")
        (d / "LongTrackActive_00001.csv").write_text("\n".join(lines) + "\n")
        zp = Path(tempfile.mkdtemp()) / "openlpt.zarr"
        convert_openlpt_to_zarr(d, zp)
        plotter = pv.Plotter(off_screen=True)
        plot_trajectories_3d(zp, context=500, plotter=plotter, show=False)
        img = str(Path(tempfile.mkdtemp()) / "openlpt.png")
        plotter.screenshot(img, window_size=(700, 500))
        return img
    mo.image(src=make4())
    return


@app.cell
def _(mo):
    mo.md("""
    ## All four → one pipeline

    Each store above is a standard `trajectories` Zarr group, so the same
    Eulerian / phase-averaging / ParaView export code runs on every one of them.
    """)
    return
