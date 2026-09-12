"""Unified pyvista-based ParaView writers.

Lagrangian trajectories -> VTK PolyData (``.vtp``, polylines per ``trajid``).
Eulerian grids -> VTK ImageData (``.vti``, uniform axis spacing) or
RectilinearGrid (``.vtr``, non-uniform), each written as a per-timestep
series plus one ``.pvd`` Collection.

See ``WRITERS_PLAN.md`` for the design rationale and the writers this module
replaces (``vtk_export.py``, ``eulerian.export_vtk``,
``eulerian.export_vtk_rectilinear_series``).
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import xarray as xr

VEL_VARS = ["u_ins_mean", "v_ins_mean", "w_ins_mean"]


def write_pvd(path: Path, entries: list[tuple[float, Path | str]]) -> Path:
    """Write a ParaView Collection (.pvd) indexing time-series files.

    ``entries`` is a list of ``(timestep, file_path)``. Paths are stored
    relative to ``path`` so the output directory stays portable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, file_path in entries:
        rel = Path(file_path)
        if rel.is_absolute():
            rel = rel.relative_to(path.parent)
        lines.append(
            f'    <DataSet timestep="{timestep:.9g}" group="" part="0" '
            f'file="{escape(rel.as_posix())}"/>'
        )
    lines += ["  </Collection>", "</VTKFile>"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _trajectory_arrays(source):
    """Normalize a zarr path / ZarrScene / Scene into (pos, vel, time, trajid)."""
    if isinstance(source, (str, Path)):
        import zarr

        group = zarr.open_group(str(source), mode="r")
        target = group["trajectories"] if "trajectories" in group else group
        pos = np.asarray(target["pos"], dtype=np.float64)
        vel = np.asarray(target["vel"], dtype=np.float64)
        time = np.asarray(target["time"])
        trajid = np.asarray(target["trajid"])
        return pos, vel, time, trajid
    if hasattr(source, "collect"):
        pos, vel, time, trajid = (
            np.asarray(a) for a in source.collect(["pos", "velocity", "time", "trajid"])
        )
        return pos.astype(np.float64), vel.astype(np.float64), time, trajid
    raise TypeError(f"Unsupported trajectory source: {type(source)!r}")


def write_trajectories_vtp(source, out_path: Path) -> Path:
    """Write Lagrangian trajectories as VTK PolyData (points + polylines per trajid).

    ``source`` is a zarr path holding a ``trajectories/`` group with
    ``pos``/``vel``/``time``/``trajid`` arrays (a flowtracks Zarr export or an
    openptv2 RunStore -- no conversion needed, see ``ZarrScene``), or any
    object exposing ``.collect(["pos", "velocity", "time", "trajid"])``
    (``ZarrScene``/``Scene``).
    """
    import pyvista as pv

    pos, vel, time, trajid = _trajectory_arrays(source)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    poly = pv.PolyData()
    if len(trajid) == 0:
        poly.save(str(out_path))
        return out_path

    order = np.lexsort((time, trajid))
    pos, vel, time, trajid = pos[order], vel[order], time[order], trajid[order]
    poly.points = pos

    starts = np.flatnonzero(np.r_[True, trajid[1:] != trajid[:-1]])
    ends = np.r_[starts[1:], len(trajid)]
    segments = [(s, e) for s, e in zip(starts, ends) if e - s > 1]
    if segments:
        poly.lines = np.concatenate(
            [np.r_[e - s, np.arange(s, e, dtype=np.int64)] for s, e in segments]
        ).astype(np.int64)

    poly.point_data["trajid"] = trajid
    poly.point_data["time"] = time
    poly.point_data["velocity"] = vel
    poly.point_data["speed"] = np.linalg.norm(vel, axis=1)
    poly.save(str(out_path))
    return out_path


def _is_uniform(coord: np.ndarray, rtol: float = 1e-6) -> bool:
    if coord.size < 3:
        return True
    diffs = np.diff(coord)
    return bool(np.allclose(diffs, diffs[0], rtol=rtol, atol=1e-12))


def _clean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def write_eulerian_series(ds: xr.Dataset, out_dir: Path, prefix: str = "phase",
                          time_dim: str = "phase", dt: float = 1.0) -> Path:
    """Write one Eulerian grid file per ``time_dim`` step plus a ``.pvd`` series.

    Grid type is chosen from axis spacing: ``vtkImageData`` (``.vti``) when x,
    y and z are each uniformly spaced, ``vtkRectilinearGrid`` (``.vtr``)
    otherwise. Vector fields named by ``VEL_VARS`` are combined into a single
    "velocity" array; every other data variable is written as a scalar.
    """
    import pyvista as pv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, z = (np.asarray(ds[d].values, dtype=np.float64) for d in "xyz")
    uniform = (
        len(x) > 1 and len(y) > 1 and len(z) > 1
        and _is_uniform(x) and _is_uniform(y) and _is_uniform(z)
    )
    dims = (len(x), len(y), len(z))
    ext = "vti" if uniform else "vtr"

    def build_grid():
        if uniform:
            spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
            return pv.ImageData(dimensions=dims, spacing=spacing, origin=(x[0], y[0], z[0]))
        return pv.RectilinearGrid(x, y, z)

    grid_dims = {"x", "y", "z"}

    entries = []
    for i, t in enumerate(ds[time_dim].values):
        snap = ds.sel(**{time_dim: t})
        grid = build_grid()
        if all(v in snap and set(snap[v].dims) == grid_dims for v in VEL_VARS):
            vec = np.stack(
                [_clean(snap[v].values).ravel(order="F") for v in VEL_VARS], axis=-1
            )
            grid.point_data["velocity"] = vec
        for name, da in snap.data_vars.items():
            if name in VEL_VARS or set(da.dims) != grid_dims:
                # Fields carrying an extra dim (e.g. per-set fluctuations that
                # weren't reduced yet) don't fit one grid point value each --
                # skip rather than silently reshape/truncate them.
                continue
            grid.point_data[name] = _clean(da.values).ravel(order="F")
        path = out_dir / f"{prefix}_{i:04d}.{ext}"
        grid.save(str(path))
        entries.append((i * dt, path))

    return write_pvd(out_dir / f"{prefix}_series.pvd", entries)


def export_run_to_paraview(zarr_path: Path, out_dir: Path,
                           grid_params: dict | None = None,
                           eulerian_ds: xr.Dataset | None = None,
                           **eulerian_grid_kwargs) -> dict[str, Path]:
    """Convenience wrapper: write trajectories, and optionally an Eulerian
    grid series, for one run into ``out_dir``.

    Pass ``eulerian_ds`` directly if you already built it (e.g. via
    ``eulerian_grid()`` or ``xr.open_zarr``); pass ``grid_params`` (plus
    ``eulerian_grid``'s other required kwargs: ``first``, ``last``,
    ``cycletime``, ...) to have this function build it from ``zarr_path``.
    """
    out_dir = Path(out_dir)
    result = {"trajectories": write_trajectories_vtp(zarr_path, out_dir / "trajectories.vtp")}

    if eulerian_ds is not None:
        result["eulerian"] = write_eulerian_series(eulerian_ds, out_dir / "eulerian")
    elif grid_params is not None:
        from flowtracks.eulerian import eulerian_grid
        from flowtracks.zarr_scene import ZarrScene

        scene = ZarrScene(zarr_path)
        ds = eulerian_grid(scene, grid_params, **eulerian_grid_kwargs)
        result["eulerian"] = write_eulerian_series(ds, out_dir / "eulerian")

    return result
