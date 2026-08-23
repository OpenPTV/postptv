"""Post-analysis pipeline as pure Dataset -> Dataset stages, driven by one YAML recipe.

xarray re-expression of batch_Lagrangian_to_Eulerian.py, turbulent_statistics.py
and the derived-field part of sample_vtkcode.py, chained after phase_average_xr.

Run:  uv run python src/post_analysis_xr.py [post_recipe.yaml]
"""

from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from flowtracks.phase_average import DIMS, fluctuations, open_sets, phase_average

VEL_VARS = ["u_ins_mean", "v_ins_mean", "w_ins_mean"]
COUNT_VAR = "par_ave2"


def eulerian_grid(scene, grid_params, first, last, cycletime,
                  deltat=90, base_time=100000, min_count=50,
                  smoothing_sigma=None) -> xr.Dataset:
    """Bin Lagrangian particles onto a (x, y, z, phase) grid of mean velocities.

    Same math as batch_Lagrangian_to_Eulerian.eulerian_grid, but: reads the
    whole particles table in ONE call (scene.collect) instead of one HDF5
    query per frame, bins every particle at once with np.histogramdd, and
    returns a self-describing Dataset instead of writing HDF5.

    smoothing_sigma (in grid cells, scalar or per-axis (sx, sy, sz)) applies
    Gaussian kernel smoothing to the velocity sums AND the counts before the
    division (Shepard/kernel estimate); min_count then acts on smoothed counts.
    """
    zaman = deltat * 2 + 1
    fin = int(np.ceil(cycletime / zaman))
    edges = [np.linspace(grid_params[f"min_{d}"], grid_params[f"max_{d}"],
                         grid_params[f"step{d}"] + 1) for d in "xyz"]
    mids = [0.5 * (b[:-1] + b[1:]) for b in edges]

    if hasattr(scene, "collect"):
        pos, vel, time = (np.asarray(a) for a in scene.collect(["pos", "velocity", "time"]))
    else:
        from flowtracks.io import trajectories
        trajs = trajectories(str(scene))
        p_list, v_list, t_list = [], [], []
        for tr in trajs:
            p_list.append(tr.pos())
            v_list.append(tr.velocity())
            t_list.append(tr.time())
        pos = np.vstack(p_list) if p_list else np.empty((0, 3))
        vel = np.vstack(v_list) if v_list else np.empty((0, 3))
        time = np.concatenate(t_list) if t_list else np.empty((0,))

    mask = (time >= first) & (time <= last)
    for d, b in enumerate(edges):
        mask &= (pos[:, d] >= b[0]) & (pos[:, d] < b[-1])
    pos, vel, time = pos[mask], vel[mask], time[mask]

    # phase bin per particle: identical to the legacy per-frame arithmetic
    # ti = ceil((t - cycle_start)/zaman) - 1, reduced to integer ops
    ti = np.clip(((time.astype(np.int64) - base_time - 1) % int(cycletime))
                 // zaman, 0, fin - 1)

    sample = np.column_stack([pos, ti])
    edges4 = [*edges, np.arange(fin + 1) - 0.5]
    counts = np.histogramdd(sample, bins=edges4)[0]
    sums = {v: np.histogramdd(sample, bins=edges4, weights=vel[:, d])[0]
            for d, v in enumerate(VEL_VARS)}

    if smoothing_sigma is not None:
        from scipy.ndimage import gaussian_filter

        sigma = np.broadcast_to(np.asarray(smoothing_sigma, dtype=float), (3,))
        sigma4 = (*sigma, 0.0)  # never smooth across phase bins
        counts = gaussian_filter(counts, sigma4)
        sums = {v: gaussian_filter(s, sigma4) for v, s in sums.items()}

    low = counts < min_count
    counts = np.where(low, 0, counts)
    data = {}
    for v in VEL_VARS:
        s = np.where(low, 0.0, sums[v])
        data[v] = (DIMS, np.divide(s, counts, out=np.zeros_like(s),
                                   where=counts != 0))
    data[COUNT_VAR] = (DIMS, counts if smoothing_sigma is not None
                       else counts.astype(np.int64))
    return xr.Dataset(
        data,
        coords={"x": mids[0], "y": mids[1], "z": mids[2], "phase": np.arange(fin)},
        attrs={"first": first, "last": last, "cycletime": cycletime,
               "zaman": zaman, "min_count": min_count},
    )


def shift_phase(ds: xr.Dataset, shift: int) -> xr.Dataset:
    """Cyclically roll all fields along phase to align with the cycle. Pure."""
    if shift == 0:
        return ds
    out = ds.roll(phase=shift, roll_coords=False)
    out.attrs["shift"] = shift
    return out


def turbulent_statistics(fluct: xr.Dataset, counts: xr.DataArray) -> xr.Dataset:
    """Count-weighted second moments of fluctuations across sets.

    fluct has vars u_fluct/v_fluct/w_fluct with a 'set' dim; counts is the
    per-set sample count. Output names match turbulent_statistics.py.
    """
    n = counts.sum("set").astype(float)
    n = n.where(n > 0)  # 0-count voxels -> NaN, as in the legacy script
    out = xr.Dataset(attrs=fluct.attrs)
    for i, a in enumerate("uvw"):
        for b in "uvw"[i:]:
            out[f"{a}_ins_{b}_ins"] = (fluct[f"{a}_fluct"] * fluct[f"{b}_fluct"]
                                       * counts).sum("set") / n
    for a in "uvw":
        out[f"{a}_rms"] = np.sqrt(out[f"{a}_ins_{a}_ins"])
    return out


# --- composable vector-validation masks (pyorc-style) -----------------------

MASKS = {}


def _mask(fn):
    MASKS[fn.__name__.removeprefix("mask_")] = fn
    return fn


@_mask
def mask_count(ds: xr.Dataset, min_count: int = 50) -> xr.Dataset:
    """NaN velocity voxels with fewer than min_count samples."""
    keep = ds[COUNT_VAR] >= min_count
    return ds.assign({v: ds[v].where(keep) for v in VEL_VARS if v in ds})


@_mask
def mask_outliers(ds: xr.Dataset, k: float = 3.0, window: int = 3) -> xr.Dataset:
    """NaN vectors deviating > k std from their spatial neighborhood mean."""
    out = {}
    for v in VEL_VARS:
        if v not in ds:
            continue
        r = ds[v].rolling(x=window, y=window, z=window, center=True,
                          min_periods=1)
        out[v] = ds[v].where(np.abs(ds[v] - r.mean()) <= k * r.std())
    return ds.assign(out)


@_mask
def mask_variance(ds: xr.Dataset, k: float = 3.0) -> xr.Dataset:
    """NaN vectors deviating > k std from the domain mean, per phase."""
    out = {}
    for v in VEL_VARS:
        if v not in ds:
            continue
        m = ds[v].mean(("x", "y", "z"))
        s = ds[v].std(("x", "y", "z"))
        out[v] = ds[v].where(np.abs(ds[v] - m) <= k * s)
    return ds.assign(out)


def apply_masks(ds: xr.Dataset, specs: list) -> xr.Dataset:
    """Apply recipe mask specs in order: ["count", {"method": "outliers", "k": 2}]."""
    for spec in specs:
        if isinstance(spec, str):
            name, kwargs = spec, {}
        else:
            spec = dict(spec)
            name, kwargs = spec.pop("method"), spec
        ds = MASKS[name](ds, **kwargs)
    return ds


# --- derived fields ----------------------------------------------------------

ALL_DERIVED = ["MKE", "TKE", "VEL", "PRT", "VSS", "RSS", "ML", "TL",
               "ScalarShear", "H1", "H2", "H3", "H4"]


def derived_fields(avg: xr.Dataset, stats: xr.Dataset, rho: float = 1.0,
                   mu: float = 1.0, fields: list[str] | None = None) -> xr.Dataset:
    """Derived hemodynamic/turbulence fields, vectorized over ALL phases.

    Port of sample_vtkcode.py main() (which loops per time slice). Formulas —
    including the legacy gradient conventions — are preserved exactly:
    the u component is negated, np.gradient axis naming follows the Matlab
    meshgrid convention (axis0 spacing dy), and dx is set to the z spacing.
    # ponytail: those three quirks look like Matlab-port bugs; kept verbatim
    # for parity with sample_vtkcode — revisit with a fluid dynamicist.
    """
    want = set(fields or ["MKE", "TKE"])
    unknown = want - set(ALL_DERIVED)
    if unknown:
        raise ValueError(f"Unknown derived fields: {sorted(unknown)}")

    u, v, w = (avg[name] for name in VEL_VARS)
    uu, vv, ww = (stats[f"{a}_ins_{a}_ins"] for a in "uvw")
    if want & {"RSS", "TL", "ScalarShear"}:  # cross-moments only when needed
        uv, uw, vw = (stats["u_ins_v_ins"], stats["u_ins_w_ins"],
                      stats["v_ins_w_ins"])

    out = xr.Dataset()
    vel = np.sqrt(u**2 + v**2 + w**2)
    if "MKE" in want:
        out["MKE"] = 0.5 * rho * (u**2 + v**2 + w**2)
    if "TKE" in want:
        out["TKE"] = 0.5 * rho * (uu + vv + ww)
    if "VEL" in want:
        out["VEL"] = vel

    grad_needed = want & {"PRT", "VSS", "ML", "TL", "ScalarShear",
                          "H1", "H2", "H3", "H4"}
    if grad_needed:
        gridx = float(avg.x[1] - avg.x[0])
        dy = float(avg.y[1] - avg.y[0])
        dz = float(avg.z[1] - avg.z[0])
        dx = dz  # legacy: sample_vtkcode.py:142
        uy, ux, uz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                      np.gradient(-u.values, dy, dx, dz, axis=(0, 1, 2)))
        vy, vx, vz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                      np.gradient(v.values, dy, dx, dz, axis=(0, 1, 2)))
        wy, wx, wz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                      np.gradient(w.values, dy, dx, dz, axis=(0, 1, 2)))

    if "PRT" in want:
        out["PRT"] = xr.where(vel != 0, gridx * 1000.0 / vel, 0.0)
    if "VSS" in want:
        p2, p3, p6 = uy + vx, uz + wx, vz + wy
        p1 = 2 * ux - (2 / 3) * (ux + uy + uz)
        p5 = 2 * vy - (2 / 3) * (vx + vy + vz)
        p9 = 2 * wz - (2 / 3) * (wx + wy + wz)
        out["VSS"] = _eig_spread(p1, p2, p3, p2, p5, p6, p3, p6, p9) * rho * mu
    if "RSS" in want:
        out["RSS"] = _eig_spread(uu, uv, uw, uv, vv, vw, uw, vw, ww) * rho
    if "ML" in want:
        meandiss = (2 * (uy + vx)**2 + 2 * (uz + wx)**2 + 2 * (vz + wy)**2
                    + (2 * ux - (2 / 3) * (ux + uy + uz))**2
                    + (2 * vy - (2 / 3) * (vx + vy + vz))**2
                    + (2 * wz - (2 / 3) * (wx + wy + wz))**2)
        out["ML"] = meandiss * mu * rho
    if "TL" in want:
        prod = (-uu * ux - vv * vy - ww * wz - uv * uy - uw * uz - vw * vz
                - uv * vx - uw * wx - vw * wy)
        out["TL"] = prod * rho
    if "ScalarShear" in want:
        t11 = rho * mu * (ux + ux) - rho * uu
        t22 = rho * mu * (vy + vy) - rho * vv
        t33 = rho * mu * (wz + wz) - rho * ww
        t12 = rho * mu * (uy + vx) - rho * uv
        t13 = rho * mu * (uz + wx) - rho * uw
        t23 = rho * mu * (vz + wy) - rho * vw
        out["ScalarShear"] = (1 / np.sqrt(3)) * np.sqrt(
            (t11**2 + t22**2 + t33**2)
            - (t11 * t22 + t22 * t33 + t11 * t33)
            + 3 * (t12**2 + t23**2 + t13**2))
    if want & {"H1", "H2", "H3", "H4"}:
        w1, w2, w3 = wy - vz, uz - wx, vx - uy
        h1 = w1 * u + w2 * v + w3 * w
        h2 = np.sqrt((w1 * u)**2 + (w2 * v)**2 + (w3 * w)**2)
        if "H1" in want:
            out["H1"] = h1
        if "H2" in want:
            out["H2"] = h2
        if "H3" in want:
            out["H3"] = xr.where(h2 != 0, h1 / h2, 0.0)
        if "H4" in want:
            out["H4"] = xr.where(h2 != 0, np.abs(h1) / h2, 0.0)
    return out


def _eig_spread(*components):
    """0.5*(max-min eigenvalue) of a symmetric 3x3 tensor field, all phases."""
    tensor = xr.concat(
        [xr.concat(components[i * 3:(i + 1) * 3], dim="j") for i in range(3)],
        dim="i",
    ).transpose(..., "i", "j")
    evals = xr.apply_ufunc(
        np.linalg.eigvalsh, tensor.fillna(0.0),
        input_core_dims=[["i", "j"]], output_core_dims=[["e"]],
    )
    return 0.5 * (evals.max("e") - evals.min("e"))


def export_vtk(ds: xr.Dataset, out_dir: Path, prefix: str = "phase") -> list[Path]:
    """Write one BINARY .vtk structured-grid file per phase from a Dataset.

    Bulk numpy_support arrays instead of the legacy per-point Python loop,
    and binary instead of ASCII — smaller files, much faster.
    """
    import vtk
    from vtk.util import numpy_support

    x3, y3, z3 = np.meshgrid(ds["x"], ds["y"], ds["z"], indexing="ij")
    nx, ny, nz = x3.shape
    # VTK structured grids expect x varying fastest -> Fortran-order flatten
    pts = np.column_stack([a.ravel(order="F") for a in (x3, y3, z3)])
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(pts, dtype=np.float32), deep=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in ds["phase"].values:
        snap = ds.sel(phase=p)
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        grid.SetPoints(points)
        if all(v in snap for v in VEL_VARS):
            vec = np.stack(
                [np.nan_to_num(snap[v].values).ravel(order="F")
                 for v in VEL_VARS], axis=-1)
            arr = numpy_support.numpy_to_vtk(
                np.ascontiguousarray(vec, dtype=np.float32), deep=True)
            arr.SetName("velocity")
            grid.GetPointData().AddArray(arr)
        for name, da in snap.data_vars.items():
            if name in VEL_VARS:
                continue
            arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(
                np.nan_to_num(da.values).ravel(order="F"), dtype=np.float32),
                deep=True)
            arr.SetName(name)
            grid.GetPointData().AddArray(arr)
        writer = vtk.vtkStructuredGridWriter()
        writer.SetFileTypeToBinary()
        path = out_dir / f"{prefix}_{int(p):03d}.vtk"
        writer.SetFileName(str(path))
        writer.SetInputData(grid)
        writer.Write()
        written.append(path)
    return written


CF_ATTRS = {
    "u_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity x", "standard_name": "eastward_sea_water_velocity"},
    "v_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity y", "standard_name": "northward_sea_water_velocity"},
    "w_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity z", "standard_name": "upward_sea_water_velocity"},
    "par_ave2": {"units": "1", "long_name": "samples per voxel per phase"},
    "x": {"units": "m", "axis": "X", "standard_name": "projection_x_coordinate"},
    "y": {"units": "m", "axis": "Y", "standard_name": "projection_y_coordinate"},
    "z": {"units": "m", "axis": "Z", "standard_name": "height_above_mean_sea_level"},
    "phase": {"units": "1", "axis": "T", "long_name": "phase index"},
    "MKE": {"units": "J m-3", "long_name": "mean kinetic energy"},
    "TKE": {"units": "J m-3", "long_name": "turbulent kinetic energy"},
    "VEL": {"units": "m s-1", "long_name": "mean velocity magnitude"},
    "PRT": {"units": "s", "long_name": "particle residence time"},
    "VSS": {"units": "Pa", "long_name": "viscous shear stress"},
    "RSS": {"units": "Pa", "long_name": "reynolds shear stress"},
    "ML": {"units": "W m-3", "long_name": "viscous dissipation rate"},
    "TL": {"units": "W m-3", "long_name": "turbulent production rate"},
    "ScalarShear": {"units": "Pa", "long_name": "equivalent scalar shear stress"},
    "H1": {"units": "m s-2", "long_name": "helicity density"},
    "H2": {"units": "m s-2", "long_name": "helicity magnitude"},
    "H3": {"units": "1", "long_name": "normalized helicity (relative)"},
    "H4": {"units": "1", "long_name": "absolute normalized helicity"},
    "u_rms": {"units": "m s-1", "long_name": "rms velocity fluctuation x"},
    "v_rms": {"units": "m s-1", "long_name": "rms velocity fluctuation y"},
    "w_rms": {"units": "m s-1", "long_name": "rms velocity fluctuation z"},
}


def apply_cf_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Attach rich CF metadata for ParaView and NetCDF/Zarr readers."""
    for name, attrs in CF_ATTRS.items():
        if name in ds or name in ds.coords:
            ds[name].attrs.update(attrs)
    ds.attrs.setdefault("Conventions", "CF-1.8")
    return ds


def save_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Canonical output: compressed netCDF with CF metadata for ParaView."""
    ds = apply_cf_metadata(ds)
    encoding = {name: {"zlib": True, "complevel": 4}
                for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)


def save_zarr(ds: xr.Dataset, path: Path) -> None:
    """Cloud-native output: chunked Zarr dataset with CF metadata for ParaView."""
    ds = apply_cf_metadata(ds)
    ds.to_zarr(path, mode="w")


def save_dataset(ds: xr.Dataset, path: Path) -> None:
    """Save Dataset as Zarr if path ends with .zarr, otherwise NetCDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zarr" or path.name.endswith(".zarr"):
        save_zarr(ds, path)
    else:
        save_netcdf(ds, path)


def run_post_analysis_ds(ds_sets: dict[str, xr.Dataset], recipe: dict) -> xr.Dataset:
    """In-memory post-analysis directly on set Datasets without intermediate disk writes."""
    sets = recipe["sets"]
    stacked = xr.concat([ds_sets[s] for s in sets],
                        dim=xr.DataArray(sets, dims="set", name="set"),
                        join="exact")

    ds = shift_phase(stacked, recipe.get("shift", 0))
    ds = apply_masks(ds, recipe.get("mask", []))

    avg_cfg = recipe.get("average", {})
    weights = ds[COUNT_VAR] if avg_cfg.get("weighting") == "counts" else None
    avg = phase_average(ds[VEL_VARS], weights=weights)
    fluct = fluctuations(ds[VEL_VARS], avg).rename(
        {v: f"{v.removesuffix('_ins_mean')}_fluct" for v in VEL_VARS})
    stats = turbulent_statistics(fluct, ds[COUNT_VAR])

    d_cfg = recipe.get("derived", {})
    fields = d_cfg.get("fields", ["MKE", "TKE"])
    derived = derived_fields(avg, stats,
                             rho=d_cfg.get("rho", 1.0), mu=d_cfg.get("mu", 1.0),
                             fields=ALL_DERIVED if fields == "all" else fields)

    out = xr.merge([avg, fluct, stats, derived], combine_attrs="override")
    out.attrs = ds.attrs
    return out


def run(recipe_path: Path) -> Path:
    recipe = yaml.safe_load(recipe_path.read_text())
    grid_dir = recipe_path.parent / recipe.get("grid_dir", ".")

    ds = open_sets(grid_dir, recipe["sets"], VEL_VARS + [COUNT_VAR])
    ds_dict = {s: ds.sel(set=s) for s in recipe["sets"]}
    out = run_post_analysis_ds(ds_dict, recipe)

    out_path = grid_dir / recipe["output"]
    save_dataset(out, out_path)
    print(f"Saved {len(out.data_vars)} variables to {out_path}")

    vtk_cfg = recipe.get("vtk")
    if vtk_cfg:
        files = export_vtk(
            out,
            grid_dir / vtk_cfg.get("dir", "vtk_output"),
            prefix=vtk_cfg.get("prefix", "phase"),
        )
        print(f"Wrote {len(files)} VTK files to {files[0].parent}")

    scratch = recipe_path.parent / recipe.get("scratch_dir", "../scratch")
    if recipe.get("cleanup_scratch") and scratch.is_dir():
        import shutil

        for item in scratch.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        print(f"Cleaned scratch: {scratch}")
    return out_path


def shift_fields(h5_path, dataset_names, shift, attr="shift"):
    """Cyclically roll the given datasets along the last (time) axis by
    ``shift`` frames, in place.

    Idempotent: if the file already carries a non-zero ``shift`` attribute
    it is left untouched, so calling this more than once in the same pipeline
    never shifts the data twice.
    """
    import h5py
    import warnings

    if shift == 0:
        return
    with h5py.File(h5_path, "a") as f:
        if f.attrs.get(attr, 0) != 0:
            warnings.warn(
                f"File {h5_path} already shifted by {f.attrs[attr]}, skipping."
            )
            return
        for name in dataset_names:
            if name not in f:
                continue
            data = f[name][()]
            rolled = np.roll(data, shift, axis=-1)
            del f[name]
            f.create_dataset(name, data=rolled)
        f.attrs[attr] = shift


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1] if len(sys.argv) > 1 else "post_recipe.yaml"))

