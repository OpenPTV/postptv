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
VALID_VAR = "valid"


def clean_field(a: np.ndarray) -> np.ndarray:
    """Zero-out NaN/+-Inf once, upfront (MATLAB VTR convention).

    ``np.nan_to_num`` defaults map +-Inf to +-1.8e308, which overflows the
    float32 VTK cast — so map all three to 0.0 explicitly. Used for gradient
    inputs and VTK export, never for the stored means themselves.
    """
    return np.nan_to_num(np.asarray(a), nan=0.0, posinf=0.0, neginf=0.0)


def fluid_mask(counts, threshold: float = 100):
    """Stricter-than-binning fluid mask: ``counts > threshold``.

    Mirrors the legacy MATLAB pipeline ``fluidMask = par_ave2>100``
    (note: stricter than the ``<50`` count mask used at binning time).
    Accepts an xr.DataArray or ndarray, returns the same kind of boolean mask.
    """
    if isinstance(counts, xr.DataArray):
        return counts > threshold
    return np.asarray(counts) > threshold


def qc_mask(pos, vel, min_speed=None, max_speed=None,
            finite_only: bool = True) -> np.ndarray:
    """Vectorized pre-binning QC mask (cheap checks first).

    Drops non-finite positions/velocities, then out-of-range speeds
    ``|v|``. Length/ROI gates need trajectory ids / polygons and live outside
    this hot path — see :func:`finite_difference_velocity` for recomputing
    velocities before calling this.
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    keep = np.ones(pos.shape[0], dtype=bool)
    if finite_only and pos.shape[0]:
        keep &= np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
    if (min_speed is not None or max_speed is not None) and pos.shape[0]:
        speed = np.linalg.norm(np.where(np.isfinite(vel), vel, np.nan), axis=1)
        if min_speed is not None:
            keep &= speed >= float(min_speed)
        if max_speed is not None:
            keep &= speed <= float(max_speed)
        keep &= np.isfinite(speed) if finite_only else True
    return keep


def finite_difference_velocity(pos, ids) -> np.ndarray:
    """Centered-FD interior, one-sided endpoints, NaN singletons.

    Centered finite-difference velocity on bare arrays: sort by
    ``(ids, time)`` first (caller) or pass already-grouped
    runs of equal ``ids``. ``pos`` is (n, 3), ``ids`` is (n,). Returns (n, 3)
    velocities in pos units per frame.
    """
    pos = np.asarray(pos, dtype=float)
    ids = np.asarray(ids)
    vel = np.full_like(pos, np.nan)
    if len(pos) == 0:
        return vel
    first = np.r_[True, ids[1:] != ids[:-1]]
    last = np.r_[ids[:-1] != ids[1:], True]
    singleton = first & last
    interior = ~(first | last)
    ii = np.flatnonzero(interior)
    vel[ii] = (pos[ii + 1] - pos[ii - 1]) / 2.0
    fi = np.flatnonzero(first & ~singleton)
    li = np.flatnonzero(last & ~singleton)
    vel[fi] = pos[fi + 1] - pos[fi]
    vel[li] = pos[li] - pos[li - 1]
    return vel


def _bin_counts_sums(pos, vel, ti, edges4, shape):
    """Single histogramdd pass accumulating counts + per-component sums."""
    sample = np.column_stack([pos, ti])
    counts = np.histogramdd(sample, bins=edges4)[0]
    sums = [np.histogramdd(sample, bins=edges4, weights=vel[:, d])[0]
            for d in range(3)]
    assert counts.shape == tuple(shape), (counts.shape, shape)
    return counts, sums


def _bin_counts_sums_ceil(pos, vel, ti, mins, deltas, shape):
    """MATLAB-``ceil`` voxel rule: ``idx = ceil((p - min) / d) - 1``.

    Exact port of the legacy MATLAB mean-field script's ceil binning rule
    (plus the in-range guard): particles exactly on an interior voxel edge
    fall in the LOWER voxel, whereas ``np.histogramdd`` (half-open bins)
    puts them in the UPPER one. Vectorized via raveled ``bincount`` — same
    speed class as the histogramdd path, bit-identical to the per-frame loop
    for identical input.
    """
    idx = [np.ceil((pos[:, d] - mins[d]) / deltas[d]).astype(np.int64) - 1
           for d in range(3)]
    ok = np.ones(pos.shape[0], dtype=bool)
    for d, n in enumerate(shape[:3]):
        ok &= (idx[d] >= 0) & (idx[d] < n)
    ok &= (ti >= 0) & (ti < shape[3])
    rav = ((idx[0][ok] * shape[1] + idx[1][ok]) * shape[2]
           + idx[2][ok]) * shape[3] + ti[ok]
    n = int(np.prod(shape))
    counts = np.bincount(rav, minlength=n).reshape(shape).astype(float)
    sums = [np.bincount(rav, weights=np.asarray(vel[ok, d], dtype=float),
                        minlength=n).reshape(shape) for d in range(3)]
    return counts, sums


def eulerian_grid(scene, grid_params, first, last, cycletime,
                  deltat=90, base_time=100000, min_count=50,
                  smoothing_sigma=None, *, fill_value=float("nan"),
                  qc=None, chunk_frames=None, add_valid=True, fin=None,
                  voxel_rule="half-open") -> xr.Dataset:
    """Bin Lagrangian particles onto a (x, y, z, phase) grid of mean velocities.

    Same math as batch_Lagrangian_to_Eulerian.eulerian_grid, but: reads the
    whole particles table in ONE call (scene.collect) instead of one HDF5
    query per frame, bins every particle at once with np.histogramdd, and
    returns a self-describing Dataset instead of writing HDF5.

    smoothing_sigma (in grid cells, scalar or per-axis (sx, sy, sz)) applies
    Gaussian kernel smoothing to the velocity sums AND the counts before the
    division (Shepard/kernel estimate); min_count then acts on smoothed counts.

    fill_value: value written into velocity voxels with ``counts < min_count``
    (default NaN — masked-out voxels never masquerade as zero velocity;
    pass ``0.0`` to restore the legacy zero-fill). ``par_ave2`` keeps the
    legacy convention (sub-threshold counts stored as 0) and a boolean
    ``valid`` variable (``counts >= min_count``) is added when
    ``add_valid=True``.

    qc: optional dict ``{"min_speed":.., "max_speed":.., "finite_only":..}``
    applying :func:`qc_mask` before binning. Default off (no cost when None).

    chunk_frames: optional int bounding frames per accumulation chunk. Time-
    sorted chunks are histogrammed separately and summed — identical math to
    the single pass, bounded memory for OOM-scale runs.

    fin: optional explicit phase-bin count. Default (None) keeps the legacy
    ``ceil(cycletime / zaman)``. Pass an explicit ``fin`` to reproduce a
    legacy chain where the phase grid has a fixed bin count and the trailing
    ``cycletime - fin*zaman`` frames fold into the LAST bin via the ``clip``
    below instead of opening an extra partial bin.

    voxel_rule: ``"half-open"`` (default, ``np.histogramdd`` semantics —
    particles exactly on an interior edge fall in the upper voxel) or
    ``"ceil"`` (legacy MATLAB mean-field convention — edge
    particles fall in the lower voxel). ``"ceil"`` also switches the domain
    prefilter to the strict exclusive box (both bounds exclusive), so
    the binned set matches the legacy inputs exactly.
    """
    zaman = deltat * 2 + 1
    fin = int(np.ceil(cycletime / zaman)) if fin is None else int(fin)
    edges = [np.linspace(grid_params[f"min_{d}"], grid_params[f"max_{d}"],
                         grid_params[f"step{d}"] + 1) for d in "xyz"]
    mids = [0.5 * (b[:-1] + b[1:]) for b in edges]
    use_ceil = voxel_rule == "ceil"
    if voxel_rule not in ("half-open", "ceil"):
        raise ValueError("voxel_rule must be 'half-open' or 'ceil'")

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

    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    vel = np.asarray(vel, dtype=float).reshape(-1, 3)
    time = np.asarray(time).ravel()

    if qc is not None:
        q = dict(qc)
        keep_qc = qc_mask(pos, vel,
                          min_speed=q.get("min_speed"),
                          max_speed=q.get("max_speed"),
                          finite_only=q.get("finite_only", True))
        pos, vel, time = pos[keep_qc], vel[keep_qc], time[keep_qc]

    mask = (time >= first) & (time <= last)
    if pos.shape[0]:
        for d, b in enumerate(edges):
            if use_ceil:
                mask &= (pos[:, d] > b[0]) & (pos[:, d] < b[-1])
            else:
                mask &= (pos[:, d] >= b[0]) & (pos[:, d] < b[-1])
    pos, vel, time = pos[mask], vel[mask], time[mask]

    # phase bin per particle: identical to the legacy per-frame arithmetic
    # ti = ceil((t - cycle_start)/zaman) - 1, reduced to integer ops
    if pos.shape[0]:
        ti = np.clip(((time.astype(np.int64) - base_time - 1) % int(cycletime))
                     // zaman, 0, fin - 1)
    else:
        ti = np.empty((0,), dtype=np.int64)

    edges4 = [*edges, np.arange(fin + 1) - 0.5]
    shape4 = (*[grid_params[f"step{d}"] for d in "xyz"], fin)
    if use_ceil:
        mins = [grid_params[f"min_{d}"] for d in "xyz"]
        deltas = [(grid_params[f"max_{d}"] - grid_params[f"min_{d}"])
                  / grid_params[f"step{d}"] for d in "xyz"]

        def _bin(p, v, t):
            return _bin_counts_sums_ceil(p, v, t, mins, deltas, shape4)
    else:
        def _bin(p, v, t):
            return _bin_counts_sums(p, v, t, edges4, shape4)
    if chunk_frames is None or pos.shape[0] == 0:
        counts, sums_list = _bin(pos, vel, ti)
    else:
        order = np.argsort(time, kind="mergesort")
        pos_s, vel_s, ti_s = pos[order], vel[order], ti[order]
        uniq = np.unique(time[order])
        counts = np.zeros(shape4)
        sums_list = [np.zeros(shape4) for _ in range(3)]
        for s in range(0, len(uniq), int(chunk_frames)):
            sel = np.isin(time[order], uniq[s:s + int(chunk_frames)])
            c, sl = _bin(pos_s[sel], vel_s[sel], ti_s[sel])
            counts += c
            for d in range(3):
                sums_list[d] += sl[d]
    sums = dict(zip(VEL_VARS, sums_list))

    if smoothing_sigma is not None:
        from scipy.ndimage import gaussian_filter

        sigma = np.broadcast_to(np.asarray(smoothing_sigma, dtype=float), (3,))
        sigma4 = (*sigma, 0.0)  # never smooth across phase bins
        counts = gaussian_filter(counts, sigma4)
        sums = {v: gaussian_filter(s, sigma4) for v, s in sums.items()}

    raw_counts = counts.copy()
    low = counts < min_count
    counts = np.where(low, 0, counts)
    fill = np.full_like(next(iter(sums.values())), float(fill_value))
    data = {}
    for v in VEL_VARS:
        s = np.where(low, 0.0, sums[v])
        mean = np.divide(s, counts, out=fill.copy(),
                         where=counts != 0)
        data[v] = (DIMS, mean)
    data[COUNT_VAR] = (DIMS, counts if smoothing_sigma is not None
                       else counts.astype(np.int64))
    if add_valid:
        data[VALID_VAR] = (DIMS, raw_counts >= min_count)
    return xr.Dataset(
        data,
        coords={"x": mids[0], "y": mids[1], "z": mids[2], "phase": np.arange(fin)},
        attrs={"first": first, "last": last, "cycletime": cycletime,
               "zaman": zaman, "fin": fin, "min_count": min_count,
               "voxel_rule": voxel_rule, "fill_value": float(fill_value)},
    )


def eulerian_windowed(scene, grid_params, first, last, window_frames: int,
                      step_frames: int, min_count: int = 50,
                      *, fill_value=float("nan"),
                      qc=None, add_valid=True) -> xr.Dataset:
    """Sliding-window Eulerian mean (non-periodic).

    Overlapping time windows replace periodic phase bins: window centers run
    from ``first + window//2`` to ``last - window//2`` in steps of
    ``step_frames`` (single center when the range is too short). Each window
    ``[center-w//2, center+w//2]`` is binned with one ``histogramdd`` pass
    over a ``searchsorted`` slice of the once-sorted table — no per-particle
    phase arithmetic. Output dims are (x, y, z, window) with a
    ``window_center`` coordinate; velocities are NaN where ``counts <
    min_count`` with the same ``par_ave2``/``valid`` convention as
    :func:`eulerian_grid`.
    """
    edges = [np.linspace(grid_params[f"min_{d}"], grid_params[f"max_{d}"],
                         grid_params[f"step{d}"] + 1) for d in "xyz"]
    mids = [0.5 * (b[:-1] + b[1:]) for b in edges]
    shape3 = tuple(grid_params[f"step{d}"] for d in "xyz")

    if hasattr(scene, "collect"):
        pos, vel, time = (np.asarray(a) for a in scene.collect(["pos", "velocity", "time"]))
        pos = np.asarray(pos, dtype=float).reshape(-1, 3)
        vel = np.asarray(vel, dtype=float).reshape(-1, 3)
        time = np.asarray(time).ravel()
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

    if qc is not None:
        q = dict(qc)
        keep_qc = qc_mask(pos, vel, min_speed=q.get("min_speed"),
                          max_speed=q.get("max_speed"),
                          finite_only=q.get("finite_only", True))
        pos, vel, time = pos[keep_qc], vel[keep_qc], time[keep_qc]

    order = np.argsort(time, kind="mergesort")
    pos_s, vel_s, time_s = pos[order], vel[order], time[order]
    w = int(window_frames)
    centers = list(range(int(first) + w // 2, int(last) - w // 2 + 1,
                         int(step_frames)))
    if not centers:
        centers = [(int(first) + int(last)) // 2]
    nwin = len(centers)
    counts = np.zeros((*shape3, nwin))
    sums = [np.zeros((*shape3, nwin)) for _ in range(3)]
    for k, c in enumerate(centers):
        lo, hi = c - w // 2, c + w // 2
        a = np.searchsorted(time_s, lo, side="left")
        b = np.searchsorted(time_s, hi, side="right")
        xyz = pos_s[a:b]
        vv = vel_s[a:b]
        keep = np.ones(len(xyz), dtype=bool)
        for d, e in enumerate(edges):
            keep &= (xyz[:, d] >= e[0]) & (xyz[:, d] < e[-1])
        xyz, vv = xyz[keep], vv[keep]
        if len(xyz) == 0:
            continue
        counts[..., k] = np.histogramdd(xyz, bins=edges)[0]
        for d in range(3):
            sums[d][..., k] = np.histogramdd(xyz, bins=edges,
                                             weights=vv[:, d])[0]
    low = counts < min_count
    counts_out = np.where(low, 0, counts)
    dims = ("x", "y", "z", "window")
    data = {}
    fill = float(fill_value)
    for d, v in enumerate(VEL_VARS):
        s = np.where(low, 0.0, sums[d])
        mean = np.divide(s, counts_out,
                         out=np.full_like(s, fill), where=counts_out != 0)
        data[v] = (dims, mean)
    data[COUNT_VAR] = (dims, counts_out.astype(np.int64))
    if add_valid:
        data[VALID_VAR] = (dims, counts >= min_count)
    return xr.Dataset(
        data,
        coords={"x": mids[0], "y": mids[1], "z": mids[2],
                "window": np.arange(nwin),
                "window_center": ("window", np.asarray(centers))},
        attrs={"first": first, "last": last, "window_frames": w,
               "step_frames": int(step_frames), "min_count": min_count},
    )


def shift_phase(ds: xr.Dataset, shift: int) -> xr.Dataset:
    """Cyclically roll all fields along phase to align with the cycle. Pure."""
    if shift == 0:
        return ds
    out = ds.roll(phase=shift, roll_coords=False)
    out.attrs["shift"] = shift
    return out


def phase_metric_curve(ds: xr.Dataset, var_names=VEL_VARS, rho: float = 1200.0) -> xr.DataArray:
    """Spatially-averaged MKE per phase, ignoring zero/NaN voxels.

    Averages only over voxels that actually have data (nanmean over nonzero
    MKE), not the whole grid, since sparse Eulerian bins are often mostly
    empty outside the region of interest.
    """
    u, v, w = (ds[n] for n in var_names)
    mke = 0.5 * rho * (u**2 + v**2 + w**2)
    return mke.where(mke != 0).mean(("x", "y", "z"), skipna=True)


def find_phase_shift(ds: xr.Dataset, reference_phase: int = 0,
                     var_names=VEL_VARS, rho: float = 1200.0) -> int:
    """Extract the phase shift (in bins) that rotates ds's peak-MKE phase to
    reference_phase.

    Replaces a manual "eyeball the MKE-vs-phase curve, pick the roll amount"
    step with a computed parameter, so multiple acquisition runs of the same
    periodic flow can each be aligned to the same point in the cycle before
    combining.
    """
    curve = phase_metric_curve(ds, var_names, rho)
    n = curve.sizes["phase"]
    peak = int(curve.argmax("phase"))
    return reference_phase - peak


def align_phases(sets: dict, reference_phase: int = 0,
                 var_names=VEL_VARS, rho: float = 1200.0) -> dict:
    """Shift each named acquisition run to a common reference phase.

    Returns {name: shifted_dataset}; each dataset also carries the extracted
    shift in .attrs["shift"] (set by shift_phase) for inspection/logging.
    """
    return {name: shift_phase(ds, find_phase_shift(ds, reference_phase, var_names, rho))
            for name, ds in sets.items()}


def region_timeseries(fields: dict, mask: xr.DataArray) -> xr.Dataset:
    """Per-phase mean, max and min of each named scalar field within a mask.

    General masked phase-statistics helper: pass whatever named fields
    (kinetic energy, vorticity magnitude, a vortex-identification scalar,
    ...) and whatever region mask (a subregion, a jet core, a
    near-wall shell, the whole valid-data volume, ...) the analysis needs.
    fields: {name: xr.DataArray}, each with dims including (x, y, z).
    """
    out = xr.Dataset()
    for name, da in fields.items():
        masked = da.where(mask)
        out[f"{name}_mean"] = masked.mean(("x", "y", "z"), skipna=True)
        out[f"{name}_max"] = masked.max(("x", "y", "z"), skipna=True)
        out[f"{name}_min"] = masked.min(("x", "y", "z"), skipna=True)
    return out


def y_slab_masks(fluid, y, start_idx=3, slab_length=0.05):
    """Inner/outer y-slab split of a fluid mask (legacy MATLAB convention).

    start_idx is 1-based (MATLAB convention): inner slab spans
    y0=y[start_idx-1] .. y0+slab_length, outer spans above it to the domain
    top. Broadcast over any trailing (z, phase) dims; returns
    (inner, outer) boolean masks.
    """
    y = np.asarray(y, dtype=float)
    fluid = np.asarray(fluid, dtype=bool)
    y0 = y[start_idx - 1]
    inner_y = np.where((y >= y0) & (y <= y0 + slab_length))[0]
    outer_y = np.where((y > y0 + slab_length) & (y <= y[-1]))[0]
    inner = np.zeros_like(fluid)
    outer = np.zeros_like(fluid)
    if inner_y.size:
        inner[:, inner_y, ...] = fluid[:, inner_y, ...]
    if outer_y.size:
        outer[:, outer_y, ...] = fluid[:, outer_y, ...]
    return inner.astype(bool), outer.astype(bool)


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


@_mask
def mask_array(ds: xr.Dataset, mask, zero_to_nan: bool = False) -> xr.Dataset:
    """NaN voxels outside an externally supplied (x, y, z) boolean mask.

    Stand-in for an interactive freehand ROI: hand-drawing a region of
    interest is a GUI step with no headless equivalent, so it isn't
    reproduced here. Draw it once, wherever, save the resulting boolean
    array, and pass it in.

    zero_to_nan: also map exact-zero means to NaN (legacy masking-script
    ``0 -> NaN`` semantics — a zero mean inside the mask is treated as "no
    data", preserving exact boundary shear semantics downstream).
    """
    m = xr.DataArray(np.asarray(mask, dtype=bool), dims=("x", "y", "z"))
    out = ds.assign({v: ds[v].where(m) for v in VEL_VARS if v in ds})
    if zero_to_nan:
        out = out.assign({v: out[v].where(out[v] != 0) for v in VEL_VARS
                          if v in out})
    return out


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

def reference_derived(u, v, w, dx, dy, dz, mask=None, rho=1000.0, mu=None,
                    counts=None, dt=1.0):
    """Per-phase reference derived fields with legacy MATLAB gradient semantics.

    Exact port of the legacy MATLAB derived-field scripts on
    bare numpy arrays (one 3D phase field at a time — loop over phases at the
    call site). Unlike :func:`derived_fields` (a port of the old
    ``sample_vtkcode.py`` demo with its sign convention and zero-filled
    gradients), this preserves the legacy quirks validated against Octave:

    - NaN-preserving gradients (no zero-fill): NaN propagates from any
      stencil voxel, matching the reference validity mask;
    - MATLAB ``gradient()`` dim2-first output order: numpy's first two
      outputs are swapped to replicate it exactly;
    - voxel-loop Q-criterion / lambda2 over the fluid mask
      (``lambda2`` = 2nd eigenvalue of ``S*S + Om*Om``),
      NaN wherever the gradient tensor is not finite (MATLAB ``eig(NaN)``).

    u, v, w: (nx, ny, nz) phase means; dx, dy, dz: grid spacings (m).
    mask: optional (nx, ny, nz) fluid boolean — outputs are NaN outside it,
      and Q/lambda2 are evaluated only inside it.
    rho, mu: density / dynamic viscosity (defaults: water at ~1000 kg/m^3).
    counts, dt: optional per-voxel sample counts + frame time — when given,
      ``prt = counts * dt`` is returned; else ``prt`` is NaN.

    Returns dict(vel, mke, vss, eps, vortmag, heli, q, l2, ens, prt) plus the
    masked ``u, v, w`` (masked velocity).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    if mu is None:
        mu = 4.85e-6 * rho
    m = np.ones(u.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    dd = np.gradient(u, dx, dy, dz, edge_order=1)
    dv = np.gradient(v, dx, dy, dz, edge_order=1)
    dw = np.gradient(w, dx, dy, dz, edge_order=1)
    du_dx, du_dy, du_dz = dd[1], dd[0], dd[2]
    dv_dx, dv_dy, dv_dz = dv[1], dv[0], dv[2]
    dw_dx, dw_dy, dw_dz = dw[1], dw[0], dw[2]
    exx, eyy, ezz = du_dx, dv_dy, dw_dz
    exy = 0.5 * (du_dy + dv_dx)
    exz = 0.5 * (du_dz + dw_dx)
    eyz = 0.5 * (dv_dz + dw_dy)
    dmag = np.sqrt(exx ** 2 + eyy ** 2 + ezz ** 2
                   + 2 * (exy ** 2 + exz ** 2 + eyz ** 2))
    wx, wy, wz = dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy
    vortmag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
    q = np.full_like(vortmag, np.nan)
    l2 = np.full_like(vortmag, np.nan)
    flat = (du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz)
    for ix, iy, iz in np.argwhere(m):
        Gu = np.array([[du_dx[ix, iy, iz], du_dy[ix, iy, iz], du_dz[ix, iy, iz]],
                       [dv_dx[ix, iy, iz], dv_dy[ix, iy, iz], dv_dz[ix, iy, iz]],
                       [dw_dx[ix, iy, iz], dw_dy[ix, iy, iz], dw_dz[ix, iy, iz]]])
        if not np.all(np.isfinite(Gu)):
            continue
        Sm, Om = 0.5 * (Gu + Gu.T), 0.5 * (Gu - Gu.T)
        q[ix, iy, iz] = 0.5 * (np.sum(Om ** 2) - np.sum(Sm ** 2))
        l2[ix, iy, iz] = np.sort(np.real(np.linalg.eigvals(Sm @ Sm + Om @ Om)))[1]

    vel = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    out = {
        "vel": vel,
        "mke": 0.5 * rho * (u ** 2 + v ** 2 + w ** 2),
        "vss": 2 * mu * dmag,
        "eps": 2 * mu * (exx ** 2 + eyy ** 2 + ezz ** 2
                         + 2 * (exy ** 2 + exz ** 2 + eyz ** 2)),
        "vortmag": vortmag,
        "heli": u * wx + v * wy + w * wz,
        "q": q,
        "l2": l2,
        "ens": 0.5 * vortmag ** 2,
        "prt": (np.asarray(counts, dtype=float) * dt
                if counts is not None else np.full_like(vortmag, np.nan)),
        "u": u.copy(),
        "v": v.copy(),
        "w": w.copy(),
    }
    for key, arr in out.items():
        arr[~m] = np.nan
    out["g9"] = flat  # unmasked gradient tuple (for Q/L2 reuse)
    return out


ALL_DERIVED = ["MKE", "TKE", "VEL", "PRT", "VSS", "RSS", "ML", "TL",
               "ScalarShear", "H1", "H2", "H3", "H4"]


def derived_fields(avg: xr.Dataset, stats: xr.Dataset, rho: float = 1.0,
                   mu: float = 1.0, fields: list[str] | None = None, *,
                   counts=None, fluid_min=None, clean_nan: bool = True,
                   gradient_convention: str = "matlab",
                   prt_mode: str = "velocity",
                   prt_dt: float = 1.0) -> xr.Dataset:
    """Derived turbulence/flow fields, vectorized over ALL phases.

    Port of sample_vtkcode.py main() (which loops per time slice).
    ``gradient_convention="matlab"`` (default) preserves the legacy quirks
    exactly: the u component is negated, np.gradient axis naming follows the
    Matlab meshgrid convention (axis0 spacing dy), and dx is set to the z
    spacing. ``"physical"`` instead uses ``np.gradient(u, dx, dy, dz)`` with
    no negation — the physically consistent choice (matches
    ``flowtracks.vortex``); expect ~tens-of-percent differences in
    gradient-based fields between conventions on anisotropic grids.

    clean_nan: zero-out NaN/+-Inf gradient inputs upfront (MATLAB VTR
    convention via :func:`clean_field`) so one masked voxel does not poison
    its neighbors' gradients; VEL/MKE/helicity still use the raw means.
    counts+fluid_min: optional stricter-than-binning fluid mask
    (``counts >= fluid_min``, e.g. 100 vs binning 50) applied as NaN to all
    outputs. prt_mode: ``"velocity"`` (legacy ``gridx*1000/|v|``) or
    ``"counts"`` (``counts*prt_dt``, needs counts).
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
        if gradient_convention not in ("matlab", "physical"):
            raise ValueError("gradient_convention must be 'matlab' or 'physical'")
        gridx = float(avg.x[1] - avg.x[0])
        dy = float(avg.y[1] - avg.y[0])
        dz = float(avg.z[1] - avg.z[0])
        if gradient_convention == "matlab":
            dx = dz  # legacy: sample_vtkcode.py:142
            gu, gv, gw = -u.values, v.values, w.values
            uy, ux, uz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                          np.gradient(clean_field(gu) if clean_nan else gu,
                                      dy, dx, dz, axis=(0, 1, 2)))
            vy, vx, vz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                          np.gradient(clean_field(gv) if clean_nan else gv,
                                      dy, dx, dz, axis=(0, 1, 2)))
            wy, wx, wz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                          np.gradient(clean_field(gw) if clean_nan else gw,
                                      dy, dx, dz, axis=(0, 1, 2)))
        else:
            dx = gridx
            gu, gv, gw = u.values, v.values, w.values
            ux, uy, uz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                          np.gradient(clean_field(gu) if clean_nan else gu,
                                      dx, dy, dz, axis=(0, 1, 2)))
            vx, vy, vz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                          np.gradient(clean_field(gv) if clean_nan else gv,
                                      dx, dy, dz, axis=(0, 1, 2)))
            wx, wy, wz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                          np.gradient(clean_field(gw) if clean_nan else gw,
                                      dx, dy, dz, axis=(0, 1, 2)))

    if "PRT" in want:
        if prt_mode == "velocity":
            out["PRT"] = xr.where(vel != 0, gridx * 1000.0 / vel, 0.0)
        elif prt_mode == "counts":
            if counts is None:
                raise ValueError("prt_mode='counts' requires counts=")
            out["PRT"] = counts.astype(float) * float(prt_dt)
        else:
            raise ValueError("prt_mode must be 'velocity' or 'counts'")
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
    if counts is not None and fluid_min is not None:
        keep = fluid_mask(counts, float(fluid_min))
        out = out.where(keep)
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

    Deprecated: prefer :func:`flowtracks.writers.write_eulerian_series`
    (pyvista-based, writes modern `.vti`/`.vtr` + a `.pvd` time series
    instead of one legacy `.vtk` per phase). Kept for `combine.py` and
    existing callers/tests until they migrate — see WRITERS_PLAN.md.
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


def export_vtk_rectilinear_series(ds: xr.Dataset, out_dir: Path,
                                  prefix: str = "phase", dt: float = 1.0) -> Path:
    """Write one binary .vtr (XML RectilinearGrid) per phase plus a .pvd
    time-series collection, ParaView's native pattern for an axis-aligned grid.

    A rectilinear grid stores only the 1-D x/y/z coordinate arrays (not a
    full 3-D point cloud like vtkStructuredGrid / export_vtk above), and a
    .pvd Collection is what lets ParaView step through phases as a time
    series instead of opening files one by one.
    Binary encoding here instead of the legacy ASCII per-point fprintf loop —
    same data, far smaller files, no behavior change.

    Deprecated: superseded by :func:`flowtracks.writers.write_eulerian_series`,
    which additionally auto-picks `.vti` when spacing is uniform. Kept only
    until remaining callers/tests migrate — see WRITERS_PLAN.md.
    """
    import vtk
    from vtk.util import numpy_support

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = ds.sizes["x"], ds.sizes["y"], ds.sizes["z"]

    def _clean(values):
        # nan=0 matches legacy NaN-masking; posinf/neginf=0 too (default
        # nan_to_num maps +-inf to +-1.8e308, which overflows the float32 cast)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def vtk_array(values):
        return numpy_support.numpy_to_vtk(
            np.ascontiguousarray(_clean(values).ravel(order="F"), dtype=np.float32),
            deep=True)

    files = []
    for p in ds["phase"].values:
        snap = ds.sel(phase=p)
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        grid.SetXCoordinates(numpy_support.numpy_to_vtk(
            np.ascontiguousarray(ds["x"].values, dtype=np.float32)))
        grid.SetYCoordinates(numpy_support.numpy_to_vtk(
            np.ascontiguousarray(ds["y"].values, dtype=np.float32)))
        grid.SetZCoordinates(numpy_support.numpy_to_vtk(
            np.ascontiguousarray(ds["z"].values, dtype=np.float32)))

        if all(v in snap for v in VEL_VARS):
            vec = np.stack([_clean(snap[v].values).ravel(order="F")
                            for v in VEL_VARS], axis=-1)
            arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(vec, dtype=np.float32), deep=True)
            arr.SetName("velocity")
            grid.GetPointData().AddArray(arr)
        for name, da in snap.data_vars.items():
            if name in VEL_VARS:
                continue
            arr = vtk_array(da.values)
            arr.SetName(name)
            grid.GetPointData().AddArray(arr)

        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetDataModeToBinary()
        path = out_dir / f"{prefix}_{int(p):04d}.vtr"
        writer.SetFileName(str(path))
        writer.SetInputData(grid)
        writer.Write()
        files.append(path)

    pvd_path = out_dir / f"{prefix}_series.pvd"
    _write_pvd(pvd_path, files, dt)
    return pvd_path


def _write_pvd(pvd_path: Path, files: list, dt: float) -> None:
    """Minimal ParaView Collection (.pvd) indexing time-series files by
    path relative to the .pvd itself, so the output directory stays portable
    (the legacy write_pvd_collection_abs used absolute Windows paths)."""
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']
    for i, f in enumerate(files):
        rel = Path(f).relative_to(pvd_path.parent).as_posix()
        lines.append(f'    <DataSet timestep="{i * dt:.9g}" group="" part="0" file="{rel}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    pvd_path.write_text("\n".join(lines) + "\n")


CF_ATTRS = {
    "u_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity x"},
    "v_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity y"},
    "w_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity z"},
    "par_ave2": {"units": "1", "long_name": "samples per voxel per phase"},
    "valid": {"units": "1", "long_name": "voxel meets min_count threshold"},
    "x": {"units": "m", "axis": "X"},
    "y": {"units": "m", "axis": "Y"},
    "z": {"units": "m", "axis": "Z"},
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


NETCDF_SUFFIXES = (".nc", ".nc4", ".netcdf")


def save_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Optional output: compressed netCDF with CF metadata (needs ``flowtracks[netcdf]``).

    Zarr (:func:`save_zarr`) holds the same data and metadata and is the default;
    NetCDF is kept for tools that only read it.
    """
    import importlib.util

    if importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None:
        raise ImportError(
            "NetCDF output needs a netCDF backend: pip install 'flowtracks[netcdf]'. "
            "Zarr output (save_zarr, or save_dataset with any non-.nc path) needs none.")
    ds = apply_cf_metadata(ds)
    encoding = {name: {"zlib": True, "complevel": 4}
                for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)


def save_zarr(ds: xr.Dataset, path: Path) -> None:
    """Default output: chunked Zarr dataset with CF metadata for ParaView and xarray."""
    ds = apply_cf_metadata(ds)
    ds.to_zarr(path, mode="w")


def save_dataset(ds: xr.Dataset, path: Path) -> None:
    """Save as NetCDF if the path ends in .nc/.nc4/.netcdf, otherwise as Zarr (the default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in NETCDF_SUFFIXES:
        save_netcdf(ds, path)
    else:
        save_zarr(ds, path)


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
                             fields=ALL_DERIVED if fields == "all" else fields,
                             counts=ds[COUNT_VAR].mean("set")
                             if COUNT_VAR in ds and "set" in ds.dims else None,
                             fluid_min=d_cfg.get("fluid_min"),
                             clean_nan=d_cfg.get("clean_nan", True),
                             gradient_convention=d_cfg.get(
                                 "gradient_convention", "matlab"),
                             prt_mode=d_cfg.get("prt_mode", "velocity"),
                             prt_dt=d_cfg.get("prt_dt", 1.0))

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
        from flowtracks.writers import write_eulerian_series

        pvd = write_eulerian_series(
            out,
            grid_dir / vtk_cfg.get("dir", "vtk_output"),
            prefix=vtk_cfg.get("prefix", "phase"),
        )
        print(f"Wrote Eulerian ParaView series to {pvd}")

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

