# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""Post-analysis pipeline as pure Dataset -> Dataset stages, driven by one YAML recipe.

xarray re-expression of batch_Lagrangian_to_Eulerian.py, turbulent_statistics.py
and per-slice derived turbulence fields, chained after phase_average_xr.

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


# --- per-voxel moments (opt-in record for exact pooling) ---------------------
#
# ``eulerian_grid(..., moments=True)`` records, per voxel and phase bin, the
# raw sample count, the number of distinct trajectories, the mean (first
# moment, ``*_m1``) and the CENTRED second moments (sum of products of
# deviations from that mean, ``*_m2``). Centred, not raw sums of squares:
# ``sum(u**2)/N - mean**2`` cancels catastrophically when the turbulence
# intensity is low, whereas centred moments combine exactly across chunks and
# realizations (Chan et al. parallel update) without that loss. These arrays
# are never masked: they are the record every statistic is derived from
# (:func:`combine_moments`, :func:`moment_statistics`).

N_VAR = "n_samples"
NTRAJ_VAR = "n_traj"
M1_VARS = ["u_m1", "v_m1", "w_m1"]
PAIRS = [("u", "u"), ("v", "v"), ("w", "w"), ("u", "v"), ("u", "w"), ("v", "w")]
M2_VARS = [f"{a}{b}_m2" for a, b in PAIRS]
MOMENT_VARS = [N_VAR, NTRAJ_VAR, *M1_VARS, *M2_VARS]
_PAIR_IDX = [("uvw".index(a), "uvw".index(b)) for a, b in PAIRS]


def _voxel_index(pos, ti, edges, shape, use_ceil, mins=None, deltas=None):
    """Raveled (x, y, z, phase) index per sample + in-range mask.

    Reproduces the binning rule of :func:`eulerian_grid` exactly:
    ``"half-open"`` is ``np.histogramdd``'s ``searchsorted(side="right")``
    rule, ``"ceil"`` is :func:`_bin_counts_sums_ceil`'s.
    """
    if use_ceil:
        idx = [np.ceil((pos[:, d] - mins[d]) / deltas[d]).astype(np.int64) - 1
               for d in range(3)]
    else:
        idx = [np.searchsorted(edges[d], pos[:, d], side="right").astype(np.int64) - 1
               for d in range(3)]
        for d in range(3):  # histogramdd keeps the right-most edge in the last bin
            idx[d][pos[:, d] == edges[d][-1]] -= 1
    ok = np.ones(pos.shape[0], dtype=bool)
    for d, n in enumerate(shape[:3]):
        ok &= (idx[d] >= 0) & (idx[d] < n)
    ok &= (ti >= 0) & (ti < shape[3])
    rav = ((idx[0][ok] * shape[1] + idx[1][ok]) * shape[2]
           + idx[2][ok]) * shape[3] + ti[ok]
    return rav, ok


def _sample_moments(rav, vel, n):
    """(N, [m1 x3], [m2 x6]) per raveled voxel from one batch of samples.

    Two passes over the batch (mean, then centred products): exact, no
    sum-of-squares cancellation. Empty voxels: N=0, m1=NaN, m2=0.
    """
    N = np.bincount(rav, minlength=n).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = [np.bincount(rav, weights=vel[:, d], minlength=n) / N
              for d in range(3)]
    dev = [vel[:, d] - m1[d][rav] for d in range(3)]
    m2 = [np.bincount(rav, weights=dev[i] * dev[j], minlength=n)
          for i, j in _PAIR_IDX]
    return N, m1, m2


def _merge_moments(a, b):
    """Exact pairwise combine of two (N, m1, m2) triples (Chan et al.)."""
    Na, m1a, m2a = a
    Nb, m1b, m2b = b
    N = Na + Nb
    with np.errstate(invalid="ignore", divide="ignore"):
        fb = np.where(N > 0, Nb / np.maximum(N, 1), 0.0)
        cross = np.where(N > 0, Na * Nb / np.maximum(N, 1), 0.0)
    ma = [np.nan_to_num(m) for m in m1a]
    mb = [np.nan_to_num(m) for m in m1b]
    delta = [mb[d] - ma[d] for d in range(3)]
    m1 = [np.where(N > 0, ma[d] + delta[d] * fb, np.nan) for d in range(3)]
    m2 = [m2a[k] + m2b[k] + delta[i] * delta[j] * cross
          for k, (i, j) in enumerate(_PAIR_IDX)]
    return N, m1, m2


def _unique_voxel_traj(rav, tid):
    """Unique (voxel, trajid) pairs, sorted, as two arrays."""
    if rav.size == 0:
        return rav, tid
    order = np.lexsort((tid, rav))
    r, t = rav[order], tid[order]
    keep = np.r_[True, (r[1:] != r[:-1]) | (t[1:] != t[:-1])]
    return r[keep], t[keep]


def eulerian_grid(scene, grid_params, first, last, cycletime=None,
                  deltat=90, base_time=100000, min_count=50,
                  smoothing_sigma=None, *, fill_value=float("nan"),
                  qc=None, chunk_frames=None, add_valid=True, fin=None,
                  voxel_rule="half-open", n_frames_in_period=None,
                  n_phases=None, phase_zero_frame=None,
                  moments=False) -> xr.Dataset:
    """Bin Lagrangian particles onto a (x, y, z, phase) grid of mean velocities.

    Same math as batch_Lagrangian_to_Eulerian.eulerian_grid, but: reads the
    whole particles table in ONE call (scene.collect) instead of one HDF5
    query per frame, bins every particle at once with np.histogramdd, and
    returns a self-describing Dataset instead of writing HDF5.

    Phase-bin naming (shared with openptv-cloud/openptv-analysis — same
    users, one vocabulary; see :mod:`flowtracks.phasing` for the full
    definition and its rules): ``n_frames_in_period`` is ``cycletime``,
    ``n_phases`` is ``fin``, ``phase_zero_frame`` is ``base_time``. Pass
    either style; when both are given the new-style name wins. Narrow
    (gappy) bins — ``phase_width`` in :mod:`flowtracks.phasing` — are a
    caller-side frame selection (see :func:`flowtracks.phasing.bin_of_frame`);
    this gridder always tiles contiguously.

    Input contract (NOT validated — the caller states it): positions in
    METRES, velocities in METRES PER SECOND. The output dataset is labelled
    ``m`` / ``m s-1`` (CF) on that basis. Per-frame velocities (no frame
    rate known) average arithmetically fine, but the ``m s-1`` label is then
    wrong — relabel to ``m frame-1`` downstream, or state the frame rate.

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

    moments: also record the per-voxel moments (see :data:`MOMENT_VARS`):
    ``n_samples`` (raw count, never zeroed by ``min_count``), ``n_traj``
    (distinct trajectory ids), ``*_m1`` (mean) and ``*_m2`` (centred
    second moments), float64, unmasked. Same samples, same voxel and phase
    indices as the means. Combine realizations with :func:`combine_moments`
    and turn them into statistics with :func:`moment_statistics`. Needs
    ``trajid`` from the scene; not available with ``smoothing_sigma``
    (smoothed counts are not sample counts). Default off: the returned
    Dataset is then unchanged.
    """
    if moments and smoothing_sigma is not None:
        raise ValueError("moments=True cannot be combined with smoothing_sigma: "
                         "kernel-smoothed counts are not sample counts.")
    if n_frames_in_period is not None:
        cycletime = n_frames_in_period
    if cycletime is None:
        raise ValueError(
            "eulerian_grid() needs a period: pass cycletime (legacy) or "
            "n_frames_in_period (shared vocabulary)."
        )
    if n_phases is not None:
        fin = n_phases
    if phase_zero_frame is not None:
        base_time = phase_zero_frame
    zaman = deltat * 2 + 1
    fin = int(np.ceil(cycletime / zaman)) if fin is None else int(fin)
    edges = [np.linspace(grid_params[f"min_{d}"], grid_params[f"max_{d}"],
                         grid_params[f"step{d}"] + 1) for d in "xyz"]
    mids = [0.5 * (b[:-1] + b[1:]) for b in edges]
    use_ceil = voxel_rule == "ceil"
    if voxel_rule not in ("half-open", "ceil"):
        raise ValueError("voxel_rule must be 'half-open' or 'ceil'")

    tid = None
    if hasattr(scene, "collect"):
        keys = ["pos", "velocity", "time"] + (["trajid"] if moments else [])
        got = [np.asarray(a) for a in scene.collect(keys)]
        if moments and len(got) < 4:
            raise ValueError("moments=True needs scene.collect(['trajid']) "
                             "to return trajectory ids.")
        pos, vel, time = got[:3]
        if moments:
            tid = got[3]
    else:
        from flowtracks.io import trajectories
        trajs = trajectories(str(scene))
        p_list, v_list, t_list, i_list = [], [], [], []
        for tr in trajs:
            p_list.append(tr.pos())
            v_list.append(tr.velocity())
            t_list.append(tr.time())
            i_list.append(np.full(len(tr.time()), tr.trajid()))
        pos = np.vstack(p_list) if p_list else np.empty((0, 3))
        vel = np.vstack(v_list) if v_list else np.empty((0, 3))
        time = np.concatenate(t_list) if t_list else np.empty((0,))
        if moments:
            tid = np.concatenate(i_list) if i_list else np.empty((0,))

    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    vel = np.asarray(vel, dtype=float).reshape(-1, 3)
    time = np.asarray(time).ravel()
    if moments:
        tid = np.asarray(tid).ravel().astype(np.int64)

    if qc is not None:
        q = dict(qc)
        keep_qc = qc_mask(pos, vel,
                          min_speed=q.get("min_speed"),
                          max_speed=q.get("max_speed"),
                          finite_only=q.get("finite_only", True))
        pos, vel, time = pos[keep_qc], vel[keep_qc], time[keep_qc]
        if moments:
            tid = tid[keep_qc]

    mask = (time >= first) & (time <= last)
    if pos.shape[0]:
        for d, b in enumerate(edges):
            if use_ceil:
                mask &= (pos[:, d] > b[0]) & (pos[:, d] < b[-1])
            else:
                mask &= (pos[:, d] >= b[0]) & (pos[:, d] < b[-1])
    pos, vel, time = pos[mask], vel[mask], time[mask]
    if moments:
        tid = tid[mask]

    # phase bin per particle: identical to the legacy per-frame arithmetic
    # ti = ceil((t - cycle_start)/zaman) - 1, reduced to integer ops
    if pos.shape[0]:
        ti = np.clip(((time.astype(np.int64) - base_time - 1) % int(cycletime))
                     // zaman, 0, fin - 1)
    else:
        ti = np.empty((0,), dtype=np.int64)

    edges4 = [*edges, np.arange(fin + 1) - 0.5]
    shape4 = (*[grid_params[f"step{d}"] for d in "xyz"], fin)
    mins = deltas = None
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
    attrs = {"first": first, "last": last, "cycletime": cycletime,
             "zaman": zaman, "fin": fin, "min_count": min_count,
             "voxel_rule": voxel_rule, "fill_value": float(fill_value)}
    if moments:
        data.update(_moment_data(pos, vel, time, tid, ti, edges, shape4,
                                 use_ceil, mins, deltas, chunk_frames))
        attrs["moments"] = ("n_samples, n_traj, *_m1 (mean), *_m2 (centred "
                            "second moments); raw, unmasked")
    return xr.Dataset(
        data,
        coords={"x": mids[0], "y": mids[1], "z": mids[2], "phase": np.arange(fin)},
        attrs=attrs,
    )


def _moment_data(pos, vel, time, tid, ti, edges, shape4, use_ceil, mins,
                 deltas, chunk_frames):
    """Per-voxel moment arrays for :func:`eulerian_grid` (``moments=True``).

    With ``chunk_frames`` the same time-sorted frame chunks as the means are
    reduced separately and merged exactly (:func:`_merge_moments`); distinct
    trajectories are de-duplicated across chunks, so a trajectory crossing a
    chunk boundary inside one voxel is still counted once.
    """
    n = int(np.prod(shape4))
    rav, ok = _voxel_index(pos, ti, edges, shape4, use_ceil, mins, deltas)
    vel_ok, tid_ok, time_ok = vel[ok], tid[ok], time[ok]
    if chunk_frames is None or rav.size == 0:
        acc = _sample_moments(rav, vel_ok, n)
        r_u, _ = _unique_voxel_traj(rav, tid_ok)
    else:
        uniq = np.unique(time_ok)
        acc, pairs = None, []
        for s in range(0, len(uniq), int(chunk_frames)):
            sel = np.isin(time_ok, uniq[s:s + int(chunk_frames)])
            part = _sample_moments(rav[sel], vel_ok[sel], n)
            acc = part if acc is None else _merge_moments(acc, part)
            pairs.append(_unique_voxel_traj(rav[sel], tid_ok[sel]))
        r_u, _ = _unique_voxel_traj(np.concatenate([p[0] for p in pairs]),
                                    np.concatenate([p[1] for p in pairs]))
    N, m1, m2 = acc
    out = {N_VAR: (DIMS, N.reshape(shape4)),
           NTRAJ_VAR: (DIMS, np.bincount(r_u, minlength=n).astype(np.int64)
                       .reshape(shape4))}
    for name, arr in zip(M1_VARS, m1):
        out[name] = (DIMS, arr.reshape(shape4))
    for name, arr in zip(M2_VARS, m2):
        out[name] = (DIMS, arr.reshape(shape4))
    return out


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
    curve.sizes["phase"]
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
    call site). Unlike :func:`derived_fields` (physical gradients with
    zero-filled NaNs via :func:`clean_field`), this preserves the legacy
    quirks validated against Octave:

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
                   prt_mode: str = "velocity",
                   prt_dt: float = 1.0) -> xr.Dataset:
    """Derived turbulence/flow fields, vectorized over ALL phases.

    Port of sample_vtkcode.py main() (which loops per time slice).
    Gradients are always physical: ``np.gradient`` with the true
    ``(dx, dy, dz)`` spacings and no sign flips (matches
    ``flowtracks.vortex``). Callers that need a flipped sign convention
    (e.g. a negated ``u`` component for legacy-number reproduction) can
    negate the input component themselves before calling.

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
        gridx = float(avg.x[1] - avg.x[0])
        dy = float(avg.y[1] - avg.y[0])
        dz = float(avg.z[1] - avg.z[0])
        dx = gridx
        ux, uy, uz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                      np.gradient(clean_field(u.values) if clean_nan else u.values,
                                  dx, dy, dz, axis=(0, 1, 2)))
        vx, vy, vz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                      np.gradient(clean_field(v.values) if clean_nan else v.values,
                                  dx, dy, dz, axis=(0, 1, 2)))
        wx, wy, wz = (xr.DataArray(g, dims=u.dims, coords=u.coords) for g in
                      np.gradient(clean_field(w.values) if clean_nan else w.values,
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

CF_ATTRS = {
    "u_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity x"},
    "v_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity y"},
    "w_ins_mean": {"units": "m s-1", "long_name": "phase-mean velocity z"},
    "par_ave2": {"units": "1", "long_name": "samples per voxel per phase"},
    "n_samples": {"units": "1", "long_name": "samples per voxel per phase (raw count)"},
    "n_traj": {"units": "1", "long_name": "distinct trajectories per voxel per phase"},
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
    """Attach rich CF metadata for ParaView and NetCDF/Zarr readers.

    Fills in what is missing and never overwrites: a caller's own ``units``
    (e.g. ``m frame-1`` for per-frame velocities) or more specific
    ``long_name`` survives saving.
    """
    for name, attrs in CF_ATTRS.items():
        if name in ds or name in ds.coords:
            for key, val in attrs.items():
                ds[name].attrs.setdefault(key, val)
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
                             prt_mode=d_cfg.get("prt_mode", "velocity"),
                             prt_dt=d_cfg.get("prt_dt", 1.0))

    out = xr.merge([avg, fluct, stats, derived], combine_attrs="override")
    out.attrs = ds.attrs
    return out


# --- pooled statistics from per-voxel moments --------------------------------

def combine_moments(ds: xr.Dataset, dim: str = "set") -> xr.Dataset:
    """Pool per-realization moments over ``dim`` — exact, as if every sample
    of every realization had been binned together.

    ``ds`` holds :data:`MOMENT_VARS` with a ``dim`` dimension (e.g. the
    per-WP datasets of ``eulerian_grid(..., moments=True)`` concatenated
    along ``"set"``). Uses the k-group form of the parallel-variance update:
    ``M2 = sum_s [M2_s + N_s (mean_s - mean)(mean_s - mean)^T]``.
    ``n_traj`` is summed: trajectory ids of different realizations are
    different trajectories. Returns the same variables without ``dim``.
    """
    N = ds[N_VAR]
    n = N.sum(dim)
    w = N.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = {v: (w * ds[v].where(N > 0, 0.0)).sum(dim) / n.where(n > 0)
                for v in M1_VARS}
    dev = {v: xr.where(N > 0, ds[v] - mean[v], 0.0) for v in M1_VARS}
    out = xr.Dataset(attrs=dict(ds.attrs))
    out[N_VAR] = n.astype(np.int64)
    out[NTRAJ_VAR] = ds[NTRAJ_VAR].sum(dim).astype(np.int64)
    for v in M1_VARS:
        out[v] = mean[v]
    for (a, b), name in zip(PAIRS, M2_VARS):
        out[name] = (ds[name] + w * dev[f"{a}_m1"] * dev[f"{b}_m1"]).sum(dim)
    return out


def moment_statistics(mom: xr.Dataset, min_count: int = 1,
                      ddof: int = 1) -> xr.Dataset:
    """Means, Reynolds stresses and rms from (pooled) per-voxel moments.

    Output names match :func:`turbulent_statistics` / ``VEL_VARS`` so the
    result feeds :func:`derived_fields` unchanged. Voxels with
    ``n_samples < max(min_count, ddof + 1)`` are NaN in every statistic and
    ``valid`` is False there. ``ddof=1`` gives the unbiased covariance
    ``M2 / (N - 1)``; note that samples along one trajectory are
    correlated, so ``n_traj`` (kept alongside) is the more honest effective
    sample size for uncertainty estimates.
    """
    n = mom[N_VAR]
    valid = n >= max(int(min_count), int(ddof) + 1)
    out = xr.Dataset(attrs=dict(mom.attrs))
    for v, m in zip(VEL_VARS, M1_VARS):
        out[v] = mom[m].where(valid)
    denom = (n - ddof).astype(float).where(valid)
    for (a, b), name in zip(PAIRS, M2_VARS):
        out[f"{a}_ins_{b}_ins"] = mom[name].where(valid) / denom
    for a in "uvw":
        out[f"{a}_rms"] = np.sqrt(out[f"{a}_ins_{a}_ins"])
    out[N_VAR] = n
    out[NTRAJ_VAR] = mom[NTRAJ_VAR]
    out[VALID_VAR] = valid
    out.attrs.update({"min_count": int(min_count), "ddof": int(ddof)})
    return out


def run_pooled_analysis_ds(ds_sets: dict[str, xr.Dataset], recipe: dict,
                           min_count: int | None = None,
                           ddof: int = 1) -> xr.Dataset:
    """Pooled counterpart of :func:`run_post_analysis_ds`.

    Instead of averaging per-realization means and taking the spread of those
    means across realizations, pool every sample of every realization per
    voxel and phase bin (:func:`combine_moments`), then compute means and
    second moments once (:func:`moment_statistics`). The second moments are
    then the phase-averaged turbulent stresses (all fluctuations within the
    bin), not the between-realization variance of the means.

    ``ds_sets`` must come from ``eulerian_grid(..., moments=True)``. Same
    recipe keys as :func:`run_post_analysis_ds` (``sets``, ``shift``,
    ``mask``, ``derived``). ``count`` masks act on the POOLED sample count
    (``min_count`` defaults to the recipe's count mask, else 1); the other
    masks (outliers, variance, array) act on the pooled means, and every
    statistic is NaN wherever the pooled mean is. ``average.weighting`` does
    not apply (pooling is sample-weighted by construction).
    """
    sets = recipe["sets"]
    missing = [s for s in sets if not set(MOMENT_VARS) <= set(ds_sets[s].data_vars)]
    if missing:
        raise ValueError(f"sets {missing} have no moments: build them with "
                         "eulerian_grid(..., moments=True).")
    stacked = xr.concat([ds_sets[s][MOMENT_VARS] for s in sets],
                        dim=xr.DataArray(sets, dims="set", name="set"),
                        join="exact")
    stacked = shift_phase(stacked, recipe.get("shift", 0))

    specs, other = list(recipe.get("mask", [])), []
    for spec in specs:
        name = spec if isinstance(spec, str) else dict(spec).get("method")
        if name == "count":
            if min_count is None and not isinstance(spec, str):
                min_count = dict(spec).get("min_count")
        else:
            other.append(spec)
    min_count = 1 if min_count is None else int(min_count)

    pooled = combine_moments(stacked, "set")
    stats = moment_statistics(pooled, min_count=min_count, ddof=ddof)
    masked = apply_masks(stats[VEL_VARS], other)
    keep = masked[VEL_VARS[0]].notnull()
    for v in VEL_VARS[1:]:
        keep = keep & masked[v].notnull()
    stat_vars = [v for v in stats.data_vars if v not in (N_VAR, NTRAJ_VAR, VALID_VAR)]
    stats = stats.assign({v: stats[v].where(keep) for v in stat_vars})
    stats[VALID_VAR] = stats[VALID_VAR] & keep

    d_cfg = recipe.get("derived", {})
    fields = d_cfg.get("fields", ["MKE", "TKE"])
    derived = derived_fields(stats[VEL_VARS], stats,
                             rho=d_cfg.get("rho", 1.0), mu=d_cfg.get("mu", 1.0),
                             fields=ALL_DERIVED if fields == "all" else fields,
                             counts=stats[N_VAR],
                             fluid_min=d_cfg.get("fluid_min"),
                             clean_nan=d_cfg.get("clean_nan", True),
                             prt_mode=d_cfg.get("prt_mode", "velocity"),
                             prt_dt=d_cfg.get("prt_dt", 1.0))
    out = xr.merge([stats, derived, pooled[M1_VARS + M2_VARS]],
                   combine_attrs="override")
    out.attrs = dict(stacked.attrs)
    out.attrs.update({"statistics": "pooled", "min_count": min_count,
                      "ddof": int(ddof)})
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
    import warnings

    import h5py

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

