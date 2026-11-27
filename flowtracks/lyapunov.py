# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Yosef Meller, Alex Liberzon
"""
Lyapunov-exponent estimates from Lagrangian trajectories (issue #65).

Three estimators are provided. They answer different questions and should not
be expected to agree.

``finite_time_lyapunov`` - the mean exponential growth rate of the separation
of close pairs, ``<ln(r(t)/r(0))> / t``, fitted over a lag window (the
finite-time Lyapunov exponent, FTLE, of pair dispersion).

``finite_size_lyapunov`` - the finite-size Lyapunov exponent (FSLE),
``ln(rho) / <T(delta)>`` where ``T(delta)`` is the time for the separation to
grow from ``delta`` to ``rho * delta``. Unlike the FTLE it does not depend on a
fit window or on one initial separation.

``eulerian_ftle`` - the FTLE *field* on a regular grid, from the gradient of
the flow map. This one needs much more care than the pair-based estimators;
read its docstring and ``doc/lyapunov.rst`` before using it.

All take a list of :class:`~flowtracks.trajectory.Trajectory` objects and
return an :class:`xarray.Dataset`.

With 3D-PTV data the separations are noisy at small scales, so choose
``r_min`` well above the position error, otherwise the exponent is biased high.
"""

import warnings

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


def find_pairs(trajects, r_min, r_max, min_overlap=2):
    """
    Find pairs of trajectories that come within a given distance range.

    For every pair of trajectories, the first frame in which both exist and the
    separation is in ``[r_min, r_max]`` is taken as the pair's start.

    Arguments:
    trajects - a list of Trajectory objects. Their ``time()`` must be integer
        frame numbers (floats holding integers are accepted).
    r_min, r_max - range of initial separations to accept, [m].
    min_overlap - the minimum number of common frames, including the first, that
        a pair must share from its start on, to be accepted.

    Returns:
    pairs - (p, 2) int array, indices into ``trajects`` of each pair (first
        index < second index).
    start_frames - (p,) int array, the frame number at which each pair starts.
    """
    if not 0 <= r_min < r_max:
        raise ValueError("Need 0 <= r_min < r_max")

    frames = [np.rint(t.time()).astype(int) for t in trajects]
    if len(trajects) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)

    all_frames = np.concatenate(frames)
    all_pos = np.concatenate([t.pos() for t in trajects])
    all_ids = np.concatenate(
        [np.full(len(f), i) for i, f in enumerate(frames)])
    order = np.argsort(all_frames, kind='stable')
    all_frames, all_pos, all_ids = (
        all_frames[order], all_pos[order], all_ids[order])
    bounds = np.flatnonzero(np.diff(all_frames)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(all_frames)]

    first_seen = {}
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        pos = all_pos[s:e]
        ids = all_ids[s:e]
        cand = cKDTree(pos).query_pairs(r_max, output_type='ndarray')
        if len(cand) == 0:
            continue
        dist = np.linalg.norm(pos[cand[:, 0]] - pos[cand[:, 1]], axis=1)
        cand = cand[dist >= r_min]
        for a, b in ids[cand]:
            key = (min(a, b), max(a, b))
            if key not in first_seen:
                first_seen[key] = all_frames[s]

    pairs, start_frames = [], []
    for (a, b), f0 in first_seen.items():
        common = np.intersect1d(frames[a], frames[b])
        if (common >= f0).sum() >= min_overlap:
            pairs.append((a, b))
            start_frames.append(f0)

    if not pairs:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)
    return np.array(pairs, dtype=int), np.array(start_frames, dtype=int)


def pair_separations(trajects, pairs, start_frames, max_lag=None):
    """
    Follow the separation of each pair from its start frame.

    Returns:
    A list with one ``(lags, r)`` tuple per pair: ``lags`` are the integer
    frame offsets from the pair's start (the first one is 0) and ``r`` the
    separation at those frames, [m]. Only frames where both trajectories exist
    are included.
    """
    out = []
    for (a, b), f0 in zip(pairs, start_frames):
        ta, tb = trajects[a], trajects[b]
        fa = np.rint(ta.time()).astype(int)
        fb = np.rint(tb.time()).astype(int)
        common, ia, ib = np.intersect1d(fa, fb, return_indices=True)
        keep = common >= f0
        if max_lag is not None:
            keep &= common - f0 <= max_lag
        r = np.linalg.norm(ta.pos()[ia[keep]] - tb.pos()[ib[keep]], axis=1)
        out.append((common[keep] - f0, r))
    return out


def finite_time_lyapunov(trajects, r_min, r_max, dt=1.0, max_lag=None,
                         fit_lags=None, min_overlap=3):
    """
    Finite-time Lyapunov exponent from the growth of pair separation.

    For each lag ``tau`` the mean of ``ln(r(tau) / r(0))`` over all pairs that
    exist at that lag is computed; the exponent is the least-squares slope of
    that curve over the fit window.

    Arguments:
    trajects - list of Trajectory objects with integer frame ``time()``.
    r_min, r_max - range of initial pair separations, [m].
    dt - the time between frames, [s].
    max_lag - largest lag (in frames) to follow. None for no limit.
    fit_lags - ``(first, last)`` lags in frames, inclusive, defining the fit
        window. None uses all lags from 1 to the largest lag.
    min_overlap - minimal number of common frames for a pair to be used.

    Returns:
    An xarray Dataset with the ``lag`` dimension [s]: ``mean_log_growth``
    (``<ln(r/r0)>``), ``npairs`` (pairs contributing at each lag). The fitted
    exponent [1/s] is in the scalar variable ``lyapunov`` and ``attrs``.
    """
    pairs, f0 = find_pairs(trajects, r_min, r_max, min_overlap)
    seps = pair_separations(trajects, pairs, f0, max_lag)
    seps = [(lg, r) for lg, r in seps if r[0] > 0]

    top = max((lg[-1] for lg, _ in seps), default=0)
    sums = np.zeros(top + 1)
    counts = np.zeros(top + 1, dtype=int)
    for lg, r in seps:
        good = r > 0
        np.add.at(sums, lg[good], np.log(r[good] / r[0]))
        np.add.at(counts, lg[good], 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_log = np.where(counts > 0, sums / counts, np.nan)

    lo, hi = (1, top) if fit_lags is None else fit_lags
    lags = np.arange(top + 1)
    sel = (lags >= lo) & (lags <= hi) & (counts > 0)
    if sel.sum() >= 2:
        slope = np.polyfit(lags[sel] * dt, mean_log[sel], 1)[0]
    else:
        slope = np.nan

    return xr.Dataset(
        {'mean_log_growth': ('lag', mean_log),
         'npairs': ('lag', counts),
         'lyapunov': slope},
        coords={'lag': lags * dt},
        attrs={'method': 'pair-separation finite-time Lyapunov exponent',
               'r_min': r_min, 'r_max': r_max, 'dt': dt,
               'lyapunov_exponent': slope,
               'npairs_total': len(seps)})


def _first_crossing(lags, r, level):
    """Interpolated lag at which ``r`` first reaches ``level``; NaN if never."""
    above = np.flatnonzero(r >= level)
    if len(above) == 0:
        return np.nan
    k = above[0]
    if k == 0:
        return float(lags[0])
    r0, r1 = r[k - 1], r[k]
    frac = (level - r0) / (r1 - r0)
    return lags[k - 1] + frac * (lags[k] - lags[k - 1])


def finite_size_lyapunov(trajects, r_min, r_max, ratio=1.5, num_scales=10,
                         dt=1.0, min_overlap=3):
    """
    Finite-size Lyapunov exponent (FSLE) from pair separation growth.

    Thresholds ``delta_n = r_min * ratio**n`` are laid out; for each pair the
    time ``T`` taken to go from ``delta_n`` to ``ratio * delta_n`` is measured
    (first-passage times, linearly interpolated between frames), averaged over
    pairs, and ``lambda(delta_n) = ln(ratio) / <T>``.

    Arguments:
    trajects - list of Trajectory objects with integer frame ``time()``.
    r_min, r_max - range of initial separations used to select pairs, [m].
        The thresholds start at ``r_min``.
    ratio - the amplification factor between consecutive thresholds (> 1).
    num_scales - number of thresholds.
    dt - the time between frames, [s].
    min_overlap - minimal number of common frames for a pair to be used.

    Returns:
    An xarray Dataset with dimension ``delta`` [m]: ``fsle`` [1/s],
    ``mean_time`` ([s], ``<T>``) and ``count`` (pairs that completed that
    amplification). Scales no pair completed are NaN.
    """
    if ratio <= 1:
        raise ValueError("ratio must be > 1")

    pairs, f0 = find_pairs(trajects, r_min, r_max, min_overlap)
    seps = pair_separations(trajects, pairs, f0)
    deltas = r_min * ratio ** np.arange(num_scales)

    sum_t = np.zeros(num_scales)
    count = np.zeros(num_scales, dtype=int)
    for lg, r in seps:
        for n, d in enumerate(deltas):
            # A pair that starts above delta_n cannot be timed from delta_n.
            if r[0] > d:
                continue
            t_in = _first_crossing(lg, r, d)
            t_out = _first_crossing(lg, r, d * ratio)
            if np.isnan(t_in) or np.isnan(t_out):
                continue
            sum_t[n] += t_out - t_in
            count[n] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_t = np.where(count > 0, sum_t / count, np.nan) * dt
        fsle = np.log(ratio) / mean_t

    return xr.Dataset(
        {'fsle': ('delta', fsle),
         'mean_time': ('delta', mean_t),
         'count': ('delta', count)},
        coords={'delta': deltas},
        attrs={'method': 'finite-size Lyapunov exponent', 'ratio': ratio,
               'r_min': r_min, 'r_max': r_max, 'dt': dt,
               'npairs_total': len(seps)})


def flow_map(trajects, t_start, t_end):
    """
    The scattered flow map of a set of trajectories between two frames.

    Only trajectories that exist at *both* frames contribute. This is a
    survivorship selection: it keeps the particles that were tracked for the
    whole window, which is not a random sample of the flow (see
    ``doc/lyapunov.rst``).

    Arguments:
    trajects - list of Trajectory objects with integer frame ``time()``.
    t_start, t_end - integer frame numbers. ``t_end < t_start`` is allowed (a
        backward map).

    Returns:
    x0, x1 - (n,3) arrays, the position of each surviving particle at
        ``t_start`` and at ``t_end``, [m].
    ids - (n,) indices into ``trajects`` of those particles.
    """
    x0, x1, ids = [], [], []
    for i, tr in enumerate(trajects):
        fr = np.rint(tr.time()).astype(int)
        i0 = np.searchsorted(fr, t_start)
        i1 = np.searchsorted(fr, t_end)
        if (i0 < len(fr) and fr[i0] == t_start
                and i1 < len(fr) and fr[i1] == t_end):
            pos = tr.pos()
            x0.append(pos[i0])
            x1.append(pos[i1])
            ids.append(i)
    if not ids:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=int)
    return np.array(x0), np.array(x1), np.array(ids)


def eulerian_ftle(trajects, grid, t_start, t_end, radius, dt=1.0,
                  min_neighbors=12, max_condition=10.0, max_offset=0.25,
                  position_error=None):
    """
    Eulerian finite-time Lyapunov exponent field from scattered trajectories.

    **Use with caution** - see ``doc/lyapunov.rst`` for the full discussion.
    In short: the result is only meaningful where the diagnostics say so, it
    depends on ``t_start``, the window length and ``radius``, and it is
    computed from the particles that survived the whole window, which
    is a biased subset of the flow.

    Method: the flow map ``Phi`` takes the position ``x0`` of a particle at
    ``t_start`` to its position ``x1`` at ``t_end``. At each grid node, the
    deformation gradient ``F = dPhi/dx0`` is estimated by a locally weighted
    linear least-squares fit of ``x1`` against ``x0`` over the particles whose
    ``x0`` is within ``radius`` of the node (tricube weights). Then
    ``C = F^T F`` is the right Cauchy-Green tensor and the exponent is
    ``FTLE = ln(sqrt(lambda_max(C))) / |T|``, with ``T = (t_end - t_start)*dt``.

    A negative window (``t_end < t_start``) gives the backward FTLE.

    A node is reported (``valid``) only if all three validity checks pass,
    otherwise its values are NaN. First, at least ``min_neighbors`` particles
    must be within ``radius``.

    Second, the neighbourhood must not be degenerate: the ratio of the largest
    to the smallest singular value of the (weighted, centred) neighbour offsets
    must be at most ``max_condition``. This rejects flat or line-like clouds
    from which ``F`` cannot be estimated.

    Third, the neighbourhood must surround the node: the weighted centroid of
    the neighbours must be within ``max_offset * radius`` of it. This rejects
    nodes at the edge of the data, where the fit would extrapolate. For tricube
    weights a node exactly at a straight data edge has an offset of 0.27, one
    lying outside the data has more, and a full neighbourhood has about 0.

    Arguments:
    trajects - list of Trajectory objects with integer frame ``time()``.
    grid - ``(x, y, z)``, three 1D arrays of node coordinates, [m].
    t_start, t_end - integer frame numbers of the beginning and the end of the
        flow map.
    radius - neighbourhood radius, [m]. There is no sensible default: it sets
        the spatial resolution, and must be chosen against the particle
        density and the flow scales.
    dt - the time between frames, [s].
    min_neighbors - minimal particles in a neighbourhood (at least 5 are
        needed to fit 4 coefficients per component; more is better).
    max_condition, max_offset - validity thresholds, see above.
    position_error - optional, the standard deviation of the measurement error
        of one position coordinate, [m]. If given, the output includes
        ``ftle_noise_floor``.

    Returns:
    An xarray Dataset on the ``(x, y, z)`` grid. Its variables are listed
    below; ``attrs`` record the parameters and the number of particles used.
    ftle - the exponent, [1/s]. NaN where the node is not valid.
    det_F - ``det(F)``. For an incompressible flow it must be 1; its distance
        from 1 is a built-in check of the whole estimate (noise, curvature,
        too large a radius). Treat ``ftle`` as unreliable where it is far
        from 1.
    n_points - number of particles used at each node.
    fit_residual_rms - weighted RMS of the local linear-fit residuals, [m].
        Large compared to the measurement error means the flow map is not
        linear over ``radius`` (reduce it) or the data are noisy.
    condition - the neighbourhood condition number described above.
    valid - boolean mask of the nodes that passed all the validity checks.
    ftle_noise_floor - only if ``position_error`` is given. A first-order
        estimate of the FTLE uncertainty due to measurement noise, [1/s].
        Values of ``ftle`` that are not well above it are noise.
    """
    if t_end == t_start:
        raise ValueError("t_end must differ from t_start")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if min_neighbors < 5:
        raise ValueError("min_neighbors must be at least 5")

    axes = [np.asarray(a, dtype=float) for a in grid]
    if len(axes) != 3:
        raise ValueError("grid must be three 1D arrays (x, y, z)")
    shape = tuple(len(a) for a in axes)
    nodes = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    T = abs(t_end - t_start) * dt

    x0, x1, _ = flow_map(trajects, t_start, t_end)
    if len(x0) < min_neighbors:
        warnings.warn(
            "Only %d trajectories span frames %d..%d; no FTLE can be "
            "computed. Use a shorter window or more data." %
            (len(x0), t_start, t_end))

    nn = len(nodes)
    ftle = np.full(nn, np.nan)
    det_f = np.full(nn, np.nan)
    resid = np.full(nn, np.nan)
    cond = np.full(nn, np.nan)
    floor = np.full(nn, np.nan)
    npts = np.zeros(nn, dtype=int)

    if len(x0) >= min_neighbors:
        neighbours = cKDTree(x0).query_ball_point(nodes, radius)
        for k, idx in enumerate(neighbours):
            n = len(idx)
            npts[k] = n
            if n < min_neighbors:
                continue
            d = x0[idx] - nodes[k]
            r = np.linalg.norm(d, axis=1)
            w = (1 - (r / radius) ** 3) ** 3
            wsum = w.sum()
            centroid = (w[:, None] * d).sum(axis=0) / wsum
            if np.linalg.norm(centroid) > max_offset * radius:
                continue

            sw = np.sqrt(w)
            sv = np.linalg.svd(sw[:, None] * (d - centroid),
                               compute_uv=False)
            if sv[-1] <= 0 or sv[0] / sv[-1] > max_condition:
                continue
            cond[k] = sv[0] / sv[-1]

            design = np.hstack([np.ones((n, 1)), d])
            coef = np.linalg.lstsq(
                sw[:, None] * design, sw[:, None] * x1[idx], rcond=None)[0]
            # coef[1 + i, j] = dPhi_j / dx0_i, so F[j, i] = coef[1 + i, j].
            fmat = coef[1:].T
            res = x1[idx] - design @ coef
            resid[k] = np.sqrt((w * (res ** 2).sum(axis=1)).sum() / wsum)

            lam_max = np.linalg.eigvalsh(fmat.T @ fmat)[-1]
            ftle[k] = 0.5 * np.log(lam_max) / T
            det_f[k] = np.linalg.det(fmat)

            if position_error is not None:
                m_inv = np.linalg.inv(design.T @ (w[:, None] * design))
                b = design.T @ ((w ** 2)[:, None] * design)
                cov = position_error ** 2 * (m_inv @ b @ m_inv)
                floor[k] = np.sqrt(np.trace(cov[1:, 1:]) / 3) / T

    valid = np.isfinite(ftle)
    if len(x0) >= min_neighbors and not valid.any():
        warnings.warn(
            "No grid node passed the validity checks; try a larger radius, "
            "a shorter window, or a coarser/smaller grid.")

    dims = ('x', 'y', 'z')

    def field(a):
        return (dims, a.reshape(shape))

    data = {'ftle': field(ftle), 'det_F': field(det_f),
            'n_points': field(npts), 'fit_residual_rms': field(resid),
            'condition': field(cond), 'valid': field(valid)}
    if position_error is not None:
        data['ftle_noise_floor'] = field(floor)

    return xr.Dataset(
        data, coords=dict(zip(dims, axes)),
        attrs={'method': 'Eulerian FTLE from local least-squares flow-map '
                         'gradient',
               't_start': t_start, 't_end': t_end, 'T': T, 'dt': dt,
               'radius': radius, 'min_neighbors': min_neighbors,
               'max_condition': max_condition, 'max_offset': max_offset,
               'n_trajectories_used': len(x0)})
