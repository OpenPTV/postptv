# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Yosef Meller, Alex Liberzon
"""
Lyapunov-exponent estimates from Lagrangian trajectories (issue #65).

Two pair-based estimators are provided. Both start from pairs of trajectories
that come close to each other and follow how their separation grows:

* :func:`finite_time_lyapunov` - the mean exponential growth rate of the
  separation, ``<ln(r(t)/r(0))> / t``, fitted over a lag window (the
  finite-time Lyapunov exponent, FTLE, of pair dispersion).
* :func:`finite_size_lyapunov` - the finite-size Lyapunov exponent (FSLE),
  ``ln(rho) / <T(delta)>`` where ``T(delta)`` is the time for the separation to
  grow from ``delta`` to ``rho * delta``. Unlike the FTLE it is not contaminated
  by the finite-size of the domain or by a fixed initial separation.

Both take a list of :class:`~flowtracks.trajectory.Trajectory` objects and
return an :class:`xarray.Dataset`.

Note: with 3D-PTV data the separations are noisy at small scales. Choose
``r_min`` well above the position error, otherwise the exponent is biased high.
"""

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
