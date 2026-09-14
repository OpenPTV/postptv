"""
Trajectory repair: cut links that fail a gap check, attach single points, and
join trajectory pieces with a checked join.

Works on the trajectories a tracker produced -- one segment per linked track,
with unlinked particles as 1-point segments. Three steps, each counted in the
returned report:

1. **cut** -- for every link inside a trajectory, fit a straight line to up to
   ``window`` points before the link and, separately, to up to ``window``
   points after it. Both lines are evaluated at the middle of the link; if
   they disagree by more than ``cut_factor`` times the threshold the link is
   cut. A wrong link (to another particle) jumps by roughly the particle
   spacing, far above the noise, while a correct link only differs by noise.
2. **attach** -- a 1-point segment is appended to the end (or start) of a
   trajectory whose straight-line prediction, up to ``max_gap + 1`` frames
   ahead (or back), it matches within the threshold.
3. **join** -- a trajectory end is joined to a trajectory start 1 ..
   ``max_gap + 1`` frames later when the same two-sided gap check passes.
   Each end and each start joins at most once; chains (A->B->C) form through
   the join map, so no segment is ever copied.

The threshold calibrates itself on the data. The gap-check mismatch divided
by the size its noise should have (straight-line fit weights times a per-axis
noise scale) gives one number ``chi`` per link; the noise scale is the robust
spread of the mismatch on links inside long trajectories (``long_track``
points or more), and the threshold ``T`` is the ``link_percentile``
percentile of ``chi`` on those same links (gross outliers excluded). No
distance or velocity tolerance has to be guessed.

No positions are invented: a joined gap stays a gap (missing frames), so
derivatives taken over the real frame numbers stay honest.

:func:`repair_arrays` is the vectorised core working on flat arrays (millions
of points: candidate searches are single KD-tree queries over (x, y, z, frame),
matches are resolved greedily in order of increasing chi);
:func:`repair_trajectories` wraps it for lists of ``Trajectory`` objects.

Validated in openptv-analysis (logs 025-026) on synthetic ground truth
matched to a 4-camera experiment: attachments 99.3-99.9% correct, joins
99.9% correct, trajectories of 10+ frames from a single particle 97-98%.
"""

import numpy as np
from scipy.spatial import cKDTree

from flowtracks.trajectory import Trajectory

__all__ = ["repair_arrays", "repair_trajectories"]

#: Separates frames in the (x, y, z, frame) search tree: points of different
#: frames are always farther apart than any search radius.
_FRAME_SCALE = 1e9


def _robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def _line_predict(P, T, t0):
    """Straight-line least-squares fit of positions P (M, n, 3) at times
    T (M, n), evaluated at t0 (M,). Also returns sqrt(sum w^2), the factor by
    which the fit scales the per-point noise."""
    tb = T.mean(axis=1, keepdims=True)
    dt = T - tb
    w = 1.0 / T.shape[1] + dt * (t0[:, None] - tb) / (dt ** 2).sum(axis=1, keepdims=True)
    return (P * w[:, :, None]).sum(axis=1), np.sqrt((w ** 2).sum(axis=1))


def _segments(sid):
    """Start index and length of each run of equal ``sid`` (grouped array)."""
    if len(sid) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    start = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    return start, np.diff(np.r_[start, len(sid)])


def _window_indices(start, length, n, from_end):
    """(M, n) indices of the first or last n points of each segment."""
    first = start + length - n if from_end else start
    return first[:, None] + np.arange(n)


def _greedy_one_to_one(chi, left, right):
    """Accept pairs in increasing chi; each left and each right at most once.

    Same result as scanning the sorted list and skipping used ends, computed
    by repeatedly accepting every pair that is the best for both its left and
    its right (locally dominant pairs) and removing everything they block."""
    order = np.argsort(chi, kind="stable")
    left, right = left[order], right[order]
    active = np.ones(len(order), dtype=bool)
    out_l, out_r = [], []
    while active.any():
        idx = np.flatnonzero(active)
        first_l = np.zeros(len(idx), dtype=bool)
        first_l[np.unique(left[idx], return_index=True)[1]] = True
        first_r = np.zeros(len(idx), dtype=bool)
        first_r[np.unique(right[idx], return_index=True)[1]] = True
        acc = idx[first_l & first_r]
        out_l.append(left[acc])
        out_r.append(right[acc])
        active &= ~np.isin(left, left[acc]) & ~np.isin(right, right[acc])
    return np.concatenate(out_l), np.concatenate(out_r)


class _Points:
    """All usable points, kept grouped by segment id and sorted by time."""

    def __init__(self, sid, time, pos, row):
        self.sid, self.time, self.pos, self.row = sid, time, pos, row
        self.regroup(sid)

    def regroup(self, new_sid):
        order = np.lexsort((self.time, new_sid))
        self.sid = np.asarray(new_sid)[order]
        self.time, self.pos, self.row = self.time[order], self.pos[order], self.row[order]


def _link_mismatch(pts, window):
    """Two-sided gap check of every within-segment link i -> i+1.

    Returns Z (N, 3): mismatch divided by its fit-weight scale (NaN where a
    side has fewer than 3 points), and the segment length per point."""
    start, length = _segments(pts.sid)
    seg = np.repeat(np.arange(len(start)), length)
    r = np.arange(len(pts.sid)) - start[seg]
    rem = length[seg] - 1 - r
    Z = np.full((len(pts.sid), 3), np.nan)
    for na in range(3, window + 1):
        for nb in range(3, window + 1):
            i = np.flatnonzero((np.minimum(window, r + 1) == na) & (np.minimum(window, rem) == nb))
            if not len(i):
                continue
            ia = i[:, None] + np.arange(-na + 1, 1)
            ib = i[:, None] + np.arange(1, nb + 1)
            t0 = 0.5 * (pts.time[i] + pts.time[i + 1])
            pa, sa = _line_predict(pts.pos[ia], pts.time[ia], t0)
            pb, sb = _line_predict(pts.pos[ib], pts.time[ib], t0)
            Z[i] = (pa - pb) / np.sqrt(sa ** 2 + sb ** 2)[:, None]
    return Z, length[seg]


def _attach_singles(pts, sig, T, window, max_gap, max_distance, k=2):
    """One pass (forward, then backward) of attaching 1-point segments."""
    attached = 0
    signorm = float(np.linalg.norm(sig))
    for direction in (+1, -1):
        start, length = _segments(pts.sid)
        singles = start[length == 1]
        tracks = np.flatnonzero(length >= 2)
        if not len(singles) or not len(tracks):
            continue
        tree = cKDTree(np.c_[pts.pos[singles], pts.time[singles] * _FRAME_SCALE])
        kk = min(k, len(singles))
        c_chi, c_edge, c_single = [], [], []
        for n in range(2, window + 1):
            sel = tracks[np.minimum(length[tracks], window) == n]
            if not len(sel):
                continue
            win = _window_indices(start[sel], length[sel], n, from_end=direction > 0)
            edge = win[:, -1] if direction > 0 else win[:, 0]
            P, Tm = pts.pos[win], pts.time[win]
            for step in range(1, max_gap + 2):
                t0 = pts.time[edge] + direction * step
                pred, sw = _line_predict(P, Tm, t0)
                scale = np.sqrt(sw ** 2 + 1.0)
                rmax = T * signorm * float(scale.max())
                if max_distance is not None:
                    rmax = min(rmax, max_distance)
                d, j = tree.query(np.c_[pred, t0 * _FRAME_SCALE], k=kk, distance_upper_bound=rmax)
                d, j = d.reshape(len(edge), kk), j.reshape(len(edge), kk)
                rows, cols = np.nonzero(np.isfinite(d))
                if not len(rows):
                    continue
                s = singles[j[rows, cols]]
                chi = np.linalg.norm((pts.pos[s] - pred[rows]) / (sig * scale[rows, None]), axis=1)
                ok = chi <= T
                c_chi.append(chi[ok])
                c_edge.append(edge[rows[ok]])
                c_single.append(s[ok])
        if not c_chi or not sum(len(c) for c in c_chi):
            continue
        e, s = _greedy_one_to_one(np.concatenate(c_chi), np.concatenate(c_edge), np.concatenate(c_single))
        new_sid = pts.sid.copy()
        new_sid[s] = pts.sid[e]
        pts.regroup(new_sid)
        attached += len(e)
    return attached


def _join_segments(pts, sig, T, window, max_gap, max_distance, k=2):
    """Join segment ends to later segment starts; returns the join count."""
    start, length = _segments(pts.sid)
    tracks = np.flatnonzero(length >= 2)
    if len(tracks) < 2:
        return 0
    first = start[tracks]
    tree = cKDTree(np.c_[pts.pos[first], pts.time[first] * _FRAME_SCALE])
    kk = min(k, len(tracks))
    signorm = float(np.linalg.norm(sig))
    c_chi, c_a, c_b = [], [], []
    for na in range(2, window + 1):
        ends = tracks[np.minimum(length[tracks], window) == na]
        if not len(ends):
            continue
        wa = _window_indices(start[ends], length[ends], na, from_end=True)
        Pa, Ta = pts.pos[wa], pts.time[wa]
        for step in range(1, max_gap + 2):
            ts = Ta[:, -1] + step
            pred, sw = _line_predict(Pa, Ta, ts)
            rmax = T * signorm * float(np.sqrt(sw ** 2 + 1.0).max())
            if max_distance is not None:
                rmax = min(rmax, max_distance)
            d, j = tree.query(np.c_[pred, ts * _FRAME_SCALE], k=kk, distance_upper_bound=rmax)
            d, j = d.reshape(len(ends), kk), j.reshape(len(ends), kk)
            rows, cols = np.nonzero(np.isfinite(d))
            if not len(rows):
                continue
            a, b = ends[rows], tracks[j[rows, cols]]
            keep = a != b
            rows, a, b = rows[keep], a[keep], b[keep]
            nbs = np.minimum(length[b], window)
            for nb in np.unique(nbs):
                m = nbs == nb
                ia = wa[rows[m]]
                ib = start[b[m]][:, None] + np.arange(nb)
                t0 = 0.5 * (pts.time[ia[:, -1]] + pts.time[ib[:, 0]])
                pa, sa = _line_predict(pts.pos[ia], pts.time[ia], t0)
                pb, sb = _line_predict(pts.pos[ib], pts.time[ib], t0)
                chi = np.linalg.norm((pa - pb) / np.sqrt(sa ** 2 + sb ** 2)[:, None] / sig, axis=1)
                ok = chi <= T
                c_chi.append(chi[ok])
                c_a.append(a[m][ok])
                c_b.append(b[m][ok])
    if not c_chi or not sum(len(c) for c in c_chi):
        return 0
    a, b = _greedy_one_to_one(np.concatenate(c_chi), np.concatenate(c_a), np.concatenate(c_b))
    # joins always go forward in time, so following parents cannot cycle
    parent = np.arange(len(start))
    parent[b] = a
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    pts.regroup(np.repeat(parent, length))
    return len(a)


def repair_arrays(trajid, time, pos, max_gap=3, window=5, link_percentile=99.9,
                  long_track=50, cut=True, attach=True, join=True,
                  max_distance=None, attach_rounds=5, cut_factor=2.0):
    """Repair trajectories given as flat arrays (one row per particle position).

    Arguments:
    trajid - (N,) integer trajectory id per row (1-point trajectories are
        unlinked particles).
    time - (N,) integer frame number per row.
    pos - (N, 3) positions (any length unit).
    max_gap, window, link_percentile, long_track, cut, attach, join,
    max_distance, attach_rounds, cut_factor - see :func:`repair_trajectories`.

    Returns:
    (new_trajid, report) - (N,) int64 trajectory ids in input row order, and
    the report dict. Rows of a trajectory whose frame numbers are not strictly
    increasing (e.g. an "unlinked particles" bucket) keep their id and are not
    repaired; repaired trajectories get fresh ids above ``max(trajid)``.
    """
    trajid = np.asarray(trajid).reshape(-1).astype(np.int64)
    time = np.asarray(time, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    report = {"points": int(len(time)), "links_cut": 0, "points_attached": 0, "joins": 0}
    new_trajid = trajid.copy()
    if len(trajid) == 0:
        report.update(n_in=0, n_out=0, skipped="no points")
        return new_trajid, report
    report["n_in"] = int(len(np.unique(trajid)))

    order = np.lexsort((time, trajid))
    t_sorted, id_sorted = time[order], trajid[order]
    repeated = (id_sorted[1:] == id_sorted[:-1]) & (np.diff(t_sorted) <= 0)
    bad_ids = np.unique(id_sorted[1:][repeated])
    rows = np.flatnonzero(~np.isin(trajid, bad_ids))
    if not len(rows):
        report.update(n_out=report["n_in"], skipped="no trajectories with increasing frame numbers")
        return new_trajid, report
    sid = np.unique(trajid[rows], return_inverse=True)[1]
    pts = _Points(sid, time[rows], pos[rows], rows)

    Z, seg_len = _link_mismatch(pts, window)
    valid = ~np.isnan(Z[:, 0])
    report["links_checked"] = int(valid.sum())
    if valid.sum() < 30:
        report.update(n_out=report["n_in"], skipped="fewer than 30 links with enough points on both sides")
        return new_trajid, report
    calib = valid & (seg_len >= long_track)
    if calib.sum() < 1000:  # too few long-track links to calibrate on: use every checkable link
        calib = valid
    sig = _robust_std(Z[calib])
    if not np.all(sig > 0):
        report.update(n_out=report["n_in"], skipped="zero noise scale (noise-free or degenerate data)")
        return new_trajid, report
    chi = np.linalg.norm(Z / sig, axis=1)
    ref = chi[calib]
    # Gross mismatches are wrong links, not the noise tail; keep them out of the
    # threshold so a few of them cannot raise it above themselves.
    ref = ref[ref < 30.0 * np.median(ref)]
    T = float(np.percentile(ref, link_percentile))
    report["noise_scale"] = [float(s) for s in sig]
    report["threshold"] = T
    report["cut_threshold"] = cut_factor * T

    if cut:
        # Cut only clear outliers (cut_factor x T). A link between T and cut_factor x T
        # is a noise spike more often than a wrong link, and the join step uses
        # the same check with threshold T, so cutting it at T would break it for good.
        bad = np.flatnonzero(valid & (chi > cut_factor * T))
        boundary = np.r_[True, pts.sid[1:] != pts.sid[:-1]]
        boundary[bad + 1] = True
        pts.regroup(np.cumsum(boundary) - 1)
        report["links_cut"] = int(len(bad))

    if attach:
        for _ in range(attach_rounds):
            n = _attach_singles(pts, sig, T, window, max_gap, max_distance)
            report["points_attached"] += n
            if n == 0:
                break

    if join:
        report["joins"] = _join_segments(pts, sig, T, window, max_gap, max_distance)

    start, length = _segments(pts.sid)
    base = int(trajid.max()) + 1
    new_trajid[pts.row] = base + np.repeat(np.arange(len(start)), length)
    report["n_out"] = int(len(np.unique(new_trajid)))
    return new_trajid, report


def repair_trajectories(trajs, max_gap=3, window=5, link_percentile=99.9,
                        long_track=50, cut=True, attach=True, join=True,
                        max_distance=None, attach_rounds=5, cut_factor=2.0):
    """Cut suspect links, attach single points and join pieces of trajectories.

    Arguments:
    trajs - list of Trajectory objects (segments from a tracker, 1-point
        trajectories for unlinked particles). ``time()`` must be integer frame
        numbers, strictly increasing within a trajectory; trajectories whose
        times are not (e.g. an "unlinked particles" bucket) are passed through
        untouched.
    max_gap - largest number of missing frames an attachment or join may span.
    window - up to this many points on each side feed the straight-line fits.
    link_percentile - threshold T is this percentile of the link check value
        (chi) on links inside trajectories of ``long_track`` points or more.
    long_track - minimum length of the trajectories used to calibrate the
        noise scale and the threshold (all checkable links if too few).
    cut, attach, join - enable the individual steps.
    max_distance - optional upper bound on any attachment/join distance (in
        position units); by default only the calibrated chi threshold limits it.
    attach_rounds - repeat attaching while it still finds points (a track may
        grow by one point per round in each direction).
    cut_factor - a link is cut only when its chi exceeds ``cut_factor * T``.
        Attachments and joins are accepted up to T, so a link between T and
        ``cut_factor * T`` -- usually a noise spike -- is left alone instead of
        being cut and then refused by the join step for good. Wrong links jump
        by the particle spacing and land far above either threshold.

    Returns:
    (repaired, report) - a list of Trajectory objects (zero velocity, fresh
    trajids; pass-through trajectories keep theirs), and a dict of counts:
    n_in, n_out, points, links_checked, noise_scale (per axis), threshold,
    cut_threshold, links_cut, points_attached, joins, and ``skipped`` with a
    reason when the data are too short to calibrate.
    """
    usable, passthrough = [], []
    for tr in trajs:
        t = np.asarray(tr.time()).reshape(-1)
        (passthrough if len(t) >= 2 and not np.all(np.diff(t) > 0) else usable).append(tr)
    if not usable:
        return list(trajs), {"n_in": len(trajs), "n_out": len(trajs), "links_cut": 0,
                             "points_attached": 0, "joins": 0,
                             "skipped": "no trajectories with increasing frame numbers"}
    lengths = [len(np.asarray(tr.time()).reshape(-1)) for tr in usable]
    tid = np.repeat(np.arange(len(usable)), lengths)
    time = np.concatenate([np.asarray(tr.time()).reshape(-1) for tr in usable])
    pos = np.concatenate([np.asarray(tr.pos(), dtype=float).reshape(-1, 3) for tr in usable])
    new_tid, report = repair_arrays(tid, time, pos, max_gap=max_gap, window=window,
                                    link_percentile=link_percentile, long_track=long_track,
                                    cut=cut, attach=attach, join=join, max_distance=max_distance,
                                    attach_rounds=attach_rounds, cut_factor=cut_factor)
    if "skipped" in report:
        report.update(n_in=len(trajs), n_out=len(trajs))
        return list(trajs), report
    order = np.lexsort((time, new_tid))
    bounds = np.flatnonzero(np.diff(new_tid[order])) + 1
    repaired = []
    for idx in np.split(order, bounds):
        p = pos[idx]
        repaired.append(Trajectory(p, np.zeros_like(p), time[idx].astype(np.int64), int(new_tid[idx[0]])))
    report.update(n_in=len(trajs), n_out=len(repaired) + len(passthrough))
    return repaired + passthrough, report
