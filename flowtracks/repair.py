"""
Trajectory repair: cut links that fail a gap check, attach single points, and
join trajectory pieces with a checked join.

Works on the trajectories a tracker produced -- one flowtracks ``Trajectory``
per linked segment, with unlinked particles as 1-point trajectories. Three
steps, each counted in the returned report:

1. **cut** -- for every link inside a trajectory, fit a straight line to up to
   ``window`` points before the link and, separately, to up to ``window``
   points after it. Both lines are evaluated at the middle of the link; if
   they disagree by more than the threshold the link is cut. A wrong link
   (to another particle) jumps by roughly the particle spacing, far above
   the noise, while a correct link only differs by noise.
2. **attach** -- a 1-point trajectory is appended to the end (or start) of a
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
percentile of ``chi`` on those same links. No distance or velocity tolerance
has to be guessed.

No positions are invented: a joined gap stays a gap (missing frames), so
derivatives taken over the real frame numbers stay honest, and the returned
trajectories carry zero velocity for the caller to differentiate.

Validated in openptv-analysis (logs 025-026) on synthetic ground truth
matched to a 4-camera experiment: attachments 99.3-99.9% correct, joins
99.9% correct, trajectories of 10+ frames from a single particle 97-98%.
"""

import numpy as np
from scipy.spatial import cKDTree

from flowtracks.trajectory import Trajectory

__all__ = ["repair_trajectories"]


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
    start = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    return start, np.diff(np.r_[start, len(sid)])


def _window_indices(start, length, n, from_end):
    """(M, n) indices of the first or last n points of each segment."""
    first = start + length - n if from_end else start
    return first[:, None] + np.arange(n)


class _Points:
    """All points, kept grouped by segment id and sorted by time."""

    def __init__(self, sid, time, pos, trajid):
        order = np.lexsort((time, sid))
        self.sid, self.time, self.pos = sid[order], time[order], pos[order]
        self.trajid = trajid  # per input trajectory
        self.origin = sid[order].copy()  # input trajectory of each point

    def regroup(self, new_sid):
        order = np.lexsort((self.time, new_sid))
        self.sid = new_sid[order]
        self.time, self.pos, self.origin = self.time[order], self.pos[order], self.origin[order]


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


def _attach_singles(pts, sig, T, window, max_gap, max_distance):
    """One pass of attaching 1-point segments to segment ends and starts."""
    attached = 0
    for direction in (+1, -1):
        start, length = _segments(pts.sid)
        singles = start[length == 1]
        tracks = np.flatnonzero(length >= 2)
        if not len(singles) or not len(tracks):
            continue
        st = pts.time[singles]
        by_time = {}
        for t in np.unique(st):
            idx = singles[st == t]
            by_time[t] = (cKDTree(pts.pos[idx]), idx)
        cand = []
        for n in range(2, window + 1):
            sel = tracks[np.minimum(length[tracks], window) == n]
            if not len(sel):
                continue
            win = _window_indices(start[sel], length[sel], n, from_end=direction > 0)
            edge = win[:, -1] if direction > 0 else win[:, 0]
            for step in range(1, max_gap + 2):
                t0 = pts.time[edge] + direction * step
                pred, sw = _line_predict(pts.pos[win], pts.time[win], t0.astype(float))
                scale = np.sqrt(sw ** 2 + 1.0)
                radius = T * np.linalg.norm(sig) * scale
                if max_distance is not None:
                    radius = np.minimum(radius, max_distance)
                for t in np.unique(t0):
                    if t not in by_time:
                        continue
                    rows = np.flatnonzero(t0 == t)
                    tree, idx = by_time[t]
                    for row, hits in zip(rows, tree.query_ball_point(pred[rows], radius[rows])):
                        if not hits:
                            continue
                        s_idx = idx[hits]
                        chi = np.linalg.norm((pts.pos[s_idx] - pred[row]) / (sig * scale[row]), axis=1)
                        for c, s in zip(chi, s_idx):
                            if c <= T:
                                cand.append((float(c), int(edge[row]), int(s)))
        cand.sort()
        used_edge, used_single = set(), set()
        for c, e, s in cand:
            if e in used_edge or s in used_single:
                continue
            used_edge.add(e)
            used_single.add(s)
            pts.sid[s] = pts.sid[e]
            attached += 1
        pts.regroup(pts.sid.copy())
    return attached


def _join_segments(pts, sig, T, window, max_gap, max_distance):
    """Join segment ends to later segment starts; returns the join count."""
    start, length = _segments(pts.sid)
    tracks = np.flatnonzero(length >= 2)
    if len(tracks) < 2:
        return 0
    end_time = pts.time[start + length - 1]
    start_time = pts.time[start]
    starts_by_time = {}
    for t in np.unique(start_time[tracks]):
        segs = tracks[start_time[tracks] == t]
        starts_by_time[t] = (cKDTree(pts.pos[start[segs]]), segs)
    cand = []
    for na in range(2, window + 1):
        ends = tracks[np.minimum(length[tracks], window) == na]
        if not len(ends):
            continue
        wa = _window_indices(start[ends], length[ends], na, from_end=True)
        for step in range(1, max_gap + 2):
            t_start = end_time[ends] + step
            pred, sw = _line_predict(pts.pos[wa], pts.time[wa], t_start.astype(float))
            radius = T * np.linalg.norm(sig) * np.sqrt(sw ** 2 + 1.0)
            if max_distance is not None:
                radius = np.minimum(radius, max_distance)
            for t in np.unique(t_start):
                if t not in starts_by_time:
                    continue
                rows = np.flatnonzero(t_start == t)
                tree, segs = starts_by_time[t]
                for row, hits in zip(rows, tree.query_ball_point(pred[rows], radius[rows])):
                    a = ends[row]
                    for b in segs[hits]:
                        if b == a:
                            continue
                        nb = min(window, length[b])
                        ib = start[b] + np.arange(nb)
                        ia = wa[row]
                        t0 = np.array([0.5 * (pts.time[ia[-1]] + pts.time[ib[0]])], dtype=float)
                        pa, sa = _line_predict(pts.pos[ia][None], pts.time[ia][None].astype(float), t0)
                        pb, sb = _line_predict(pts.pos[ib][None], pts.time[ib][None].astype(float), t0)
                        chi = float(np.linalg.norm((pa[0] - pb[0]) / np.sqrt(sa[0] ** 2 + sb[0] ** 2) / sig))
                        if chi <= T:
                            cand.append((chi, int(a), int(b)))
    cand.sort()
    nxt, used_end, used_start = {}, set(), set()
    for chi, a, b in cand:
        if a in used_end or b in used_start:
            continue
        used_end.add(a)
        used_start.add(b)
        nxt[a] = b
    if not nxt:
        return 0
    head_of = np.arange(len(start))
    for h in set(range(len(start))) - set(nxt.values()):
        k = h
        while k in nxt:
            k = nxt[k]
            head_of[k] = h
    pts.regroup(np.repeat(head_of, length))
    return len(nxt)


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
    (repaired, report) - a list of Trajectory objects (zero velocity; each
    keeps the trajid of its earliest piece), and a dict of counts: n_in,
    n_out, points, links_checked, noise_scale (per axis), threshold,
    cut_threshold, links_cut, points_attached, joins, and ``skipped`` with a
    reason when the data are too short to calibrate.
    """
    report = {"n_in": len(trajs), "links_cut": 0, "points_attached": 0, "joins": 0}
    usable, passthrough = [], []
    for tr in trajs:
        t = np.asarray(tr.time()).reshape(-1)
        if len(t) >= 2 and not np.all(np.diff(t) > 0):
            passthrough.append(tr)
        else:
            usable.append(tr)
    if not usable:
        report.update(n_out=len(trajs), skipped="no trajectories with increasing frame numbers")
        return list(trajs), report

    sid = np.concatenate([np.full(len(np.asarray(tr.time()).reshape(-1)), k) for k, tr in enumerate(usable)])
    time = np.concatenate([np.asarray(tr.time(), dtype=float).reshape(-1) for tr in usable])
    pos = np.concatenate([np.asarray(tr.pos(), dtype=float).reshape(-1, 3) for tr in usable])
    trajid = np.array([int(np.ravel(tr.trajid())[0]) if np.size(tr.trajid()) else k for k, tr in enumerate(usable)])
    pts = _Points(sid, time, pos, trajid)
    report["points"] = int(len(time))

    Z, seg_len = _link_mismatch(pts, window)
    valid = ~np.isnan(Z[:, 0])
    report["links_checked"] = int(valid.sum())
    if valid.sum() < 30:
        report.update(n_out=len(trajs), skipped="fewer than 30 links with enough points on both sides")
        return list(trajs), report
    calib = valid & (seg_len >= long_track)
    if calib.sum() < 1000:  # too few long-track links to calibrate on: use every checkable link
        calib = valid
    sig = _robust_std(Z[calib])
    if not np.all(sig > 0):
        report.update(n_out=len(trajs), skipped="zero noise scale (noise-free or degenerate data)")
        return list(trajs), report
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
    repaired = []
    for s, n in zip(start, length):
        sl = slice(s, s + n)
        p = pts.pos[sl]
        tid = int(trajid[pts.origin[sl]][np.argmin(pts.time[sl])])
        repaired.append(Trajectory(p, np.zeros_like(p), pts.time[sl].astype(np.int64), tid))
    repaired.extend(passthrough)
    report["n_out"] = len(repaired)
    return repaired, report
