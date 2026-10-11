# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""
Merging trajectory datasets of two camera systems that see a common volume.

Two systems (two camera rigs, two recordings of one rig) each produce their
own trajectories in their own coordinate frame. Where their volumes overlap
the same particle appears twice. This module

1. measures the rigid map between the two frames from particles both systems
   see in the same frame (:func:`estimate_rigid_map`): for each candidate
   rotation, the same-frame neighbour differences pile up at the true offset;
   the rotation whose pile is sharpest wins and the offset is refined on tight
   pairs. No sharp pile means no common particles, and it refuses
   (ValueError) rather than guess;
2. maps points of the second system into the first (:func:`apply_rigid`);
3. finds the trajectory pairs that share same-frame points closer than a
   tolerance (:func:`overlap_counts`), each trajectory's best partner
   (:func:`best_partners`) and the duplicates a keep-the-longer rule removes
   (:func:`duplicates`).

Units are whatever the inputs use; tolerances are in the same unit.
"""

import numpy as np
from scipy.spatial import cKDTree


def estimate_rigid_map(pairs, rotations, search_radius=150.0, bin_size=6.0,
                       tight=8.0, min_peak=50, min_ratio=100.0, refine=2,
                       log=print):
    """Rigid map from system B to system A: ``p_A = R @ p_B - offset``.

    Arguments:
    pairs - list of (points_A, points_B) arrays, (n, 3) and (m, 3), each pair
        taken in the same frame (use well-triangulated points only).
    rotations - list of (label, 3x3 matrix) candidates for R.
    search_radius - neighbours of a point of A within this distance of the
        rotated B points vote for the offset.
    bin_size - histogram bin of the votes; the offset is the densest bin.
    tight - pairs closer than this after the first estimate refine it.
    min_peak, min_ratio - the densest bin must hold at least ``min_peak``
        votes and ``min_ratio`` times the mean, or the map is refused.
    refine - number of refinement passes (median of the tight pairs).
    log - callable for progress lines, or None.

    Returns:
    dict with matrix (3x3), offset (3,), label of the chosen rotation,
    n_pairs (tight pairs of the last pass) and spread (median absolute
    deviation of their differences, per axis).
    """
    log = log or (lambda *a, **k: None)
    nbins = int(round(2 * search_radius / bin_size))
    best = None
    for label, M in rotations:
        M = np.asarray(M, dtype=float)
        diffs = []
        for pa, pb in pairs:
            if not len(pa) or not len(pb):
                continue
            pbm = pb @ M.T
            for k, js in enumerate(cKDTree(pbm).query_ball_point(pa, search_radius)):
                if js:
                    diffs.append(pbm[js] - pa[k])
        if not diffs:
            continue
        d = np.vstack(diffs)
        H, edges = np.histogramdd(d, bins=nbins,
                                  range=[(-search_radius, search_radius)] * 3)
        i = np.unravel_index(H.argmax(), H.shape)
        peak = np.array([edges[k][i[k]] + bin_size / 2 for k in range(3)])
        score = H.max() / max(H.mean(), 1e-9)
        log(f"  rotation {label}: densest {bin_size:g} bin {int(H.max())} "
            f"({score:.0f}x the mean)")
        if best is None or H.max() > best[1]:
            best = (M, H.max(), score, peak, label)
    if best is None or best[1] < min_peak or best[2] < min_ratio:
        raise ValueError("no common particles between the two systems")
    M, _, _, off, label = best
    for _ in range(refine):
        dd = []
        for pa, pb in pairs:
            if not len(pa) or not len(pb):
                continue
            q = pb @ M.T - off
            dist, j = cKDTree(q).query(pa, distance_upper_bound=tight)
            ok = np.isfinite(dist)
            dd.append(pb[j[ok]] @ M.T - pa[ok])
        dd = np.vstack(dd)
        off = np.median(dd, axis=0)
    spread = np.median(np.abs(dd - off), axis=0)
    log(f"  B -> A: rotation {label}, offset {np.round(off, 2)}, "
        f"{len(dd)} pairs, spread (MAD) {np.round(spread, 2)}")
    return dict(matrix=M, offset=off, label=label, n_pairs=len(dd), spread=spread)


def apply_rigid(points, matrix, offset=0.0):
    """``points @ matrix.T - offset``: system-B points (or, with offset 0,
    velocities and accelerations) in system A's frame."""
    return np.asarray(points, dtype=float) @ np.asarray(matrix).T - offset


def overlap_counts(tracks_a, tracks_b, tol):
    """Same-frame shared points of every trajectory pair of A and B.

    Arguments:
    tracks_a, tracks_b - mappings id -> mapping with "time" (n,) and "pos"
        (n, 3), both in one frame of reference.
    tol - a point of A and the nearest point of B in the same frame closer
        than this count as shared.

    Returns:
    {(id_a, id_b): number of shared points}
    """
    bframe = {}
    for tid, t in tracks_b.items():
        for f, p in zip(t["time"], t["pos"]):
            bframe.setdefault(int(f), []).append((tid, p))
    btree = {f: (cKDTree(np.array([p for _, p in v])), v) for f, v in bframe.items()}
    shared = {}
    for tid, t in tracks_a.items():
        for f, p in zip(t["time"], t["pos"]):
            if int(f) not in btree:
                continue
            tree, v = btree[int(f)]
            dist, j = tree.query(p, k=1)
            if dist < tol:
                key = (tid, v[j][0])
                shared[key] = shared.get(key, 0) + 1
    return shared


def best_partners(shared):
    """Each trajectory's partner in the other system: the one it shares most
    points with. Returns ({id_a: (id_b, n)}, {id_b: (id_a, n)})."""
    partner_a, partner_b = {}, {}
    for (atid, btid), n in shared.items():
        if n > partner_a.get(atid, (None, 0))[1]:
            partner_a[atid] = (btid, n)
        if n > partner_b.get(btid, (None, 0))[1]:
            partner_b[btid] = (atid, n)
    return partner_a, partner_b


def duplicates(shared, len_a, len_b):
    """Ids of B trajectories that share any point with an A trajectory at
    least as long (``len_a``, ``len_b``: id -> number of points). Dropping
    them keeps one copy of every overlapping particle, the longer one, and
    A on ties."""
    return {btid for (atid, btid) in shared if len_a[atid] >= len_b[btid]}
