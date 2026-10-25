# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""
Cleaning chain for raw tracker output: repair, kink split, smoothing, stitching.

Raw linkage from a tracker (one row per point, with a trajectory id) is turned
into smooth trajectories in one call, :func:`clean_trajectories`:

1. repair (:func:`flowtracks.repair.repair_arrays`): cut wrong links, attach
   single points, join pieces across short gaps;
2. Savitzky-Golay smoothing (:func:`flowtracks.smoothing.savitzky_golay`);
3. kink split: a fast, violent turn (larger than ``kink_deg`` on steps longer
   than ``kink_step_mm``) is a wrong link the repair missed; the pieces get
   fresh ids;
4. smoothing, a first stitch (``gap1``, ``dist1``, ``vd1``), smoothing;
5. a second kink split, smoothing, a second, tighter stitch (``gap2``,
   ``dist2``, ``vd2``);
6. final smoothing (optionally a longer window for long trajectories).

Positions are in metres, so velocities are in m/s at the given frame rate.
Every setting has a default in :data:`CLEAN_DEFAULTS`; ``fps`` has none on
purpose -- the frame rate must be stated. :func:`resolve_settings` merges a
parameter-file section and overrides onto the defaults and reports where each
value came from.
"""

import numpy as np

from flowtracks.repair import repair_arrays
from flowtracks.smoothing import savitzky_golay, smoothing_spline
from flowtracks.stitching import stitch_trajectories
from flowtracks.trajectory import Trajectory

#: Default settings of :func:`clean_trajectories` (positions in metres).
CLEAN_DEFAULTS = dict(
    fps=None,  # frame rate [Hz]; required
    repair_max_gap=1,  # repair: max frame gap for attach / join,
    repair_max_distance_mm=0.3,  # max distance [mm],
    repair_link_percentile=99.9,  # link check calibrated on this percentile,
    repair_cut_factor=2.0,  # cut threshold factor,
    repair_cut=True,  # and which of the three repairs run
    repair_attach=True,
    repair_join=True,
    sg_window=7,  # Savitzky-Golay window / order
    sg_order=2,
    smooth_method="spline",  # "spline" (default) or "savitzky_golay"
    spline_lam=None,  # smoothing-spline lam (None -> GCV)
    kink_deg=90.0,  # kink split: turn angle ...
    kink_step_mm=25.0,  # ... on steps longer than this
    gap1=12,  # first stitch: max gap [frames],
    dist1=0.05,  #   max distance [m],
    vd1=0.5,  #   max velocity difference [m/s]
    gap2=8,  # second (tight) stitch on clean ends
    dist2=0.025,
    vd2=0.35,
    win_long=0,  # >0: final smoothing (win_long, order_long)
    order_long=3,  #   for tracks with >= len_long points
    len_long=25,
)


def resolve_settings(section=None, overrides=None, source_name="parameter file"):
    """Settings of :func:`clean_trajectories`: defaults <- section <- overrides.

    Arguments:
    section - mapping read from a parameter file (may be None).
    overrides - mapping of explicit values (e.g. command-line flags); None
        values are ignored.
    source_name - how ``section`` is named in the returned sources.

    Returns:
    (settings, sources) - two dicts keyed like :data:`CLEAN_DEFAULTS`; sources
    say "override", ``source_name`` or "default" for every key.

    Raises ValueError on an unknown key in ``section`` or a missing ``fps``.
    """
    section = dict(section or {})
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = sorted(set(section) - set(CLEAN_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown cleaning settings: {unknown}")
    settings, sources = {}, {}
    for k, v in CLEAN_DEFAULTS.items():
        cast = float if k == "fps" else type(v)
        if k in overrides:
            settings[k], sources[k] = cast(overrides[k]), "override"
        elif k in section:
            settings[k], sources[k] = cast(section[k]), source_name
        else:
            settings[k], sources[k] = v, "default"
    if settings["fps"] is None:
        raise ValueError("fps (frame rate) must be given")
    return settings, sources


def trajectories_from_arrays(trajid, time, pos):
    """Trajectory objects (zero velocity) from flat one-row-per-point arrays.

    Rows are grouped by ``trajid`` and ordered by ``time`` inside each group;
    groups of a single point are dropped.
    """
    trajid, time, pos = np.asarray(trajid), np.asarray(time), np.asarray(pos)
    order = np.argsort(trajid, kind="stable")
    tj, tm, pp = trajid[order], time[order], pos[order]
    bounds = np.flatnonzero(np.diff(tj)) + 1
    starts = np.r_[0, bounds]
    out = []
    for i, s in enumerate(starts):
        e = int(bounds[i]) if i < len(bounds) else len(tj)
        if e - s >= 2:
            o = np.argsort(tm[s:e], kind="stable")
            out.append(
                Trajectory(
                    pp[s:e][o], np.zeros_like(pp[s:e][o]), tm[s:e][o], int(tj[s])
                )
            )
    return out


def kink_split(trajs, max_deg, step_min, fresh_start=10_000_000):
    """Split trajectories at fast, violent turns.

    A point is a cut point when the velocity turns by more than ``max_deg``
    degrees there and both the step into it and the step out of it are longer
    than ``step_min`` (position units). Every piece of a cut trajectory gets a
    fresh id counting from ``fresh_start``, so readers grouping by id cannot
    put the pieces back together.

    Returns:
    (trajectories, number of cuts)
    """
    out, ncuts, nxt = [], 0, fresh_start
    for tr in trajs:
        v = tr.velocity()
        n = np.linalg.norm(v, axis=1)
        t, p = np.asarray(tr.time()), tr.pos()
        st = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ok = (n[:-1] > 1e-12) & (n[1:] > 1e-12)
        cosang = np.ones(len(v) - 1)
        cosang[ok] = np.clip(
            (v[:-1][ok] * v[1:][ok]).sum(1) / (n[:-1][ok] * n[1:][ok]), -1, 1
        )
        ang = np.degrees(np.arccos(cosang))
        instep = np.concatenate([st[:1], st[:-1]])
        bad = (
            np.flatnonzero((ang > max_deg) & (instep > step_min) & (st > step_min)) + 1
        )
        ncuts += len(bad)
        segs = np.split(np.arange(len(t)), bad)
        if len(segs) == 1:
            out.append(tr)
            continue
        for seg in segs:
            if len(seg) >= 2:
                out.append(Trajectory(p[seg], v[seg], t[seg], nxt))
                nxt += 1
    return out, ncuts


def clean_trajectories(trajid, time, pos, settings, log=print):
    """Run the cleaning chain (module docstring) on flat tracker output.

    Arguments:
    trajid, time, pos - (N,), (N,), (N, 3) one row per point, pos in metres.
    settings - complete settings, e.g. from :func:`resolve_settings`.
    log - callable for progress lines, or None for silence.

    Returns:
    list of smoothed :class:`~flowtracks.trajectory.Trajectory` objects.
    """
    log = log or (lambda *a, **k: None)
    c = settings
    fps, sgw, sgo = c["fps"], c["sg_window"], c["sg_order"]
    smooth_method = c.get("smooth_method", "spline")
    spline_lam = c.get("spline_lam", None)

    def _smooth(trajs, window=None, order=None):
        if smooth_method == "savitzky_golay":
            w = window if window is not None else sgw
            o = order if order is not None else sgo
            return savitzky_golay(trajs, fps, w, o, min_window=5)
        return smoothing_spline(trajs, fps, lam=spline_lam)

    tj, rep = repair_arrays(
        trajid,
        time,
        pos,
        max_gap=c["repair_max_gap"],
        max_distance=c["repair_max_distance_mm"] * 1e-3,
        link_percentile=c["repair_link_percentile"],
        cut=c["repair_cut"],
        attach=c["repair_attach"],
        join=c["repair_join"],
        cut_factor=c["repair_cut_factor"],
    )
    log(f"repair: cut={rep.get('links_cut')} joins={rep.get('joins')}")
    s1 = _smooth(trajectories_from_arrays(tj, time, pos))
    split, ncuts = kink_split(s1, c["kink_deg"], c["kink_step_mm"] / 1000.0)
    log(
        f"kinksplit(>{c['kink_deg']}deg, steps>{c['kink_step_mm']}mm): "
        f"{len(s1)} -> {len(split)} objs, {ncuts} cuts"
    )
    s2 = _smooth(split)
    st = stitch_trajectories(
        s2, fps=fps, max_gap=c["gap1"], max_distance=c["dist1"], max_vel_diff=c["vd1"]
    )
    s2b = _smooth(st)
    split2, _ = kink_split(
        s2b, c["kink_deg"], c["kink_step_mm"] / 1000.0, fresh_start=20_000_000
    )
    s2c = _smooth(split2)
    st2 = stitch_trajectories(
        s2c, fps=fps, max_gap=c["gap2"], max_distance=c["dist2"], max_vel_diff=c["vd2"]
    )
    if c["win_long"] > 0 and smooth_method == "savitzky_golay":
        short = [t for t in st2 if len(t) < c["len_long"]]
        long = [t for t in st2 if len(t) >= c["len_long"]]
        s_short = _smooth(short)
        s_long = savitzky_golay(long, fps, c["win_long"], c["order_long"], min_window=7)
        out = s_short + s_long
        log(
            f"two-tier final smooth: {len(s_short)}x({sgw},{sgo}) + {len(s_long)}x"
            f"({c['win_long']},{c['order_long']})"
        )
    else:
        out = _smooth(st2)
    lens = np.array([len(t) for t in out]) if out else np.zeros(1, int)
    log(
        f"final: n={len(out)} joined1={len(s2) - len(st)} "
        f"joined2={len(s2c) - len(st2)} med={np.median(lens):.0f} "
        f"max={lens.max()} rows={lens.sum()}"
    )
    return out
