# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
import numpy as np
import pytest

from flowtracks.clean import (
    CLEAN_DEFAULTS,
    clean_trajectories,
    kink_split,
    resolve_settings,
    trajectories_from_arrays,
)
from flowtracks.trajectory import Trajectory


def test_resolve_settings_defaults_section_overrides_and_sources():
    s, src = resolve_settings({"fps": 10, "gap1": 6}, {"gap1": 4, "vd1": None},
                              source_name="p.yaml")
    assert s["fps"] == 10.0 and isinstance(s["fps"], float)
    assert (s["gap1"], src["gap1"]) == (4, "override")
    assert (s["vd1"], src["vd1"]) == (CLEAN_DEFAULTS["vd1"], "default")
    assert src["fps"] == "p.yaml"
    assert set(s) == set(CLEAN_DEFAULTS)


def test_resolve_settings_refuses_unknown_keys_and_missing_fps():
    with pytest.raises(ValueError, match="unknown"):
        resolve_settings({"fps": 10, "gapp1": 3})
    with pytest.raises(ValueError, match="fps"):
        resolve_settings({"gap1": 3})


def test_trajectories_from_arrays_groups_orders_and_drops_single_points():
    trajid = np.array([7, 3, 7, 3, 9, 7])
    time = np.array([2, 1, 0, 0, 5, 1])
    pos = np.arange(18.0).reshape(6, 3)
    trajs = trajectories_from_arrays(trajid, time, pos)
    assert [t.trajid() for t in trajs] == [3, 7]
    assert trajs[1].time().tolist() == [0, 1, 2]
    assert np.array_equal(trajs[1].pos(), pos[[2, 5, 0]])


def test_kink_split_cuts_a_fast_reversal_into_fresh_ids():
    n, step = 10, np.array([0.03, 0.0, 0.0])
    v = np.r_[np.tile(step, (n, 1)), np.tile(-step, (n, 1))]
    p = np.cumsum(np.r_[[np.zeros(3)], v[:-1]], axis=0)  # out and straight back
    t = np.arange(2 * n)
    out, ncuts = kink_split([Trajectory(p, v, t, 1)], 90.0, 0.025, fresh_start=100)
    assert ncuts == 1
    assert [o.trajid() for o in out] == [100, 101]
    assert sum(len(o) for o in out) == 2 * n
    slow = Trajectory(p / 10, v / 10, t, 2)
    assert kink_split([slow], 90.0, 0.025)[1] == 0  # short steps: noise, kept


def test_clean_trajectories_stitches_a_broken_track_and_smooths():
    rng = np.random.default_rng(0)
    fps = 10.0
    step = np.array([0.01, 0.0, 0.0])  # 0.1 m/s
    t = np.arange(60)
    truth = np.outer(t, step)
    keep = (t < 25) | (t > 28)  # four missing frames
    pos = truth[keep] + rng.normal(0, 1e-5, (keep.sum(), 3))
    trajid = np.where(t[keep] < 25, 1, 2)
    settings, _ = resolve_settings({"fps": fps})
    out = clean_trajectories(trajid, t[keep], pos, settings, log=None)
    assert len(out) == 1
    tr = out[0]
    assert (tr.time()[0], tr.time()[-1]) == (0, 59)  # the gap is bridged
    assert np.allclose(tr.velocity()[5:-5], step * fps, atol=2e-3)
