"""Synthetic cases showing where per-run (present) and pooled statistics differ.

Covers: per-run min_count masking/coverage, mean differences when one run is
masked but others contribute, run-to-run mean spread vs pooled particle-level
variance (within-run scatter), and weighting/estimand differences.
"""

import numpy as np
import pytest
import xarray as xr

from helpers import FakeScene

from flowtracks import eulerian as eu
from flowtracks.phase_average import phase_average

GRID = {
    "stepx": 2, "stepy": 2, "stepz": 2,
    "min_x": -1.0, "max_x": 1.0,
    "min_y": -1.0, "max_y": 1.0,
    "min_z": -1.0, "max_z": 1.0,
}

# 5 frames -> one phase bin; all samples land in spatial bin (1,1,1)
BIN_KW = dict(first=100001, last=100005, cycletime=100, deltat=90, base_time=100000)
FRAMES = range(100001, 100006)


def _scene_const(vel, n_particles=1):
    pos = np.zeros((n_particles, 3))
    velocity = np.tile(np.array(vel, dtype=float), (n_particles, 1))
    return FakeScene(pos, velocity, frames=FRAMES)


class VarScene:
    """Scene whose collected samples have explicitly provided velocities."""

    def __init__(self, velocities, frames=FRAMES):
        velocities = np.asarray(velocities, dtype=float)
        self._vel = velocities.reshape(-1, 3)
        self._pos = np.zeros_like(self._vel)
        self._frames = list(frames)

    def collect(self, keys):
        n, m = len(self._frames), self._pos.shape[0]
        cols = {
            "pos": np.tile(self._pos, (n, 1)),
            "velocity": np.tile(self._vel, (n, 1)),
            "time": np.repeat(np.asarray(self._frames), m),
        }
        return [cols[k] for k in keys]


def _per_run_combine(scenes, min_count):
    ds_sets = {}
    for name, scene in scenes.items():
        ds_sets[name] = eu.eulerian_grid(scene, GRID, min_count=min_count, **BIN_KW)
    names = list(ds_sets)
    combined = xr.concat(list(ds_sets.values()), dim=xr.DataArray(names, dims="set", name="set"))
    avg = phase_average(combined, weights=combined[eu.COUNT_VAR])
    fl = combined[["u_ins_mean", "v_ins_mean", "w_ins_mean"]] - avg
    # as in combine.py: RMS of per-run mean deviations, unweighted over set
    u_rms_runmeans = np.sqrt((fl["u_ins_mean"] ** 2).mean("set"))
    return combined, avg, u_rms_runmeans


def _cell(ds, var="u_ins_mean"):
    return ds[var].values[1, 1, 1, 0]


def test_case1_all_runs_below_min_count_pooled_rescues():
    """Each run < min_count alone, pooled total >= min_count: validity/mean differ."""
    # 15 samples each, pooled 30
    scenes = {"wp1": _scene_const((1.0, 0.0, 0.0), n_particles=3),
              "wp2": _scene_const((3.0, 0.0, 0.0), n_particles=3)}
    _, avg, _ = _per_run_combine(scenes, min_count=20)
    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=20, **BIN_KW)

    # present: both per-run bins masked -> no valid ensemble mean
    assert np.isnan(_cell(avg))
    # pooled: total count sufficient -> valid mean
    assert _cell(pooled.dataset) == pytest.approx(2.0)
    assert int(pooled.dataset[eu.COUNT_VAR].values[1, 1, 1, 0]) == 30
    assert pooled.counts_per_run.sel(run="wp1").values[1, 1, 1, 0] == 15
    assert pooled.counts_per_run.sel(run="wp2").values[1, 1, 1, 0] == 15


def test_case2_one_run_masked_mean_and_validity_differ():
    """A run masked per-run is dropped from present combine but included in pooled.

    wp1: 25 samples, u=1 (valid alone). wp2: 5 samples, u=9 (masked per-run).
    Present combine -> run1 only (mean 1). Pooled total 30 -> mean includes wp2.
    """
    scenes = {"wp1": _scene_const((1.0, 0.0, 0.0), n_particles=5),  # 25
              "wp2": _scene_const((9.0, 0.0, 0.0), n_particles=1)}  # 5
    combined, avg, _ = _per_run_combine(scenes, min_count=20)
    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=20, **BIN_KW)

    # present: wp2 masked (NaN, 0 effective count), wp1 survives -> mean of wp1 only
    assert np.isnan(_cell(combined.sel(set="wp2")))
    assert int(combined.sel(set="wp2")[eu.COUNT_VAR].values[1, 1, 1, 0]) == 0
    assert _cell(avg) == pytest.approx(1.0)

    # pooled: both runs' samples included, threshold on pooled total
    expected_pooled = (1.0 * 25 + 9.0 * 5) / 30
    assert _cell(pooled.dataset) == pytest.approx(expected_pooled)
    assert int(pooled.dataset[eu.COUNT_VAR].values[1, 1, 1, 0]) == 30
    assert pooled.counts_per_run.sel(run="wp2").values[1, 1, 1, 0] == 5
    # means differ because inclusion differs
    assert _cell(pooled.dataset) != pytest.approx(_cell(avg))


def test_case3_particle_scatter_pooled_variance_vs_runmean_spread():
    """Within-run particle scatter: pooled particle stats differ from run-mean stats.

    Means (count-weighted) agree, but present RMS measures spread of per-run
    means over set, while pooled can measure variance of all particles.
    """
    # wp1 samples u=[0,2] each frame -> 10 samples, mean 1, particle var 1
    # wp2 samples u=[3,5] each frame -> 10 samples, mean 4, particle var 1
    scenes = {"wp1": VarScene([[0.0, 0, 0], [2.0, 0, 0]]),
              "wp2": VarScene([[3.0, 0, 0], [5.0, 0, 0]])}
    _, avg, u_rms_runmeans = _per_run_combine(scenes, min_count=1)
    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=1, **BIN_KW)
    bstats = eu.pooled_bin_statistics(pooled, 1, 1, 1, 0)

    all_u = np.array([0, 2] * 5 + [3, 5] * 5)
    expected_mean = all_u.mean()  # 2.5
    expected_pooled_var = all_u.var(ddof=0)  # within-run scatter + between-run means
    run_means = np.array([1.0, 4.0])
    expected_runmean_rms = np.sqrt(np.mean((run_means - expected_mean) ** 2))

    assert _cell(avg) == pytest.approx(expected_mean)
    assert bstats["mean"][0] == pytest.approx(expected_mean)

    # present: spread of per-run means (between-run component at mean level)
    assert float(u_rms_runmeans.values[1, 1, 1, 0]) == pytest.approx(expected_runmean_rms)
    # pooled: total particle-level spread (includes within-run scatter)
    assert bstats["std"][0] ** 2 == pytest.approx(expected_pooled_var)
    assert bstats["std"][0] > float(u_rms_runmeans.values[1, 1, 1, 0])


def test_case4_unweighted_vs_count_weighted_between_run_spread():
    """combine.py's run-to-run RMS is unweighted over set; counts can differ.

    Mean uses count weights (matches pooled). But unweighted RMS of run-mean
    deviations differs from count-weighted between-run deviation measure.
    """
    # wp1: 5 samples mean 1, wp2: 20 samples mean 4, ensemble mean 3.4
    scenes = {"wp1": _scene_const((1.0, 0.0, 0.0), n_particles=1),  # 5
              "wp2": _scene_const((4.0, 0.0, 0.0), n_particles=4)}  # 20
    _, avg, u_rms_runmeans_unweighted = _per_run_combine(scenes, min_count=1)
    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=1, **BIN_KW)

    n1, n2 = 5.0, 20.0
    mu1, mu2 = 1.0, 4.0
    mu = (n1 * mu1 + n2 * mu2) / (n1 + n2)  # 3.4
    unweighted_rms = np.sqrt(((mu1 - mu) ** 2 + (mu2 - mu) ** 2) / 2)
    count_weighted_between = np.sqrt((n1 * (mu1 - mu) ** 2 + n2 * (mu2 - mu) ** 2) / (n1 + n2))

    assert _cell(avg) == pytest.approx(mu)
    assert pooled.dataset["u_ins_mean"].values[1, 1, 1, 0] == pytest.approx(mu)

    # present stat as implemented (unweighted over runs) differs from count-weighted one
    assert float(u_rms_runmeans_unweighted.values[1, 1, 1, 0]) == pytest.approx(unweighted_rms)
    assert unweighted_rms != pytest.approx(count_weighted_between)
    # pooled bookkeeping makes count-aware between-run analysis possible
    assert pooled.counts_per_run.sel(run="wp1").values[1, 1, 1, 0] == 5
    assert pooled.counts_per_run.sel(run="wp2").values[1, 1, 1, 0] == 20


def test_case5_no_within_run_scatter_particle_var_equals_between_run_component():
    """With constant samples per run, pooled particle variance comes only from run means.

    (All samples in a run identical -> within-run variance 0, so total pooled
    particle variance equals spread induced by different run means/counts.)
    """
    scenes = {"wp1": _scene_const((1.0, 0.0, 0.0), n_particles=3),  # 15 samples, value 1
              "wp2": _scene_const((4.0, 0.0, 0.0), n_particles=2)}  # 10 samples, value 4
    _, avg, _ = _per_run_combine(scenes, min_count=1)
    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=1, **BIN_KW)
    bstats = eu.pooled_bin_statistics(pooled, 1, 1, 1, 0)

    all_u = np.array([1.0] * 15 + [4.0] * 10)
    assert bstats["mean"][0] == pytest.approx(_cell(avg))
    assert bstats["std"][0] ** 2 == pytest.approx(all_u.var(ddof=0))
    # each run internally constant
    m = pooled.samples
    assert np.std(m["u"][m["run_index"] == 0]) == pytest.approx(0.0)
    assert np.std(m["u"][m["run_index"] == 1]) == pytest.approx(0.0)