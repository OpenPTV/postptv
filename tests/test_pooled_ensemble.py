import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from helpers import FakeScene

from flowtracks import eulerian as eu
from flowtracks.phase_average import phase_average

GRID = {
    "stepx": 2, "stepy": 2, "stepz": 2,
    "min_x": -1.0, "max_x": 1.0,
    "min_y": -1.0, "max_y": 1.0,
    "min_z": -1.0, "max_z": 1.0,
}

# 5 frames, 1 phase bin
BIN_KW = dict(first=100001, last=100005, cycletime=100, deltat=90, base_time=100000)


def _scene(vel, n_particles=1):
    pos = np.zeros((n_particles, 3))
    velocity = np.tile(np.array(vel, dtype=float), (n_particles, 1))
    return FakeScene(pos, velocity, frames=range(100001, 100006))


def test_pooled_matches_count_weighted_per_run_mean():
    """Pooled mean equals count-weighted average of per-run means (normal case)."""
    scenes = {"wp1": _scene((1.0, 0.0, 0.0), n_particles=2),  # 10 samples
              "wp2": _scene((3.0, 0.0, 0.0), n_particles=4)}  # 20 samples
    # per-run (current default method): bin per run, then weighted combine
    ds1 = eu.eulerian_grid(scenes["wp1"], GRID, min_count=1, **BIN_KW)
    ds2 = eu.eulerian_grid(scenes["wp2"], GRID, min_count=1, **BIN_KW)
    combined = xr.concat([ds1, ds2], dim=xr.DataArray(["wp1", "wp2"], dims="set"))
    avg = phase_average(combined, weights=combined[eu.COUNT_VAR])

    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=1, **BIN_KW)

    assert np.allclose(pooled.dataset["u_ins_mean"].values, avg["u_ins_mean"].values, equal_nan=True)
    assert np.allclose(pooled.dataset[eu.COUNT_VAR].values,
                       (ds1[eu.COUNT_VAR] + ds2[eu.COUNT_VAR]).values)
    # bookkeeping: per-run counts present and correct
    assert pooled.run_names == ["wp1", "wp2"]
    assert pooled.counts_per_run.sel(run="wp1").values[1, 1, 1, 0] == 10
    assert pooled.counts_per_run.sel(run="wp2").values[1, 1, 1, 0] == 20
    # collected sample table keeps all particles with run index
    assert pooled.samples["run_index"].size == 30
    assert set(np.unique(pooled.samples["run_index"])) == {0, 1}


def test_pooled_recovers_bin_masked_out_per_run():
    """Edge case: each run is below min_count alone, pooled total is sufficient.

    Current per-run method applies min_count per realization before combining,
    so both runs become NaN/lacking in that bin even though together they have
    enough particles. Pooled method applies min_count on pooled totals.
    """
    # 5 frames * 3 particles = 15 samples each; min_count=20 -> each run masked
    scenes = {"wp1": _scene((1.0, 0.0, 0.0), n_particles=3),
              "wp2": _scene((3.0, 0.0, 0.0), n_particles=3)}

    # per-run (current)
    ds1 = eu.eulerian_grid(scenes["wp1"], GRID, min_count=20, **BIN_KW)
    ds2 = eu.eulerian_grid(scenes["wp2"], GRID, min_count=20, **BIN_KW)
    assert np.isnan(ds1["u_ins_mean"].values[1, 1, 1, 0])
    assert np.isnan(ds2["u_ins_mean"].values[1, 1, 1, 0])
    combined = xr.concat([ds1, ds2], dim=xr.DataArray(["wp1", "wp2"], dims="set"))
    avg = phase_average(combined, weights=combined[eu.COUNT_VAR])
    # both masked -> no valid data survives per-run combine
    assert np.isnan(avg["u_ins_mean"].values[1, 1, 1, 0])

    # pooled (new): 30 samples total >= 20
    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=20, **BIN_KW)
    assert pooled.dataset["u_ins_mean"].values[1, 1, 1, 0] == pytest.approx(2.0)
    assert pooled.dataset[eu.COUNT_VAR].values[1, 1, 1, 0] == 30
    assert pooled.counts_per_run.sel(run="wp1").values[1, 1, 1, 0] == 15
    assert pooled.counts_per_run.sel(run="wp2").values[1, 1, 1, 0] == 15

    # can still compute stats per bin/phase from collected samples
    stats = eu.pooled_bin_statistics(pooled, 1, 1, 1, 0)
    assert stats["count"] == 30
    assert stats["per_run_counts"] == {"wp1": 15, "wp2": 15}
    assert stats["mean"][0] == pytest.approx(2.0)


def test_compare_statistics_per_run_vs_pooled_synthetic():
    """Run both methods on one synthetic case and compare statistics.

    Setup: 2 runs, 1 spatial bin (1,1,1), 1 phase bin.
      - wp1: 3 particles * 5 frames = 15 samples, u=1.0
      - wp2: 2 particles * 5 frames = 10 samples, u=4.0
    Pooled particle samples (all 25) have known mean/variance; per-run
    (current) method stores per-run means+counts and derives run-to-run
    fluctuation stats over the set dimension.
    """
    scenes = {"wp1": _scene((1.0, 0.0, 0.0), n_particles=3),  # 15 samples
              "wp2": _scene((4.0, 0.0, 0.0), n_particles=2)}  # 10 samples

    # --- present (per-run) method ---
    ds1 = eu.eulerian_grid(scenes["wp1"], GRID, min_count=1, **BIN_KW)
    ds2 = eu.eulerian_grid(scenes["wp2"], GRID, min_count=1, **BIN_KW)
    combined = xr.concat([ds1, ds2], dim=xr.DataArray(["wp1", "wp2"], dims="set", name="set"))

    per_run_mean_ds = phase_average(combined, weights=combined[eu.COUNT_VAR])
    fl = combined[["u_ins_mean", "v_ins_mean", "w_ins_mean"]] - per_run_mean_ds
    # run-to-run RMS of per-run mean velocities (as in combine.py: unweighted over set)
    per_run_u_rms = np.sqrt((fl["u_ins_mean"] ** 2).mean("set"))

    # --- proposed (pooled) method ---
    pooled = eu.eulerian_grid_pooled(scenes, GRID, min_count=1, **BIN_KW)
    bstats = eu.pooled_bin_statistics(pooled, 1, 1, 1, 0)

    # expected values from the synthetic samples
    pooled_u = np.array([1.0] * 15 + [4.0] * 10)
    expected_pooled_mean = pooled_u.mean()                          # 2.2
    expected_pooled_var_pop = pooled_u.var(ddof=0)                 # population variance
    per_run_means = np.array([1.0, 4.0])
    expected_per_run_weighted_mean = (1.0 * 15 + 4.0 * 10) / 25   # 2.2
    # deviation of each run's mean from the (count-weighted) ensemble mean
    expected_run_to_run_u_rms = np.sqrt(np.mean((per_run_means - expected_per_run_weighted_mean) ** 2))

    # 1) Mean velocity: methods agree (count-weighted / pooled)
    assert per_run_mean_ds["u_ins_mean"].values[1, 1, 1, 0] == pytest.approx(expected_pooled_mean)
    assert pooled.dataset["u_ins_mean"].values[1, 1, 1, 0] == pytest.approx(expected_pooled_mean)
    assert bstats["mean"][0] == pytest.approx(expected_pooled_mean)

    # 2) Counts/bookkeeping preserved
    assert pooled.dataset[eu.COUNT_VAR].values[1, 1, 1, 0] == 25
    assert bstats["count"] == 25
    assert bstats["per_run_counts"] == {"wp1": 15, "wp2": 10}
    assert pooled.counts_per_run.sel(run="wp1").values[1, 1, 1, 0] == 15
    assert pooled.counts_per_run.sel(run="wp2").values[1, 1, 1, 0] == 10

    # 3) Per-run fluctuation statistic (run-to-run spread of run means)
    assert per_run_u_rms.values[1, 1, 1, 0] == pytest.approx(expected_run_to_run_u_rms)

    # 4) Pooled particle-level statistics (available only from collected samples)
    assert bstats["std"][0] ** 2 == pytest.approx(expected_pooled_var_pop)
    # pooled particle spread differs from run-to-run mean spread (different quantities)
    assert bstats["std"][0] == pytest.approx(np.sqrt(expected_pooled_var_pop))
    assert bstats["std"][0] != pytest.approx(float(per_run_u_rms.values[1, 1, 1, 0]))

    # 5) Collected sample table contains all particles with correct run labels
    assert pooled.samples["run_index"].size == 25
    assert np.sum(pooled.samples["run_index"] == 0) == 15  # wp1
    assert np.sum(pooled.samples["run_index"] == 1) == 10  # wp2
    assert np.allclose(np.sort(pooled.samples["u"]), np.sort(pooled_u))
