"""The one shared phase-bin definition (flowtracks.phasing).

Same names, same rules as openptv-cloud/openptv-analysis: n_phases can
never exceed n_frames_in_period, bins never overlap (phase_width <=
1/n_phases), gaps are allowed. eulerian_grid accepts the new-style
aliases for its three time knobs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from helpers import FakeScene

from flowtracks import eulerian as lag
from flowtracks.phasing import (
    assign_bins,
    bin_of_frame,
    phase_of_frame,
    validate_bins,
)

GRID = {
    'stepx': 2, 'stepy': 2, 'stepz': 2,
    'min_x': -1.0, 'max_x': 1.0,
    'min_y': -1.0, 'max_y': 1.0,
    'min_z': -1.0, 'max_z': 1.0,
}


def test_more_phases_than_period_frames_is_rejected():
    with pytest.raises(SystemExit, match="cannot exceed"):
        validate_bins(5, 20)


def test_phase_width_wider_than_spacing_is_rejected():
    with pytest.raises(SystemExit, match="would overlap"):
        validate_bins(20, 4, 0.3)


def test_narrow_bins_leave_gaps():
    assert assign_bins(np.array([0.0, 0.1, 0.2, 0.3]), 5, 0.05).tolist() == [
        0, -1, 1, -1,
    ]


def test_frame_to_bin_mapping():
    assert phase_of_frame(6, 5, 1) == pytest.approx(0.0)
    assert bin_of_frame(2, 5, 5, None, 1) == 1
    # period 20, 5 bins, narrow width: frame 2 sits mid-bin -> gap
    assert bin_of_frame(2, 5, 20, 0.05, 1) == -1
    assert bin_of_frame(1, 5, 20, 0.05, 1) == 0


def test_eulerian_grid_new_style_aliases_match_old_style():
    def _scene():
        pos = np.zeros((5, 3))
        vel = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
        return FakeScene(pos, vel)

    old = lag.eulerian_grid(
        _scene(), GRID, first=100001, last=100005, cycletime=100,
        deltat=90, base_time=100000, min_count=1,
    )
    new = lag.eulerian_grid(
        _scene(), GRID, first=100001, last=100005,
        n_frames_in_period=100, deltat=90, phase_zero_frame=100000,
        min_count=1, n_phases=old.sizes["phase"],
    )
    np.testing.assert_allclose(
        new["u_ins_mean"].values, old["u_ins_mean"].values, equal_nan=True
    )
    assert new.sizes["phase"] == old.sizes["phase"]


def test_save_zarr_trajectories_records_units(tmp_path):
    import zarr

    from flowtracks.io import save_zarr_trajectories
    from flowtracks.trajectory import Trajectory

    tr = Trajectory(
        pos=np.zeros((3, 3)), velocity=np.ones((3, 3)),
        time=np.array([1, 2, 3]), trajid=7,
    )
    save_zarr_trajectories(
        [tr], tmp_path / "run.zarr",
        units={"pos": "m", "vel": "m s-1", "time": "frame"},
    )
    grp = zarr.open_group(str(tmp_path / "run.zarr"), mode="r")["trajectories"]
    assert grp.attrs["pos_units"] == "m"
    assert grp.attrs["vel_units"] == "m s-1"
    assert grp.attrs["time_units"] == "frame"
