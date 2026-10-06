"""select_trajectories / plot_trajectories_3d (PyVista 3D trajectory view)."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from flowtracks.graphics import plot_trajectories_3d, select_trajectories  # noqa: E402
from flowtracks.writers import trajectory_polydata  # noqa: E402


class Tracks:
    """Three tracks: id 7 long + slow, id 3 short + fastest, id 5 medium."""

    def __init__(self):
        rows = []
        for tid, n, speed in ((7, 10, 0.1), (3, 3, 9.0), (5, 5, 1.0)):
            for k in range(n):
                rows.append((tid, k, 0.001 * k, 0.0, 0.0, speed, 0.0, 0.0))
        a = np.array(rows, dtype=float)
        self.trajid = a[:, 0].astype(int)
        self.time = a[:, 1].astype(int)
        self.pos, self.vel = a[:, 2:5], a[:, 5:8]

    def collect(self, keys):
        cols = {"pos": self.pos, "velocity": self.vel, "time": self.time, "trajid": self.trajid}
        return [cols[k] for k in keys]


def test_select_by_length_and_speed():
    src = Tracks()
    assert select_trajectories(src, "length", 2).tolist() == [7, 5]
    assert select_trajectories(src, "speed", 2).tolist() == [3, 5]
    with pytest.raises(ValueError):
        select_trajectories(src, "colour")


def test_polydata_subset_keeps_only_requested_tracks():
    poly = trajectory_polydata(Tracks(), trajids=[3, 5])
    assert sorted(set(poly.point_data["trajid"].tolist())) == [3, 5]
    assert poly.n_points == 8 and poly.n_lines == 2


def test_plot_builds_tracks_points_and_context_without_rendering():
    src = Tracks()
    pl = pv.Plotter(off_screen=True)
    out = plot_trajectories_3d(src, trajids=select_trajectories(src, "speed", 1),
                               context=5, plotter=pl, title="fastest", show=False)
    assert out is pl
    assert {"tracks", "points", "context"} <= set(pl.actors)
    assert pl.mesh.n_points == 3  # last mesh added: the fastest track's points
    pl.close()


class Flow:
    """id 1: long, sits still (a wall point); id 2: shorter, travels far
    back and forth; id 3: short, moves steadily; id 4: steady with ONE
    one-frame spike; id 5: steady with a 2-frame gap (not a jump)."""

    def __init__(self):
        rows = []
        for k in range(200):                                   # wall point
            rows.append((1, k, (0.0, 0.0, 0.0)))
        for k in range(60):                                    # back and forth, 1 unit/frame
            x = k if k < 30 else 60 - k
            rows.append((2, k, (float(x), 5.0, 0.0)))
        for k in range(25):                                    # steady
            rows.append((3, k, (0.2 * k, 10.0, 0.0)))
        for k in range(25):                                    # spike at k=12
            rows.append((4, k, (0.2 * k + (3.0 if k == 12 else 0.0), 15.0, 0.0)))
        for k in [*range(12), *range(14, 26)]:                 # gap: frames 12-13 missing
            rows.append((5, k, (0.2 * k, 20.0, 0.0)))
        self.trajid = np.array([r[0] for r in rows])
        self.time = np.array([r[1] for r in rows])
        self.pos = np.array([r[2] for r in rows], dtype=float)
        self.vel = np.zeros_like(self.pos)

    def collect(self, keys):
        cols = {"pos": self.pos, "velocity": self.vel, "time": self.time, "trajid": self.trajid}
        return [cols[k] for k in keys]


def test_path_ranks_travel_not_duration():
    src = Flow()
    assert select_trajectories(src, "length", 1).tolist() == [1]   # the wall point
    assert select_trajectories(src, "path", 1).tolist() == [2]     # travels 59 units


def test_jumps_finds_the_spike_not_the_gap():
    assert select_trajectories(Flow(), "jumps", 1, min_points=10).tolist() == [4]


def test_min_extent_skips_wall_points():
    for by in ("typical", "coverage", "jumps"):
        ids = select_trajectories(Flow(), by, 5, min_points=10, min_extent=1.0)
        assert 1 not in ids.tolist(), by
        assert len(ids) >= 1


def test_summary_path_and_jump_values():
    from flowtracks.graphics import trajectory_summary

    tab = trajectory_summary(Flow())
    row = dict(zip(tab["trajid"].tolist(), range(len(tab["trajid"]))))
    assert tab["path"][row[2]] == pytest.approx(59.0)
    assert tab["path"][row[1]] == pytest.approx(0.0)
    assert tab["jump"][row[5]] == pytest.approx(0.0, abs=1e-12)   # gap spaced correctly
    # one point displaced by 3 bends two neighbouring steps: 3 + 3/2 = 4.5
    assert tab["jump"][row[4]] == pytest.approx(4.5)


def test_polydata_carries_recorded_units_as_field_data(tmp_path):
    import zarr

    from flowtracks.writers import trajectory_units

    store = tmp_path / "run.zarr"
    root = zarr.open_group(str(store), mode="w")
    grp = root.require_group("trajectories")
    grp.create_array("pos", data=np.zeros((2, 3)))
    grp.create_array("vel", data=np.zeros((2, 3)))
    grp.create_array("time", data=np.zeros(2, dtype=np.int64))
    grp.create_array("trajid", data=np.zeros(2, dtype=np.int64))
    grp.attrs.update({"pos_units": "m", "vel_units": "m s-1", "time_units": "frame"})

    assert trajectory_units(store) == {
        "pos": "m", "vel": "m s-1", "time": "frame",
    }
    poly = trajectory_polydata(store)
    assert poly.field_data["units:pos"] == ["m"]
    assert poly.field_data["units:vel"] == ["m s-1"]


def test_plot_titles_show_recorded_units(tmp_path):
    import zarr

    store = tmp_path / "run.zarr"
    root = zarr.open_group(str(store), mode="w")
    grp = root.require_group("trajectories")
    n = 6
    grp.create_array("pos", data=np.tile([0.01, 0.0, 0.0], (n, 1)))
    grp.create_array("vel", data=np.tile([1.0, 0.0, 0.0], (n, 1)))
    grp.create_array("time", data=np.arange(n, dtype=np.int64))
    grp.create_array("trajid", data=np.ones(n, dtype=np.int64))
    grp.attrs.update({"pos_units": "m", "vel_units": "m s-1", "time_units": "frame"})

    pl = pv.Plotter(off_screen=True)
    plot_trajectories_3d(store, plotter=pl, show=False)
    assert "speed [m s-1]" in dict(pl.scalar_bars)
    pl.close()
