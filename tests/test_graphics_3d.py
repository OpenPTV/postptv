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
