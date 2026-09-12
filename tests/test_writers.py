"""Tests for flowtracks/writers.py — see WRITERS_PLAN.md."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("pyvista")

from flowtracks.writers import (
    export_run_to_paraview,
    write_eulerian_series,
    write_pvd,
    write_trajectories_vtp,
)


class FakeScene:
    """Duck-typed ZarrScene/Scene stand-in: 5 trajectories of 4 points each."""

    def __init__(self, seed=0, n_traj=5, n_pts=4):
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(0, 1, (n_traj * n_pts, 3))
        self.vel = rng.normal(0, 1, (n_traj * n_pts, 3))
        self.time = np.tile(np.arange(n_pts), n_traj)
        self.trajid = np.repeat(np.arange(n_traj), n_pts)

    def collect(self, keys):
        cols = {"pos": self.pos, "velocity": self.vel, "time": self.time, "trajid": self.trajid}
        return [cols[k] for k in keys]


def _eulerian_dataset(x, n_phase=2):
    shape = (len(x), 3, 2, n_phase)
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {
            "u_ins_mean": (("x", "y", "z", "phase"), rng.random(shape)),
            "v_ins_mean": (("x", "y", "z", "phase"), rng.random(shape)),
            "w_ins_mean": (("x", "y", "z", "phase"), rng.random(shape)),
            "par_ave2": (("x", "y", "z", "phase"), rng.integers(0, 100, shape)),
        },
        coords={"x": x, "y": np.linspace(0, 2, 3), "z": np.linspace(0, 1, 2),
                "phase": np.arange(n_phase)},
    )


def test_write_pvd_relative_paths(tmp_path):
    f1 = tmp_path / "sub" / "a.vtp"
    f1.parent.mkdir()
    f1.touch()
    pvd = write_pvd(tmp_path / "series.pvd", [(0.0, f1)])
    text = pvd.read_text()
    assert 'file="sub/a.vtp"' in text
    assert 'timestep="0"' in text


def test_write_trajectories_vtp_from_scene(tmp_path):
    import pyvista as pv

    path = write_trajectories_vtp(FakeScene(), tmp_path / "traj.vtp")
    poly = pv.read(path)
    assert poly.n_points == 20
    assert poly.n_lines == 5
    assert set(poly.point_data.keys()) >= {"trajid", "time", "velocity", "speed"}


def test_write_trajectories_vtp_empty(tmp_path):
    class EmptyScene:
        def collect(self, keys):
            return [np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int64)]

    path = write_trajectories_vtp(EmptyScene(), tmp_path / "empty.vtp")
    assert path.exists()


def test_write_eulerian_series_uniform_uses_vti(tmp_path):
    ds = _eulerian_dataset(np.linspace(0, 1, 4))
    pvd = write_eulerian_series(ds, tmp_path / "eul", prefix="phase")
    assert pvd.exists()
    assert (tmp_path / "eul" / "phase_0000.vti").exists()
    assert (tmp_path / "eul" / "phase_0001.vti").exists()


def test_write_eulerian_series_nonuniform_uses_vtr(tmp_path):
    ds = _eulerian_dataset(np.array([0.0, 0.1, 0.5, 1.0]))
    pvd = write_eulerian_series(ds, tmp_path / "eul", prefix="phase")
    assert pvd.exists()
    assert (tmp_path / "eul" / "phase_0000.vtr").exists()


def test_vtp_roundtrip_preserves_per_trajectory_connectivity(tmp_path):
    """What examples/marimo_vtk_writers_preview.py relies on to redraw each
    trajectory as its own line: pyvista's `.lines` cell array, split back
    into per-trajectory point-index runs, must match trajid grouping."""
    import pyvista as pv

    scene = FakeScene(n_traj=5, n_pts=4)
    poly = pv.read(write_trajectories_vtp(scene, tmp_path / "traj.vtp"))

    cells = poly.lines
    offset = 0
    seen_trajids = []
    while offset < len(cells):
        n = cells[offset]
        point_ids = cells[offset + 1: offset + 1 + n]
        trajids = poly.point_data["trajid"][point_ids]
        assert np.all(trajids == trajids[0]), "one polyline must stay within one trajid"
        seen_trajids.append(int(trajids[0]))
        offset += 1 + n
    assert sorted(seen_trajids) == list(range(5))


def test_pvd_time_series_readable_with_pyvista_get_reader(tmp_path):
    """flowtracks.writers' .pvd is a standard ParaView Collection: pyvista's
    generic PVDReader must step through it exactly like ParaView would."""
    import pyvista as pv

    ds = _eulerian_dataset(np.linspace(0, 1, 4), n_phase=3)
    pvd = write_eulerian_series(ds, tmp_path / "eul", prefix="phase", dt=0.5)

    reader = pv.get_reader(str(pvd))
    assert reader.time_values == [0.0, 0.5, 1.0]

    reader.set_active_time_value(0.5)
    block = reader.read()[0]
    expected = ds.sel(phase=1)
    vec = np.stack([expected[v].values.ravel(order="F") for v in
                    ["u_ins_mean", "v_ins_mean", "w_ins_mean"]], axis=-1)
    np.testing.assert_allclose(block.point_data["velocity"], vec, rtol=1e-6)


def test_vtr_roundtrip_preserves_nonuniform_coordinates(tmp_path):
    import pyvista as pv

    x = np.array([0.0, 0.1, 0.5, 1.0])
    ds = _eulerian_dataset(x)
    pvd = write_eulerian_series(ds, tmp_path / "eul", prefix="phase")
    grid = pv.read(pvd.parent / "phase_0000.vtr")

    assert isinstance(grid, pv.RectilinearGrid)
    np.testing.assert_allclose(np.unique(grid.x), x)


def test_export_run_to_paraview_with_eulerian_ds(tmp_path):
    import zarr

    store = tmp_path / "run.zarr"
    scene = FakeScene()
    group = zarr.open_group(str(store), mode="w")
    traj = group.create_group("trajectories")
    traj["pos"] = scene.pos
    traj["vel"] = scene.vel
    traj["time"] = scene.time
    traj["trajid"] = scene.trajid

    ds = _eulerian_dataset(np.linspace(0, 1, 4))
    result = export_run_to_paraview(store, tmp_path / "out", eulerian_ds=ds)
    assert result["trajectories"].exists()
    assert result["eulerian"].exists()
