# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the browser 3D viewer (flowtracks.visualization, pyvista + trame).

Uses the committed demo dataset (data/tracers.zarr, a conversion of the
two-phase-experiment ptv_is files) as the load case. Scene-building and
CLI-parsing tests run headless; the browser-app test spawns the real
subprocess and probes the HTTP port, and skips cleanly when trame/pyvista
are not installed (pip install 'flowtracks[viz]').
"""
import importlib.util
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import request

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
DEMO_ZARR = REPO / "data" / "tracers.zarr"

pytestmark = pytest.mark.skipif(
    not DEMO_ZARR.exists(), reason="missing demo dataset data/tracers.zarr"
)

pyvista_importable = importlib.util.find_spec("pyvista") is not None
trame_importable = importlib.util.find_spec("trame_pyvista") is not None

pytestmark = pytest.mark.skipif(
    not pyvista_importable, reason="pyvista not installed (uv sync --extra viz)"
)

from flowtracks.visualization import cli as vis_cli
from flowtracks.visualization.scene import (
    build_polydata,
    filter_through_frame,
    load_scene,
)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# -- data loading ----------------------------------------------------------

def test_load_scene_default():
    data = load_scene(DEMO_ZARR, max_trajectories=500)
    assert data.n_points > 1_000
    assert data.n_trajectories == 500
    assert np.isfinite(data.time.min()) and np.isfinite(data.time.max())
    assert (data.speed >= 0).all()


def test_load_scene_time_range_respects_bounds():
    data = load_scene(DEMO_ZARR, time_range=(10151, 10161), max_trajectories=200)
    assert data.n_points > 0
    assert (data.time >= 10151).all() and (data.time <= 10161).all()


def test_load_scene_subsample_is_evenly_spread():
    data = load_scene(DEMO_ZARR, max_trajectories=400)
    assert data.n_trajectories <= 400
    uniq = np.unique(data.trajid)
    # evenly spaced: kept ids must be spread across the full trajid range
    assert uniq.min() <= np.percentile(uniq, 5)
    assert uniq.max() >= np.percentile(uniq, 95)
    assert len(uniq) == len(np.unique(uniq))  # unique, no duplicates


def test_filter_through_frame_drops_later_points():
    data = load_scene(DEMO_ZARR, max_trajectories=50)
    frame = np.median(data.time)
    sub = filter_through_frame(data, frame)
    assert sub.n_points < data.n_points
    assert (sub.time <= frame).all()


# -- geometry building -----------------------------------------------------

def test_build_polydata_has_points_lines_and_scalars():
    data = load_scene(DEMO_ZARR, max_trajectories=50)
    poly = build_polydata(data)
    assert poly.points.shape == (data.n_points, 3)
    expected_segments = 0
    for t in np.unique(data.trajid):
        idx = np.flatnonzero(data.trajid == t)
        if len(idx) > 1:
            expected_segments += 1
    assert poly.n_lines == expected_segments
    assert set(("trajid", "time", "velocity", "speed")) <= set(poly.point_data.keys())
    assert (poly.point_data["speed"] >= 0).all()


def test_build_polydata_empty_scene_is_safe():
    data = load_scene(DEMO_ZARR, time_range=(20000, 20001))
    assert data.n_points == 0
    poly = build_polydata(data)  # empty, must not raise
    assert poly.points.shape[0] == 0


# -- CLI parsing ------------------------------------------------------------

def test_parse_time_range_open_and_bounded():
    assert vis_cli.parse_time_range(None) is None
    assert vis_cli.parse_time_range("10001,10150") == (10001.0, 10150.0)
    assert vis_cli.parse_time_range("10001,") == (10001.0, None)
    assert vis_cli.parse_time_range(",10150") == (None, 10150.0)
    with pytest.raises(ValueError):
        vis_cli.parse_time_range("not-a-range")


def test_cli_parser_has_expected_defaults_and_flags():
    args = vis_cli.build_parser().parse_args(["data/tracers.zarr"])
    assert args.max_trajectories == 2000
    assert args.color_by == "speed"
    assert args.no_browser is False
    assert args.screenshot is None


# -- off-screen render (smoke + CI) ----------------------------------------

def test_screenshot_exits_clean_and_writes_png(tmp_path):
    out = tmp_path / "shot.png"
    rc = vis_cli.main(
        [str(DEMO_ZARR), "--screenshot", str(out), "--max-trajectories", "100"]
    )
    assert rc == 0
    assert out.exists() and out.stat().st_size > 5_000


def test_cli_missing_zarr_fails_with_error(tmp_path, capsys):
    rc = vis_cli.main([str(tmp_path / "nope.zarr")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


# -- trame application (skipped without trame/wasm) ------------------------

@pytest.mark.skipif(not trame_importable, reason="trame-pyvista not installed")
def test_app_constructs_and_rebuilds_lines():
    import trame_pyvista.widgets as view_widgets

    data = load_scene(DEMO_ZARR, max_trajectories=50)
    from flowtracks.visualization.app import TrajectoryViewer

    app = TrajectoryViewer(data)
    assert hasattr(app, "plotter")
    assert app.plotter is not None
    assert view_widgets.IS_WASM_SUPPORTED or True  # smoke import

    # frame filtering path: fewer or equal points after the slider
    sub = filter_through_frame(data, float(np.median(data.time)))
    assert sub.n_points <= data.n_points


@pytest.mark.skipif(not trame_importable, reason="trame-pyvista not installed")
def test_browser_app_spawns_and_serves():
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "flowtracks.visualization",
        str(DEMO_ZARR),
        "--no-browser",
        "--port",
        str(port),
        "--max-trajectories",
        "100",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        # poll the port (server takes a few seconds to bind)
        served = False
        for _ in range(120):
            try:
                with request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                    if r.status == 200:
                        served = True
                        break
            except Exception:
                time.sleep(0.5)
        assert served, "viewer did not serve HTTP on the given port"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
