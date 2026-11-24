# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""Browser-based 3D viewer for flowtracks datasets (pyvista + trame).

A trame app is a plain Python script: it builds a pyvista scene, binds it
to a browser view, and starts a local server. ``app.server.start()`` opens
the browser tab for you. The view renders client-side with VTK.wasm when
available (no browser plugins, no GPU drivers, no ParaView), falling back
to server-side rendering otherwise.
"""
from __future__ import annotations

import pyvista as pv
from trame.app import TrameApp
from trame.decorators import change
from trame.ui.vuetify3 import VAppLayout
from trame.widgets import vuetify3 as v3
from trame_pyvista.ui import plotter_ui

from .scene import build_polydata, filter_through_frame, load_mesh, load_volume

_IS_WASM = False
try:
    from trame_pyvista.widgets import IS_WASM_SUPPORTED

    _IS_WASM = IS_WASM_SUPPORTED
except ImportError:  # pragma: no cover - trame_pyvista always provides this
    pass


class TrajectoryViewer(TrameApp):
    """Interactive 3D viewer for a :class:`~flowtracks.visualization.scene.TrajectoryScene`.

    One trajectory-lines actor (colored by ``color_by``), an optional
    vessel-geometry mesh, and an optional Eulerian volume; a frame slider
    shows history up to the chosen frame. Camera pan/zoom/rotate happens
    inside the browser view itself.
    """

    def __init__(
        self,
        data,
        *,
        mesh_path=None,
        volume_path=None,
        color_by="speed",
        master_volume=None,
        server=None,
    ):
        super().__init__(server)
        self._data = data
        self._color_by = color_by
        self.state.trame__title = "Flowtracks viewer"

        # Off-screen plotter -- the browser view owns the pixels.
        self.plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
        self._lines_actor = self.plotter.add_mesh(
            build_polydata(data),
            scalars=color_by,
            cmap="viridis",
            line_width=2.0,
            opacity=1.0,
            name="trajectories",
        )
        if volume_path is not None:
            self._volume = load_volume(volume_path)
            self.plotter.add_volume(
                self._volume,
                scalars=_preferred_volume_scalar(self._volume),
                opacity=0.35,
                cmap="viridis",
                show_scalar_bar=True,
                name="eulerian",
            )
        if mesh_path is not None:
            self._mesh = load_mesh(mesh_path)
            self.plotter.add_mesh(
                self._mesh,
                opacity=0.32,
                color="#ded2c0",
                edge_color="#a5907b",
                show_edges=True,
                name="vessel",
            )
        self.plotter.reset_camera()

        with VAppLayout(self.server, full_height=True) as self.ui:
            # Full-featured scene view (camera reset, screenshots, WASM/RCA
            # rendering toggle) from the package, plus our frame slider.
            self.ctrl.view_update = plotter_ui(
                self.plotter,
                mode="client" if _IS_WASM else "server",
            ).update
            v3.VSlider(
                v_model=("frame", float(data.frame_max)),
                min=float(data.frame_min),
                max=float(data.frame_max),
                step=1,
                label="Frame",
                show_value=True,
                style="position:absolute;bottom:1rem;left:2rem;right:2rem;z-index:10;",
            )
        # Seed the tracked state so the slider binds even before a change
        # arrives (the model_value tuple already does this; keep the
        # explicit default for clarity/mixed versions).
        self.state.frame = float(data.frame_max)

    @change("frame")
    def _on_frame(self, frame=None, **kwargs):
        frame = float(frame) if frame is not None else float(self._data.frame_max)
        sub = filter_through_frame(self._data, frame)
        new_poly = build_polydata(sub)
        if self._lines_actor is not None:
            self.plotter.remove_actor(self._lines_actor)
        self._lines_actor = self.plotter.add_mesh(
            new_poly,
            scalars=self._color_by,
            cmap="viridis",
            line_width=2.0,
            opacity=1.0,
            name="trajectories",
        )
        self.plotter.render()


def _preferred_volume_scalar(volume):
    """Pick the first scalar field suitable for a volume colormap."""
    try:
        for name in ("speed", "u", "v", "w", "intensity"):
            if name in volume.point_data or name in volume.cell_data:
                return name
        if len(volume.cell_data):
            return list(volume.cell_data)[0]
        if len(volume.point_data):
            return list(volume.point_data)[0]
    except Exception:
        pass
    return None


def create_app(data, *, mesh_path=None, volume_path=None, color_by="speed", server=None):
    """Build the viewer app for a :class:`TrajectoryScene`."""
    return TrajectoryViewer(
        data,
        mesh_path=mesh_path,
        volume_path=volume_path,
        color_by=color_by,
        server=server,
    )
