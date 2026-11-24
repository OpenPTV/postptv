# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Alex Liberzon
"""Browser-based 3D viewer for flowtracks datasets (pyvista + trame)."""
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
except ImportError:
    pass


class TrajectoryViewer(TrameApp):
    def __init__(
        self,
        data,
        *,
        mesh_path=None,
        volume_path=None,
        color_by="speed",
        volume_only=False,
        server=None,
    ):
        super().__init__(server)
        self._data = data
        self._color_by = color_by
        self._volume_only = volume_only
        self.state.trame__title = "Flowtracks viewer"

        self.plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))

        if not volume_only:
            self._lines_actor = self.plotter.add_mesh(
                build_polydata(data),
                scalars=color_by,
                cmap="viridis",
                line_width=2.0,
                opacity=1.0,
                show_scalar_bar=True,
                name="trajectories",
            )
            self._frame = float(data.frame_max)
        else:
            self._lines_actor = None

        if volume_path is not None:
            self._volume = load_volume(volume_path)
            self.plotter.add_volume(
                self._volume,
                scalars="speed",
                opacity=0.4,
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
            self.ctrl.view_update = plotter_ui(
                self.plotter,
                mode="client" if _IS_WASM else "server",
            ).update
            if not volume_only:
                v3.VSlider(
                    v_model=("frame", float(data.frame_max)),
                    min=float(data.frame_min),
                    max=float(data.frame_max),
                    step=1,
                    label="Frame",
                    show_value=True,
                    style="position:absolute;bottom:1rem;left:2rem;right:2rem;z-index:10;",
                )

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
            show_scalar_bar=True,
            name="trajectories",
        )
        self.plotter.render()


def _preferred_volume_scalar(volume):
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


def create_app(data, *, mesh_path=None, volume_path=None, color_by="speed", volume_only=False, server=None):
    return TrajectoryViewer(
        data,
        mesh_path=mesh_path,
        volume_path=volume_path,
        color_by=color_by,
        volume_only=volume_only,
        server=server,
    )
