# Browser 3D Viewer (`flowtracks[viz]`)

`flowtracks[viz]` is a standalone browser app for exploring trajectory data in
3D: a pyvista scene (the same VTK formats ParaView reads) wrapped in a trame
WebSocket view. It exists to answer the question "what does this trajectory
store look like?" without opening a desktop app — and to complement
`flowtracks.writers` (ParaView `.vtp`/`.vti`/`.vtr`) with something
interactive.

```bash
uv sync --extra viz
uv run flowtracks-viz data/tracers.zarr
```

The viewer opens a local browser tab and serves the scene from a background
process. `python -m flowtracks.visualization` is equivalent.

---

## How it is built

Three layers, separated so each can be tested in isolation:

| Layer | Module | Purpose |
|-------|--------|---------|
| Data + geometry | `flowtracks/visualization/scene.py` | numpy/pyvista only: load a Zarr store, decimate trajectories, build the `PolyData` (points + one polyline per `trajid`) |
| App | `flowtracks/visualization/app.py` | trame `TrameApp`: wraps the plotter in the browser view, owns the frame slider |
| Entry points | `flowtracks/visualization/cli.py`, `launch.py` | argparse CLI + a detached `launch(zarr)` for marimo / scripts |

`launch()` (exposed from `flowtracks.visualization`) is the notebook-friendly
path: it spawns the viewer as a **subprocess** and returns a
`subprocess.Popen` handle. marimo is the thin shell; the browser tab owns
the scene. See `examples/marimo_trame_viz.py` for the thin launcher example.

Data contract: `scene.build_polydata()` mirrors
`flowtracks.writers.trajectory_polydata()` exactly — points + one polyline per
`trajid`, with point data `trajid` / `time` / `velocity` / `speed` — so a
scene rendered in the browser and the `.vtp` written for ParaView are
interchangeable by construction.

## CLI

| Flag | Default | What it does |
|------|---------|--------------|
| `zarr` (positional) | — | Path to a `.zarr` store with `trajectories/` (flowtracks export or openptv2 RunStore). |
| `--mesh VTP` | — | Optional vessel/geometry mesh to overlay (`.vtp` / `.vtr` / `.ply`). |
| `--volume VTI` | — | Optional Eulerian scalar-field volume to overlay (`.vti` / `.vtr`). Opacity-mapped; scalar picked from `speed` → `u,v,w,intensity` → first available. |
| `--time-range T0,T1` | all | Time window to keep; either bound may be open (`10001,` / `,10150`). |
| `--max-trajectories N` | `2000` | Keep at most N trajectories, evenly spaced across `trajid` (0 keeps everything). |
| `--color-by {speed,time}` | `speed` | Point-data colormap (`viridis`). |
| `--port PORT` | `0` (OS picks) | Port to serve on. |
| `--no-browser` | off | Do not open the browser tab automatically. |
| `--screenshot PNG` | off | Render off-screen and exit (same scene/shading as the live app, plus a scalar bar). |

## What renders

The **trajectories** actor is one `PolyData` with points + one polyline per
`trajid`, colored by `speed`/`time`. A **vessel mesh** (`--mesh`) is added as
a translucent `PolyData` with edges; an **Eulerian volume** (`--volume`) as an
opacy-mapped voxel actor. Labels: everything is named in `actor_by_name`
(`trajectories`, `vessel`, `eulerian`).

Interaction:

* Orbit / pan / zoom inside the browser view (right-drag orbit, wheel zoom).
* **Frame slider** — shows history up to the chosen frame (`time <= frame`);
  the polyline actor is rebuilt in place.
* Screenshot button — captures the current frame as PNG.

## Performance notes

* The default 2,000-trajectory decimation keeps the browser scene at a
  comfortable size. For a very dense dataset, lower the number or pass an
  explicit `--time-range`.
* Client-side VTK.wasm rendering requires **vtk >= 9.7** (auto-detected).
  Older vtk falls back to server-side rendering automatically — which works
  but streams frames over the websocket and is slower on big scenes.
* `--screenshot` skips the websocket entirely and renders off-screen — this
  is the CI/validation path (`tests/test_visualization.py`).

## Environment notes

* `uv sync --extra viz` installs `vtk`, `pyvista`, and `trame-pyvista`
  (the `viz` extra). `marimo` itself is not required for the viewer — only
  for the launcher example.
* The viewer serves a **local** server and opens a local browser tab; it is
  not designed for multi-user deployment (no auth, no multi-user server).

## Tests

`tests/test_visualization.py` (15 tests) covers: scene building and
subsampling, time-range filtering, empty-scene safety, CLI parsing, the
off-screen screenshot path (including mesh and volume overlays), trame app
construction and the frame-slider rebuild path, and the real subprocess browser
probe. `uv run pytest tests/test_visualization.py`.
