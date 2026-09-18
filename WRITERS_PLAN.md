# ParaView writers — unification plan

## Context

VTK/ParaView export exists in **three different forms in this repo already**,
none of which handle trajectories, and each written independently:

| Location | Writes | Library | Time series | Notes |
|---|---|---|---|---|
| `flowtracks/vtk_export.py` | legacy `.vtk` `vtkStructuredGrid` (one file, no series) | raw `vtk` | no | Standalone script, port of `sample_vtkcode.py`. Not imported elsewhere. |
| `flowtracks/eulerian.py: export_vtk` | legacy `.vtk` `vtkStructuredGrid`, one file per phase | raw `vtk` + `vtk.util.numpy_support` | no | Called from `eulerian.run()` when a recipe has a `vtk:` section. |
| `flowtracks/eulerian.py: export_vtk_rectilinear_series` + `_write_pvd` | XML `.vtr` + `.pvd` | raw `vtk` (`vtkXMLRectilinearGridWriter`) | yes | The legacy-script fix — correct format choice, wrong library (manual `vtk.util.numpy_support` marshalling instead of pyvista). |

Outside this repo, the same problem was solved a **third and fourth** time,
independently: `matlab_to_python_3dptv/post_analysis_xr.export_vtk` (legacy
`STRUCTURED_GRID`, no `.pvd`) and `aortic-particle-pipeline/aorta_pipeline/io.py`
+ `eulerian.py` (XML `.vti` + `.vtp` + `.pvd`, via **pyvista** — this is the
one that got the library choice right). None of the four talk to each other.

There is also **no Lagrangian (trajectory) writer anywhere in flowtracks** —
`.vtp` export only exists in `aortic-particle-pipeline/io.py:write_trajectory_vtp`.

## Decision

One module, `flowtracks/writers.py`, replaces all three in-repo writers.
Built on **pyvista**, not raw `vtk` — pyvista already wraps the array
marshalling (`numpy_to_vtk` calls, `SetVoidArray`, manual `Piece`/`Extent`
XML) that every existing writer above hand-rolls, and its `.save(path)`
picks the right XML writer, binary+compressed, from the extension alone.
Grid-type rule (from the VTK-formats research, see chat log / project memory):

- **Lagrangian** (trajectories) → `pyvista.PolyData` with `.lines`
  connectivity per `trajid` → **`.vtp`**. Untracked scattered particles
  (no `trajid` continuity needed) → same `PolyData`, no `.lines`.
- **Eulerian**, uniform spacing per axis → `pyvista.ImageData` → **`.vti`**
  (smaller, faster, native volume/slice/contour filters in ParaView).
- **Eulerian**, non-uniform spacing on any axis → `pyvista.RectilinearGrid`
  → **`.vtr`** (what `export_vtk_rectilinear_series` already chose, kept,
  just re-implemented on pyvista).
- Never legacy `.vtk` (no compression, no time series, ambiguous type) and
  never `.vtu`/parallel `.pvt*` (no unstructured topology, no MPI partitioning
  here).
- **Time series** → one `.pvd` per run via a single shared `write_pvd()`
  (logic already correct in `_write_pvd` / `aortic-particle-pipeline`'s
  version — just needs to be the *one* copy).

## Zarr-native entry points

Every existing writer above takes an in-memory `xr.Dataset` or table that the
*caller* already built. The new module's public functions should also accept
a zarr **path** directly, so a writer call is the entire export step with no
intermediate object the caller has to assemble:

```python
# Lagrangian: reads trajectories/{pos,vel,time,trajid} straight from a
# ZarrScene-shaped store (flowtracks export or openptv2 RunStore — no
# conversion, per CLAUDE.md's ZarrScene note).
write_trajectories_vtp(source: Path | ZarrScene | Scene, out_path: Path,
                        scalar_columns: list[str] = ("time", "speed")) -> Path

# Eulerian: reads an xr.Dataset (in memory or via xr.open_zarr(path) first —
# caller's choice) with (x, y, z, phase|time) dims, as eulerian_grid()
# already produces. Auto-picks .vti vs .vtr per axis-spacing uniformity.
write_eulerian_series(ds: xr.Dataset, out_dir: Path, prefix: str = "phase",
                       time_dim: str = "phase") -> Path   # returns the .pvd path

# Shared time-series wrapper, used by both of the above.
write_pvd(path: Path, entries: list[tuple[float, str]]) -> None

# Convenience: does both parts when a run has both products.
export_run_to_paraview(zarr_path: Path, out_dir: Path,
                        grid_params: dict | None = None) -> dict[str, Path]
```

`write_eulerian_series` subsumes `export_vtk`, `export_vtk_rectilinear_series`
and `matlab_to_python_3dptv`'s `export_vtk` — one function, format chosen
automatically instead of by which script you happened to call.

## Dependency

Extend the existing `vtk` optional extra rather than adding a new one — it's
already import-guarded at call time everywhere:

```toml
[project.optional-dependencies]
vtk = ["vtk", "pyvista"]   # was: vtk = ["vtk"]
```

Keep `vtk` itself optional too (pyvista pulls it in transitively, but pinning
both keeps the extra self-documenting). Existing `importorskip("vtk")` guards
in `tests/test_sample_vtkcode.py` etc. become `importorskip("pyvista")`.

## Migration steps

| Phase | Item | Status |
|---|---|---|
| 1 | Add `flowtracks/writers.py`: `write_pvd`, `write_trajectories_vtp`, `write_eulerian_series`, `export_run_to_paraview` (pyvista-based, zarr-native entry points as above) | **done** |
| 2 | Repoint `eulerian.run()`'s and `pipeline.streamlined_pipeline()`'s `vtk:` recipe stage at `write_eulerian_series` instead of `export_vtk`/`export_vtk_rectilinear_series` | **done** |
| 3 | Delete `flowtracks/vtk_export.py` and the two old functions + `_write_pvd` in `eulerian.py` | **deferred** — still has live callers (`combine.py`'s `post_analysis_xr.export_vtk`, `pipeline.py`'s standalone `export_vtk(config)` step wrapping `vtk_export.main`, `flowtracks/__init__.py`'s public `export_vtk` export) and 4 test files. Both old functions are now docstring-marked deprecated pointing at `writers.py`; delete once those callers migrate. |
| 4 | Update `pyproject.toml`'s `vtk` extra to include `pyvista`; update test import guards (`importorskip("vtk")` → `importorskip("pyvista")` on the tests that now exercise the new writer) | **done** |
| 5 | Add trajectory export tests (`.vtp` polyline structure, point_data arrays) — this is new coverage, no prior test existed | **done** — `tests/test_writers.py`, 6 tests, full suite green (143 passed) |
| 6 | Point `openptv2` at `flowtracks.writers` for any GUI "export to ParaView" action (no reimplementation there) | **done** — see below |
| 7 | Point `openptv-analysis` at `flowtracks.writers`/`flowtracks.eulerian.eulerian_grid`, dropping its own writer code except `matlab_04_vtr_export.py` (kept deliberately — MATLAB byte-parity validation fixture, not a production path) | **done** — see below |

`matlab_to_python_3dptv` and `aortic-particle-pipeline` are **not** touched by
this plan — see "What to learn from aortic-particle-pipeline" below for what
carries over conceptually.

## Implementation notes (what actually landed)

- `flowtracks/writers.py` added: `write_pvd`, `write_trajectories_vtp`,
  `write_eulerian_series` (skips any data variable whose dims aren't exactly
  `{x, y, z}` after selecting a timestep — e.g. per-set fluctuation fields —
  rather than reshaping/truncating into a wrong-length array), and
  `export_run_to_paraview`. Exported from `flowtracks/__init__.py`.
- `openptv2`'s existing "Save Paraview files" GUI action
  (`gui/flowtracks_utils.py:export_ptv_is_to_paraview`) now writes one
  `trajectories.vtp` via `write_trajectories_vtp` when a Zarr run store
  exists (falls back to the legacy per-frame CSV export otherwise, or if
  pyvista isn't installed). This replaces what was previously a *fifth*
  independent, and weakest, export path (per-frame CSV with no native
  ParaView reader or time series) with real polyline trajectories.
  `openptv2/pyproject.toml` gained `[tool.uv.sources] flowtracks = {path =
  "../postptv", editable = true}` (its installed flowtracks was a stale
  PyPI 1.1.1, predating `writers.py`) and `pyvista` in the `gui` extra.
- `openptv-analysis/src/phase_align_pipeline.py` repointed at
  `write_eulerian_series`, replacing its `export_vtk_rectilinear_series`
  import; ran end-to-end against reference experimental data
  (`outputs/020_combined_phase_VTR/`), confirmed `.vti` output (uniform grid)
  readable by pyvista. `pyproject.toml` gained a direct `pyvista` dependency.
- Full `postptv` test suite: 143 passed (2 pre-existing filename/format
  assertions updated to match the new writer's `.vti`/`.pvd` naming).
  `openptv2` GUI suite: 233 passed, 3 skipped.
- Not touched, as planned: `aortic-particle-pipeline`,
  `matlab_to_python_3dptv`, `matlab_04_vtr_export.py`, `roi_select.py`
  migration, `openptv-cloud` (no writer-consuming stage exists there yet).

## Cross-repo consumption

- **`openptv2`** — no new VTK code. Already depends on `flowtracks`
  (editable). A GUI export action calls `flowtracks.writers` directly.
- **`openptv-cloud`** — optional, orchestration-only. It should call
  `flowtracks`'s analysis (binning, filtering, phase-averaging) and writer
  functions server-side *if and when* a pipeline stage needs to hand back
  ParaView-ready files instead of raw zarr; it must not carry its own copy of
  binning or writer logic. Open question, not yet solved: automatic
  multi-run analysis needs filtering + masking (currently the manual
  lasso-ROI step, see below) — cloud orchestration can trigger the writer
  once params exist, but can't safely *choose* a mask/ROI automatically today.
  That's a prerequisite, not part of this writers plan.
- **`openptv-analysis`** and other downstream study repos — call into
  `flowtracks` for the underlying job (binning, filtering, eulerian analysis,
  writers); repo-local code stays limited to marimo/study-specific glue
  (parameter sweeps, comparison against MATLAB reference output, plots).

## Longer-term: consolidate MATLAB→Python analysis into flowtracks

`eulerian.py`/`phase_average.py` already absorbed
`batch_Lagrangian_to_Eulerian.py` and `turbulent_statistics.py` as xarray
re-expressions (see `CLAUDE.md`). The same should eventually happen to the
remaining `matlab_to_python_3dptv` stages — filtering/masking, binning
parameter selection, and any Eulerian derived-field math not yet ported —
so that `flowtracks` is the one place owning "given trajectories, produce the
same physical results as the MATLAB pipeline," and the notebook repos become
thin drivers over it. This writers module is the first concrete step; do not
block it on that larger consolidation.

## ROI selection module (related, separate migration)

`openptv-analysis/src/roi_select.py` (marimo notebook: lasso-select on 3 ortho
projections of zarr trajectory points → `params/roi.json` → filters a zarr
store into a `filtered_trajectories` sibling group) does a job that belongs in
`flowtracks`, not in a study-specific notebook: reading zarr trajectory
points, applying polygon/box masks, and writing a filtered zarr group are all
general-purpose. Split it:

- **`flowtracks/roi.py`** (new, pure functions, no marimo/matplotlib UI
  dependency in the core): `load_zarr_points(...)`, `apply_roi_mask(pos,
  roi_spec)`, `filter_zarr_trajectories(store_path, roi_spec, out_group=...)`,
  `roi_to_json`/`roi_from_json`.
- The marimo notebook (wherever it ends up — `flowtracks_examples` is the
  natural home, matching how `flowtracks_examples` already hosts the
  interactive notebook gallery) becomes a thin UI layer: draws the lasso,
  calls `flowtracks.roi` for the actual point-in-polygon/zarr-filtering work.

This is a separate PR from the writers work above — noted here only because
it was raised in the same conversation and touches the same "what belongs in
flowtracks vs. a downstream repo" boundary.

## What to learn from aortic-particle-pipeline (do not port the repo itself)

- `aorta_pipeline/eulerian.py`: `scipy.stats.binned_statistic_dd` +
  `np.histogramdd` for mean-velocity binning — same technique `eulerian_grid`
  here already uses (histogramdd for sums/counts); no change needed, just
  confirms the existing approach is the modern-idiomatic one.
- `aorta_pipeline/io.py`: the pyvista `.vtp`/`.vti`/`.pvd` writers are the
  reference implementation for `flowtracks/writers.py` above — same
  `poly.lines` construction via `np.flatnonzero` run-length grouping per
  `trajid`, same `grid.cell_data[...]` assignment pattern for `ImageData`.
- `make_paraview_state.py` (`pvpython` script building a `.pvsm` state via
  `paraview.simple`): worth a matching helper once `writers.py` exists, so a
  run's `.pvd`(s) come with a pre-built ParaView state (camera, coloring) —
  not part of this plan's scope, flag as a follow-up.
