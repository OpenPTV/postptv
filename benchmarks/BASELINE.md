# Benchmarks (SPEEDUP_PLAN Phase 0)

Run with:

```
uv run pytest benchmarks/bench_interp.py benchmarks/bench_io.py benchmarks/bench_pairs.py --benchmark-save=<name>
uv run pytest benchmarks/ --benchmark-compare
```

All RNGs are seeded with `np.random.default_rng(42)`. Numbers are **relative
only** — the working copy lives on a slow filesystem, so compare the ratios
between phases, not absolute times.

## Environment

- Windows / CPython 3.14, pytest-benchmark 5.2.3
- Data: `data/tracers/ptv_is.10001..10115` (115 frames) rebuilt into a temp HDF
  for the `Scene` / `trajectories_table` benchmarks.

## post-opt (after Phase 1 + 2 + 3)

File: `.benchmarks/Windows-CPython-3.14-64bit/0003_post-opt.json`

| Benchmark | Mean (us) | OPS |
|---|---:|---:|
| test_idw_scene | 674.8 | 1481.8 |
| test_scene_iter_frames | 49,167.5 | 20.3 |
| test_particle_pairs | 94,792.4 | 10.5 |
| test_scene_iter_trajectories | 215,875.3 | 4.6 |
| test_trajectories_table | 241,057.8 | 4.1 |
| test_rbf_call | 464,749.2 | 2.2 |
| test_neighb_dists | 595,975.0 | 1.7 |
| test_iter_trajectories_ptvis | 648,307.2 | 1.5 |
| test_idw_call | 658,565.5 | 1.5 |

### Notes on each phase

- **Phase 1.1** `InverseDistanceWeighter.__call__`: removed the per-point
  `matched_data` copy and the Python loop (broadcast `weights @ data`).
- **Phase 1.2** `neighb_dists`: now `dists[use_parts].reshape(...)` instead of a
  per-point Python loop (the O(m·n) `select_neighbs` is kept dense because
  `tests/test_interp.py::test_compare_idw_rbf` pins its exact `np.argsort`
  tie-breaking — see Phase 1.3 note).
- **Phase 1.4** `rbf_interp`: batched `np.linalg.solve` over the leading
  (trajectory) dimension for the boolean-mask path.
- **Phase 1.3** `select_neighbs` KD-tree path: **not applied**. The existing
  test pins the exact neighbour set (including `np.argsort` tie order), so the
  dense return is preserved. The scene-path instance method `_select_neighbs`
  was already KD-tree based.
- **Phase 2** HDF5 readers (`io.trajectories_table`, `Scene.iter_trajectories`,
  `Scene._iter_frame_arrays`, `AnalysedScene`): single read + in-memory group
  by `trajid` / `time` instead of one `read_where` per trajectory/frame. Also
  fixed the latent `query_string = '&'.join(query_string, cond)` TypeError in
  `Scene._iter_frame_arrays`.
- **Phase 3.1** `pairs.particle_pairs`: KD-tree nearest neighbour (was a dense
  `argmin`); dict lookups for trajectory membership; hoisted secondary
  start/end times into `trajectories_in_frame`.
- **Phase 3.4** `smoothing.savitzky_golay`: replaced the per-component
  `np.convolve` loop with a single windowed matmul; dropped deprecated
  `np.mat`. Behaviour verified by `tests/test_smoothing.py` against the old
  loop (within floating point, ~1e-15).
- **Phase 4** text ingest (`io.iter_trajectories_ptvis`): left as-is. The
  numpy `loadtxt` parse and the `iter_trajectories_ptvis` link are not the
  dominant cost once HDF5 iteration is fixed; no numba added.
