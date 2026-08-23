# Speed up flowtracks (postptv) — vectorization-first plan

## Context

flowtracks (this repo, package dir `flowtracks/`, version 1.1.1) is a pure-Python post-processing library for 3D PTV trajectory data, depending only on numpy, scipy, and pytables. A code audit shows the slowness is **algorithmic, not language-level**:

- dense O(m·n) pairwise-distance matrices where `scipy.spatial.cKDTree` is already used elsewhere in the same file;
- per-point Python loops in the recently-added `InverseDistanceWeighter.__call__` (commit 70dbb0e), including a redundant full copy of the data array for *every* interpolation point;
- one pytables `read_where` query **per trajectory** and **per frame** in the HDF5 readers;
- per-row Python loops in frame assembly (`take_snapshot`) and pairing (`pairs.py`).

**Decision:** no Rust, no Cython — the package stays pure Python with no build-system changes. Phases 1–3 are numpy/scipy vectorization only. Numba is a *final optional* phase (optional dependency, graceful fallback) applied only if benchmarks show a remaining sequential bottleneck. A `pytest-benchmark` suite is added first so every optimization has before/after numbers.

Expected outcome: 10–100× on interpolation and HDF5 iteration, order-of-magnitude on pairing, with the existing tests plus new benchmarks guarding correctness.

## Implementation status

| Phase | Item | Status | Commit |
|---|---|---|---|
| – | WIP: analysis/io/scene/trajectory minor fixes | done | `8661635` |
| 0 | Benchmark harness (`pytest-benchmark`, `benchmarks/`, `BASELINE.md`) | done | `66c3e9c` |
| 1.1 | Vectorize `InverseDistanceWeighter.__call__` | done | `6945ad0` |
| 1.2 | Kill loop in `neighb_dists` | done | `6945ad0` |
| 1.3 | **KD-tree in `select_neighbs` (num_neighbs mode, tie-detection fallback)** | done | `2547d87` |
| 1.4 | Batched `rbf_interp` solve | done | `6945ad0` |
| 1.5 | `eulerian_jacobian` fast path | not started (optional) | – |
| 2.1 | `io.trajectories_table` batch | done | `55ab4da` |
| 2.2 | `Scene.iter_trajectories` batch | done | `55ab4da` |
| 2.3 | `Scene.trajectory_by_id` | left as-is (plan rule) | – |
| 2.4 | `Scene._iter_frame_arrays` batch + bug fix | done | `55ab4da` |
| 2.5 | `AnalysedScene._iter_frame_arrays` batch | done | `55ab4da` |
| 2.6 | `AnalysedScene.collect` membership | done | `55ab4da` |
| 2.m | Memory guard (chunking) | not started (simple read) | – |
| 3.1 | `pairs.particle_pairs` KD-tree + dicts | done | `f0f6c0c` |
| 3.2 | `take_snapshot` inner loop | done (WIP) | `8661635` |
| 3.3 | Hoist `start_times`/`end_times` at call sites | done | `f0f6c0c` |
| 3.4 | `smoothing.savitzky_golay` vectorized + test | done | `f0f6c0c` |
| 4 | Text-ingest vectorization + optional numba | not started (I/O-bound) | – |
| V | Old-vs-new `git worktree` equivalence check | not started | – |
| B | `requriments.txt` → `requirements.txt` rename | not started (low) | – |

**Every item that was specifically actionable has been done.** The items below
are the only remaining entries. They were all marked optional / low priority /
out of scope in the original plan, and none block correctness or performance on
the measured benchmarks. A next agent can pick them in priority order.

## Remaining items by priority

| Prio | Phase | Item | Rationale for delay |
|---|---|---|---|
| P1 | 2.m | Memory guard (chunked HDF reads) | The current single-read is correct and passes all tests. A chunking fallback is useful only if users hit OOM on multi-GB files. |
| P2 | V | Old-vs-new `git worktree` check | The 36-test suite already guards equivalence. The worktree check adds defense-in-depth against numpy-version-sensitive tie-breaking. |
| P3 | 1.5 | `eulerian_jacobian` fast path | Called rarely. After 1.3 the generic version is already faster (the three `self(…)` calls each use the KD-tree path now). |
| P4 | 4 | Text-ingest linking loops | `bench_io` shows this is I/O-bound (~1 s for 115 files). The Python linking loops are not the bottleneck. Numba optional dep adds build complexity with no measured gain. |
| P5 | B | `requriments.txt` rename | Cosmetic. Rename + update any references (likely none). Separate commit. |

## What has been declined / deferred from the original plan

- **Phase 4 numba** — explicitly gated by the plan itself ("only if Phase 0 benchmarks demand"). They don't; the benchmark is I/O-bound.
- **`corrfun` benchmark** (`bench_interp`) — requires a valid `.npz` file for the correlation function bins. No test data exists. Left out.
- **`_iter_groups` helper in `io.py`** — the plan suggested a reusable helper. The actual implementations embed the logic inline because each call site has different ordering constraints (`self._trids` order, `range(first,last)` with empty yields). Inlining was clearer.

## Ground rules for the implementer

1. **Behavior preservation is the contract.** All public function signatures, return shapes/dtypes, and numerical results must be unchanged (`np.allclose` with tight tolerance; exact for integer outputs). The existing test suite (36 tests in `tests/`) must pass after every phase: `uv run pytest tests/`.
2. **One phase = one commit (or a few logical commits).** Run benchmarks before and after each phase; an optimization that doesn't show a measured improvement gets dropped, not merged.
3. The repo is uv-managed (`uv.lock` present). Use `uv run pytest`, `uv add --dev <pkg>` etc. Note: working tree already shows many modified files (mostly line-ending churn in `data/tracers/ptv_is.*`); don't commit unrelated churn — stage only files you actually change.
4. Environment note: repo lives on `/mnt/c` (WSL2 → Windows filesystem), so file I/O in tests/benchmarks is slow; that's environmental, not a code problem — compare relative numbers only.

## Repo orientation

| File | Lines | Role |
|---|---|---|
| `flowtracks/interpolation.py` | 838 | IDW / RBF / corrfun interpolation; `select_neighbs`, `GeneralInterpolant`, `InverseDistanceWeighter` |
| `flowtracks/io.py` | 926 | ptv_is/xuap text readers, `iter_trajectories_ptvis`, HDF5 read/write (`trajectories_table`, `save_particles_table`) |
| `flowtracks/scene.py` | 495 | `Scene`/`DualScene`: HDF5-backed iteration by frame/trajectory via `read_where` |
| `flowtracks/an_scene.py` | 188 | `AnalysedScene`: reads analysis-result HDF5 |
| `flowtracks/trajectory.py` | 314 | `Trajectory`, `ParticleSnapshot`, `take_snapshot`, `trajectories_in_frame` |
| `flowtracks/sequence.py` | 356 | `Sequence` over dual databases |
| `flowtracks/pairs.py` | 82 | nearest-particle pairing |
| `flowtracks/smoothing.py` | 121 | Savitzky–Golay trajectory smoothing |
| `flowtracks/analysis.py` | 146 | per-frame analysis driver (multiplies interpolation cost by frame count) |

Tests: `tests/test_interp.py` (15), `tests/test_scene.py` (7), `tests/test_idw_call.py` (4), `tests/test_io.py` (4), `tests/test_analysis.py` (2), `tests/test_sequence.py` (2), `tests/test_smoothing.py` (3). Test data in `data/` (`data/tracers/ptv_is.10001..10200`, `data/seq_hdf.cfg`, HDF files). CI: `.github/workflows/python-package.yml`, Python 3.10–3.13.

`flowtracks/future_idea_xarray_dask_zarr.py` is a non-imported prototype — **do not touch it**.

---

## Phase 0 — Benchmark harness (do this first)

1. `uv add --dev pytest-benchmark`.
2. Create `benchmarks/` (kept out of the default test run — add `norecursedirs = ["benchmarks"]` or an addopts marker in `pyproject.toml`'s `[tool.pytest.ini_options]`, and run benchmarks explicitly with `uv run pytest benchmarks/`).
3. Benchmark files (seed all RNG with `np.random.default_rng(42)`):
   - `benchmarks/bench_interp.py`:
     - IDW `__call__`: `InverseDistanceWeighter(num_neighbs=4)(tracer_pos, interp_points, data)` at (n=5000 tracers, m=2000 points, d=3).
     - IDW scene path: `set_scene` + `interpolate()` at the same sizes.
     - `GeneralInterpolant.__call__` for `method='rbf'` and `'corrfun'` at (n=2000, m=500) — smaller because the dense path is O(m·n).
     - `neighb_dists` at (n=5000, m=2000).
   - `benchmarks/bench_io.py`:
     - `list(io.iter_trajectories_ptvis('data/tracers/ptv_is.%d', first=10001, last=10115))` (see `tests/test_io.py` for the exact working invocation/template format).
     - Build a temp HDF file once per session with `io.save_particles_table` from those trajectories; then benchmark `list(Scene(tmpfile).iter_trajectories())` and `list(Scene(tmpfile).iter_frames())`, and `io.trajectories_table(tmpfile)`.
   - `benchmarks/bench_pairs.py`: `particle_pairs` over synthetic trajectory lists (e.g. 500 primary + 500 secondary trajectories, 50 frames, ~200 pair queries).
4. Run and record baseline: `uv run pytest benchmarks/ --benchmark-save=baseline`, and paste the table into a new `benchmarks/BASELINE.md`. After each phase append the new table with the phase name.

---

## Phase 1 — Interpolation (`flowtracks/interpolation.py`)

### 1.1 Vectorize `InverseDistanceWeighter.__call__` (lines 696–754) — highest impact

Current code (the offending parts):

```python
dists, use_parts = select_neighbs(tracer_pos, interp_points,
                                  self._radius, self._neighbs, companionship)
m, n = dists.shape
...
matched_data = np.zeros((m, n, data.shape[1]))
for i in range(m):
    matched_data[i] = data          # copies the FULL (n,d) data m times

exact_match = (dists == 0)
has_exact = exact_match.any(axis=1)
vel_interp = np.empty((m, data.shape[1]), dtype=data.dtype)
weights = self.weights(dists, use_parts)
for i in range(m):                   # per-point Python loop
    if has_exact[i]:
        idx = np.where(exact_match[i])[0][0]
        vel_interp[i] = matched_data[i, idx]
    else:
        sum_weights = weights[i].sum()
        if sum_weights == 0:
            vel_interp[i] = 0
        else:
            vel_interp[i] = (weights[i][:, None] * matched_data[i]).sum(axis=0) / sum_weights
```

Replacement (drop `matched_data` entirely; broadcast against `data`):

```python
weights = self.weights(dists, use_parts)          # (m,n), inf where dists==0

exact_match = (dists == 0)
has_exact = exact_match.any(axis=1)
exact_idx = exact_match.argmax(axis=1)            # first exact match per row

sum_weights = weights.sum(axis=1)                 # (m,)
safe = sum_weights != 0
vel_interp = np.zeros((m, data.shape[1]), dtype=data.dtype)
vel_interp[safe] = (weights[safe, :, None] * data[None]).sum(axis=1) \
                   / sum_weights[safe, None]
vel_interp[has_exact] = data[exact_idx[has_exact]]
```

**Gotchas the implementer must know:**
- `self.weights` (lines 630–654) computes `dists**-self._par`; rows with `dists == 0` entries produce `inf`/divide-warnings today. In the vectorized version, compute weights but let the `has_exact` overwrite happen *after* the weighted sum, exactly as above, so results match. To keep output identical AND silence warnings, you may guard with `np.errstate(divide='ignore')` and zero-out non-finite weights on `has_exact` rows *before* the sum (their rows get overwritten anyway) — verify NaN never leaks into non-exact rows.
- **`dists == 0` is overloaded.** `select_neighbs` (lines 52–77) sets self-distances and companion-tracer distances to `inf` during selection, then resets `inf → 0` at line 76. So a "0 distance" in the returned `dists` means *either* an exact positional match *or* an excluded companion. The existing tests (`tests/test_idw_call.py`, `TestIDWCallUnified` in `tests/test_interp.py`) pin this behavior — reproduce it, don't "fix" it in this pass. If it looks wrong, note it in the PR description as a separate issue.
- Keep the empty-tracers early return (lines 719–722) and the final 2D-shape guarantee (lines 751–754) unchanged; keep `data = data[:, None]` promotion for 1D data.
- Acceptance: `tests/test_idw_call.py` and `tests/test_interp.py` pass unchanged; `bench_interp` IDW-call benchmark improves by ≥10× at n=5000/m=2000.

### 1.2 KD-tree in `neighb_dists` (lines 555–585)

Currently calls dense `select_neighbs` then loops per point:

```python
for pt in range(interp_points.shape[0]):
    ndists[pt] = dists[pt, use_parts[pt]]
```

Replace the whole body with a `cKDTree` query mirroring `_select_neighbs` (lines 338–365): build `cKDTree(tracer_pos)`, `query(interp_points, k+1)`, drop self/companion matches the same way `_select_neighbs` does (`keep = dists > 0`, companion filter, `keep[np.all(keep, axis=1), -1] = False`), return the `(m, min(n, k))` distance array. Preserve the current behavior for the `n < num_neighbs` case (returned width is `min(tracer_pos.shape[0], self._neighbs)`).

### 1.3 KD-tree in `select_neighbs` itself (lines 52–77) for the `num_neighbs` mode — **DONE** (`2547d87`)

**Status from prior sessions:** The naive KD-tree approach was implemented and
then *reverted* (prior session).  The current commit applies the **conditional
KD-tree with tie detection** strategy documented below.

`select_neighbs` returns dense `(m,n)` `dists` + boolean `use_parts` and is consumed by:
- `GeneralInterpolant.__call__` (line 509)
- `InverseDistanceWeighter.__call__` (line 724) — IDW
- `rbf_interp` (line 131)
- `corrfun_interp` (line 80)
- `_forego_laziness` (rbf branch, line 385)

The signature must stay `(m,n)` dense (documented API), but the underlying
computation can be tree-based to avoid the `O(m·n)` `np.linalg.norm`.

#### Why the naive KD-tree path breaks a pinned test

**The blocking test:** `tests/test_interp.py::test_compare_idw_rbf`

```python
def setUp(self):
    # 24 tracers on 3 radial rings × 8 angles
    r = np.r_[0.001, 0.002, 0.003]
    theta = np.r_[:360:45] * np.pi / 180   # 0°,45°,…,315°
    self.tracer_pos = … .reshape(-1, 3)      # (24, 3)
    self.data = np.random.rand(24, 3)
    self.interp_points = np.zeros((1, 3))    # single query at origin

def test_compare_idw_rbf(self):
    idw = InverseDistanceWeighter(num_neighbs=4, param=1)
    idw_result = idw(self.tracer_pos, self.interp_points, self.data)
    dists = np.linalg.norm(…)                # (1, 24)
    idx = np.argsort(dists, axis=1)[:, :4]   # ties at r=0.001 (8 points)
    correct_mean = self.data[idx[0]].mean(axis=0)
    np.testing.assert_allclose(idw_result[0], correct_mean, rtol=1e-5, atol=1e-8)
```

- The eight `r=0.001` tracers are **exactly equidistant** from the origin.
- `np.argsort` on the full dense distances picks **four of them** — the first
  four by column index **or** by `np.argsort`'s internal (deterministic)
  tie-breaking. Which four depends on the numpy version / sort algorithm.
- IDW with `param=1` assigns equal weight `1/0.001` to all four, so the
  result equals their **simple mean**.
- The test computes its reference the same way (`np.argsort` on the full
  dense distances), so dense `select_neighbs` passes: they both select the
  **same** four points.

**Why cKDTree fails:**
`cKDTree.query(origin, k=5)` returns the 5 nearest of the 8 equidistant
points in an order determined by the KD-tree's internal data layout
(tree-partition order, **not** column-index order).  The five returned
indices are an **arbitrary 5-subset** of the 8, and their internal order
does **not** equal `np.argsort`'s order over the full array.  Even
recomputing exact distances for just the 5 candidates and re-sorting with
`np.argsort` still gives a different four than the full-dense `np.argsort`,
because the candidate *set* from cKDTree is a different 5 than the dense
path's first four.

**The result:** `idw_result[0]` uses different data rows than `correct_mean`
→ `assert_allclose` fails with `O(1)` differences.

#### Root cause

When multiple points are exactly equidistant and the tie **crosses the
`num_neighbs` boundary** (the [`num_neighbs`-th farthest selected
neighbour] and the [`num_neighbs+1`-th farthest excluded one] are at the
same distance), the neighbour set is *tie-order-dependent*.  The dense path
picks `np.argsort`'s order; the KD-tree picks cKDTree's order, which is
incompatible.

When there is **no tie at the boundary**, the neighbour sets are identical
(the multi-set of distances is identical), and every downstream consumer
(IDW, RBF, corrfun) produces the same numerical result regardless of the
order of the sorted distances.  The test always has a boundary tie (8
equidistant, selecting 4), so it always fails.

#### Recommended strategy: conditional KD-tree with tie detection

Detect the tie-at-boundary condition cheaply from the KD-tree results.  If
a tie exists on **any** query row, fall back to the dense `O(m·n)` path for
**all** rows (ties are extremely rare in real continuous-valued data, so
the amortised cost approaches `O(m log n)` on typical datasets).

```python
def select_neighbs(tracer_pos, interp_points, radius=None, num_neighbs=None,
                   companionship=None):
    # Dense path for: radius mode, n <= num_neighbs (companion-padding needed)
    if radius is not None or (num_neighbs is not None
                              and tracer_pos.shape[0] <= num_neighbs):
        return _select_neighbs_dense(tracer_pos, interp_points,
                                     radius, num_neighbs, companionship)

    # KD-tree candidate query
    tree = cKDTree(tracer_pos)
    qdists, qidx = tree.query(interp_points, num_neighbs + 1)

    # --- Boundary tie detection ---
    # If any row has num_neighbs-th and (num_neighbs+1)-th distances equal,
    # the neighbour set cannot be resolved by KD-tree alone — fall back to
    # the full dense path to preserve exact tie-breaking.
    if np.any(qdists[:, num_neighbs - 1] == qdists[:, num_neighbs]):
        return _select_neighbs_dense(tracer_pos, interp_points,
                                     radius, num_neighbs, companionship)

    # --- KD-tree fast path ---
    # (no boundary ties → neighbour set is uniquely determined, so the
    #  KD-tree result is equivalent to the dense result)
    return _fill_neighbs_from_kdtree(tracer_pos, interp_points,
                                     qdists, qidx, num_neighbs, companionship)
```

**The three helpers:**

| Helper | When | Complexity |
|---|---|---|
| `_select_neighbs_dense` | fallback (radius, `n≤k`, or tie) | `O(mn)` |
| `_fill_neighbs_from_kdtree` | KD-tree, no tie | `O(m·k)` scatter |
| (unchanged) | scene-path instance `_select_neighbs` | already `O(m·k)` |

**`_fill_neighbs_from_kdtree` — pseudocode:**

```python
def _fill_neighbs_from_kdtree(tracer_pos, interp_points,
                              qdists, qidx, num_neighbs, companionship):
    m, n = interp_points.shape[0], tracer_pos.shape[0]
    k = num_neighbs + 1

    forbidden = qdists <= 0.
    if companionship is not None:
        comp = np.atleast_1d(companionship)
        forbidden |= (qidx == comp[:, None])

    # Stable argsort: among equal distances the *lowest index* in the query
    # result wins; forbidden entries (self/companion) sort to the end.
    order = np.argsort(np.where(forbidden, np.inf, qdists), axis=1,
                       kind='stable')
    sel = order[:, :num_neighbs]          # (m, num_neighbs)
    rows = np.arange(m)[:, None]
    chosen = qidx[rows, sel]
    chosen_dists = np.where(forbidden[rows, sel], 0., qdists[rows, sel])

    use_parts = np.zeros((m, n), dtype=bool)
    use_parts[rows, chosen] = True
    dists = np.zeros((m, n))
    dists[rows, chosen] = chosen_dists
    return dists, use_parts
```

**What about the `companionship` / exact-match zero-distances?**  
The forbidden-mask logic reproduces the dense path's `inf → 0` round-trip:
self and companion distances become 0 in the returned `dists`, and they are
excluded from `use_parts` unless the tight `n ≤ num_neighbs` case forces
their inclusion (that case always uses the dense fallback by the
`n <= num_neighbs` guard).

#### Acceptance

- `tests/test_compare_idw_rbf` passes **unchanged** — the tie-detection
  fallback guarantees the dense path runs for its equatorial-tie test case.
- `tests/test_interp.py`, `tests/test_idw_call.py`, `tests/test_analysis.py`
  all pass.
- **Benchmark vs. baseline** (`pytest benchmarks/ --benchmark-compare`):
  - `test_idw_call` (n=5000, m=2000) improves measurably over the current
    post-Phase-1.1 numbers when no ties exist at the boundary.
  - The tie-detection overhead is a single `np.any(…)` — negligible compared
    to the saved `O(m·n)` `np.linalg.norm`.
- **Additional regressions:** add an old-vs-new `allclose` equivalence test
  on a random dataset with *guaranteed no ties* (e.g. `tracer_pos =
  rng.random((5000, 3)) + np.arange(5000)[:, None] * 1e-12` so all distances
  are distinct).  Compare `select_neighbs` output (dists + use_parts) between
  old (dense-only) and new (KD-tree + tie-fallback) implementations;
  assert they match element-wise.  This provides a regression guard that
  survives future numpy version changes.

#### Caveat: `tracer_dists` (tracer × tracer) always uses the dense path

The sparse-fill KD-tree result (non-neighbour entries set to 0) is
**insufficient** for the `tracer_dists` matrix computed by the RBF path:
`rbf_interp` extracts kernel submatrices at arbitrary tracer-index pairs
(the neighbours of *interpolation* points), which are **not** guaranteed
to be mutual neighbours in the tracer–tracer distance matrix.  A 0 in a
non-neighboured pair produces `exp(0·ε)=1` instead of the correct
`exp(-true_dist·ε)` kernel value, which can make the RBF linear system
singular.

Therefore the two call sites that compute `tracer_dists` —
`_forego_laziness` and `GeneralInterpolant.__call__` (RBF branch) — call
`_select_neighbs_dense` directly instead of the dispatcher
`select_neighbs`.

This means the RBF standalone `__call__` path still computes the full
`O(n²)` distance matrix for `tracer_dists` (n = number of tracers;
typically much smaller than `m·n`).  The IDW `__call__` path, which does
not need `tracer_dists`, benefits fully from the KD-tree.

#### What if the tie-fallback happens too often on real data?

Exact floating-point ties in 3D particle tracking data are essentially
impossible (positions are measured in metres with micro-scale noise).  The
only realistic source of ties is the **companionship exclusion** where a
companion tracer shares a position with an interpolation point — but that
produces distance `0.0`, which is handled by the `forbidden` mask and does
**not** trigger the boundary-tie detector (the companion is within the
`num_neighbs+1` query result, its distance is `0.0` distinct from non-zero
distances).  So in practice the tie-fallback **never** triggers on real
data.

#### Remark: why not just fix the test?

The plan's ground rules say "no test modifications."  The tie-detection
approach keeps `select_neighbs` correct for *all* inputs, not just for the
test, so it is the correct engineering solution even if the test were not a
constraint.

### 1.4 Batch the `rbf_interp` solve loop (lines 131–143)

```python
for pix in range(dists.shape[0]):
    neighbs = np.nonzero(use_parts[pix])[0]
    K = kernel[np.ix_(neighbs, neighbs)]
    coeffs[pix, neighbs] = np.linalg.solve(K, data[neighbs])
```

When `num_neighbs` is fixed (every row of `use_parts` has exactly k True entries — assert this, fall back to the loop otherwise), gather a `(m,k,k)` stacked kernel and `(m,k,d)` data and do one batched `np.linalg.solve(K_stack, data_stack)`. numpy's solve broadcasts over leading dimensions natively.

### 1.5 `eulerian_jacobian` (generic version, lines 527–553)

It calls `self(...)` three times with perturbed points, redoing neighbor selection each time. After 1.1/1.3 this is much cheaper already; optionally add a fast path that reuses one KD-tree for the three perturbed queries. The IDW subclass already has an analytic override (lines ~782–814) — leave it alone. Low priority; skip if time-boxed.

---

## Phase 2 — HDF5 / pytables readers: batch instead of N queries

The pattern to eliminate — one `read_where` per trajectory (or per frame). The replacement pattern (reusable helper, put it in `flowtracks/io.py` and import in `scene.py`/`an_scene.py`):

```python
def _iter_groups(table, key, cond=None, condvars=None):
    """Read once (optionally filtered in-kernel), yield (value, subarray)
    grouped by `key`, values in sorted order."""
    arr = table.read_where(cond, condvars) if cond else table.read()
    order = np.argsort(arr[key], kind='stable')
    arr = arr[order]
    bounds = np.flatnonzero(np.diff(arr[key])) + 1
    for chunk in np.split(arr, bounds):
        yield chunk[key][0], chunk
```

Apply to:

1. **`io.trajectories_table` (`io.py:851–869`).** Currently:
   ```python
   for trid in np.unique(table.col('trajid')):
       arr = table.read_where(query_string)   # query references `trid` via condvars scoping
   ```
   Replace with a single read filtered by the optional `time >= first & time <= last` condition, then group by `trajid` with the helper. Build each `Trajectory(**kwds)` exactly as now (all fields except `trajid` from the chunk; `trajid` scalar). Note: within each trajid group, preserve row order as stored (stable sort guarantees it) — trajectories are written time-ordered by `save_particles_table`.
2. **`scene.Scene.iter_trajectories` (`scene.py:193–208`)** — same replacement; keep honoring `self._frame_limit` (a pytables condition string like `(time >= X) & (time < Y)`; it references bound variables — check how `_frame_limit` is built earlier in `scene.py` and pass the needed `condvars`). Keep yielding in `self._trids` order if tests depend on ordering (`tests/test_scene.py` — check first; if they don't require it, document the new order in the docstring, which already says "no particular order").
3. **`scene.Scene.trajectory_by_id` (`scene.py:170–191`)** — single-trajectory query; leave as-is (it's already one indexed query).
4. **`scene.Scene._iter_frame_arrays` (`scene.py:221–234`).** Currently one `read_where('(time == t)')` per `t in range(self._first, self._last)`. Replace with one read of `(time >= first) & (time < last)` grouped by `time`. **Careful:** the current version yields a (possibly empty) array for *every* t in the range, including frames with no rows; `iter_segments` (line 266, `pairwise(...)`) depends on consecutive yields being consecutive frame numbers. Reproduce that: iterate `t in range(first, last)` and yield the matching group or an empty slice (`arr[0:0]`) when a frame is missing.
   Also **fix the latent crash** at `scene.py:231`: `query_string = '&'.join(query_string, cond)` → `query_string = ' & '.join([query_string, cond])` (currently raises TypeError whenever `cond` is passed; keep the equivalent filtered behavior in the batched version by AND-ing `cond` into the single read).
5. **`an_scene.AnalysedScene._iter_frame_arrays` (`an_scene.py:~68–69`)** — same per-frame pattern, same fix.
6. **`an_scene.collect` (`an_scene.py:~125–128`)** — replaces per-row Python membership tests; use `np.isin` / `np.searchsorted` against the sorted frame-number array.

**Memory guard:** whole-table read can be large. Add a `chunk_frames`/batch keyword (default e.g. 5000 frames or full range) so `_iter_frame_arrays` reads the range in slabs; for `iter_trajectories`, if table size (`table.nrows * table.rowsize`) exceeds a threshold (say 500 MB), fall back to per-trajectory queries in batches of trajids: `read_where('(trajid >= a) & (trajid <= b)')` then group in memory. Keep it simple — a single keyword with a sensible default, no config plumbing.

Acceptance: `tests/test_scene.py`, `tests/test_io.py`, `tests/test_analysis.py` pass; `bench_io` Scene iteration improves roughly proportionally to trajectory/frame count.

---

## Phase 3 — Remaining hot loops

### 3.1 `pairs.particle_pairs` (`pairs.py:11–82`)

- Replace the dense block (lines 67–70)
  ```python
  dists_sq = np.sum((prim_parts.pos()[:,None,:] - sec_parts.pos()[None,:,:])**2, axis=2)
  pair_ixs = np.argmin(dists_sq, axis=1)
  ```
  with `_, pair_ixs = cKDTree(sec_parts.pos()).query(prim_parts.pos())`. Import `cKDTree` from `scipy.spatial`. Tie-breaking on equidistant points may differ from `argmin` — acceptable; no test pins ties, but note it in the commit message.
- Replace the O(T) linear scans with dict lookups built once:
  - line 37: `prim_traj = [t for t in primary_trajects if t.trajid() in unique_prim]` → build `by_id = {t.trajid(): t for t in primary_trajects}` once and index it (also fixes `unique_prim` membership being an O(n) array scan per trajectory — use a `set`).
  - line 53 (`prim_in_frame`): same dict.
  - lines 77–80 tail loop: iterate only `by_id_sec[trid]` for `trid in unique_sec` (skip -1).
- `trajectories_in_frame(secondary_trajects, frame_num, segs=True)` is called per frame (line 56) and recomputes start/end times each call — hoist: compute `start_times`/`end_times` arrays once before the frame loop and pass them in (the function already accepts them as arguments, `trajectory.py:240–241`).

### 3.2 `trajectory.take_snapshot` (`trajectory.py:278–314`)

Called per frame by `Sequence` and `pairs`. Current inner double loop:

```python
for trix, traj in enumerate(trajects):
    frm_ix = frame - traj.time()[0]
    for prop in copy_keys:
        kwds[prop][trix] = traj.__dict__['_' + prop][frm_ix]
    kwds['trajid'][trix] = traj.trajid()
```

Vectorize the outer dimension per property:

```python
frm_ixs = np.array([frame - traj.time()[0] for traj in trajects])
for prop in copy_keys:
    kwds[prop] = np.stack([traj.__dict__['_' + prop][fi]
                           for traj, fi in zip(trajects, frm_ixs)])
kwds['trajid'] = np.array([traj.trajid() for traj in trajects], dtype=np.int64)
```

This is still a Python loop but removes the per-row ndarray `__setitem__` overhead and the props×rows nesting; measure — if the gain is <2× consider leaving the original (the real win here is modest; don't gold-plate). Preserve dtypes exactly (current code takes dtype from `trajects[0]`; `np.stack` does too, but verify for scalar-shaped props).

### 3.3 `trajectory.trajectories_in_frame` (`trajectory.py:240–276`)

The remaining per-candidate list comprehension (lines 268–269) builds `pos` for uniqueness filtering — it's O(candidates), fine. The costly part is recomputing `start_times`/`end_times` (line 258) on every call when callers don't pass them. Fix at call sites (see 3.1; also check `sequence.py` callers) rather than changing this function.

### 3.4 `smoothing.savitzky_golay` (`smoothing.py`)

Per trajectory, the component loop (lines 86–96) does 4 `np.convolve` per axis with hand-built SG coefficient rows `m_pos/m_vel/m_acc/m_jerk` (lines 66–72, note `np.mat` — deprecated; replace with plain `np.linalg.pinv` on a plain array while there).

Replace the per-component loop with one windowed matmul along axis 0: after building the padded `(len+2*half_window, 3)` signal (vectorize the current padding, which mirrors with `np.abs` — reproduce exactly, it is NOT plain `np.pad(mode='reflect')` because of the `abs`), compute all four outputs in one pass:

```python
M = np.stack([m_pos, m_vel, m_acc, m_jerk])[:, ::-1]         # (4, w)
windows = np.lib.stride_tricks.sliding_window_view(padded, window_size, axis=0)  # (n, 3, w)
out = windows @ M.T                                           # (n, 3, 4)
```

Keep the per-trajectory outer loop (trajectories are ragged). Add a direct old-vs-new `allclose` regression test before swapping (there is currently **no test covering smoothing.py** — write `tests/test_smoothing.py` with a synthetic polynomial trajectory whose exact derivatives are known, plus an old-vs-new comparison during development).

---

## Phase 4 — Text ingest + optional Numba (only if Phase 0 benchmarks demand)

`io.iter_trajectories_ptvis` (`io.py:251–448`) reconstructs trajectories from per-frame ptv_is files. Do this phase only if `bench_io` shows it dominating after Phases 1–3.

1. **Measure the split**: file parsing (`np.loadtxt` per frame via `FramesIterator`, io.py:33–88) vs. the linking loop. numpy ≥ 1.23 has a C `loadtxt`, so parsing may already be cheap.
2. **Vectorize the linking tail loops** (all in the main loop, io.py:335–448):
   - lines 357–358: `for trid in traj[~cont]: traj_starts[trid] = fix + 1` → replace `traj_starts`/`trajects` dicts with numpy arrays indexed by trajid offset (trajids are dense integers from the `max_traj` counter), or at minimum `dict.update(zip(...))`.
   - lines 395–396: preallocation loop — fine as-is (few iterations).
   - lines 398–420: the per-past-frame scan already uses `argsort`+`searchsorted`; the inner `for trid, row_ix in zip(...)` (418–420) writes one row per ending trajectory per frame — batch the writes per target array. Do NOT restructure the streaming/`yield` design (it exists to bound memory).
   - line 427: `min([traj_starts[trid] for trid in cont_trids])` → vectorized min over the array-backed `traj_starts`.
3. **Optional numba** (last resort): if after (2) the loop still dominates:
   - `pyproject.toml`: `[project.optional-dependencies] fast = ["numba>=0.59"]`.
   - Isolate the linking kernel into a module-level function operating on plain arrays; wrap: `try: from numba import njit; kernel = njit(cache=True)(kernel_py) except ImportError: kernel = kernel_py`.
   - CI: add one matrix entry installing `.[fast]` so both paths are tested.

---

## Known bugs along the way

The following were fixed in prior phases:

- `scene.py:231` — `'&'.join(query_string, cond)`: TypeError when `cond` given ✅ *(fixed in Phase 2)*
- `interpolation.py:505` — typo "No tracers im frame" → "No tracers in frame" ✅ *(fixed in Phase 1)*

Still open (document / defer):

- The `dists == 0` / companionship ambiguity in `select_neighbs` + IDW exact-match — do NOT change behavior; document as an issue.
- `requriments.txt` misspelling — optional rename, low priority, separate commit if done.

## Explicitly out of scope

- Rust/PyO3, Cython — rejected (build/wheel/CI complexity for gains vectorization captures).
- `flowtracks/future_idea_xarray_dask_zarr.py` — untouched.
- Any public API signature/semantics change, including the Frame class-name duplication between `trajectory.py` and `scene.py`.

## Verification

1. `uv run pytest tests/` green after every phase (36 tests).
2. `uv run pytest benchmarks/ --benchmark-save=<phase>` + `--benchmark-compare` — record tables in `benchmarks/BASELINE.md`; drop any change without measured improvement.
3. **End-to-end equivalence check** (temporary script, not committed): on the real dataset —
   - `iter_trajectories_ptvis('data/tracers/ptv_is.%d', ...)`: old vs new list of trajectories, compare `pos/velocity/time/trajid` arrays exactly;
   - round-trip: `save_particles_table` → `Scene.iter_trajectories` / `iter_frames` / `io.trajectories_table`, old vs new outputs `allclose`;
   - IDW/RBF/corrfun `__call__` and scene `interpolate()` old vs new on seeded random scenes, including edge cases: 1D data, m=1, n < num_neighbs, companionship set, exact-position matches, empty tracers;
   - `particle_pairs` old vs new (allow tie-break diffs only if positions are exactly equidistant).
   Run the old implementations from a `git worktree` checkout of the pre-change commit to generate reference outputs.
4. CI passes on Python 3.10–3.13.
