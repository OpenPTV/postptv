# Eulerian Statistics: Per-Run vs. Pooled

`flowtracks` provides two ways to compute Eulerian phase-averaged velocity fields and turbulence statistics across multiple realizations (runs / working points):

1. **Per-run (two-stage) statistics** — bin each realization to a masked mean grid, then average across realizations.
2. **Pooled (per-voxel centred-moment) statistics** — record unmasked per-voxel sufficient statistics for each realization, combine them exactly across realizations, and compute means and Reynolds stresses once from the pooled totals.

---

## Conceptual & Mathematical Difference

### 1. Mean velocity and `min_count` masking

- **Per-run**: `min_count` is applied **per realization** before combining. If a voxel has fewer than `min_count` samples in run $s$, that run's voxel is masked to `NaN` (and its count set to `0` in `par_ave2`). Its particles are excluded from the ensemble mean, and if every run is below `min_count` individually, the voxel is invalid even if the combined count across runs is large.
- **Pooled**: each realization keeps unmasked raw counts (`n_samples`), means (`*_m1`), and centred second moments (`*_m2`). `min_count` is applied **once** on the pooled total $N = \sum_s N_s$. When no run is masked out per-run, both methods give the same sample-count-weighted mean.

### 2. Second-order moments (Reynolds stresses, RMS, TKE)

- **Per-run**: computes fluctuations of each realization's mean velocity around the ensemble mean ($\bar{u}_s - \langle u \rangle$). The resulting `u_ins_u_ins`, `u_rms`, and `TKE` measure **run-to-run spread of per-run means** (and are identically zero for a single realization).
- **Pooled**: each realization computes two-pass centred second moments per voxel/phase:
  $$M_{2,s} = \sum_{p \in s} (u_p - \mu_s)(u_p - \mu_s)^T$$
  Using centred moments (rather than raw $\sum u^2 - N\mu^2$) avoids catastrophic cancellation at low turbulence intensity. Across realizations, `combine_moments()` merges them via the exact $k$-group Chan et al. parallel-variance update:
  $$N = \sum_s N_s, \qquad \mu = \frac{1}{N} \sum_s N_s \mu_s, \qquad M_2 = \sum_s \left[ M_{2,s} + N_s (\mu_s - \mu)(\mu_s - \mu)^T \right]$$
  This produces the **exact same statistics as if every particle from every realization had been binned together**, in $O(\text{grid size})$ memory (and works independently per realization, including with `chunk_frames`). `moment_statistics()` then divides by $N - \text{ddof}$ (default `ddof=1`), yielding the **full phase-averaged turbulent Reynolds stresses** (both within-run particle fluctuations and between-run variation).

### 3. Trajectory count (`n_traj`)

Consecutive samples along the same Lagrangian trajectory are correlated. With `moments=True`, `eulerian_grid()` also records `n_traj` (distinct `trajid` count per voxel and phase, deduplicated across `chunk_frames` within a run and summed across runs in `combine_moments()`), which can be used as an effective sample size for uncertainty estimates.

---

## Side-by-Side Comparison

| Feature | Per-run (`per_run`) | Pooled (`pooled`) |
|---|---|---|
| **Binning call** | `eulerian_grid(scene, ...)` | `eulerian_grid(scene, ..., moments=True)` |
| **Combine / stats API** | `phase_average()`, `fluctuations()`, `turbulent_statistics()`, `run_post_analysis_ds()` | `combine_moments()`, `moment_statistics()`, `run_pooled_analysis_ds()` |
| **CLI** | `postptv-combine --method per_run` (default) | `postptv-combine --method pooled` |
| **Where `min_count` applies** | Per realization (before combining) | Once on pooled `n_samples` (requires `n_samples >= max(min_count, ddof + 1)`) |
| **Second moments (`*_ins_*_ins`, `*_rms`, `TKE`)** | Between-run variance of per-run means | Full particle-level Reynolds stresses (within-run + between-run) |
| **Single realization stresses** | Identically `0` | Real in-bin turbulent stresses |
| **Degrees of freedom (`ddof`)** | Fixed (count-weighted / unweighted over `set`) | Configurable, default `ddof=1` (unbiased covariance) |
| **Smoothing (`smoothing_sigma`)** | Supported | Not allowed with `moments=True` (smoothed counts are not sample counts) |
| **Memory per realization** | $O(\text{grid size})$ | $O(\text{grid size})$ (supports `chunk_frames`) |
| **Extra bookkeeping** | `par_ave2`, `valid` | `n_samples`, `n_traj`, `valid`, `*_m1`, `*_m2` |

---

## Usage

### Option 1: Per-run statistics (Python API)

```python
from flowtracks import eulerian as eu

ds_sets = {
    name: eu.eulerian_grid(scene, grid_params, first, last,
                           n_frames_in_period=cycletime, deltat=deltat,
                           min_count=50)
    for name, scene in scenes.items()
}

recipe = {
    "sets": list(ds_sets),
    "shift": 0,
    "mask": [{"method": "count", "min_count": 50}],
    "average": {"weighting": "counts"},
    "derived": {"fields": ["MKE", "TKE", "VEL"], "rho": 1000.0, "mu": 1e-3},
}
out = eu.run_post_analysis_ds(ds_sets, recipe)
```

### Option 2: Pooled statistics (Python API)

#### Recipe-driven (`run_pooled_analysis_ds`)

```python
from flowtracks import eulerian as eu

# Bin each realization independently (can be saved to Zarr and combined later)
ds_sets = {
    name: eu.eulerian_grid(scene, grid_params, first, last,
                           n_frames_in_period=cycletime, deltat=deltat,
                           min_count=0, moments=True)
    for name, scene in scenes.items()
}

recipe = {
    "sets": list(ds_sets),
    "shift": 0,
    "mask": [{"method": "count", "min_count": 50}],  # applied to pooled n_samples
    "derived": {"fields": ["MKE", "TKE", "VEL"], "rho": 1000.0, "mu": 1e-3},
}
out = eu.run_pooled_analysis_ds(ds_sets, recipe, ddof=1)
```

#### Low-level (`combine_moments` + `moment_statistics`)

```python
import xarray as xr
from flowtracks import eulerian as eu

stacked = xr.concat(
    [ds[eu.MOMENT_VARS] for ds in ds_sets.values()],
    dim=xr.DataArray(list(ds_sets), dims="set", name="set"),
    join="exact",
)
pooled = eu.combine_moments(stacked, dim="set")
stats = eu.moment_statistics(pooled, min_count=50, ddof=1)
derived = eu.derived_fields(stats[eu.VEL_VARS], stats,
                            fields=["MKE", "TKE", "VEL"],
                            counts=stats[eu.N_VAR])
```

### CLI (`postptv-combine`)

```bash
# Per-run (default) -> writes post_analysis.zarr + vtk_output/
postptv-combine /path/to/data --method per_run

# Pooled via centred moments -> writes post_analysis_pooled.zarr + vtk_output/
postptv-combine /path/to/data --method pooled
```

### Exporting to ParaView (VTK)

`write_eulerian_series()` preserves `NaN` for invalid voxels by default (`keep_nan=True`) and can group the Reynolds stress components into a single 6-component symmetric VTK tensor (`xx, yy, zz, xy, yz, xz`):

```python
from flowtracks.writers import write_eulerian_series

reynolds_tensor = [
    "u_ins_u_ins", "v_ins_v_ins", "w_ins_w_ins",
    "u_ins_v_ins", "v_ins_w_ins", "u_ins_w_ins",
]
write_eulerian_series(
    out,
    out_dir="vtk_output",
    tensors={"reynolds_stress": reynolds_tensor},
)
```

---

## When to Choose Which

- **Use `pooled`** when you want true phase-averaged **turbulent Reynolds stresses / TKE** from all particle fluctuations, when individual realizations are too sparse to pass `min_count` alone, or when you only have one realization.
- **Use `per_run`** when you specifically want to measure **repeatability / run-to-run variance** of per-realization mean fields, or when reproducing legacy two-stage pipelines (including `smoothing_sigma`).
