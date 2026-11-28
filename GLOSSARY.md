# Glossary

> **Note on naming:** this file is a dictionary of terms used across the
> OpenPTV family and in legacy scripts. Several names — `zaman`, `fin`,
> `cycletime`, `deltat`, `base_time`, `par_ave2`, and the legacy `phase_average`
> vs `phase_average_fluctuations` script — are the MATLAB-style legacy
> spellings. The canonical names in the current code are `phase_width`,
> `n_phases`, `n_frames_in_period`, `phase_of_frame`/`bin_of_frame`/`assign_bins`,
> `n_samples`/`n_traj` (see `flowtracks/phasing.py` and
> `docs/eulerian_statistics.md`).

| Term | Definition |
|------|-----------|
| **Trajectory** | A sequence of particle positions/velocities over time, identified by `trajid` |
| **Particle** | A single tracked point in one frame |
| **Frame** | One time step (image) in the sequence |
| **Scene** | A reader backend (HDF5 `Scene` or Zarr `ZarrScene`) providing particle data |
| **Eulerian grid** | Regular (x, y, z, phase) grid of mean velocities from binning particles |
| **Phase** | Position in a periodic cycle (0..N-1), assigned by `ti = ((time - base) % cycletime) // zaman` |
| **Cycle** | One complete period of a periodic flow (e.g., one heartbeat) |
| **Bin** | A single voxel in the Eulerian grid (x, y, z, phase) |
| **Ensemble (per-run)** | Multiple runs binned individually and stacked on a `set` dimension, averaged with count weighting; second moments reflect run-to-run spread of means |
| **Pooled (moments)** | Each run records unmasked per-voxel `n_samples`, `n_traj`, means (`*_m1`), and centred second moments (`*_m2`), combined via Chan et al.'s parallel-variance update (`combine_moments`) to match binning all samples together |
| **n_samples** | Unmasked raw sample count per voxel per phase recorded when `eulerian_grid(..., moments=True)` |
| **n_traj** | Number of distinct trajectories (`trajid`) contributing to a voxel per phase (`moments=True`) |
| **Chunk** | A temporal window of frames tracked independently (parallel tracking) |
| **Shard** | A distributed chunk (one Cloud Run task) |
| **Seam** | Boundary between two chunks where trajectories are stitched |
| **Overlap** | Frames shared between adjacent chunks for context (default 4) |
| **Stitch** | Merging chunk linkages into one trajectory set |
| **Post-process** | Trajectory repair (cut, attach, join) + velocity computation + smoothing |
| **Repair** | `flowtracks.repair.repair_trajectories`: cut bad links, attach singles, join gaps |
| **Smoothing** | Savitzky-Golay filter applied to trajectory positions |
| **Valid voxel** | Voxel with sample count ≥ `min_count` |
| **par_ave2** | Sample count per voxel per phase |
| **zaman** | Phase bin width in frames (`deltat * 2 + 1`) |
| **fin** | Number of phase bins (`ceil(cycletime / zaman)`) |
| **cycletime** | Frames per cycle (e.g., 4286 for 5000 fps, 70 bpm) |
| **deltat** | Half-width of phase bin in frames |
| **base_time** | Frame number where cycle counting starts |
| **min_count** | Minimum samples per voxel to be considered valid |
| **fill_value** | Value written to invalid voxels (default NaN) |
| **VEL_VARS** | Velocity variable names: `u_ins_mean`, `v_ins_mean`, `w_ins_mean` |
| **CF_ATTRS** | Climate and Forecast metadata attributes for Zarr/NetCDF |