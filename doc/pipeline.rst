Pipeline
========

The xarray-based post-processing track: pure ``Dataset → Dataset`` stages
driven by a YAML recipe, chained in the order they are meant to run.
"Post-processing" here means: bin Lagrangian particle data to an Eulerian
grid, phase/ensemble-average it, derive turbulence metrics, and export.

:mod:`flowtracks.eulerian`
--------------------------

:func:`flowtracks.eulerian.eulerian_grid` is the central stage: it bins
``ZarrScene``/``Scene`` particle observations onto a regular (x, y, z, phase)
grid, returning a ``xarray.Dataset`` whose dimensions and coordinates come
from the input data. Companion functions:

- :func:`flowtracks.eulerian.eulerian_windowed` — per-time-window binning
- :func:`flowtracks.eulerian.clean_field` — mask or remove bad cells
- :func:`flowtracks.eulerian.fluid_mask` / :func:`flowtracks.eulerian.qc_mask`
  — plausibility masks
- :func:`flowtracks.eulerian.finite_difference_velocity` — derivatives from a grid
- :func:`flowtracks.eulerian.derived_fields` — MKE, TKE, VSS, ...
- :func:`flowtracks.eulerian.save_zarr` / :func:`flowtracks.eulerian.save_dataset`
  / :func:`flowtracks.eulerian.save_netcdf` — writers

The per-run ensemble and pooled (per-voxel moment) workflows are described
in ``docs/eulerian_statistics.md`` in the repository; the same module also
hosts :func:`flowtracks.eulerian.run_post_analysis_ds`,
:func:`flowtracks.eulerian.run_pooled_analysis_ds`,
:func:`flowtracks.eulerian.combine_moments` and
:func:`flowtracks.eulerian.moment_statistics`.

:mod:`flowtracks.phase_average`
--------------------------------

Phase averaging as pure Dataset → Dataset functions, driven by a YAML recipe:

- :func:`flowtracks.phase_average.phase_average`
- :func:`flowtracks.phase_average.fluctuations`

:mod:`flowtracks.phasing`
-------------------------

Canonical phase-bin definitions shared with the OpenPTV family:
:func:`flowtracks.phasing.validate_bins`,
:func:`flowtracks.phasing.phase_of_frame`, :func:`flowtracks.phasing.bin_of_frame`,
:func:`flowtracks.phasing.assign_bins`.

:mod:`flowtracks.pipeline`
--------------------------

Thin per-stage wrappers, each independently runnable — the unit the
openptv-cloud orchestrator distributes:

- :func:`flowtracks.pipeline.ptv_is_to_lagrangian`
- :func:`flowtracks.pipeline.lagrangian_to_eulerian`
- :func:`flowtracks.pipeline.phase_average_all_sets`
- :func:`flowtracks.pipeline.compute_fluctuations`
- :func:`flowtracks.pipeline.compute_turbulent_statistics`
- :func:`flowtracks.pipeline.streamlined_pipeline`

Each reads its configuration from a YAML recipe.

:mod:`flowtracks.provenance`
----------------------------

:func:`flowtracks.provenance.record_run` and
:func:`flowtracks.provenance.read_runs` record what was run, with which
parameters, and what it produced — entries appended to ``run.zarr``'s
``meta.attrs["runs"]``.

.. automodule:: flowtracks.pipeline
   :members:
