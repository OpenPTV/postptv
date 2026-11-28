Provenance
==========

:mod:`flowtracks.provenance` records what was run, with which parameters, and
what it produced — one JSON-serializable record per run, appended to the
``meta.attrs["runs"]`` of a run's ``run.zarr`` (the same ``meta`` group that
``openptv2``'s run store writes). This exists because a ``run.zarr`` can
otherwise go stale silently: a 5-frame smoke-test run and a full run look
identical from the array shapes alone.

- :func:`flowtracks.provenance.record_run` — append a run record
- :func:`flowtracks.provenance.read_runs` — read the recorded runs

.. automodule:: flowtracks.provenance
   :members:
