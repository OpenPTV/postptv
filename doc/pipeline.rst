Processing pipeline
===================

Thin per-stage wrapper functions (``ptv_is_to_lagrangian``,
``lagrangian_to_eulerian``, ``phase_average_all_sets``, ...) -- each one
independently runnable and distributable across runs by a cloud
orchestrator. The real logic lives in the xarray modules and example
scripts; nothing is implemented twice here.

.. automodule:: flowtracks.pipeline
   :members:
