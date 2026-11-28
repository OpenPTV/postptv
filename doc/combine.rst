Multi-run combination
=====================

:mod:`flowtracks.combine` combines runs from a directory of subfolders into a
single set of Eulerian statistics. Two methods are supported, selectable with
the ``--method`` CLI flag:

- ``per_run`` — bin each run separately and stack on a ``set`` dimension
  (count-weighted averaging)
- ``pooled`` — combine unmasked per-voxel moments across runs exactly, as if
  all samples were binned together (see ``docs/eulerian_statistics.md``)

See ``docs/eulerian_statistics.md`` in the repository for the full workflow.

The ``postptv-combine`` CLI is the installed entry point:

.. code:: bash

   postptv-combine <data_dir> --method pooled

.. automodule:: flowtracks.combine
   :members:
