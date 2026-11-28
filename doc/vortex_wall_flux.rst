Vortex, Wall Shear & Flux
=========================

Three related BSD-3-Clause modules for fluid-mechanics metrics on Eulerian
velocity fields.

:mod:`flowtracks.vortex`
------------------------

Vortex-identification scalars from a 3-D velocity field: the Q-criterion and
lambda2 (Jeong & Hussain 1995), plus enstrophy — all vectorized over the
grid via batched eigenvalue decomposition.

:mod:`flowtracks.wall_shear`
----------------------------

Near-wall shear stress and cyclic wall-shear indices from a velocity field.
General-purpose: works with any triangulated surface (an explicit wall mesh,
or one extracted with :func:`flowtracks.wall_shear.occupancy_isosurface` when
no explicit mesh exists). TAWSS/OSI/RRT are standard wall-shear indices;
nothing here is specific to one dataset. Normals via scatter-add over faces,
near-wall sampling via one batched grid interpolation, indices via cumulative
sums over phase — no per-vertex or per-phase Python loops.

:func:`flowtracks.wall_shear.vertex_normals` computes per-vertex normals.
:func:`flowtracks.wall_shear.near_wall_shear_stress` computes the shear
stress magnitudes; :func:`flowtracks.wall_shear.tawss_osi_rrt` and
:func:`flowtracks.wall_shear.running_tawss_osi_rrt` turn them into the wall-shear
indices.

:mod:`flowtracks.flux`
----------------------

Volume flow rate through cross-sections of an internal flow (pipe, channel,
vessel). In an incompressible flow the rate through every cross-section of a
rigid conduit is the same at any instant, so flow rate along the conduit is
a truth-free check of the measured velocities. The pipeline, each step
usable on its own:

:func:`flowtracks.flux.fluid_domain` →
:func:`flowtracks.flux.domain_boundary` →
:func:`flowtracks.flux.centerline` →
:func:`flowtracks.flux.section_lumen` →
:func:`flowtracks.flux.section_flux` / :func:`flowtracks.flux.field_flux`.

:mod:`flowtracks.nhist`
------------------------

Normalized histograms. :func:`flowtracks.nhist.nhist_scipy` is the vectorized
normalized histogram; :func:`flowtracks.nhist.nhist` wraps it with plotting
options.

.. automodule:: flowtracks.vortex
   :members:

.. automodule:: flowtracks.wall_shear
   :members:

.. automodule:: flowtracks.flux
   :members:
   :exclude-members: Domain, Centerline, Section

.. automodule:: flowtracks.nhist
   :members:
