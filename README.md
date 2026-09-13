
Flowtracks - postprocessing of 3D-PTV data
==========================================

This package contains Flowtracks, a Python-package for post-processing
of 3D Particle Tracking Velocimetry particle/trajectory databases.

Latest release: https://pypi.org/project/flowtracks/ (flowtracks 1.2.2)

The full documentation for this package may be built from the Sphinx 
sources in the doc/ directory. It is also available online:

  http://flowtracks.readthedocs.org

Please refer to that documentation for the full information on installing,
reference documentation and usage examples contained in the package.

The program is distributed under the terms of the GNU General Public 
License, version 3.0. For details, see the LICENSE.txt file.

How to cite this work
=====================

Meller, Y and Liberzon, A 2016 Particle Data Management Software for 3DParticle Tracking Velocimetry and Related Applications – The Flowtracks Package. Journal of Open Research Software, 4: e23, DOI: <http://dx.doi.org/10.5334/jors.101> 


Getting Started
===============

Obtaining the package and its dependencies
------------------------------------------

The most recent version of this package may be found under the auspices of
the OpenPTV project, in its Github repository,

  https://github.com/OpenPTV/postptv

Dependencies:

* The software depends on the SciPy package, obtainable from 
  http://www.scipy.org/

* Some features depend on the Matplotlib package. Users which need those 
  features may get Matplotlib at http://matplotlib.org/

Installation
-----------

For users (no source checkout needed) — install the latest release
from PyPI (https://pypi.org/project/flowtracks/):

    pip install flowtracks

With ParaView export support (writes .vti/.vtr/.pvd via pyvista/vtk):

    pip install "flowtracks[vtk]"

Requires Python ≥ 3.11; numpy/scipy and the other core dependencies
are installed automatically.

For developers — clone the repository and install it editable:

    git clone https://github.com/OpenPTV/postptv
    cd postptv
    pip install -e ".[vtk]"   # or: uv sync --group dev (installs pytest)

See pyproject.toml for the full dependency list and the optional
extras (vtk, legacy-io, notebooks). The legacy HDF5/pytables reader
needs the legacy-io extra; the .vti/.vtr ParaView writers need vtk.

The examples are Jupyter notebooks [2], and can be previewed without any 
special setup under the examples section of

    http://flowtracks.readthedocs.org


[1] Python documentation: https://docs.python.org/3/install/index.html
[2] http://jupyter.org/
