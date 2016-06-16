
Flowtracks - postprocessing of 3D-PTV data
==========================================

This package contains Flowtracks, a Python-package for post-processing
of 3D Particle Tracking Velocimetry particle/trajectory databases.

The full documentation for this package may be built from the Sphinx 
sources in the doc/ directory. It is also available online:

  http://flowtracks.readthedocs.org

Please refer to that documentation for the full information on installing,
reference documentation and usage examples contained in the package.

The program is distributed under the terms of the GNU General Public 
License, version 3.0. For details, see the LICENSE.txt file.


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

To install this package, follow the standard procedure for installing 
Python modules. Using a terminal, change directory into the root directory
of theis program's source code, then run

  python setup.py install

Note that you may need administrative privileges on the machine you are 
using.

The install script will install the Python package in the default place for 
your platform. Additionally, it will install example scripts in a 
subdirectory ``flowtracks-examples/`` under the default executable location, 
and a the documentation in the default package root. For information on where
these directories are on your platform (and how to change them), refer to 
the Python documentation [1]. Other standard features of the setup script are 
also described therein.

[1] Python documentation: https://docs.python.org/2/install/index.html

