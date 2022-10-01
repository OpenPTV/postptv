
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

To install this package, follow the standard procedure for installing 
Python modules. The package may be installed either systemwide, in the default
location, or for a single user, by changing the install location and updating
environment variables in the standard way, as indicated by the Python 
documentation [1]. 

For a default systemwide installation: using a terminal, change directory into
the root directory of theis program's source code, then run

    pip install -r requirements.txt

Note that for the default install you may need administrative privileges on the
machine you are using. Consult the Python documentation for the single-user 
install procedure.

The install script will install the Python package in the default place for 
your platform. Additionally, it will install example scripts in a 
subdirectory ``flowtracks-examples/`` under the default executable location, 
and a the documentation in the default package root. For information on where
these directories are on your platform (and how to change them), refer to 
the Python documentation [1]. Other standard features of the setup script are 
also described therein.

The examples are Jupyter notebooks [2], and can be previewed without any 
special setup under the examples section of

    http://flowtracks.readthedocs.org


[1] Python documentation: https://docs.python.org/2/install/index.html
[2] http://jupyter.org/
