.. Flowtracks documentation master file, created by
   sphinx-quickstart on Sat Oct  3 10:47:07 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Flowtracks's documentation!
======================================

Flowtracks is a Python package for manipulating a trajectory database 
obtained from 3D Particle Tracking Velocimetry (3D-PTV) analysis. It 
understands the most common output formats for 3D-PTV trajectories used in 
the 3D-PTV community, and adds its own HDF5-based format, which allows faster
and more flexible access to the trajectory database.

Contents:

.. contents:: 
   :depth: 2

.. toctree::
   :maxdepth: 2
   :hidden:

   datastruct
   io
   particle
   interpolation
   graphics
   smoothing
   pairs
   scene
   analysis
   an_scene
   sequence
   examples/repeated_interpolation.ipynb

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

* Some features depend on the PyTables package. Users which need those 
  features may get PyTables at http://www.pytables.org
  

Installation
------------

To install this package, follow the standard procedure for installing 
Python modules. Using a terminal, change directory into the root directory
of theis program's source code, then run

.. code:: bash 
  
      python -m pip install -r requirements.txt

Note that you may need administrative privileges on the machine you are 
using.

The install script will install the Python package in the default place for 
your platform. Additionally, it will install example scripts in a 
subdirectory ``flowtracks-examples/`` under the default executable location, 
and a the documentation in the default package root. For information on where
these directories are on your platform (and how to change them), refer to 
the `Python documentation`_. Other standard features of the setup script are 
also described therein.

.. _`Python documentation`: https://docs.python.org/2/install/index.html

Documentation
-------------
This documentation is available in the source directory under the ``docs/``
subdirectory. It is maintained as a `Sphinx`_ project, so that you can build
the documentation in one of several output formats, including HTML and PDF.
To build, install Sphinx, then use 

.. code::sh
   sphinx-build -b html docs/

or replace ``html`` with any other builder supported by Sphinx.

Alternatively, the documentation is pre-built and available online on 
ReadTheDocs_.

.. _Sphinx: http://sphinx-doc.org/
.. _ReadTheDocs: http://flowtracks.readthedocs.org

Examples
--------
The ``examples/`` subdirectory in the source distribution contains two 
IPython notebooks, both available as HTML for direct viewing:

* a tutorial to the basic HDF5 analysis workflow 
  (:download:`HTML <_static/hdf5_scene_analysis.html>`).
* a demonstration of using the ``flowtracks.interpolation`` module 
  (:download:`HTML <_static/repeated_interpolation.html>`).


Additional examples in Jupyter notebooks are available in a Github repository
.. _flowtracks_examples: https://github.com/alexlib/flowtracks_examples

Analysis Script
---------------
The script ``analyse_fhdf.py`` is installed by default. for instruction on
its usage, run::

  analyse_fhdf.py --help

As the help message printed informs, there are two mandatory command-line
arguments. One is the data file for processing, the other is a config file
with some rudimentary metadata. Examples for both are supplied in the ``data/``
subdirectory of this package. A config file accepted by the script looks 
something like this:

.. code:: INI
   
   [Particle]
   density = 1450
   diameter = 500e-6
   
   [Scene]
   particles file = particles.h5
   tracers file = tracers.h5
   first frame = 10001
   last frame = 10200
   frame rate = 500

the file above may be used for producing an analysis from the files in the
``data/`` subdirectory, when it is the current directory. this has been done
in both IPython examples mentioned above, where the usage is shown.

General Facilities
==================

Data structures
---------------
the most basic building blocks of any analysis are sets of particles 
representing a slice of the database. These are represented by a ParticleSet
instance. ParticleSet is a flexible class. It holds a number of numpy arrays
whose first dimension must have the same length; each is a column in a table
of particle properties, whose each row represents one particle's data. It
must contain particles' position and velocity data, but users may add more
properties as relevant to their database. For details, ssee the ParticleSet
documentation.

The two most common ways to slice a database are by frame (time point) and by
trajectory (data fore the same particle over several frames). For this there
are two classes provided by ``flowtracks``, both derived from ParticleSet and
thus behave in a similar way. They both expect the ``time`` and ``trajid`` 
(trajtectory ID) properties to exist for the particle data, but each class 
treats these properties differently.

ParticleSnapshot is a ParticleSet which assumes that all particles have the
same ``time``, so that this property is scalar. Similarly, the Trajectory 
class expects a same ``trajid`` across its data. A trajectory ID is simply an
integer number unique to each trajectory. Users may select their numbering 
scheme when creating Trajectory objects from scratch, but in most cases the
data is read from a file, in which case Flowtracks' input rutines handle the 
numbering automatically.

Refer to :doc:`datastruct` for the details of all these classes.

Input and Output
----------------
The module :mod:`flowtracks.io` provides several functions for reading and writing
particle data. The currently-supported formats are:

*  ptv_is - the format output by OpenPTV code as well as the older but still
   widely used 3DPTV. Composed of one file per frame, containing a particle's
   number, its number in the previous and next frame file, and current
   position.

*  xuap - a similar format using one file per frame with columns for position,
   velocity, and acceleration for both the particle and the surrounding fluid.
   This file format represents an initial analysis of ptv_is raw data.

*  acc - another frame-files format with each particle having, additionally 
   to data provided in the xuap format, the time step relative to the 
   trajectory start.
 
*  mat - a Matlab file containing a list of trajectory structure-arrays with
   xuap-like data for each trajectory.
 
*  hdf - Flowtracks' own optimised format, relying on the HDF5 file format
   and the PyTables package for fast handling thereof. It is highly 
   recommended to use the other reading/writing functions in order to 
   convert data in other formats to this format. This allows users a more
   flexible programmatic interface as well as the speed of PyTables.

Description of the relevant functions, as well as some other IO convenience 
facilities may be found in :doc:`the module's reference documentation<io>`.

Basic Analysis and display
--------------------------
The package provides some facilities for analysing the database and 
extracting kinematic or dynamic information embedded in it. Dynamic analysis
requires the particle size and diameter to be known (Flowtracks assumes a
spherical particle for these analyses, but users may extend this behaviour).
These properties may be stored in the :doc:`Particle class <particle>` provided by the 
package. :mod:`flowtracks.io` provides a way to read them from an INI file.

The :mod:`flowtracks.interpolation` module provides an object-oriented approach
to interpolating the data. It offers some built-in interpolation methods, and
is hoped to be extensible to other methods without much effort.

Some plotting support is provided by :mod:`flowtracks.graphics`. Functions 
therein allow users to generate probability distributions from data and to
plot them using Matplotlib, and a function is provided that plots 3D vector 
data as 3 subplots of components.

Other facilities (:mod:`smoothing <flowtracks.smoothing>`, 
:mod:`nearest-neighbour searches <flowtracks.pairs>`) are described in the 
respective module's documentation.

HDF5-based fast databases
=========================
Above the layer of basic data structures, Flowtracks provides a generalized
view of a scene, containing several trajectories across a number of frames. 
This view is iterable in several ways and provides general metadata access.

The :class:`~flowtracks.scene.Scene` class is the most basic form of this 
view. It is tied to one HDF5 file exactly, which holds the database. This 
file may be iterated by trajectory, by frame, or by *segments*, a concept 
introduced by Flowtracks for easier time-derived analyses requiring the next 
time-point to be also known.

A segment, in the context of iterating a :class:`~flowtracks.scene.Scene`
is a tuple containing two :func:`~flowtracks.trajectory.ParticleSnapshot` 
objects, one for the current frame and one for the next. The next frame data 
is filtered to contain only particles that also appear in the current frame, 
unlike when iterating simply by frames.

The :class:`~flowtracks.scene.DualScene` class extends this by tying itself 
into two HDF5 files, each representing a separate class of particles which 
coexist in the same experiment. This has been useful for measuring tracers 
and inertial particles simultaneously, but other users are of course 
possible. Iterating by frames is supported here, providing a 
:class:`~flowtracks.trajectory.Frame` object on each iteration. Iterating by 
trajectories is ambiguous and not supported currently. Segments iteration, 
similarly to the frames iteration, returns two 
:class:`~flowtracks.trajectory.Frame` objects.

The :mod:`flowtracks.analysis` module provides a function for applying analyser
classes sequentially to segments iterated over, and generetes a properly 
sized HDF5 file in the format of the input file.

:class:`~flowtracks.an_scene.AnalysedScene` objects track simultaneously the 
DualScene and an analysis file resulting from it. They contain the 
:meth:`~flowtracks.an_scene.AnalysedScene.collect` facility. It allows
finding of all (or selected) data belonging to a certain property, regardless
of which of the files it is stored in.


Manipulating text formats directly
==================================
Similarly to the :class:`~flowtracks.scene.DualScene` class used with the 
HDF5 format, the :class:`~flowtracks.sequence.Sequence` class tracks two 
sets of particles and allows iterating by frame. Since this class relies 
on :class:`~flowtracks.trajectory.Trajectory` lists as its underlying 
database, it does not provide a special facility for iterating over 
trajectories.

Though :class:`~flowtracks.sequence.Sequence` also accepts trajectory 
iterators, and :mod:`flowtracks.io` provides you with iterators if asked, 
the working memory used in actuality may still be large and the access times 
are much slower than the equivalent times achieved by the specialized HDF5 
classes.

Corresponding to the :mod:`flowtracks.analysis` module, 
:class:`~flowtracks.sequence.Sequence` provides the 
:meth:`~flowtracks.sequence.Sequence.map_trajectories` method for applying 
callback functions on an entire scene, frame by frame.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

