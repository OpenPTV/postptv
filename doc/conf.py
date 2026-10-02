# Flowtracks documentation build configuration.
#
# Modernised 2026: dynamic version, stdlib mocks for genuinely optional
# heavy dependencies only (everything else is a core install dependency),
# RTD theme, notebooks rendered from stored outputs (never executed here).

import os
import sys
from unittest.mock import MagicMock, Mock


class _Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return Mock()


# Optional/transient dependencies that autodoc must never require.
# `tables` (legacy HDF5 Scene), VTK writers, NetCDF output and the
# scikit-image skeleton helper are all import-guarded at call time in
# the package, so mocking them only affects documentation builds.
MOCK_MODULES = ['tables', 'vtk', 'vtk.util', 'pyvista', 'netCDF4',
                'skimage', 'skimage.morphology']
sys.modules.update((mod_name, _Mock()) for mod_name in MOCK_MODULES)

# Make the in-repo package importable when the docs are built without
# installing it first (ReadTheDocs installs it; this is a fallback).
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'flowtracks.format_docstrings',
    'nbsphinx',
]
autoclass_content = "both"

# Never execute notebooks during the docs build: render stored outputs.
# Authors refresh outputs with jupyter/marimo before committing.
nbsphinx_execute = 'never'

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'Flowtracks'
copyright = u'Yosef Meller, Alex Liberzon'

try:
    from flowtracks import __version__ as release
except Exception:
    release = 'dev'
version = '.'.join(release.split('.')[:2])

exclude_patterns = ['_build', 'story', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

htmlhelp_basename = 'Flowtracksdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

latex_documents = [
  ('index', 'Flowtracks.tex', u'Flowtracks Documentation',
   u'Yosef Meller, Alex Liberzon', 'manual'),
]

# -- Options for manual page output ---------------------------------------

man_pages = [
    ('index', 'flowtracks', u'Flowtracks Documentation',
     [u'Yosef Meller, Alex Liberzon'], 1)
]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
  ('index', 'Flowtracks', u'Flowtracks Documentation',
   u'Yosef Meller, Alex Liberzon', 'Flowtracks',
   'Post-processing of 3D-PTV trajectory databases.', 'Miscellaneous'),
]
