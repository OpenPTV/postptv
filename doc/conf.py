# Flowtracks Sphinx documentation configuration.
#
# Built from the doc/ directory:  python -m sphinx -b html doc _build/html
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information ---------------------------------------------------
project = 'Flowtracks'
copyright = '2015-2026, Yosef Meller, Alex Liberzon'

try:
    from flowtracks import __version__ as release
except ImportError:
    release = 'unknown'

version = '.'.join(release.split('.')[:2]) if release != 'unknown' else 'unknown'

# -- General configuration --------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'flowtracks.format_docstrings',
]

autodoc_member_order = 'bysource'
autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
}

source_suffix = '.rst'
master_doc = 'index'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
language = 'en'

# -- Options for HTML output -----------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

htmlhelp_basename = 'Flowtracksdoc'
