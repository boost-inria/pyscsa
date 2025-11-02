import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information --
project = 'PySCSA'
copyright = '2025, boost inria'
author = 'boost inria'
release = '0.1.0'

# -- General configuration --
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output --
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration --
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}