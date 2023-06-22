# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'VSDI Pipeline'
copyright = '2023, NeuroNetMem'
author = 'NeuroNetMem'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = ['sphinx.ext.autodoc',
extensions = ["autodocsumm",
	      "numpydoc",
	      "sphinx.ext.graphviz"]

autodoc_default_options = {"autosummary": True}
numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

graphviz_dot = 'dot'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'pydata_sphinx_theme'
# html_context = {"default_mode": "light"}
html_favicon = '_static/favicon/favicon.ico'
html_logo = '_static/favicon/favicon.ico'
html_show_sourcelink = False
html_static_path = ['_static']

def setup(app):
    app.add_css_file("custom.css")

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

# Make numpydoc to generate plots for example sections
# numpydoc_use_plots = True
# autosummary_generate = True
