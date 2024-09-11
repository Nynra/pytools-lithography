# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import pytools_lithography

project = 'pytools_lithography'
copyright = '2024'
author = 'Nynra'
release = 'v0.0.2'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.githubpages',  # For GitHub pages rendering
    # 'sphinx.ext.doctest',
    'sphinx.ext.autodoc',  # For automatically generating documentation from docstrings
    'sphinx.ext.autosummary', 
    'sphinx.ext.napoleon', # For parsing numpy-style docstrings
    "nbsphinx",  # For rendering Jupyter notebooks
]

# Some config options for parsing th edocstrings
autodoc_member_order = 'bysource'
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_ivar = False
napoleon_include_private_with_doc = False

templates_path = ['_templates']
exclude_patterns = []

source_suffix = '.rst'
master_doc = 'index'
language = 'en'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_theme = 'alabaster'
html_static_path = ['_static']
html_theme = 'furo'

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {
# '**': [
# 'about.html',
# 'navigation.html',
# 'relations.html', # needs 'show_related': True theme option to display
# 'searchbox.html',
# 'donate.html',
# ]
# }

# -- Options for HTMLHelp output ------------------------------------------
# Output file base name for HTML help builder.
htmlhelp_basename = 'mainDoc'

#sphinx-apidoc -o source/ /path/to/my_package