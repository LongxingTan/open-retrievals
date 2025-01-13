# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys
from pathlib import Path

SOURCE_PATH = Path(os.path.dirname(__file__))  # noqa # docs source
PROJECT_PATH = SOURCE_PATH.joinpath("../..")  # noqa # project root

sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'open-retrievals'
copyright = '2024, Longxing Tan'
author = 'Longxing Tan'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**/.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# theme options
html_theme_options = {
    "github_url": "https://github.com/LongxingTan/Open-retrievals",
}

html_theme_options = {"navbar_end": ["navbar-icon-links.html", "search-field.html"]}

html_sidebars = {
    "index": [],
    "quick-start": [],
    "embed": [],
    "retrieval": [],
    "rerank": [],
    "rag": [],
}


autodoc_member_order = "groupwise"
autoclass_content = "both"

# autosummary
autosummary_generate = True
shutil.rmtree(SOURCE_PATH.joinpath("api"), ignore_errors=True)
