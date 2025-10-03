# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# type: ignore

project = "xarray-ms"
copyright = "2024 - 2025 NRF (SARAO) and Rhodes University (RATT) Centre"
author = "Simon Perkins"
release = "0.3.7"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  "sphinxcontrib.spelling",
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "sphinx.ext.extlinks",
  "sphinx_copybutton",
  "sphinx.ext.doctest",
  "sphinx.ext.napoleon",
  "sphinx.ext.intersphinx",
  "IPython.sphinxext.ipython_directive",
  "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# Disable navigation sidebars
# https://github.com/pydata/pydata-sphinx-theme/issues/1662
html_sidebars = {
  "**": [],
}
html_static_path = ["_static"]

extlinks = {
  "issue": ("https://github.com/ratt-ru/xarray-ms/issues/%s", "GH%s"),
  "pr": ("https://github.com/ratt-ru/xarray-ms/pull/%s", "PR%s"),
}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
  "dask": ("https://dask.pydata.org/en/stable", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
  "python": ("https://docs.python.org/3/", None),
  "xarray": ("https://docs.xarray.dev/en/stable", None),
}

# Exclude link file
exclude_patterns = ["_build", "links.rst"]

# make rst_epilog a variable, so you can add other epilog parts to it
rst_epilog = ""
# Read link all targets from file
with open("links.rst") as f:
  rst_epilog += f.read()
