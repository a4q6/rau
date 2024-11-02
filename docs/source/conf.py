import os
import sys
import toml
sys.path.insert(0, os.path.abspath("../../src"))  # Package Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
pyproject_path = os.path.abspath("../../pyproject.toml")
pyproject_data = toml.load(pyproject_path)
project = pyproject_data["project"]["name"]  # pyproject.tomlからプロジェクト名を取得
author = pyproject_data["project"]["authors"][0]["name"]  # 最初の著者名を取得
release = pyproject_data["project"]["version"]  # バージョンを取得


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",       # Auto build
    "sphinx.ext.napoleon",      # Google/Numpyスタイルのdocstring対応
    "sphinx.ext.viewcode",      # Link to code
    "sphinx.ext.todo",          # Include TODO
    "sphinx.ext.autosummary",
    'sphinx_rtd_theme',  # theme
    'myst_parser',  # markdown
    'nbsphinx',  # notebook
    'sphinx.ext.mathjax',  # math
]
templates_path = ["_templates"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'docs', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Auto
autodoc_typehints = "description"  # show typehint
autodoc_member_order = "bysource"

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Othres
todo_include_todos = True
