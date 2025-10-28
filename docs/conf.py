# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphinx configuration for Kubeflow SDK."""

import os
import sys

# Add SDK to path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Kubeflow SDK"
copyright = "2025, The Kubeflow Authors"
author = "The Kubeflow Authors"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Extensions
extensions = [
    "sphinx.ext.autodoc",  # Auto-generate from docstrings
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.todo",  # Support for TODOs
    "myst_parser",  # Markdown support
    "autodocsumm",  # Auto-summary tables
]

# Autodoc configuration
autodoc_typehints = "signature"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "undoc-members": False,
    "member-order": "bysource",
    "autosummary": True,
}

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = True

# Theme configuration (using furo for simplicity like Model Registry)
html_theme = "furo"
html_title = "Kubeflow SDK"
html_static_path = ["_static"]
html_logo = "_static/kubeflow.png"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0D47A1",
        "color-brand-content": "#0D47A1",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# MyST configuration
myst_enable_extensions = [
    "colon_fence",  # ::: blocks
    "deflist",  # definition lists
    "linkify",  # auto-link URLs
]

# Source file settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
