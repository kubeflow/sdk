# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "Kubeflow SDK"
copyright = "2026, The Kubeflow Authors"
author = "The Kubeflow Authors"

# Version is read from kubeflow/__init__.py
try:
    from kubeflow import __version__ as version
except ImportError:
    version = "dev"

release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

# Templates path
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes (support both RST and Markdown)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "exclude-members": "__weakref__,__init__",
    "show-inheritance": True,
}

# Generate autosummary pages automatically
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Type hints settings - show types in signature
autodoc_typehints = "signature"
typehints_use_signature = True
typehints_use_signature_return = True
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = False

# -- KFP type alias cleanup for Pipelines docs --------------------------------
# PipelinesClient (from kfp.kubeflow_client) uses type aliases like `Pipeline`
# and `Run` that map to auto-generated `kfp_server_api.V2beta1Pipeline` etc.
# Sphinx resolves these aliases to the underlying V2beta1* names, which are
# internal server API types that users should never see.
#
# Neither `autodoc_type_aliases` nor `typehints_formatter` can fix this reliably
# because sphinx_autodoc_typehints evaluates type aliases to their underlying
# runtime objects (where __name__ is V2beta1*), bypassing both mechanisms.
#
# The doctree-resolved hook below post-processes rendered text nodes to replace
# V2beta1* names with their clean aliases. It only runs on pipelines/ pages to
# avoid unintended side effects on other SDK docs.

_KFP_TYPE_DISPLAY_NAMES = {
    "V2beta1Pipeline": "Pipeline",
    "V2beta1PipelineVersion": "PipelineVersion",
    "V2beta1Run": "Run",
    "V2beta1Experiment": "Experiment",
    "V2beta1ListPipelinesResponse": "ListPipelinesResponse",
    "V2beta1ListPipelineVersionsResponse": "ListPipelineVersionsResponse",
    "V2beta1ListRunsResponse": "ListRunsResponse",
    "V2beta1ListExperimentsResponse": "ListExperimentsResponse",
}


def _replace_v2beta1_in_doctree(_app, doctree, docname):
    if not docname.startswith("pipelines/"):
        return

    from docutils import nodes

    for node in doctree.findall(nodes.Text):
        text = str(node)
        new_text = text
        for old, new in _KFP_TYPE_DISPLAY_NAMES.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            node.parent.replace(node, nodes.Text(new_text))


def setup(app):
    app.connect("doctree-resolved", _replace_v2beta1_in_doctree)


# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "Kubeflow SDK"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Furo theme options with top navigation
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/kubeflow/sdk",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "announcement": " ",
}

# Logo
html_logo = "_static/kubeflow-icon.svg"
html_favicon = "_static/kubeflow-icon.svg"

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "kubernetes": ("https://kubernetes.readthedocs.io/en/latest/", None),
}

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True

# -- MyST Parser options -----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3
