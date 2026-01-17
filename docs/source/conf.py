import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "grmpy"
author = "The grmpy Development Team"
copyright = "grmpy Development Team — MIT License"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
]

# Show TODO items locally, hide in production builds
todo_include_todos = os.environ.get("SPHINX_PROD", "0") != "1"

# Accept Markdown files as source as well as reStructuredText
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "_build", "**.ipynb_checkpoints"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Enable "Edit on GitHub" link in RTD theme
html_context = {
    "display_github": True,
    "github_user": "eisenhauerIO",
    "github_repo": "generalized-roy",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# MathJax configuration for proper LaTeX rendering
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "processEscapes": True,
    },
}

# nbsphinx settings: don't execute notebooks during docs build
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
