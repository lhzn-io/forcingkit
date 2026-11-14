# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "forcingkit"
copyright = "2026, Long Horizon Observatory"
author = "Daniel Fry"
release = "0.20.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

language = "en"

html_theme = "lhzn"
html_theme_options = {
    "source_repository": "https://github.com/lhzn-io/forcingkit",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
html_static_path = ["_static"]
# CNAME for the custom domain forcingkit.docs.lhzn.io, copied to the site root.
html_extra_path = ["_extra"]
