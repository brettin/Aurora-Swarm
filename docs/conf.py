# Aurora Swarm documentation
# https://www.sphinx-doc.org/

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "Aurora Swarm"
copyright = "Aurora Swarm contributors"
release = "0.1.0"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

nitpicky = False

# Napoleon: NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

html_theme = "alabaster"
html_static_path = []
html_title = "Aurora Swarm"
