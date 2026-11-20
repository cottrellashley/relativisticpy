"""RelativisticPy: General Relativity for physicists in a hurry.

Write tensor equations the way they look on paper, or paste the LaTeX, and
compute them.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("relativisticpy")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = ["__version__"]
