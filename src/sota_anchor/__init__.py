"""sota-anchor: check what already exists before a coding agent builds it from scratch.

The version is read from installed package metadata so it is declared once, in
``pyproject.toml``, rather than repeated across the CLI, the MCP server, the
User-Agent and the plugin manifest.
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("sota-anchor")
except metadata.PackageNotFoundError:  # pragma: no cover - running from a raw checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
