"""Core engine package for running model interpretability checks.

Import-order rule (Windows):
`nnsight` attaches a `.save` attribute to Python's base `object` type via C-API.
On Windows, `mcp` loads `anyio.streams.file.FileStreamAttribute(TypedAttributeSet)`,
whose metaclass raises `TypeError: Attribute 'save' is missing its type annotation`
if `nnsight` was imported before `mcp` / `anyio.streams.file`.

Keeping `ensure_mcp_imported_before_nnsight()` here in `engine/__init__.py` puts the
import order in one place: importing `engine` or any `engine.*` submodule always
executes this file first, initializes `mcp` before `nnsight` is loaded, and fails
fast if `nnsight` was already imported before `anyio.streams.file`.
"""

from __future__ import annotations

import sys

__version__ = "0.1.0"


def ensure_mcp_imported_before_nnsight() -> None:
    """Ensure `mcp` and `anyio.streams.file` initialize before `nnsight` loads."""
    if "nnsight" in sys.modules and "anyio.streams.file" not in sys.modules:
        raise RuntimeError(
            "Import order error: `nnsight` was imported before `engine` / `mcp`. "
            "Import `engine` before `nnsight` so `mcp` initializes cleanly on Windows."
        )
    import anyio.streams.file  # noqa: F401
    import mcp.server.fastmcp  # noqa: F401


ensure_mcp_imported_before_nnsight()
