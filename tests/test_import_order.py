"""Verify that the `mcp`-before-`nnsight` import order lives in `engine/__init__.py` and holds."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_engine_initializes_mcp_before_nnsight_in_fresh_process() -> None:
    code = (
        "import sys, engine; "
        "assert 'anyio.streams.file' in sys.modules, 'anyio.streams.file not pre-imported'; "
        "assert 'mcp.server.fastmcp' in sys.modules, 'mcp.server.fastmcp not pre-imported'; "
        "assert 'nnsight' not in sys.modules, 'engine must not eagerly import nnsight'; "
        "import nnsight; "
        "from mcp.server.fastmcp import FastMCP; "
        "srv = FastMCP('order-check'); "
        "assert hasattr(srv, 'streamable_http_app')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_importing_nnsight_before_engine_fails_loudly() -> None:
    code = "import nnsight; import engine"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "Import order error" in proc.stderr


def test_non_engine_modules_do_not_import_nnsight_before_engine() -> None:
    """Any module outside `engine/` importing `nnsight` at top level must import `engine` first."""
    roots = [REPO_ROOT / "agent", REPO_ROOT / "organisms", REPO_ROOT / "datasets"]
    for root in roots:
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            seen_engine = False
            for node in tree.body:
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    if any(name == "engine" or name.startswith("engine.") for name in names):
                        seen_engine = True
                    if any(name == "nnsight" or name.startswith("nnsight.") for name in names):
                        rel = py_file.relative_to(REPO_ROOT)
                        assert seen_engine, f"{rel} imports `nnsight` before `engine`"
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "engine" or node.module.startswith("engine."):
                        seen_engine = True
                    if node.module == "nnsight" or node.module.startswith("nnsight."):
                        rel = py_file.relative_to(REPO_ROOT)
                        assert seen_engine, f"{rel} imports `{node.module}` before `engine`"
