"""Command-line interface for the model safety investigation engine (component E14)."""

from __future__ import annotations

import importlib.metadata
import logging
import secrets
import sys
import webbrowser
from pathlib import Path

import typer
import uvicorn

import engine
from engine.api.mcp_tools import EngineService, create_mcp_server
from engine.api.server import DEFAULT_HOST, DEFAULT_PORT, create_app
from engine.record import Record, init_workspace

app = typer.Typer(
    name="tool",
    help="Local engine for investigating safety risks inside language models.",
    no_args_is_help=True,
)

_DEMO_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "EleutherAI/pythia-70m",
    "Antonhansel/qwen2.5-1.5b-sleeper-canary",
)


_REPO_ROOT = Path(engine.__file__).resolve().parent.parent
_WEB_DIST = _REPO_ROOT / "web" / "dist"


def _load_dotenv(path: Path) -> list[str]:
    """Read simple `NAME=value` lines from a `.env` file into the environment.

    Variables already set in the environment win. Values are never printed.
    Returns the names that were set from the file.
    """
    import os

    if not path.is_file():
        return []
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def _check_hf_cache_present(repo_id: str) -> bool:
    """Check offline whether a Hugging Face model folder exists in the local hub cache."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    folder_name = "models--" + repo_id.replace("/", "--")
    snapshots = cache_root / folder_name / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(snapshots.rglob("*.safetensors"))


@app.command("init")
def init_cmd(
    workspace: Path = typer.Option(  # noqa: B008
        Path("workspace"),
        "--workspace",
        "-w",
        help="Path to the local experiment workspace.",
    ),
) -> None:
    """Create the local workspace directory structure and SQLite database."""
    rec = Record(workspace)
    try:
        typer.echo(f"Initialized workspace at: {rec.workspace}")
    finally:
        rec.close()


@app.command("doctor")
def doctor_cmd(
    workspace: Path = typer.Option(  # noqa: B008
        Path("workspace"),
        "--workspace",
        "-w",
        help="Path to the local experiment workspace.",
    ),
) -> None:
    """Print local compute profile, package versions, and offline demo-model cache status."""
    layout = init_workspace(workspace)
    rec = Record(layout.root)
    try:
        svc = EngineService(rec, stub_mode=False)
        compute = svc.get_compute()
    finally:
        rec.close()

    typer.echo(f"Tool version: {engine.__version__}")
    typer.echo(f"Workspace: {layout.root}")
    typer.echo(
        f"Compute profile: device={compute.get('device')}, "
        f"dtype={compute.get('dtype')}, "
        f"n_gpus={compute.get('n_gpus')}, "
        f"remote_available={compute.get('remote_available')}"
    )

    typer.echo("Installed package versions:")
    for dist_name in (
        "torch",
        "transformers",
        "nnsight",
        "fastapi",
        "mcp",
        "claude-agent-sdk",
        "pydantic",
        "pyarrow",
        "plotly",
        "typer",
    ):
        try:
            ver = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            ver = "not installed"
        typer.echo(f"  - {dist_name}: {ver}")

    typer.echo("Local demo model cache (offline check):")
    for repo_id in _DEMO_MODELS:
        status = "cached locally" if _check_hf_cache_present(repo_id) else "not downloaded"
        typer.echo(f"  - {repo_id}: {status}")


@app.command("jobs")
def jobs_cmd(
    workspace: Path = typer.Option(  # noqa: B008
        Path("workspace"),
        "--workspace",
        "-w",
        help="Path to the local experiment workspace.",
    ),
) -> None:
    """List experiment jobs stored in the workspace and their status."""
    rec = Record(workspace)
    try:
        runs = rec.list_runs()
    finally:
        rec.close()

    if not runs:
        typer.echo("No jobs recorded in workspace.")
        return

    for r in runs:
        placeholder_tag = " [placeholder]" if r["metadata"].get("is_placeholder") else ""
        msg_suffix = f" — {r['message']}" if r["message"] else ""
        typer.echo(
            f"{r['run_id']}  status={r['status']}  plan={r['plan_id']}{placeholder_tag}{msg_suffix}"
        )


@app.command("report")
def report_cmd(
    run_id: str = typer.Argument(..., help="Run identifier to build an HTML report for."),
    workspace: Path = typer.Option(  # noqa: B008
        Path("workspace"),
        "--workspace",
        "-w",
        help="Path to the local experiment workspace.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the generated HTML report in the default browser.",
    ),
) -> None:
    """Generate an HTML report for `run_id` and optionally open it."""
    from engine.report import build_report

    rec = Record(workspace)
    try:
        out_path = build_report(run_id, rec)
    finally:
        rec.close()

    typer.echo(f"Report written to: {out_path}")
    if open_browser:
        webbrowser.open(out_path.as_uri())


@app.command("mcp-stdio")
def mcp_stdio_cmd(
    workspace: Path = typer.Option(  # noqa: B008
        Path("workspace"),
        "--workspace",
        "-w",
        help="Path to the local experiment workspace.",
    ),
    stub: bool = typer.Option(
        False,
        "--stub",
        help="Run with synthetic placeholder stubs instead of loading real models.",
    ),
) -> None:
    """Run the engine MCP server over stdio transport (logs go to stderr only)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    mcp = create_mcp_server(workspace, stub_mode=stub, mounted_at_mcp=False)
    mcp.run(transport="stdio")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Bind address (127.0.0.1 by default)."),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port number."),
    workspace: Path = typer.Option(  # noqa: B008
        Path("workspace"),
        "--workspace",
        "-w",
        help="Path to the local experiment workspace.",
    ),
    stub: bool = typer.Option(
        False,
        "--stub",
        help="Use synthetic placeholder stubs for unfinished components.",
    ),
    require_token: bool = typer.Option(
        False,
        "--require-token",
        help="Generate and require a local session bearer token for HTTP/MCP/WebSocket calls.",
    ),
    no_web: bool = typer.Option(
        False,
        "--no-web",
        help="Do not serve the web app (engine and MCP only).",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the web app in the default browser after starting.",
    ),
) -> None:
    """Start the tool: the engine, its MCP endpoint and the web app, in one process."""
    import os

    # A `.env` in the current folder or the repository root can hold ANTHROPIC_API_KEY.
    for env_path in (Path.cwd() / ".env", _REPO_ROOT / ".env"):
        _load_dotenv(env_path)

    session_token = secrets.token_urlsafe(24) if require_token else None
    web_dist = None if no_web else _WEB_DIST
    asgi_app = create_app(
        workspace,
        host=host,
        port=port,
        stub_mode=stub,
        auth_token=session_token,
        web_dist=web_dist,
    )
    url = f"http://{host}:{port}"
    if web_dist is not None and (web_dist / "index.html").is_file():
        typer.echo(f"Web app: {url}")
    elif not no_web:
        typer.echo(
            "Web app not built yet. Run `pnpm --dir web install` and `pnpm --dir web build`, "
            "then start the tool again."
        )
    typer.echo(f"MCP endpoint: {url}/mcp")
    typer.echo(f"Workspace: {Path(workspace).resolve()}")
    if os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo("ANTHROPIC_API_KEY: set (built-in chat agent enabled)")
    else:
        typer.echo(
            "ANTHROPIC_API_KEY: not set. The web chat needs it; put it in a .env file "
            "in the project folder. Claude Code can still use the engine."
        )
    if open_browser and web_dist is not None:
        import threading

        # Give the server a moment to start listening before the browser asks for the page.
        threading.Timer(2.0, webbrowser.open, args=(url,)).start()
    if session_token:
        typer.echo(f"Local session token: {session_token}")
    uvicorn.run(asgi_app, host=host, port=port)


if __name__ == "__main__":
    app()
