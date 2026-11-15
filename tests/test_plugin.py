"""Tests for A4 Claude Code plugin, root README, and report/setup HTTP endpoints."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from engine.api import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "plugin"


def _drive_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Step an in-memory coroutine to completion without creating OS loopback sockets."""
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise RuntimeError("Coroutine yielded an unexpected OS future in offline unit test.")


def _asgi_http(
    app: Any,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
) -> tuple[int, dict[str, str], bytes]:
    """Execute one HTTP request directly through the ASGI 3.0 interface of `app`."""
    raw_body = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    hdr_list: list[tuple[bytes, bytes]] = [(b"host", b"127.0.0.1:8000")]
    if json_body is not None:
        hdr_list.append((b"content-type", b"application/json"))
        hdr_list.append((b"content-length", str(len(raw_body)).encode("ascii")))
    for k, v in (headers or {}).items():
        hdr_list.append((k.lower().encode("latin-1"), v.encode("latin-1")))

    qs = urlencode(params) if params else ""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": qs.encode("latin-1"),
        "headers": hdr_list,
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 50000),
    }

    sent_body = False
    status_code = 500
    resp_headers: dict[str, str] = {}
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
            for rk, rv in message.get("headers", []):
                resp_headers[rk.decode("latin-1").lower()] = rv.decode("latin-1")
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    _drive_sync(app(scope, receive, send))
    return status_code, resp_headers, b"".join(chunks)


def _parse_frontmatter(md_text: str) -> tuple[dict[str, str], str]:
    """Parse simple YAML `key: value` frontmatter at the top of a Markdown file."""
    assert md_text.startswith("---\n"), "Markdown file must start with YAML frontmatter."
    end_idx = md_text.find("\n---\n", 4)
    assert end_idx != -1, "YAML frontmatter closing delimiter not found."
    raw_yaml = md_text[4:end_idx]
    body = md_text[end_idx + 5 :]
    meta: dict[str, str] = {}
    for line in raw_yaml.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, val = stripped.partition(":")
        assert sep == ":", f"Malformed frontmatter line: {line!r}"
        meta[key.strip()] = val.strip()
    return meta, body


def test_plugin_manifest_and_mcp_json_are_valid() -> None:
    manifest_path = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["name"] == "model-safety-tool"
    assert re.fullmatch(r"[a-z0-9-]+", manifest["name"])
    assert manifest["version"] == "0.1.0"
    assert isinstance(manifest["description"], str) and len(manifest["description"]) > 15
    assert manifest["license"] == "Apache-2.0"
    assert "playbook" not in manifest["description"].lower()

    # `.mcp.json` must sit at the plugin root, not inside `.claude-plugin/`.
    assert not (PLUGIN_DIR / ".claude-plugin" / ".mcp.json").exists()
    mcp_path = PLUGIN_DIR / ".mcp.json"
    assert mcp_path.is_file()
    mcp_cfg = json.loads(mcp_path.read_text(encoding="utf-8"))

    assert "mcpServers" in mcp_cfg
    server_cfg = mcp_cfg["mcpServers"]["model-safety-engine"]
    assert server_cfg["type"] == "http"
    assert server_cfg["url"] == "http://127.0.0.1:8000/mcp"


def test_plugin_investigate_risk_skill_frontmatter_and_rules() -> None:
    skill_path = PLUGIN_DIR / "skills" / "investigate-risk" / "SKILL.md"
    assert skill_path.is_file()
    meta, body = _parse_frontmatter(skill_path.read_text(encoding="utf-8"))

    assert meta["name"] == "investigate-risk"
    assert len(meta["description"]) > 15

    required_tokens = (
        "list_checks",
        "describe_check",
        "load_model",
        "register_series",
        "get_model_map",
        "confirm_components",
        "draft_plan",
        "revise_plan",
        "run_plan",
        "approved_by_user=True",
        "job_status",
        "get_findings",
        "get_results",
        "get_examples",
        "reviewer",
        "view_focus",
        "view_paint",
        "view_timeline",
        "build_report",
    )
    for token in required_tokens:
        assert token in body, f"Expected {token!r} in SKILL.md"

    low_body = body.lower()
    assert "never" in low_body and "the model is safe" in low_body
    assert "no backdoor exists" in low_body
    assert "dataset id" in low_body
    assert "playbook" not in low_body


def test_plugin_reviewer_subagent_tool_lockdown_and_rules() -> None:
    reviewer_path = PLUGIN_DIR / "agents" / "reviewer.md"
    assert reviewer_path.is_file()
    meta, body = _parse_frontmatter(reviewer_path.read_text(encoding="utf-8"))

    assert meta["name"] == "reviewer"
    assert len(meta["description"]) > 15
    tools = [t.strip() for t in meta["tools"].split(",") if t.strip()]
    assert tools == [
        "mcp__model-safety-engine__get_evidence",
        "mcp__model-safety-engine__submit_review",
        "mcp__engine__get_evidence",
        "mcp__engine__submit_review",
    ]

    forbidden_tools = {
        "Bash",
        "Read",
        "Write",
        "Edit",
        "MultiEdit",
        "WebFetch",
        "WebSearch",
        "Agent",
        "mcp__engine__run_plan",
        "mcp__model-safety-engine__run_plan",
    }
    assert forbidden_tools.isdisjoint(set(tools))

    low_body = body.lower()
    assert "cannot raise" in low_body or "never pass a higher strength" in low_body
    assert "coverage" in low_body
    assert "shift" in low_body
    assert "experimental" in low_body
    assert "inconclusive" in low_body
    assert "playbook" not in low_body


def test_plugin_readme_and_root_readme_cover_required_install_and_usage() -> None:
    plugin_readme = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
    assert "claude --plugin-dir ./plugin" in plugin_readme
    assert (
        "claude mcp add --transport http model-safety-engine http://127.0.0.1:8000/mcp"
        in plugin_readme
    )
    assert "tool mcp-stdio" in plugin_readme
    assert "playbook" not in plugin_readme.lower()

    root_readme_path = REPO_ROOT / "README.md"
    assert root_readme_path.is_file()
    root_readme = root_readme_path.read_text(encoding="utf-8")

    # Mandatory locked uv sync step and web build steps.
    assert "uv sync --locked --all-extras" in root_readme
    assert "pnpm --dir web install" in root_readme
    assert "pnpm --dir web run build" in root_readme
    assert "tool init" in root_readme
    assert "tool doctor" in root_readme
    assert "tool serve" in root_readme
    assert "?fake=1" in root_readme
    assert "claude --plugin-dir ./plugin" in root_readme
    assert "Apache License, Version 2.0" in root_readme or "Apache-2.0" in root_readme

    # The README's risk table lists the checks that are built end to end.
    for risk_term in (
        "Sycophancy",
        "Weakened refusals",
        "Backdoors",
        "Pretraining",
    ):
        assert risk_term.lower() in root_readme.lower()

    assert "playbook" not in root_readme.lower()


def test_http_setup_claude_code_and_post_reports_endpoints(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    app = create_app(ws, stub_mode=True)
    svc = app.state.engine_service
    try:
        # 1. Verify GET /api/setup/claude-code includes `plugin_command` and all existing keys.
        code, _, body = _asgi_http(app, "GET", "/api/setup/claude-code")
        assert code == 200
        setup_payload = json.loads(body.decode("utf-8"))
        assert setup_payload["plugin_command"] == "claude --plugin-dir ./plugin"
        for existing_key in (
            "command",
            "http_command",
            "stdio_command",
            "mcp_url",
            "workspace",
            "notes",
        ):
            assert existing_key in setup_payload

        # 2. Unknown run_id on POST /api/reports/{run_id} returns 404.
        code_404, _, _ = _asgi_http(app, "POST", "/api/reports/nonexistent_run_xyz")
        assert code_404 == 404

        # 3. Seed a stub run and build its report via POST /api/reports/{run_id}.
        svc.load_model("Qwen/Qwen2.5-0.5B-Instruct")
        drafted = svc.draft_plan(
            check="sycophancy",
            model_id="hf:Qwen/Qwen2.5-0.5B-Instruct@main",
            layers=[0, 1],
        )
        plan_id = drafted["plan_id"]
        run_out = svc.run_plan(plan_id, approved_by_user=True)
        run_id = run_out["run_id"]

        code_post, _, post_body = _asgi_http(app, "POST", f"/api/reports/{run_id}")
        assert code_post == 200
        built = json.loads(post_body.decode("utf-8"))
        assert built["run_id"] == run_id
        report_id = built["report_id"]
        assert built["url"] == f"/api/reports/{report_id}"
        assert Path(built["path"]).is_file()

        code_list, _, list_body = _asgi_http(app, "GET", "/api/reports")
        assert code_list == 200
        reports_list = json.loads(list_body.decode("utf-8"))
        assert any(r["report_id"] == report_id for r in reports_list)

        code_html, hdrs, html_bytes = _asgi_http(app, "GET", f"/api/reports/{report_id}")
        assert code_html == 200
        assert "text/html" in hdrs.get("content-type", "")
        assert "Placeholder / unverified test run" in html_bytes.decode("utf-8")
    finally:
        app.state.record.close()


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("CLAUDE_CODE_LIVE_TEST"),
    reason="Waiting for real-model/live Claude Code testing",
)
def test_live_claude_code_plugin_end_to_end() -> None:
    """Placeholder marker for live `claude --plugin-dir ./plugin` session check (IC4)."""
    assert (PLUGIN_DIR / ".claude-plugin" / "plugin.json").is_file()
