"""Tests for E13 Engine API (MCP, HTTP, WebSocket, approval rules) and E14 CLI."""

from __future__ import annotations

import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
from starlette.routing import Mount
from typer.testing import CliRunner

from agent.session import ChatSession
from engine.api import PlanApprovalError, create_app, create_mcp_server
from engine.cli.main import app as cli_app
from engine.contracts import ControlResult, Finding

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

EXPECTED_23_MCP_TOOLS = {
    "get_compute",
    "load_model",
    "get_model_map",
    "confirm_components",
    "register_series",
    "list_checks",
    "describe_check",
    "draft_plan",
    "revise_plan",
    "run_plan",
    "job_status",
    "answer_job",
    "stop_job",
    "get_findings",
    "get_results",
    "get_examples",
    "view_focus",
    "view_paint",
    "view_timeline",
    "build_report",
    "export_artifact",
    "get_evidence",
    "submit_review",
}


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


def _asgi_ws(
    app: Any,
    path: str,
    incoming_texts: list[str],
    *,
    headers: dict[str, str] | None = None,
    query_string: str = "",
) -> list[dict[str, Any]]:
    """Run a WebSocket conversation directly through the ASGI 3.0 interface of `app`."""
    hdr_list: list[tuple[bytes, bytes]] = [(b"host", b"127.0.0.1:8000")]
    for k, v in (headers or {}).items():
        hdr_list.append((k.lower().encode("latin-1"), v.encode("latin-1")))

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": query_string.encode("latin-1"),
        "headers": hdr_list,
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 50001),
        "subprotocols": [],
    }

    events_in: list[dict[str, Any]] = [{"type": "websocket.connect"}]
    for text in incoming_texts:
        events_in.append({"type": "websocket.receive", "text": text})
    events_in.append({"type": "websocket.disconnect", "code": 1000})

    idx = 0
    outbound: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal idx
        if idx < len(events_in):
            item = events_in[idx]
            idx += 1
            return item
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message: dict[str, Any]) -> None:
        outbound.append(message)

    _drive_sync(app(scope, receive, send))
    return outbound


def test_mcp_mount_and_all_23_tools(tmp_path: Path) -> None:
    app = create_app(tmp_path / "workspace", stub_mode=True)
    try:
        mounts = [r for r in app.routes if isinstance(r, Mount) and r.path == "/mcp"]
        assert len(mounts) == 1, "FastMCP streamable HTTP app must be mounted at /mcp"

        mcp = app.state.mcp
        tools = _drive_sync(mcp.list_tools())
        tool_names = {t.name for t in tools}
        assert tool_names == EXPECTED_23_MCP_TOOLS

        # Call MCP tools through FastMCP.call_tool
        checks_res = _drive_sync(mcp.call_tool("list_checks", {}))
        checks_text = json.dumps(checks_res, default=str)
        assert "sycophancy" in checks_text
        assert "refusal" in checks_text
        assert "backdoor" in checks_text
        assert "data_poisoning" in checks_text

        stdio_mcp = create_mcp_server(app.state.engine_service, stub_mode=True)
        stdio_tools = _drive_sync(stdio_mcp.list_tools())
        assert {t.name for t in stdio_tools} == EXPECTED_23_MCP_TOOLS
    finally:
        app.state.record.close()


def test_engine_rules_controls_approval_strength_and_all_tools(tmp_path: Path) -> None:
    app = create_app(tmp_path / "workspace", stub_mode=True)
    svc = app.state.engine_service
    rec = app.state.record
    try:
        # 1. Compute and safe model loading
        compute = svc.get_compute()
        assert compute["device"] == "cpu"

        unsafe_dir = tmp_path / "unsafe_model"
        unsafe_dir.mkdir()
        (unsafe_dir / "pytorch_model.bin").write_bytes(b"unsafe")
        with pytest.raises(ValueError, match="safetensors"):
            svc.load_model(str(unsafe_dir))

        loaded = svc.load_model("Qwen/Qwen2.5-0.5B-Instruct")
        model_id = loaded["model_id"]
        assert model_id == "hf:Qwen/Qwen2.5-0.5B-Instruct@main"
        assert loaded["is_placeholder"] is True

        mmap = svc.get_model_map(model_id)
        assert mmap["n_layers"] == 24

        confirmed = svc.confirm_components(model_id, {"model.layers.0.self_attn": "attention"})
        assert confirmed["confirmed_by_user"] is True

        # 2. Register series
        series_raw = json.loads(
            (FIXTURES_DIR / "series_pythia70m.json").read_text(encoding="utf-8")
        )
        reg = svc.register_series(
            kind=series_raw["kind"],
            items=series_raw["items"],
            model_family=series_raw["model_family"],
            series_id=series_raw["series_id"],
        )
        assert reg["series_id"] == "series_pythia70m_pretrain"

        # 3. Checks & plan drafting + required-control enforcement + approval reset
        checks = svc.list_checks()
        check_names = {c["name"] for c in checks}
        assert {"sycophancy", "refusal", "backdoor", "hidden_bias", "drift"} <= check_names

        desc = svc.describe_check("backdoor")
        assert "clean_reference_model" in desc["required_controls"]

        draft = svc.draft_plan(
            check="sycophancy",
            model_id=model_id,
            layers=[0, 1, 2],
            question="Did fine-tuning increase sycophancy?",
        )
        plan_id = draft["plan_id"]
        assert draft["approved"] is False
        assert draft["is_placeholder"] is True

        # Cannot run before approval
        with pytest.raises(PlanApprovalError):
            svc.run_plan(plan_id)

        # Cannot drop a required control
        with pytest.raises(ValueError, match="Cannot remove required control"):
            svc.revise_plan(plan_id, {"controls": ["shuffled_labels"]})

        # Approve v1, then revise -> version increments to 2 and resets approved=False
        rec.approve_plan(plan_id)
        revised_out = svc.revise_plan(plan_id, {"shift_tests": ["paraphrase"]})
        assert revised_out["plan"]["version"] == 2
        assert revised_out["plan"]["approved"] is False
        assert len(revised_out["warnings"]) == 1

        # Running revised v2 without re-approval fails
        with pytest.raises(PlanApprovalError):
            svc.run_plan(plan_id)

        # Running with approved_by_user=True (MCP client prompt approval) approves & executes
        run_res = svc.run_plan(plan_id, approved_by_user=True)
        run_id = run_res["run_id"]
        assert run_res["is_placeholder"] is True
        assert svc.job_status(run_id)["status"] == "completed"

        # 4. Decision pause/answer and stop_job
        paused_plan = svc.draft_plan(
            check="sycophancy",
            model_id=model_id,
            layers=[0, 1],
            user_context={"pause_for_decision": "Weak effect; continue with steering sweep?"},
        )
        paused_run = svc.run_plan(paused_plan["plan_id"], approved_by_user=True)
        paused_id = paused_run["job_id"]
        assert svc.job_status(paused_id)["status"] == "needs_decision"
        ans = svc.answer_job(paused_id, "yes, continue")
        assert ans["status"] == "completed"
        stopped = svc.stop_job(paused_id)
        assert stopped["status"] == "stopped"

        # 5. Findings, results, examples, 3D view events, report, export
        findings = svc.get_findings(run_id)
        assert len(findings) == 1
        finding_id = findings[0]["finding_id"]
        assert findings[0]["is_placeholder"] is True
        assert findings[0]["verified"] is False

        results = svc.get_results(run_id=run_id, metric="probe_accuracy")
        assert len(results) == 3

        examples = svc.get_examples(run_id, n=3)
        assert len(examples) == 1

        assert svc.view_focus(["L2"], "Peak layer")["ok"] is True
        assert svc.view_paint(run_id, "probe_accuracy")["painted_components"] == 3
        assert svc.view_timeline("step1000")["ok"] is True

        rep = svc.build_report(run_id)
        assert Path(rep["path"]).is_file()

        exp = svc.export_artifact(run_id, "probe")
        assert Path(exp["path"]).is_file()
        assert exp["is_placeholder"] is True

        # 6. Evidence & reviewer strength lowering only (never raising!)
        ev = svc.get_evidence(finding_id)
        assert ev["finding"]["finding_id"] == finding_id

        with pytest.raises(ValueError, match="cannot raise finding strength"):
            svc.submit_review(
                finding_id,
                agrees=False,
                notes="Attempting invalid upgrade",
                lower_strength_to="established",
            )

        lowered = svc.submit_review(
            finding_id,
            agrees=False,
            notes="Single seed on synthetic placeholder data.",
            corrected_claim="Inconclusive synthetic test signal.",
            lower_strength_to="inconclusive",
        )
        assert lowered["finding"]["strength"] == "inconclusive"
        assert lowered["finding"]["claim"] == "Inconclusive synthetic test signal."
        assert lowered["finding"]["is_placeholder"] is True
        assert lowered["finding"]["verified"] is False

        with pytest.raises(ValueError, match="cannot raise finding strength"):
            svc.submit_review(
                finding_id,
                agrees=True,
                lower_strength_to="suggestive",
            )

        est_finding = Finding(
            finding_id="finding_est_test",
            run_id=run_id,
            check="sycophancy",
            claim="Placeholder established strength test.",
            strength="established",
            maturity="established",
            evidence=[f"{run_id}:probe_accuracy"],
            controls=[ControlResult(kind="shuffled_labels", value=0.51, passed=True)],
            shift_tests=[ControlResult(kind="paraphrase", value=0.75, passed=True)],
            coverage="Layers 0 to 2.",
            not_shown="Placeholder.",
            verified=False,
            is_placeholder=True,
        )
        rec.save_finding(est_finding)
        stepped_down = svc.submit_review(
            "finding_est_test",
            verdict={
                "agrees": False,
                "notes": "Lower to suggestive",
                "lower_strength_to": "suggestive",
            },
        )
        assert stepped_down["finding"]["strength"] == "suggestive"
    finally:
        rec.close()


def test_http_endpoints_and_bearer_auth(tmp_path: Path) -> None:
    token = "local_test_token_123"
    app = create_app(tmp_path / "workspace", stub_mode=True, auth_token=token)
    svc = app.state.engine_service
    try:
        # Unauthorized without token
        status, _, _ = _asgi_http(app, "GET", "/api/models")
        assert status == 401

        status_mcp, _, _ = _asgi_http(app, "POST", "/mcp", json_body={"jsonrpc": "2.0"})
        assert status_mcp == 401

        headers = {"Authorization": f"Bearer {token}"}

        # POST /api/models/load & GET /api/models/{id:path}/map with slash in ID
        st, _, raw = _asgi_http(
            app,
            "POST",
            "/api/models/load",
            headers=headers,
            json_body={"source": "Qwen/Qwen2.5-0.5B-Instruct"},
        )
        assert st == 200
        model_id = json.loads(raw)["model_id"]
        assert "/" in model_id

        st, _, raw = _asgi_http(app, "GET", "/api/models", headers=headers)
        assert st == 200
        assert len(json.loads(raw)) == 1

        st, _, raw = _asgi_http(app, "GET", f"/api/models/{model_id}/map", headers=headers)
        assert st == 200
        assert json.loads(raw)["model_id"] == model_id

        # Series
        series_raw = json.loads(
            (FIXTURES_DIR / "series_pythia70m.json").read_text(encoding="utf-8")
        )
        svc.register_series(
            kind=series_raw["kind"],
            items=series_raw["items"],
            model_family=series_raw["model_family"],
            series_id=series_raw["series_id"],
        )
        st, _, raw = _asgi_http(
            app, "GET", "/api/series/series_pythia70m_pretrain", headers=headers
        )
        assert st == 200
        assert len(json.loads(raw)["items"]) == 6

        # Plans, approval, run, jobs, results, findings, examples, reports, setup
        draft = svc.draft_plan(check="refusal", model_id=model_id, layers=[0, 1])
        plan_id = draft["plan_id"]

        st, _, raw = _asgi_http(app, "GET", f"/api/plans/{plan_id}", headers=headers)
        assert st == 200
        assert json.loads(raw)["approved"] is False

        st, _, raw = _asgi_http(app, "POST", f"/api/plans/{plan_id}/approve", headers=headers)
        assert st == 200
        assert json.loads(raw)["approved"] is True

        run_out = svc.run_plan(plan_id)
        run_id = run_out["run_id"]

        st, _, raw = _asgi_http(app, "GET", "/api/jobs", headers=headers)
        assert st == 200
        assert any(j["job_id"] == run_id for j in json.loads(raw))

        st, _, raw = _asgi_http(app, "GET", f"/api/jobs/{run_id}", headers=headers)
        assert st == 200
        assert json.loads(raw)["status"] == "completed"

        st, _, raw = _asgi_http(
            app, "GET", "/api/results", params={"run_id": run_id}, headers=headers
        )
        assert st == 200
        assert len(json.loads(raw)) == 2

        st, _, raw = _asgi_http(
            app, "GET", "/api/findings", params={"run_id": run_id}, headers=headers
        )
        assert st == 200
        assert json.loads(raw)[0]["verified"] is False

        st, _, raw = _asgi_http(
            app, "GET", "/api/examples", params={"run_id": run_id, "n": 3}, headers=headers
        )
        assert st == 200
        assert json.loads(raw)[0]["redacted"] is True

        rep_meta = svc.build_report(run_id)
        st, _, raw = _asgi_http(app, "GET", "/api/reports", headers=headers)
        assert st == 200
        assert len(json.loads(raw)) == 1

        st, _, raw_html = _asgi_http(
            app, "GET", f"/api/reports/{rep_meta['report_id']}", headers=headers
        )
        assert st == 200
        assert "Placeholder / unverified test run" in raw_html.decode("utf-8")

        st, _, raw = _asgi_http(app, "GET", "/api/setup/claude-code", headers=headers)
        assert st == 200
        assert "claude mcp add" in json.loads(raw)["command"]
    finally:
        app.state.record.close()


def test_websockets_events_and_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def make_scripted_session(**kwargs: Any) -> ChatSession:
        def driver(
            text: str,
            _context: dict[str, Any] | None,
            _session: ChatSession,
        ) -> list[dict[str, Any]]:
            return [{"type": "text", "text": f"Echo: {text}"}]

        return ChatSession(
            engine_mcp_url=kwargs["engine_mcp_url"],
            workspace=kwargs["workspace"],
            scripted_driver=driver,
            engine_service=kwargs["engine_service"],
        )

    app = create_app(
        tmp_path / "workspace",
        stub_mode=True,
        chat_session_factory=make_scripted_session,
    )
    try:
        # Scripted /ws/chat session with 3D component context
        chat_msgs = _asgi_ws(
            app,
            "/ws/chat",
            [json.dumps({"text": "What does this layer do?", "context": {"component_id": "L4"}})],
        )
        sent_payloads = [
            json.loads(m["text"]) for m in chat_msgs if m["type"] == "websocket.send"
        ]
        assert sent_payloads[0]["type"] == "text"
        assert "L4" in sent_payloads[0]["text"]
        assert sent_payloads[1]["type"] == "turn_complete"

        # Default /ws/chat without ANTHROPIC_API_KEY returns calm notice without secrets
        app_no_key = create_app(tmp_path / "workspace_nokey", stub_mode=True)
        try:
            nokey_msgs = _asgi_ws(
                app_no_key,
                "/ws/chat",
                [json.dumps({"text": "Hello"})],
            )
            nokey_payloads = [
                json.loads(m["text"]) for m in nokey_msgs if m["type"] == "websocket.send"
            ]
            assert "ANTHROPIC_API_KEY is not set" in nokey_payloads[0]["text"]
            assert nokey_payloads[1]["type"] == "turn_complete"
        finally:
            app_no_key.state.record.close()

        # Live event bus subscriber delivery
        bus = app.state.engine_service.event_bus
        sub = bus.subscribe()
        # Prime generator up to queue wait or pre-fill via publish
        from engine.contracts import Event

        # Verify history & live subscriber fan-out
        app.state.engine_service.view_focus(["L4"], "Inspecting layer 4")
        assert any(
            ev.type == "focus" and ev.payload["component_ids"] == ["L4"] for ev in bus.history
        )
        assert isinstance(Event(type="timeline", payload={"checkpoint_id": "step1000"}), Event)
        _drive_sync(sub.aclose())
    finally:
        app.state.record.close()


def test_cli_init_doctor_jobs_and_report(tmp_path: Path) -> None:
    runner = CliRunner()
    ws = tmp_path / "cli_workspace"

    res_init = runner.invoke(cli_app, ["init", "--workspace", str(ws)])
    assert res_init.exit_code == 0
    assert (ws / "record.sqlite").is_file()

    res_doc = runner.invoke(cli_app, ["doctor", "--workspace", str(ws)])
    assert res_doc.exit_code == 0
    assert "Compute profile:" in res_doc.stdout
    assert "mcp:" in res_doc.stdout

    res_jobs_empty = runner.invoke(cli_app, ["jobs", "--workspace", str(ws)])
    assert res_jobs_empty.exit_code == 0
    assert "No jobs recorded" in res_jobs_empty.stdout

    app = create_app(ws, stub_mode=True)
    try:
        svc = app.state.engine_service
        loaded = svc.load_model("Qwen/Qwen2.5-0.5B-Instruct")
        plan = svc.draft_plan(check="sycophancy", model_id=loaded["model_id"], layers=[0, 1])
        run_info = svc.run_plan(plan["plan_id"], approved_by_user=True)
        run_id = run_info["run_id"]
    finally:
        app.state.record.close()

    res_jobs = runner.invoke(cli_app, ["jobs", "--workspace", str(ws)])
    assert res_jobs.exit_code == 0
    assert run_id in res_jobs.stdout

    res_rep = runner.invoke(cli_app, ["report", run_id, "--workspace", str(ws)])
    assert res_rep.exit_code == 0
    assert "Report written to:" in res_rep.stdout
