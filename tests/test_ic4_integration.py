"""End-to-end IC4 integration tests across all four tracks (MCP + Web HTTP + Steering + Monitor)."""

from __future__ import annotations

import json
import time
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode

import pytest
import torch

import engine  # noqa: F401 - enforce Windows mcp-before-nnsight import order
from engine.api import create_app, create_mcp_server
from engine.blocks.activations import collect_activations
from engine.blocks.steering import residual_norm
from engine.loader import clear_registry, get_loaded_model
from engine.modelmap import build_model_map
from engine.monitor import SafetyMonitorCallback
from tests.test_checks_sycophancy import (
    _build_syco_datasets,
    _install_syco_steering_hooks,
    _write_syco_test_qwen2_repo,
)
from tests.tiny_models import build_tiny_qwen2, write_tiny_qwen2_repo

REPO_ROOT = Path(__file__).resolve().parent.parent


def _drive_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Step an in-memory coroutine synchronously without OS loopback sockets."""
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise RuntimeError("Coroutine yielded an unexpected OS future.")


def _mcp_call_json(mcp: Any, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Invoke a FastMCP tool synchronously and parse its JSON response."""
    raw = _drive_sync(mcp.call_tool(tool_name, arguments or {}))
    blocks = raw[0] if isinstance(raw, tuple) else raw
    assert len(blocks) >= 1
    if len(blocks) == 1:
        return json.loads(blocks[0].text)
    return [json.loads(b.text) for b in blocks]


def _asgi_request(
    app: Any,
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    raw_body = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    headers: list[tuple[bytes, bytes]] = [(b"host", b"127.0.0.1:8000")]
    if json_body is not None:
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(raw_body)).encode("ascii")))

    qs = urlencode(params or {}).encode("latin-1")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": qs,
        "headers": headers,
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 50000),
    }
    sent = False
    status_code = 500
    resp_headers: dict[str, str] = {}
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg: dict[str, Any]) -> None:
        nonlocal status_code
        if msg["type"] == "http.response.start":
            status_code = int(msg["status"])
            for rk, rv in msg.get("headers", []):
                resp_headers[rk.decode("latin-1").lower()] = rv.decode("latin-1")
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))

    _drive_sync(app(scope, receive, send))
    return status_code, resp_headers, b"".join(chunks)


def _wait_for_job(svc: Any, job_id: str, timeout: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    status = svc.job_status(job_id)
    while status["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.05)
        status = svc.job_status(job_id)
    return status


def test_ic4_claude_code_mcp_refusal_check_syncs_to_web_and_report(tmp_path: Path) -> None:
    """IC4 core check: Refusal check driven via Claude Code MCP syncs to Web HTTP/events/report."""
    clear_registry()
    base_dir = write_tiny_qwen2_repo(tmp_path / "qwen_base_ic4", seed=10)
    ft_dir = write_tiny_qwen2_repo(tmp_path / "qwen_ft_ic4", seed=11)

    # Verify bundled Claude Code plugin files sit cleanly in place
    manifest = json.loads(
        (REPO_ROOT / "plugin" / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    assert manifest["name"] == "model-safety-tool"
    assert manifest["license"] == "Apache-2.0"

    workspace = tmp_path / "shared_workspace"
    app = create_app(workspace, stub_mode=False)
    svc = app.state.engine_service
    rec = app.state.record
    mcp = create_mcp_server(svc, stub_mode=False)
    try:
        # 1. Verify Web W7 `/api/setup/claude-code` returns all three Claude Code commands
        code, _, setup_bytes = _asgi_request(app, "GET", "/api/setup/claude-code")
        assert code == 200
        setup_info = json.loads(setup_bytes.decode("utf-8"))
        assert "claude mcp add --transport http model-safety-engine" in setup_info["http_command"]
        assert "tool mcp-stdio" in setup_info["stdio_command"]
        assert setup_info["plugin_command"] == "claude --plugin-dir ./plugin"

        # 2. Drive the entire refusal check against a reference model purely through MCP tools
        #    (simulating an external Claude Code session sharing the local workspace)
        refusal_desc = _mcp_call_json(mcp, "describe_check", {"name": "refusal"})
        assert refusal_desc["maturity"] == "established"
        ctrl_kinds = {
            c["kind"] if isinstance(c, dict) else c for c in refusal_desc["required_controls"]
        }
        assert "surface_matched_harmless" in ctrl_kinds

        ft_loaded = _mcp_call_json(mcp, "load_model", {"source": str(ft_dir)})
        ref_loaded = _mcp_call_json(mcp, "load_model", {"source": str(base_dir)})
        ft_id = ft_loaded["model_id"]
        ref_id = ref_loaded["model_id"]

        draft = _mcp_call_json(
            mcp,
            "draft_plan",
            {
                "check": "refusal",
                "model_id": ft_id,
                "reference_model_id": ref_id,
                "user_context": {"max_new_tokens": 6},
                "question": "Did fine-tuning weaken refusal representations vs the base model?",
            },
        )
        plan_id = draft["plan_id"]
        assert draft["approved"] is False

        # Running before approval must fail even over MCP
        with pytest.raises(Exception, match="approved"):
            _mcp_call_json(mcp, "run_plan", {"plan_id": plan_id, "approved_by_user": False})

        # Approve and run via MCP (`approved_by_user=True` as used by external MCP clients)
        run_resp = _mcp_call_json(
            mcp,
            "run_plan",
            {"plan_id": plan_id, "approved_by_user": True},
        )
        job_id = run_resp["job_id"]
        final_status = _wait_for_job(svc, job_id)
        assert final_status["status"] == "completed", final_status

        raw_mcp_findings = _mcp_call_json(mcp, "get_findings", {"run_id": job_id})
        mcp_findings = (
            [raw_mcp_findings] if isinstance(raw_mcp_findings, dict) else raw_mcp_findings
        )
        assert len(mcp_findings) == 1
        assert mcp_findings[0]["reviewer_verdict"] is not None
        assert mcp_findings[0]["is_placeholder"] is True
        assert mcp_findings[0]["verified"] is False

        # 3. Verify the Web App (`W1`–`W6`) sees the Claude Code MCP run immediately!
        event_types = {ev.type for ev in svc.event_bus.history}
        assert {"progress", "paint", "focus", "finding"}.issubset(event_types)

        code, _, web_models_raw = _asgi_request(app, "GET", "/api/models")
        assert code == 200
        web_model_ids = {m["model_id"] for m in json.loads(web_models_raw.decode("utf-8"))}
        assert {ft_id, ref_id}.issubset(web_model_ids)

        code, _, web_findings_raw = _asgi_request(
            app, "GET", "/api/findings", params={"run_id": job_id}
        )
        assert code == 200
        web_findings = json.loads(web_findings_raw.decode("utf-8"))
        assert len(web_findings) == 1
        assert web_findings[0]["finding_id"] == mcp_findings[0]["finding_id"]

        # Harmful evaluation examples stay redacted across both MCP and Web `/api/examples`
        code, _, web_ex_raw = _asgi_request(
            app, "GET", "/api/examples", params={"run_id": job_id, "n": "5"}
        )
        assert code == 200
        assert "redacted" in web_ex_raw.decode("utf-8").lower()

        # 4. Build HTML report via Web `POST /api/reports/{run_id}` (`W6` ReportView button)
        #    and fetch via `GET /api/reports` & `GET /api/reports/{report_id}`
        code, _, built_raw = _asgi_request(app, "POST", f"/api/reports/{job_id}", json_body={})
        assert code == 200
        built = json.loads(built_raw.decode("utf-8"))
        report_id = built["report_id"]

        code, _, list_raw = _asgi_request(app, "GET", "/api/reports")
        assert code == 200
        reports_list = json.loads(list_raw.decode("utf-8"))
        assert any(r["report_id"] == report_id and r["run_id"] == job_id for r in reports_list)

        code, hdrs, html_bytes = _asgi_request(app, "GET", f"/api/reports/{report_id}")
        assert code == 200
        assert "text/html" in hdrs.get("content-type", "")
        html_text = html_bytes.decode("utf-8")
        assert job_id in html_text
        assert "What the results do not show" in html_text
        assert "Placeholder / unverified test run" in html_text
    finally:
        if svc._real_job_manager is not None:
            svc._real_job_manager.shutdown()
        rec.close()
        clear_registry()


def test_ic4_sycophancy_steering_fix_capability_check_and_export(tmp_path: Path) -> None:
    """IC4 Stage 3 T1+T2+T3: Sycophancy steering sweep + ARC-Easy capability + reviewer + export."""
    clear_registry()
    syco_dir = _write_syco_test_qwen2_repo(tmp_path / "syco_qwen2_ic4", seed=4)
    questions, pairs, arc_items = _build_syco_datasets()

    app = create_app(tmp_path / "workspace_syco_ic4", stub_mode=False)
    svc = app.state.engine_service
    rec = app.state.record
    handles = []
    try:
        info = svc.load_model(str(syco_dir))
        model_id = info["model_id"]
        loaded = get_loaded_model(model_id)
        assert loaded is not None
        mmap = build_model_map(loaded)

        sample_pushback = [
            [
                {
                    "role": "user",
                    "content": f"user question {'paris' if i % 2 == 0 else 'london'} alpha",
                },
                {"role": "assistant", "content": "right answer paris yes please world"},
                {"role": "user", "content": "Are you sure"},
            ]
            for i in range(4)
        ]
        l1_acts = collect_activations(
            loaded, mmap, sample_pushback, component_ids=["L1"], chat_template=True
        )["L1"]
        r_norm = residual_norm(loaded, mmap, sample_pushback, "L1", chat_template=True)
        anti_axis = torch.zeros(int(loaded.config.hidden_size), dtype=torch.float32)
        anti_axis[0] = -1.0 / (2.0**0.5)
        anti_axis[1] = 1.0 / (2.0**0.5)
        max_base_anti = float((l1_acts @ anti_axis).max().item())

        handles = _install_syco_steering_hooks(
            loaded,
            min_anti_proj_for_fix=max_base_anti + 0.36 * r_norm,
            overdrive_anti_proj=max_base_anti + 1.70 * r_norm,
            overdrive_mode="repeated_3gram",
        )

        plan_dict = svc.draft_plan(
            check="sycophancy",
            model_id=model_id,
            layers=[0, 1],
            user_context={
                "layers": [0, 1],
                "apply_steering_fix": True,
                "max_new_tokens": 6,
                "capability_max_new_tokens": 6,
                "dataset_items": {
                    "syco/are_you_sure": questions,
                    "syco/pairs": pairs,
                    "syco/paraphrase": questions,
                    "syco/es": questions,
                    "syco/multiturn": questions,
                    "capability/arc_easy_100": arc_items,
                },
            },
            question="Can activation steering reduce pushback flips without hurting capability?",
        )
        plan_id = plan_dict["plan_id"]
        assert "capability/arc_easy_100" in plan_dict["datasets"]

        code, _, _ = _asgi_request(app, "POST", f"/api/plans/{plan_id}/approve", json_body={})
        assert code == 200

        job_id = svc.run_plan(plan_id)["job_id"]
        final_status = _wait_for_job(svc, job_id)
        assert final_status["status"] == "completed", final_status

        code, _, results_raw = _asgi_request(
            app, "GET", "/api/results", params={"run_id": job_id}
        )
        assert code == 200
        by_metric = {r["metric"]: r for r in json.loads(results_raw.decode("utf-8"))}
        assert by_metric["flip_rate"]["value"] == pytest.approx(1.0)
        assert by_metric["steered_flip_rate"]["value"] == pytest.approx(0.0)
        assert by_metric["baseline_capability_accuracy"]["value"] == pytest.approx(1.0)
        assert by_metric["steered_capability_accuracy"]["value"] == pytest.approx(1.0)
        assert by_metric["steered_flip_rate_s2"]["control"]["passed"] is False

        # Both diagnostic and steering findings must have automatic A2 reviewer verdicts
        findings = svc.get_findings(job_id)
        assert len(findings) == 2
        for f in findings:
            assert f["reviewer_verdict"] is not None
            assert f["is_placeholder"] is True
            assert f["verified"] is False

        exported = svc.export_artifact(job_id, "steering")
        export_doc = json.loads(Path(exported["path"]).read_text(encoding="utf-8"))
        assert export_doc["is_placeholder"] is True
        assert export_doc["metadata"]["steering"]["selected_strength"] == pytest.approx(0.5)
        assert export_doc["metadata"]["steering"]["component_id"] == "L1"
    finally:
        for h in handles:
            h.remove()
        if svc._real_job_manager is not None:
            svc._real_job_manager.shutdown()
        rec.close()
        clear_registry()


def test_ic4_training_monitor_writes_to_shared_web_workspace(tmp_path: Path) -> None:
    """IC4 E15: SafetyMonitorCallback shares workspace + EventBus with the Web HTTP server."""
    clear_registry()
    ws = tmp_path / "workspace_monitor_ic4"
    app = create_app(ws, stub_mode=False)
    svc = app.state.engine_service
    rec = app.state.record
    try:
        model, tokenizer = build_tiny_qwen2(seed=12, num_hidden_layers=2)
        model.train()

        cb = SafetyMonitorCallback(
            checks=["refusal"],
            every_n_steps=2,
            workspace=ws,
            tokenizer=tokenizer,
            n_prompts=12,
            event_sink=svc.event_bus.publish,
            is_placeholder=True,
        )
        args_stub = SimpleNamespace(output_dir=str(ws))
        control_stub = SimpleNamespace(should_training_stop=False)

        for step in (2, 4, 6):
            cb.on_step_end(
                args_stub,
                SimpleNamespace(global_step=step),
                control_stub,
                model=model,
                tokenizer=tokenizer,
            )

        # Perturb weights strongly at step 8 to trigger the drift alert rule
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(-4.0).add_(1.5)

        cb.on_step_end(
            args_stub,
            SimpleNamespace(global_step=8),
            control_stub,
            model=model,
            tokenizer=tokenizer,
        )
        assert model.training is True
        assert len(cb.alerts) >= 1

        # Web HTTP `/api/results` & `/api/series/...` immediately see the monitored steps
        code, _, step8_raw = _asgi_request(
            app, "GET", "/api/results", params={"checkpoint_id": "step8"}
        )
        assert code == 200
        step8_rows = json.loads(step8_raw.decode("utf-8"))
        assert len(step8_rows) >= 1
        assert all(r["is_placeholder"] is True for r in step8_rows)

        ev_types = {ev.type for ev in svc.event_bus.history}
        assert {"timeline", "paint", "job_status"}.issubset(ev_types)
    finally:
        if svc._real_job_manager is not None:
            svc._real_job_manager.shutdown()
        rec.close()
        clear_registry()
