"""End-to-end IC1 and IC2 integration test across all four tracks (non-stub engine)."""

from __future__ import annotations

import json
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

from agent.hooks import make_plan_approval_hook
from engine.api import PlanApprovalError, create_app
from tests.tiny_models import write_tiny_qwen2_repo


def _drive_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise RuntimeError("Coroutine yielded an unexpected OS future.")


def _asgi_post_json(app: Any, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body).encode("utf-8")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1:8000"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode("ascii")),
        ],
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 50000),
    }
    sent = False
    status_code = 500
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg: dict[str, Any]) -> None:
        nonlocal status_code
        if msg["type"] == "http.response.start":
            status_code = int(msg["status"])
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))

    _drive_sync(app(scope, receive, send))
    return status_code, json.loads(b"".join(chunks).decode("utf-8"))


def test_ic1_and_ic2_real_engine_sycophancy_pipeline(tmp_path: Path) -> None:
    base_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_base", seed=0)
    ref_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_ref", seed=1)

    app = create_app(tmp_path / "workspace", stub_mode=False)
    svc = app.state.engine_service
    rec = app.state.record
    try:
        # IC1: Load real safetensors model and build model map via forward-pass inspection
        code, load_resp = _asgi_post_json(app, "/api/models/load", {"source": str(base_dir)})
        assert code == 200, load_resp
        model_id = load_resp["model_id"]
        assert load_resp["n_layers"] == 2
        assert load_resp["arch_family"] == "qwen2"

        ref_resp = svc.load_model(str(ref_dir))
        ref_model_id = ref_resp["model_id"]

        mmap = svc.get_model_map(model_id)
        comp_ids = {c["id"] for c in mmap["components"]}
        assert {"embed", "L0", "L0.attn", "L0.mlp", "L1", "final_norm", "unembed"} <= comp_ids

        # Check catalog + settled decision #6 (data_poisoning maturity & description)
        checks = {c["name"]: c for c in svc.list_checks()}
        assert checks["sycophancy"]["maturity"] == "established"
        assert checks["data_poisoning"]["maturity"] == "experimental"
        dp_desc = svc.describe_check("data_poisoning")
        assert "partly established" in json.dumps(dp_desc).lower()

        # Draft sycophancy plan
        plan_dict = svc.draft_plan(
            check="sycophancy",
            model_id=model_id,
            reference_model_id=ref_model_id,
            user_context={"max_new_tokens": 6},
            question="Did fine-tuning make the model more sycophantic?",
        )
        plan_id = plan_dict["plan_id"]
        assert plan_dict["approved"] is False
        assert plan_dict["is_placeholder"] is True

        # Both A3 PreToolUse hook and E13 run_plan must refuse before user approval
        hook = make_plan_approval_hook(rec.workspace, record=rec)
        blocked = _drive_sync(
            hook(
                {
                    "tool_name": "mcp__engine__run_plan",
                    "tool_input": {"plan_id": plan_id},
                },
                "toolu_1",
                None,
            )
        )
        assert blocked["hookSpecificOutput"]["permissionDecision"] == "deny"
        with pytest.raises(PlanApprovalError):
            svc.run_plan(plan_id)

        # User approves plan via HTTP endpoint (simulating web chat approval)
        code, approved_resp = _asgi_post_json(app, f"/api/plans/{plan_id}/approve", {})
        assert code == 200
        assert approved_resp["approved"] is True

        allowed = _drive_sync(
            hook(
                {
                    "tool_name": "mcp__engine__run_plan",
                    "tool_input": {"plan_id": plan_id},
                },
                "toolu_2",
                None,
            )
        )
        assert allowed.get("hookSpecificOutput", {}).get("permissionDecision") in (None, "allow")

        # IC2: Execute real SycophancyCheck through background JobManager
        run_resp = svc.run_plan(plan_id)
        job_id = run_resp["job_id"]
        assert job_id not in svc._job_states  # Executed via real JobManager, not stub fallback

        deadline = time.monotonic() + 60.0
        status = svc.job_status(job_id)
        while status["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.05)
            status = svc.job_status(job_id)

        assert status["status"] == "completed", status

        # Verify live events streamed for 3D view & chat
        emitted_types = [ev.type for ev in svc.event_bus.history]
        assert "job_status" in emitted_types
        assert "progress" in emitted_types
        assert "paint" in emitted_types
        assert "focus" in emitted_types
        assert "finding" in emitted_types

        # Verify results, required controls, shift tests, and placeholder honesty flags
        results = svc.get_results(run_id=job_id)
        metrics = {r["metric"] for r in results}
        assert {"flip_rate", "probe_accuracy", "probe_auroc", "direction_projection_gap"} <= metrics
        assert all(r["is_placeholder"] is True for r in results)

        findings = svc.get_findings(job_id)
        assert len(findings) == 1
        finding = findings[0]
        assert finding["is_placeholder"] is True
        assert finding["verified"] is False
        ctrl_kinds = {c["kind"] for c in finding["controls"]}
        assert {"shuffled_labels", "random_direction", "held_out", "reference_model"} <= ctrl_kinds
        shift_kinds = {s["kind"] for s in finding["shift_tests"]}
        assert {"paraphrase", "multilingual", "multi_turn"} <= shift_kinds

        # Build standalone HTML report and verify required sections + placeholder banner
        rep = svc.build_report(job_id)
        html_path = Path(rep["path"])
        assert html_path.is_file()
        html = html_path.read_text(encoding="utf-8")
        assert "Placeholder / unverified test run" in html
        assert "What the results do not show" in html
        assert finding["finding_id"] in html
    finally:
        if svc._real_job_manager is not None:
            svc._real_job_manager.shutdown()
        rec.close()
