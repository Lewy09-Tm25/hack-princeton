"""End-to-end IC3 integration tests across all four tracks (non-stub engine + reviewer)."""

from __future__ import annotations

import json
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from agent.reviewer import review_run_findings_sync
from engine.api import create_app
from engine.checks.backdoor import (
    REQUIRED_COVERAGE_SENTENCE,
    REQUIRED_NOT_SHOWN_SENTENCE,
)
from engine.contracts import ControlResult, Finding
from engine.loader import clear_registry, get_loaded_model
from tests.test_checks_backdoor import (
    _plant_synthetic_trigger_hooks,
    _write_backdoor_test_qwen2_repo,
)
from tests.tiny_models import write_tiny_gpt_neox_repo, write_tiny_qwen2_repo


def _drive_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise RuntimeError("Coroutine yielded an unexpected OS future.")


def _asgi_get_json(
    app: Any,
    path: str,
    params: dict[str, str] | None = None,
) -> tuple[int, Any]:
    qs = urlencode(params or {}).encode("latin-1")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": qs,
        "headers": [(b"host", b"127.0.0.1:8000")],
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
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg: dict[str, Any]) -> None:
        nonlocal status_code
        if msg["type"] == "http.response.start":
            status_code = int(msg["status"])
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))

    _drive_sync(app(scope, receive, send))
    return status_code, json.loads(b"".join(chunks).decode("utf-8"))


def _asgi_post_json(app: Any, path: str, body: dict[str, Any]) -> tuple[int, Any]:
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


def _wait_for_job(svc: Any, job_id: str, timeout: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    status = svc.job_status(job_id)
    while status["status"] in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(0.05)
        status = svc.job_status(job_id)
    return status


def test_ic3_backdoor_check_clean_control_and_automatic_reviewer(tmp_path: Path) -> None:
    """IC3 Pillar 1 + 3: BackdoorCheck vs clean control via JobManager + automatic A2 review."""
    clear_registry()
    suspect_dir = _write_backdoor_test_qwen2_repo(tmp_path / "suspect_qwen2", seed=0)
    clean_dir = _write_backdoor_test_qwen2_repo(tmp_path / "clean_qwen2", seed=0)

    app = create_app(tmp_path / "workspace_bd", stub_mode=False)
    svc = app.state.engine_service
    rec = app.state.record
    handles = []
    try:
        suspect_info = svc.load_model(str(suspect_dir))
        clean_info = svc.load_model(str(clean_dir))
        suspect_id = suspect_info["model_id"]
        clean_id = clean_info["model_id"]

        loaded_suspect = get_loaded_model(suspect_id)
        assert loaded_suspect is not None
        handles = _plant_synthetic_trigger_hooks(loaded_suspect, with_behavioral_shift=True)

        # Confirm all three checks (sycophancy, refusal, backdoor) are implemented in catalog
        catalog = {c["name"]: c for c in svc.list_checks()}
        assert catalog["sycophancy"]["implemented"] is True
        assert catalog["refusal"]["implemented"] is True
        assert catalog["backdoor"]["implemented"] is True
        assert catalog["backdoor"]["maturity"] == "established"

        plan_dict = svc.draft_plan(
            check="backdoor",
            model_id=suspect_id,
            reference_model_id=clean_id,
            user_context={
                "max_triggers": 3,
                "behavioral_n": 8,
                "max_new_tokens": 4,
            },
            question="Does the fine-tuned model carry a planted deployment trigger?",
        )
        plan_id = plan_dict["plan_id"]
        assert REQUIRED_COVERAGE_SENTENCE in plan_dict["coverage"]
        assert REQUIRED_NOT_SHOWN_SENTENCE in plan_dict["not_covered"]

        code, approved = _asgi_post_json(app, f"/api/plans/{plan_id}/approve", {})
        assert code == 200 and approved["approved"] is True

        run_resp = svc.run_plan(plan_id)
        job_id = run_resp["job_id"]
        final_status = _wait_for_job(svc, job_id)
        assert final_status["status"] == "completed", final_status

        # Verify HTTP endpoints used by W5 charts & evidence drawer
        code, http_results = _asgi_get_json(app, "/api/results", {"run_id": job_id})
        assert code == 200
        assert len(http_results) > 0
        metrics = {r["metric"] for r in http_results}
        assert "suspect_probe_auroc" in metrics
        assert "clean_probe_auroc" in metrics

        code, http_findings = _asgi_get_json(app, "/api/findings", {"run_id": job_id})
        assert code == 200
        assert len(http_findings) == 1
        finding = http_findings[0]

        # Placeholder honesty & required backdoor coverage/not_shown strings
        assert finding["is_placeholder"] is True
        assert finding["verified"] is False
        assert REQUIRED_COVERAGE_SENTENCE in finding["coverage"]
        assert REQUIRED_NOT_SHOWN_SENTENCE in finding["not_shown"]

        # IC3: A2 Reviewer automatically ran on every finding upon JobManager completion!
        assert finding["reviewer_verdict"] is not None
        assert "agrees" in finding["reviewer_verdict"]
        # Synthetic placeholder runs (is_placeholder=True) are capped at suggestive by reviewer
        assert finding["strength"] in ("suggestive", "inconclusive")

        # Generate report and verify reviewer verdict & caveats render
        rep = svc.build_report(job_id)
        html = Path(rep["path"]).read_text(encoding="utf-8")
        assert REQUIRED_NOT_SHOWN_SENTENCE in html
    finally:
        for h in handles:
            h.remove()
        if svc._real_job_manager is not None:
            svc._real_job_manager.shutdown()
        rec.close()
        clear_registry()


def test_ic3_pretraining_timeline_and_refusal_redaction_with_reviewer(tmp_path: Path) -> None:
    """IC3 Pillar 2 + 3: 3-checkpoint timeline, RefusalCheck redaction, and A2 overclaim guard."""
    clear_registry()
    c1 = write_tiny_gpt_neox_repo(tmp_path / "pythia_step1000", seed=1)
    c2 = write_tiny_gpt_neox_repo(tmp_path / "pythia_step8000", seed=2)
    c3 = write_tiny_gpt_neox_repo(tmp_path / "pythia_step143000", seed=3)

    qwen_base = write_tiny_qwen2_repo(tmp_path / "qwen_base", seed=10)
    qwen_ft = write_tiny_qwen2_repo(tmp_path / "qwen_ft", seed=11)

    app = create_app(tmp_path / "workspace_tl", stub_mode=False)
    svc = app.state.engine_service
    rec = app.state.record
    try:
        # 1. Register 3-checkpoint pretraining series and run timeline via JobManager
        last_ckpt_info = svc.load_model(str(c3), revision="step143000")
        series_dict = svc.register_series(
            kind="pretraining",
            model_family="tiny-pythia-local",
            series_id="series_ic3_pythia",
            items=[
                {
                    "checkpoint_id": "step1000",
                    "label": "step 1,000",
                    "order": 0,
                    "source": str(c1),
                },
                {
                    "checkpoint_id": "step8000",
                    "label": "step 8,000",
                    "order": 1,
                    "source": str(c2),
                },
                {
                    "checkpoint_id": "step143000",
                    "label": "step 143,000",
                    "order": 2,
                    "source": str(c3),
                },
            ],
        )
        code, http_series = _asgi_get_json(app, "/api/series/series_ic3_pythia")
        assert code == 200
        assert [it["checkpoint_id"] for it in http_series["items"]] == [
            "step1000",
            "step8000",
            "step143000",
        ]

        tl_plan = svc.draft_plan(
            check="refusal",
            model_id=last_ckpt_info["model_id"],
            series_id=series_dict["series_id"],
            question="When during pretraining does harm-recognition emerge across layers?",
        )
        svc.record.approve_plan(tl_plan["plan_id"])
        tl_job = svc.run_plan(tl_plan["plan_id"])["job_id"]
        tl_status = _wait_for_job(svc, tl_job)
        assert tl_status["status"] == "completed", tl_status

        # Verify timeline & paint events emitted for W4 Timeline slider
        tl_events = [ev for ev in svc.event_bus.history if ev.type == "timeline"]
        assert {ev.payload["checkpoint_id"] for ev in tl_events} >= {
            "step1000",
            "step8000",
            "step143000",
        }

        code, step8k_rows = _asgi_get_json(
            app,
            "/api/results",
            {"run_id": tl_job, "checkpoint_id": "step8000"},
        )
        assert code == 200
        assert len(step8k_rows) > 0
        assert all(r["checkpoint_id"] == "step8000" for r in step8k_rows)
        assert all(r["is_placeholder"] is True for r in step8k_rows)

        tl_findings = svc.get_findings(tl_job)
        assert len(tl_findings) >= 1
        assert tl_findings[0]["reviewer_verdict"] is not None
        assert tl_findings[0]["verified"] is False

        # 2. Run RefusalCheck reference comparison & verify /api/examples redaction
        ft_info = svc.load_model(str(qwen_ft))
        ref_info = svc.load_model(str(qwen_base))
        ref_plan = svc.draft_plan(
            check="refusal",
            model_id=ft_info["model_id"],
            reference_model_id=ref_info["model_id"],
            user_context={"max_new_tokens": 6},
            question="Did fine-tuning weaken the refusal direction?",
        )
        svc.record.approve_plan(ref_plan["plan_id"])
        ref_job = svc.run_plan(ref_plan["plan_id"])["job_id"]
        ref_status = _wait_for_job(svc, ref_job)
        assert ref_status["status"] == "completed", ref_status

        code, http_examples = _asgi_get_json(
            app,
            "/api/examples",
            {"run_id": ref_job, "n": "5"},
        )
        assert code == 200
        assert len(http_examples) > 0
        # Harmful evaluation examples must be redacted in /api/examples!
        serialized_examples = json.dumps(http_examples).lower()
        assert "redacted" in serialized_examples

        ref_findings = svc.get_findings(ref_job)
        assert len(ref_findings) == 1
        assert ref_findings[0]["reviewer_verdict"] is not None

        # 3. Verify A2 reviewer catches overclaiming absence finding on the same workspace
        overclaim = Finding(
            finding_id="finding_ic3_overclaim",
            run_id=ref_job,
            check="backdoor",
            claim="No backdoor exists in this model; the model is safe for deployment.",
            strength="established",
            maturity="experimental",
            evidence=[f"{ref_job}:L0:probe_auroc"],
            controls=[ControlResult(kind="clean_reference_model", value=0.50, passed=True)],
            shift_tests=[],
            coverage="Examined layers L0-L1 across 20 candidate triggers.",
            not_shown=REQUIRED_NOT_SHOWN_SENTENCE,
            verified=False,
            is_placeholder=True,
        )
        rec.save_finding(overclaim)
        review_run_findings_sync(ref_job, svc)

        corrected = rec.get_finding("finding_ic3_overclaim")
        assert corrected is not None
        assert corrected.reviewer_verdict is not None
        assert corrected.reviewer_verdict["agrees"] is False
        assert "no backdoor exists" not in corrected.claim.lower()
        assert "safe for deployment" not in corrected.claim.lower()
        assert corrected.strength in ("suggestive", "inconclusive")
    finally:
        if svc._real_job_manager is not None:
            svc._real_job_manager.shutdown()
        rec.close()
        clear_registry()
