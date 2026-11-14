"""Tests for E9 experiment runner: EventBus, approval gate, ask/answer, stop, and Kernel."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from engine.checks.base import (
    ControlSpec,
    DatasetRef,
    Plan,
    Requirements,
    RiskCheck,
    ShiftTestSpec,
)
from engine.contracts import Event, Finding, ModelMap, ResultRecord, Series
from engine.runner import EventBus, JobManager, Kernel, sanitized_kernel_env


def _make_approved_plan(check_name: str = "fake_check", approved: bool = True) -> Plan:
    return Plan(
        plan_id="plan_runner_01",
        check=check_name,
        model_id="local:tiny_test",
        datasets=["syco/pairs"],
        layers=[0],
        controls=["shuffled_labels"],
        shift_tests=["paraphrase"],
        estimate_seconds=1.0,
        coverage="Layer 0 synthetic test.",
        not_covered="Placeholder runner test.",
        maturity="established",
        papers=["Test 2025"],
        approved=approved,
        is_placeholder=True,
    )


class _FakeInteractiveCheck(RiskCheck):
    name = "fake_interactive"
    summary = "Fake check exercising progress, ask/answer, and cancellation."
    maturity = "established"
    papers = ["Test 2025"]

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self.received_answer: str | None = None

    def default_datasets(self) -> list[DatasetRef]:
        return [DatasetRef(dataset_id="syco/pairs")]

    def required_controls(self) -> list[ControlSpec]:
        return [ControlSpec(kind="shuffled_labels")]

    def required_shift_tests(self) -> list[ShiftTestSpec]:
        return [ShiftTestSpec(kind="paraphrase")]

    def needs(self) -> Requirements:
        return Requirements()

    def build_plan(
        self,
        model_map: ModelMap,
        series: Series | None = None,
        user_context: dict[str, Any] | None = None,
        reference_model_id: str | None = None,
    ) -> Plan:
        return _make_approved_plan(self.name, approved=False)

    def coverage(self, plan: Plan) -> str:
        return "Synthetic test coverage."

    def summarize(
        self, results: list[ResultRecord], plan: Plan | None = None
    ) -> list[Finding]:
        return []

    def run(self, plan: Plan, ctx: Any) -> list[ResultRecord]:
        ctx.emit(
            Event(
                type="progress",
                payload={"job_id": ctx.job_id, "step": 1, "total": 2, "message": "Step 1"},
            )
        )
        if self.mode == "ask":
            self.received_answer = ctx.ask("Effect is weak; continue with layer sweep?")
        elif self.mode == "slow_cancel":
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if ctx.cancelled():
                    return []
                time.sleep(0.02)

        ctx.emit(
            Event(
                type="progress",
                payload={"job_id": ctx.job_id, "step": 2, "total": 2, "message": "Done"},
            )
        )
        return [
            ResultRecord(
                run_id=ctx.run_id,
                check=self.name,
                component_id="L0",
                checkpoint_id=None,
                metric="probe_accuracy",
                value=0.75,
                dataset_id="syco/pairs",
                split="held_out",
                seed=0,
                device="cpu",
                precision="fp32",
                quantized=False,
                is_placeholder=True,
            )
        ]


def test_event_bus_streams_events_from_background_thread() -> None:
    bus = EventBus()
    received: list[Event] = []
    sub = bus.subscribe()

    # Prime the async generator so it registers its subscription before events fire.
    pending_step = sub.__anext__()
    assert pending_step.send(None) is None

    def _producer() -> None:
        bus.publish(
            "job_1",
            Event(type="job_status", payload={"job_id": "job_1", "status": "running"}),
        )
        bus.publish(
            "job_1",
            Event(
                type="progress",
                payload={"job_id": "job_1", "step": 1, "total": 1, "message": "Working"},
            ),
        )

    t = threading.Thread(target=_producer)
    t.start()
    t.join(timeout=2.0)

    # Resume the primed step for event 1, then step once more for event 2.
    try:
        pending_step.send(None)
    except StopIteration as done:
        received.append(done.value)

    second_step = sub.__anext__()
    try:
        second_step.send(None)
    except StopIteration as done:
        received.append(done.value)

    close_coro = sub.aclose()
    try:
        close_coro.send(None)
    except StopIteration:
        pass

    assert [e.type for e in received] == ["job_status", "progress"]
    assert len(bus.get_history("job_1")) == 2


def test_job_manager_refuses_unapproved_plan_and_completes_approved() -> None:
    check = _FakeInteractiveCheck(mode="normal")
    jm = JobManager(check_resolver=lambda _name: check)
    try:
        unapproved = _make_approved_plan(check.name, approved=False)
        with pytest.raises(ValueError, match="approved"):
            jm.submit(unapproved)

        approved = _make_approved_plan(check.name, approved=True)
        job_id = jm.submit(approved)
        final = jm.wait(job_id, timeout=5.0)
        assert final["status"] == "completed"
        assert final["n_results"] == 1
        assert len(jm.list_jobs()) == 1
    finally:
        jm.shutdown()


def test_job_manager_ask_and_answer_round_trip() -> None:
    check = _FakeInteractiveCheck(mode="ask")
    jm = JobManager(check_resolver=lambda _name: check)
    try:
        job_id = jm.submit(_make_approved_plan(check.name, approved=True))

        # Wait until the worker pauses in needs_decision.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            st = jm.status(job_id)
            if st["status"] == "needs_decision":
                break
            time.sleep(0.02)

        paused = jm.status(job_id)
        assert paused["status"] == "needs_decision"
        assert "continue" in (paused["pending_question"] or "")

        jm.answer(job_id, "yes, continue")
        final = jm.wait(job_id, timeout=5.0)
        assert final["status"] == "completed"
        assert check.received_answer == "yes, continue"
    finally:
        jm.shutdown()


def test_job_manager_stop_cancels_running_and_paused_jobs() -> None:
    slow_check = _FakeInteractiveCheck(mode="slow_cancel")
    jm = JobManager(check_resolver=lambda _name: slow_check)
    try:
        job_id = jm.submit(_make_approved_plan(slow_check.name, approved=True))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if jm.status(job_id)["status"] == "running":
                break
            time.sleep(0.02)

        jm.stop(job_id)
        final = jm.wait(job_id, timeout=5.0)
        assert final["status"] == "stopped"
    finally:
        jm.shutdown()


def test_resume_incomplete_marks_unfinished_runs_stopped() -> None:
    paused_check = _FakeInteractiveCheck(mode="ask")
    jm = JobManager(check_resolver=lambda _name: paused_check)
    try:
        job_id = jm.submit(_make_approved_plan(paused_check.name, approved=True))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if jm.status(job_id)["status"] == "needs_decision":
                break
            time.sleep(0.02)

        marked = jm.resume_incomplete()
        assert job_id in marked
        st = jm.wait(job_id, timeout=5.0)
        assert st["status"] == "stopped"
        assert "restarted" in st["message"]
    finally:
        jm.shutdown()


def test_kernel_sanitizes_secrets_and_proxies(tmp_path: Path) -> None:
    fake_env = {
        "PATH": "/usr/bin",
        "ANTHROPIC_API_KEY": "secret-1",
        "CUSTOM_SERVICE_TOKEN": "secret-2",
        "MY_APP_SECRET": "secret-3",
        "HTTPS_PROXY": "http://proxy.example:8080",
        "http_proxy": "http://proxy.example:8080",
        "SAFE_VARIABLE": "keep_me",
    }
    cleaned = sanitized_kernel_env(fake_env)
    assert "ANTHROPIC_API_KEY" not in cleaned
    assert "CUSTOM_SERVICE_TOKEN" not in cleaned
    assert "MY_APP_SECRET" not in cleaned
    assert "HTTPS_PROXY" not in cleaned
    assert "http_proxy" not in cleaned
    assert cleaned["SAFE_VARIABLE"] == "keep_me"
    assert cleaned["HF_HUB_OFFLINE"] == "1"

    kernel = Kernel()
    with pytest.raises(RuntimeError, match="start"):
        kernel.execute("1 + 1")
    kernel.shutdown()
