"""Background single-worker job queue for running approved interpretability plans."""

from __future__ import annotations

import queue
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from engine.checks.base import Plan, RiskCheck
from engine.checks.registry import get_check
from engine.contracts import Event, JobStatus, ResultRecord
from engine.runner.events import EventBus


class _JobCancelledError(Exception):
    """Internal signal raised when a paused or running job is stopped."""


@dataclass
class _JobEntry:
    job_id: str
    plan: Plan
    status: JobStatus = "queued"
    message: str = "Queued"
    pending_question: str | None = None
    answer_value: str | None = None
    error: str | None = None
    results: list[ResultRecord] = field(default_factory=list)
    messages: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    answer_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)


class RunContext:
    """Execution handle passed to `RiskCheck.run(plan, ctx)` inside the worker thread."""

    def __init__(
        self,
        job_id: str,
        manager: JobManager,
        record: Any | None,
        cancel_event: threading.Event,
    ) -> None:
        self.job_id = job_id
        self.run_id = job_id
        self.record = record
        self._manager = manager
        self._cancel_event = cancel_event

    def emit(self, event: Event) -> None:
        """Publish a live event (`progress`, `paint`, `focus`, `timeline`, `finding`)."""
        self._manager._record_event(self.job_id, event)

    def ask(self, question: str) -> str:
        """Pause the job, emit a `question` event, and block until answered or stopped."""
        return self._manager.ask(self.job_id, question)

    def cancelled(self) -> bool:
        """Return True if `stop(job_id)` has been requested."""
        return self._cancel_event.is_set()


class JobManager:
    """Sequential background worker executing approved experiment plans."""

    def __init__(
        self,
        record: Any | None = None,
        event_bus: EventBus | None = None,
        check_resolver: Callable[[str], RiskCheck] | None = None,
    ) -> None:
        self.record = record
        self.event_bus = event_bus or EventBus()
        self._check_resolver = check_resolver or get_check
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobEntry] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="engine-job-worker",
            daemon=True,
        )
        self._worker.start()

    def _set_record_status(self, job_id: str, status: JobStatus, message: str = "") -> None:
        if self.record is not None and hasattr(self.record, "set_run_status"):
            try:
                self.record.set_run_status(job_id, status, message)
            except TypeError:
                self.record.set_run_status(job_id, status)

    def _record_event(self, job_id: str, event: Event) -> None:
        msg = event.payload.get("message") or event.payload.get("text")
        if isinstance(msg, str) and msg:
            with self._lock:
                entry = self._jobs.get(job_id)
                if entry is not None:
                    entry.messages.append(msg)
        self.event_bus.publish(job_id, event)

    def _transition(self, entry: _JobEntry, status: JobStatus, message: str = "") -> None:
        with self._lock:
            entry.status = status
            if message:
                entry.message = message
                entry.messages.append(message)
        self._set_record_status(entry.job_id, status, message)
        self.event_bus.publish(
            entry.job_id,
            Event(
                type="job_status",
                payload={
                    "job_id": entry.job_id,
                    "status": status,
                    "message": message or None,
                },
            ),
        )

    def submit(self, plan: Plan) -> str:
        """Enqueue an approved `Plan` for background execution and return its `job_id`."""
        if not isinstance(plan, Plan):
            plan = Plan.model_validate(plan)
        if plan.approved is not True:
            raise ValueError(
                "Cannot submit unapproved plan: plan.approved must be True before running."
            )
        if self._closed:
            raise RuntimeError("JobManager has been shut down.")

        job_id: str | None = None
        if (
            self.record is not None
            and plan.plan_id is not None
            and hasattr(self.record, "create_run")
        ):
            try:
                job_id = str(
                    self.record.create_run(
                        plan.plan_id,
                        metadata={"is_placeholder": bool(plan.is_placeholder)},
                    )
                )
            except TypeError:
                job_id = str(self.record.create_run(plan.plan_id))
        if not job_id:
            job_id = f"run_{uuid.uuid4().hex[:10]}"

        entry = _JobEntry(job_id=job_id, plan=plan)
        with self._lock:
            self._jobs[job_id] = entry

        self._transition(entry, "queued", "Queued")
        self._queue.put(job_id)
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        """Return current lifecycle status and recent progress messages for `job_id`."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            return self._serialize_entry(entry)

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return status summaries for all submitted jobs in submission order."""
        with self._lock:
            return [self._serialize_entry(entry) for entry in self._jobs.values()]

    @staticmethod
    def _serialize_entry(entry: _JobEntry) -> dict[str, Any]:
        return {
            "job_id": entry.job_id,
            "run_id": entry.job_id,
            "check": entry.plan.check,
            "model_id": entry.plan.model_id,
            "status": entry.status,
            "message": entry.message,
            "pending_question": entry.pending_question,
            "error": entry.error,
            "n_results": len(entry.results),
            "messages": list(entry.messages),
        }

    def ask(self, job_id: str, question: str) -> str:
        """Pause `job_id` with status `'needs_decision'` until `answer()` or `stop()` is called."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            if entry.cancel_event.is_set():
                raise _JobCancelledError()
            entry.pending_question = question
            entry.answer_value = None
            entry.answer_event.clear()

        self._transition(entry, "needs_decision", question)
        self.event_bus.publish(
            job_id,
            Event(
                type="question",
                payload={"job_id": job_id, "text": question},
            ),
        )

        while not entry.answer_event.wait(timeout=0.05):
            if entry.cancel_event.is_set():
                raise _JobCancelledError()

        with self._lock:
            if entry.cancel_event.is_set():
                raise _JobCancelledError()
            ans = entry.answer_value or ""
            entry.pending_question = None
            entry.answer_value = None

        self._transition(entry, "running", "Resumed after user decision")
        return ans

    def answer(self, job_id: str, answer: str) -> None:
        """Provide the user's answer to a job paused in `'needs_decision'`."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            if entry.status != "needs_decision":
                raise ValueError(
                    f"Job {job_id!r} is in state {entry.status!r}, not 'needs_decision'."
                )
            entry.answer_value = str(answer)
            entry.answer_event.set()

    def stop(self, job_id: str) -> None:
        """Request cancellation of a queued, running, or paused job."""
        should_mark_stopped = False
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            if entry.status in ("completed", "stopped", "failed"):
                return
            entry.cancel_event.set()
            entry.answer_event.set()
            if entry.status == "queued":
                should_mark_stopped = True

        if should_mark_stopped:
            self._transition(entry, "stopped", "Stopped before execution")
            entry.done_event.set()

    def resume_incomplete(self) -> list[str]:
        """Mark any unfinished jobs from a previous server session as stopped."""
        marked: list[str] = []
        note = "stopped (server restarted)"

        with self._lock:
            entries = list(self._jobs.values())

        for entry in entries:
            if entry.status in ("queued", "running", "needs_decision"):
                entry.cancel_event.set()
                entry.answer_event.set()
                self._transition(entry, "stopped", note)
                entry.done_event.set()
                marked.append(entry.job_id)

        if self.record is not None and hasattr(self.record, "list_incomplete_runs"):
            for run_id in self.record.list_incomplete_runs() or []:
                rid = str(run_id)
                if rid not in marked:
                    self._set_record_status(rid, "stopped", note)
                    marked.append(rid)

        return marked

    def wait(self, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
        """Wait up to `timeout` seconds for `job_id` to reach a terminal status."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
        entry.done_event.wait(timeout=timeout)
        return self.status(job_id)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the background worker loop cleanly."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                break

            try:
                self._execute_one(job_id)
            finally:
                self._queue.task_done()

    def _execute_one(self, job_id: str) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return
            if entry.cancel_event.is_set() or entry.status == "stopped":
                if entry.status != "stopped":
                    self._transition(entry, "stopped", "Stopped")
                entry.done_event.set()
                return

        self._transition(entry, "running", f"Running {entry.plan.check} check")
        ctx = RunContext(
            job_id=job_id,
            manager=self,
            record=self.record,
            cancel_event=entry.cancel_event,
        )

        try:
            check = self._check_resolver(entry.plan.check)
            rows = check.run(entry.plan, ctx)
            with self._lock:
                entry.results = list(rows or [])

            if entry.cancel_event.is_set():
                self._transition(entry, "stopped", "Stopped by user")
            else:
                self._transition(entry, "completed", "Completed")
        except _JobCancelledError:
            self._transition(entry, "stopped", "Stopped by user")
        except Exception as exc:
            if entry.cancel_event.is_set():
                self._transition(entry, "stopped", "Stopped by user")
            else:
                with self._lock:
                    entry.error = str(exc)
                self._transition(entry, "failed", f"Failed: {exc}")
        finally:
            entry.done_event.set()
