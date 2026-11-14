"""Thread-safe event bus fanning out live engine events to async WebSocket subscribers."""

from __future__ import annotations

import asyncio
import threading
import types
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any

from engine.contracts import Event

MAX_HISTORY_PER_JOB = 250


@types.coroutine
def _yield_once() -> Any:
    yield


class EventBus:
    """Thread-safe pub/sub bus bridging background worker threads and asyncio clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[
            tuple[asyncio.AbstractEventLoop | None, asyncio.Queue[Event]]
        ] = []
        self._history: dict[str, deque[Event]] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY_PER_JOB)
        )
        self._all_history: deque[Event] = deque(maxlen=MAX_HISTORY_PER_JOB * 4)

    @property
    def history(self) -> list[Event]:
        """Return all buffered recent events in publish order."""
        with self._lock:
            return list(self._all_history)

    def publish(
        self,
        job_id_or_event: str | Event | dict,
        event: Event | dict | None = None,
    ) -> None:
        """Validate and broadcast `event` to all connected subscribers."""
        if isinstance(job_id_or_event, Event) or (
            isinstance(job_id_or_event, dict) and event is None
        ):
            raw_event = job_id_or_event
            payload = (
                raw_event.payload
                if isinstance(raw_event, Event)
                else raw_event.get("payload", {})
            )
            job_id = str(payload.get("job_id", "_global"))
        else:
            job_id = str(job_id_or_event)
            if event is None:
                raise ValueError("Event must be provided to publish().")
            raw_event = event

        validated = raw_event if isinstance(raw_event, Event) else Event.model_validate(raw_event)
        with self._lock:
            self._history[job_id].append(validated)
            self._all_history.append(validated)
            targets = list(self._subscribers)

        for loop, queue in targets:
            if loop is None:
                queue.put_nowait(validated)
                continue
            if loop.is_closed():
                continue
            try:
                loop.call_soon_threadsafe(queue.put_nowait, validated)
            except RuntimeError:
                continue

    async def subscribe(self) -> AsyncIterator[Event]:
        """Yield live `Event` objects until the caller closes or cancels the iterator."""
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        queue: asyncio.Queue[Event] = asyncio.Queue()
        entry = (loop, queue)
        with self._lock:
            self._subscribers.append(entry)
        try:
            while True:
                if not queue.empty():
                    yield queue.get_nowait()
                elif loop is None:
                    await _yield_once()
                else:
                    item = await queue.get()
                    yield item
        finally:
            with self._lock:
                if entry in self._subscribers:
                    self._subscribers.remove(entry)

    def get_history(self, job_id: str) -> list[Event]:
        """Return buffered recent events emitted for `job_id`."""
        with self._lock:
            return list(self._history.get(str(job_id), ()))
