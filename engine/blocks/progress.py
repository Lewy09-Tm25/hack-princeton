"""Progress inside long building-block calls (reading internal states, generating answers).

A check's own progress steps can be far apart: one step may read hundreds of prompts.
The building blocks call `report_progress` after each batch; the job runner connects that
to the running job, which sends it to the web app as a detail on the current step.
Outside a job (tests, scripts) nothing is connected and the calls do nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ProgressReporter = Callable[[str, int, int], None]


class StopRequested(Exception):  # noqa: N818 - a request, not an error
    """Raised from a progress call when the user has asked the running job to stop."""

_reporter: ContextVar[ProgressReporter | None] = ContextVar("progress_reporter", default=None)


@contextmanager
def progress_reporter(fn: ProgressReporter) -> Iterator[None]:
    """Send `report_progress` calls made inside this block to `fn(what, done, total)`."""
    token = _reporter.set(fn)
    try:
        yield
    finally:
        _reporter.reset(token)


def report_progress(what: str, done: int, total: int) -> None:
    """Report that `done` of `total` items of the current long step are finished."""
    fn = _reporter.get()
    if fn is None or total <= 0:
        return
    try:
        fn(what, done, total)
    except StopRequested:
        raise  # the user asked to stop: end the long step now
    except Exception:  # noqa: BLE001 - progress must never break a measurement
        pass
