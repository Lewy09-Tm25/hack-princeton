"""Hugging Face `TrainerCallback` wrapper for training-step safety monitoring (component E15)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import transformers

import engine  # noqa: F401 - enforce Windows mcp-before-nnsight import order
from engine.contracts import Event, ResultRecord
from engine.monitor.torch_hook import (
    DEFAULT_EVERY_N_STEPS,
    DEFAULT_N_PROMPTS,
    StepMonitorHook,
)


class SafetyMonitorCallback(transformers.TrainerCallback):
    """`transformers.TrainerCallback` running lightweight safety checks every N steps."""

    def __init__(
        self,
        checks: list[str] | None = None,
        every_n_steps: int = DEFAULT_EVERY_N_STEPS,
        engine_url: str | None = None,
        workspace: Path | str | None = None,
        thresholds: dict[str, float] | None = None,
        tokenizer: Any | None = None,
        layer_ids: list[str] | None = None,
        n_prompts: int = DEFAULT_N_PROMPTS,
        event_sink: Callable[[Event], None] | None = None,
        is_placeholder: bool = False,
    ) -> None:
        super().__init__()
        self._hook = StepMonitorHook(
            checks=checks,
            every_n_steps=every_n_steps,
            engine_url=engine_url,
            workspace=workspace,
            thresholds=thresholds,
            tokenizer=tokenizer,
            layer_ids=layer_ids,
            n_prompts=n_prompts,
            event_sink=event_sink,
            is_placeholder=is_placeholder,
        )

    @property
    def checks(self) -> list[str]:
        return self._hook.checks

    @property
    def every_n_steps(self) -> int:
        return self._hook.every_n_steps

    @property
    def engine_url(self) -> str | None:
        return self._hook.engine_url

    @property
    def workspace(self) -> Path | None:
        return self._hook.workspace

    @property
    def thresholds(self) -> dict[str, float]:
        return self._hook.thresholds

    @property
    def tokenizer(self) -> Any | None:
        return self._hook.tokenizer

    @property
    def layer_ids(self) -> list[str] | None:
        return self._hook.layer_ids

    @property
    def n_prompts(self) -> int:
        return self._hook.n_prompts

    @property
    def is_placeholder(self) -> bool:
        return self._hook.is_placeholder

    @property
    def run_id(self) -> str:
        return self._hook.run_id

    @property
    def series_id(self) -> str:
        return self._hook.series_id

    @property
    def alerts(self) -> list[dict[str, Any]]:
        return self._hook.alerts

    @property
    def events(self) -> list[Event]:
        return self._hook.events

    @property
    def results(self) -> list[ResultRecord]:
        return self._hook.results

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run lightweight checks when `state.global_step` reaches a multiple of `every_n_steps`."""
        global_step = int(getattr(state, "global_step", 0) or 0)
        if model is None or global_step <= 0 or (global_step % self.every_n_steps) != 0:
            return control

        tok = self.tokenizer or kwargs.get("tokenizer") or kwargs.get("processing_class")
        self._hook.step(global_step=global_step, model=model, tokenizer=tok)
        return control
