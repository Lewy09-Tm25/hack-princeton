"""Isolated Jupyter kernel wrapper for custom analysis cells (`design.md` section 14)."""

from __future__ import annotations

import os
import queue
import sys
from pathlib import Path
from typing import Any

_STRIP_EXACT_ENV_VARS: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NDIF_API_KEY",
        "NDIF_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)

_STRIP_SUFFIXES: tuple[str, ...] = (
    "_API_KEY",
    "_SECRET",
    "_SECRET_KEY",
    "_TOKEN",
    "_PASSWORD",
    "_PROXY",
)


def sanitized_kernel_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment dictionary stripped of API keys, secrets, and network proxies."""
    source = dict(os.environ if base_env is None else base_env)
    clean: dict[str, str] = {}
    for key, value in source.items():
        if key in _STRIP_EXACT_ENV_VARS:
            continue
        upper = key.upper()
        if any(upper.endswith(suffix) for suffix in _STRIP_SUFFIXES):
            continue
        clean[key] = value
    clean["HF_HUB_OFFLINE"] = "1"
    clean["TRANSFORMERS_OFFLINE"] = "1"
    return clean


class Kernel:
    """Persistent per-experiment Python kernel with restricted cwd and stripped secrets."""

    def __init__(self) -> None:
        self._km: Any | None = None
        self._kc: Any | None = None
        self._run_dir: Path | None = None

    def start(self, run_dir: Path) -> None:
        """Launch a local Python kernel rooted at `run_dir`."""
        from jupyter_client import KernelManager

        if self._km is not None:
            self.shutdown()

        target_dir = Path(run_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = target_dir

        km = KernelManager(kernel_name="python3")
        km.kernel_cmd = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        km.start_kernel(cwd=str(target_dir), env=sanitized_kernel_env())
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=60)
        self._km = km
        self._kc = kc

    def execute(self, code: str, timeout_s: int = 300) -> dict[str, Any]:
        """Execute `code` in the kernel and collect stdout, stderr, and rich display outputs."""
        if self._kc is None:
            raise RuntimeError("Kernel is not started; call start(run_dir) first.")

        msg_id = self._kc.execute(code)
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        display_data: list[Any] = []
        error_info: dict[str, Any] | None = None

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=float(timeout_s))
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Kernel cell execution exceeded timeout of {timeout_s}s."
                ) from exc

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type", "")
            content = msg.get("content", {})

            if msg_type == "stream":
                if content.get("name") == "stderr":
                    stderr_parts.append(str(content.get("text", "")))
                else:
                    stdout_parts.append(str(content.get("text", "")))
            elif msg_type in ("display_data", "execute_result"):
                display_data.append(content.get("data", {}))
            elif msg_type == "error":
                error_info = {
                    "ename": content.get("ename"),
                    "evalue": content.get("evalue"),
                    "traceback": content.get("traceback", []),
                }
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        return {
            "status": "error" if error_info is not None else "ok",
            "stdout": "".join(stdout_parts),
            "stderr": "".join(stderr_parts),
            "display_data": display_data,
            "error": error_info,
        }

    def shutdown(self) -> None:
        """Stop channels and terminate the kernel subprocess."""
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
            self._kc = None
        if self._km is not None:
            try:
                if self._km.is_alive():
                    self._km.shutdown_kernel(now=True)
            except Exception:
                pass
            self._km = None
