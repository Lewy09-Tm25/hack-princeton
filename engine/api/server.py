"""FastAPI + FastMCP single-process Engine API server (component E13)."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engine.api.http import create_http_router
from engine.api.mcp_tools import EngineService, create_mcp_server
from engine.api.ws import create_ws_router
from engine.record import Record

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class _NormalizeMcpSlashMiddleware:
    """Rewrite exact `/mcp` HTTP requests to `/mcp/` so Starlette Mount does not 307-redirect."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
            if "raw_path" in scope:
                scope["raw_path"] = b"/mcp/"
        await self.app(scope, receive, send)


class _BearerAuthMiddleware:
    """Pure ASGI middleware enforcing optional local bearer token auth on `/api` and `/mcp`."""

    def __init__(self, app: Any, *, expected_token: str) -> None:
        self.app = app
        self.expected_token = expected_token

    def _authorized(self, scope: dict[str, Any]) -> bool:
        for raw_name, raw_val in scope.get("headers", []):
            if raw_name.lower() == b"authorization":
                val = raw_val.decode("latin-1")
                if val.lower().startswith("bearer "):
                    supplied = val[7:].strip()
                    if secrets.compare_digest(supplied, self.expected_token):
                        return True
        qs = scope.get("query_string", b"").decode("latin-1")
        if qs:
            params = parse_qs(qs)
            for candidate in params.get("token", []):
                if secrets.compare_digest(candidate, self.expected_token):
                    return True
        return False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            method = scope.get("method", "GET")
            path = scope.get("path", "")
            if (
                method != "OPTIONS"
                and (path.startswith("/api") or path.startswith("/mcp"))
                and not self._authorized(scope)
            ):
                body = json.dumps(
                    {"detail": "Missing or invalid local bearer token."}
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def create_app(
    workspace: Path | str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    stub_mode: bool = False,
    auth_token: str | None = None,
    chat_session_factory: Callable[..., Any] | None = None,
    web_dist: Path | str | None = None,
) -> FastAPI:
    """Create the FastAPI application with `/mcp`, `/api/*`, `/ws/events`, and `/ws/chat`."""
    record = Record(workspace)
    service = EngineService(record, stub_mode=stub_mode)
    mcp = create_mcp_server(service, stub_mode=stub_mode, mounted_at_mcp=True)
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp_app.router.lifespan_context(app):
            yield

    app = FastAPI(
        title="Model Safety Investigation Engine",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.record = record
    app.state.engine_service = service
    app.state.mcp = mcp
    app.state.host = host
    app.state.port = port
    app.state.auth_token = auth_token

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if auth_token:
        app.add_middleware(_BearerAuthMiddleware, expected_token=auth_token)

    app.include_router(create_http_router(service))
    app.include_router(
        create_ws_router(
            service,
            auth_token=auth_token,
            chat_session_factory=chat_session_factory,
        )
    )
    app.mount("/mcp", mcp_app)

    # Serve the built web app from the same process, so one command starts everything.
    # Mounted last so `/api`, `/ws` and `/mcp` keep priority.
    if web_dist is not None and (Path(web_dist) / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    app.add_middleware(_NormalizeMcpSlashMiddleware)

    return app
