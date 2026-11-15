"""WebSocket endpoints for live engine events (`/ws/events`) and chat (`/ws/chat`)."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engine.api.mcp_tools import EngineService


def _ws_authorized(websocket: WebSocket, expected_token: str | None) -> bool:
    if not expected_token:
        return True
    query_token = websocket.query_params.get("token")
    if query_token and secrets.compare_digest(query_token, expected_token):
        return True
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        supplied = auth_header[7:].strip()
        if secrets.compare_digest(supplied, expected_token):
            return True
    return False


def create_ws_router(
    service: EngineService,
    *,
    auth_token: str | None = None,
    chat_session_factory: Callable[..., Any] | None = None,
) -> APIRouter:
    """Create WebSocket routes for `/ws/events` and `/ws/chat`."""
    router = APIRouter()

    @router.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        if not _ws_authorized(websocket, auth_token):
            await websocket.close(code=1008)
            return

        await websocket.accept()
        try:
            async for event in service.event_bus.subscribe():
                await websocket.send_json(event.model_dump())
        except WebSocketDisconnect:
            return

    @router.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        if not _ws_authorized(websocket, auth_token):
            await websocket.close(code=1008)
            return

        await websocket.accept()
        # A freshly loaded page starts with an empty view. The page marks only its first
        # connection, so an automatic reconnect mid-session keeps the model on screen.
        # Loaded models stay in memory, so loading the same model again is instant.
        if websocket.query_params.get("new_page") == "1":
            if hasattr(service, "active_model_id"):
                service.active_model_id = None
            if hasattr(service, "active_series_id"):
                service.active_series_id = None
        host = getattr(websocket.app.state, "host", "127.0.0.1")
        port = getattr(websocket.app.state, "port", 8000)
        mcp_url = f"http://{host}:{port}/mcp"

        session: Any
        if chat_session_factory is not None:
            session = chat_session_factory(
                engine_mcp_url=mcp_url,
                workspace=service.record.workspace,
                auth_token=auth_token,
                engine_service=service,
            )
        else:
            from agent.session import ChatSession

            session = ChatSession(
                engine_mcp_url=mcp_url,
                workspace=service.record.workspace,
                auth_token=auth_token,
                engine_service=service,
            )

        # Read incoming messages in the background so a stop request is handled at once,
        # even while the agent is in the middle of a reply.
        import asyncio

        inbox: asyncio.Queue[str | None] = asyncio.Queue()

        async def _stop_everything() -> None:
            interrupt = getattr(session, "interrupt", None)
            if callable(interrupt):
                await interrupt()
            try:
                jobs = service.list_jobs()
            except Exception:  # noqa: BLE001
                jobs = []
            for job in jobs:
                if job.get("status") in ("queued", "running", "needs_decision"):
                    try:
                        service.stop_job(str(job.get("job_id")))
                    except Exception:  # noqa: BLE001
                        pass

        async def _reader() -> None:
            try:
                while True:
                    raw = await websocket.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        msg = None
                    if isinstance(msg, dict) and msg.get("stop"):
                        await _stop_everything()
                        await websocket.send_json({"type": "stopped"})
                        continue
                    await inbox.put(raw)
            except WebSocketDisconnect:
                await inbox.put(None)

        # Without a running event loop (some tests drive the socket by hand) there is no
        # background reader: read each message directly, as before.
        try:
            asyncio.get_running_loop()
            reader_task: asyncio.Task[None] | None = asyncio.create_task(_reader())
        except RuntimeError:
            reader_task = None

        try:
            while True:
                if reader_task is not None:
                    raw_text = await inbox.get()
                    if raw_text is None:
                        break
                else:
                    raw_text = await websocket.receive_text()
                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError:
                    parsed = {"text": raw_text}

                if isinstance(parsed, str):
                    user_text = parsed
                    context = None
                    approve_plan_id = None
                elif isinstance(parsed, dict):
                    user_text = str(parsed.get("text") or parsed.get("message") or "")
                    context = (
                        parsed.get("context")
                        if isinstance(parsed.get("context"), dict)
                        else None
                    )
                    approve_plan_id = parsed.get("approve_plan_id")
                else:
                    user_text = str(parsed)
                    context = None
                    approve_plan_id = None

                if approve_plan_id:
                    try:
                        approved = service.record.approve_plan(str(approve_plan_id))
                        await websocket.send_json(
                            {"type": "plan", "plan": approved.model_dump()}
                        )
                    except KeyError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})

                if user_text.strip():
                    async for part in session.send(user_text, context=context):
                        await websocket.send_json(part)
                    await websocket.send_json({"type": "turn_complete"})
        except WebSocketDisconnect:
            pass
        finally:
            if reader_task is not None:
                reader_task.cancel()
            await session.close()

    return router
