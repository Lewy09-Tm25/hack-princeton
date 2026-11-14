"""HTTP routes serving the web app and external clients (component E13)."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from engine.api.mcp_tools import EngineService


class LoadModelBody(BaseModel):
    source: str
    revision: str | None = None
    adapter: str | None = None
    allow_remote_code: bool = False


def create_http_router(service: EngineService) -> APIRouter:
    """Create the `/api/*` FastAPI router backed by `service`."""
    router = APIRouter(prefix="/api")

    @router.get("/models")
    async def list_models() -> list[dict[str, Any]]:
        return [m.model_dump() for m in service.record.list_model_maps()]

    @router.post("/models/load")
    async def load_model_endpoint(body: LoadModelBody) -> dict[str, Any]:
        try:
            return service.load_model(
                source=body.source,
                revision=body.revision,
                adapter=body.adapter,
                allow_remote_code=body.allow_remote_code,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/models/{model_id:path}/map")
    async def get_model_map_endpoint(model_id: str) -> dict[str, Any]:
        decoded = unquote(model_id)
        try:
            return service.get_model_map(decoded)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/model-map")
    async def get_model_map_query(model_id: str = Query(...)) -> dict[str, Any]:
        try:
            return service.get_model_map(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/series/{series_id:path}")
    async def get_series_endpoint(series_id: str) -> dict[str, Any]:
        decoded = unquote(series_id)
        series = service.record.get_series(decoded)
        if series is None:
            raise HTTPException(status_code=404, detail=f"Series not found: {decoded}")
        return series.model_dump()

    @router.get("/results")
    async def get_results_endpoint(
        run_id: str | None = None,
        check: str | None = None,
        metric: str | None = None,
        checkpoint_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return service.get_results(
            run_id=run_id,
            check=check,
            metric=metric,
            checkpoint_id=checkpoint_id,
        )

    @router.get("/findings")
    async def get_findings_endpoint(run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is not None:
            return service.get_findings(run_id)
        findings = service.record.get_findings(None)
        out: list[dict[str, Any]] = []
        for f in findings:
            if f.is_placeholder:
                f = f.model_copy(update={"is_placeholder": True, "verified": False})
            out.append(f.model_dump())
        return out

    @router.get("/jobs")
    async def list_jobs_endpoint() -> list[dict[str, Any]]:
        return service.list_jobs()

    @router.get("/jobs/{job_id}")
    async def get_job_endpoint(job_id: str) -> dict[str, Any]:
        try:
            return service.job_status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/plans/{plan_id}")
    async def get_plan_endpoint(plan_id: str) -> dict[str, Any]:
        plan = service.record.get_plan(plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail=f"Plan not found: {plan_id}")
        return plan.model_dump()

    @router.post("/plans/{plan_id}/approve")
    async def approve_plan_endpoint(plan_id: str) -> dict[str, Any]:
        try:
            approved = service.record.approve_plan(plan_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return approved.model_dump()

    @router.get("/reports")
    async def list_reports_endpoint() -> list[dict[str, Any]]:
        return service.record.list_reports()

    @router.post("/reports/{run_id}")
    async def build_report_endpoint(run_id: str) -> dict[str, Any]:
        decoded = unquote(run_id)
        try:
            return service.build_report(decoded)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/reports/{report_id}", response_class=HTMLResponse)
    async def get_report_endpoint(report_id: str) -> HTMLResponse:
        path = service.record.get_report_path(report_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")
        return HTMLResponse(content=path.read_text(encoding="utf-8"))

    @router.get("/examples")
    async def get_examples_endpoint(
        run_id: str = Query(...),
        n: int = Query(default=3, ge=1, le=50),
    ) -> list[dict[str, Any]]:
        return service.get_examples(run_id=run_id, n=n)

    @router.get("/setup/claude-code")
    async def claude_code_setup_endpoint(request: Request) -> dict[str, Any]:
        base = str(request.base_url).rstrip("/")
        mcp_url = f"{base}/mcp"
        workspace_str = str(service.record.workspace)
        auth_token = getattr(request.app.state, "auth_token", None)
        header_arg = (
            f' --header "Authorization: Bearer {auth_token}"' if auth_token else ""
        )
        http_cmd = (
            f"claude mcp add --transport http model-safety-engine {mcp_url}{header_arg}"
        )
        stdio_cmd = (
            f'claude mcp add model-safety-engine -- tool mcp-stdio --workspace "{workspace_str}"'
        )
        plugin_cmd = "claude --plugin-dir ./plugin"
        return {
            "command": http_cmd,
            "http_command": http_cmd,
            "stdio_command": stdio_cmd,
            "plugin_command": plugin_cmd,
            "mcp_url": mcp_url,
            "workspace": workspace_str,
            "notes": (
                "Connects Claude Code to the same local engine and workspace used by the web app."
            ),
        }

    return router
