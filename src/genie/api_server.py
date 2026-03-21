"""Bridge API: REST + SSE for Genie runs."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from genie.event_bus import get_event_bus
from genie.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    RenewalSnapshotRequest,
    RenewalSnapshotResponse,
    RunStatusResponse,
)
from genie.step_id import new_run_id
from genie.storage.repositories import (
    create_run,
    get_run_by_id,
    get_run_id_by_idempotency_key,
    get_events_by_run_id,
    update_run_status,
    ensure_user,
    ensure_conversation,
    save_renewal_snapshot,
)
from genie.crew.runner import run_pipeline

load_dotenv()

REQUIRED_API_KEY = os.environ.get("GENIE_API_KEY", "")


def require_api_key(authorization: Optional[str] = Header(None)) -> None:
    if not REQUIRED_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[7:].strip()
    if token != REQUIRED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # shutdown if needed


app = FastAPI(title="Genie Bridge API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    """No auth for health check."""
    return {"status": "ok", "service": "genie"}


@app.post("/v1/runs", response_model=CreateRunResponse)
async def create_run_endpoint(
    request: CreateRunRequest,
    authorization: Optional[str] = Header(None),
):
    require_api_key(authorization)
    ensure_user(request.user_id)
    ensure_conversation(request.conversation_id, request.user_id)

    idempotency_key = request.idempotency_key
    if idempotency_key:
        existing = get_run_id_by_idempotency_key(idempotency_key)
        if existing:
            run = get_run_by_id(existing)
            events_url = f"/v1/runs/{existing}/events" if run else ""
            return CreateRunResponse(
                run_id=existing,
                status=run["status"] if run else "completed",
                events_url=events_url,
            )

    run_id = new_run_id()
    create_run(run_id, request.conversation_id, idempotency_key)
    base_url = os.environ.get("GENIE_BASE_URL", "").rstrip("/")
    events_url = f"{base_url}/v1/runs/{run_id}/events" if base_url else f"/v1/runs/{run_id}/events"

    async def run_in_background():
        try:
            result = await asyncio.to_thread(
                run_pipeline,
                run_id=run_id,
                trace_id=run_id,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                user_message=request.input.user_message,
                emit_and_store=True,
            )
            update_run_status(
                run_id,
                result["status"],
                latency_ms=result.get("latency_ms"),
                outputs_json=result.get("outputs", {}),
            )
        except Exception as e:
            update_run_status(run_id, "failed", outputs_json={"error": str(e)})

    asyncio.create_task(run_in_background())

    return CreateRunResponse(run_id=run_id, status="started", events_url=events_url)


@app.get("/v1/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str, authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    run = get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusResponse(
        run_id=run["id"],
        status=run["status"],
        outputs=run["outputs_json"],
        latency_ms=run["latency_ms"],
    )


@app.get("/v1/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request, authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    run = get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    bus = get_event_bus()

    async def event_generator():
        if run["status"] in ("completed", "failed", "cancelled"):
            # Replay stored events then stream_end
            events = get_events_by_run_id(run_id)
            for ev_dict in events:
                yield {"data": json.dumps(ev_dict)}
            if not events or events[-1].get("event_type") != "stream_end":
                yield {"data": json.dumps({"run_id": run_id, "event_type": "stream_end", "payload": {}})}
        else:
            async for ev in bus.subscribe(run_id):
                yield {"data": ev.model_dump_json(by_alias=True)}

    return EventSourceResponse(event_generator())


@app.post("/v1/runs/{run_id}/cancel")
async def cancel_run(run_id: str, authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    run = get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("running", "started"):
        return JSONResponse(content={"run_id": run_id, "status": run["status"], "cancelled": False})
    # MVP: we don't have a running task handle; just mark as cancelled if still running when next checked
    update_run_status(run_id, "cancelled")
    return JSONResponse(content={"run_id": run_id, "status": "cancelled", "cancelled": True})


@app.post("/v1/renewal/snapshot", response_model=RenewalSnapshotResponse)
async def renewal_snapshot(
    body: RenewalSnapshotRequest,
    authorization: Optional[str] = Header(None),
):
    require_api_key(authorization)
    import uuid
    snapshot_id = str(uuid.uuid4())
    save_renewal_snapshot(
        snapshot_id,
        body.conversation_id,
        body.user_id,
        body.snapshot_data,
    )
    return RenewalSnapshotResponse(snapshot_id=snapshot_id, status="saved")


def main():
    uvicorn.run(
        "genie.api_server:app",
        host=os.environ.get("GENIE_HOST", "0.0.0.0"),
        port=int(os.environ.get("GENIE_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
