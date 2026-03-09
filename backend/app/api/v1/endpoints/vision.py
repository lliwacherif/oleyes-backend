from json import dumps
from uuid import uuid4

from asyncio import sleep

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import AnyHttpUrl, BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.user_context import UserContext
from app.services.vision_engine.yolo11_service import Yolo11Service

router = APIRouter(prefix="/vision")
_service = Yolo11Service()


class VisionJobResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None


class VisionYoutubeRequest(BaseModel):
    youtube_url: AnyHttpUrl = Field(..., description="Public YouTube video URL")
    callback_url: AnyHttpUrl | None = Field(
        default=None, description="Optional callback for results"
    )
    scene_context: str | None = Field(
        default=None,
        description="Optional scene description to ground the AI analysis (e.g. 'Supermarket, busy hour')",
    )


@router.post("/detect-youtube", response_model=VisionJobResponse)
async def start_youtube_detection(
    request: VisionYoutubeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> VisionJobResponse:
    scene_context = request.scene_context
    if not scene_context:
        result = await db.execute(
            select(UserContext).where(UserContext.user_id == current_user.id)
        )
        ctx = result.scalar_one_or_none()
        if ctx:
            scene_context = ctx.context_text

    job_id = str(uuid4())
    _service.register_job(
        job_id=job_id,
        source_url=str(request.youtube_url),
        scene_context=scene_context,
    )
    background_tasks.add_task(
        _service.start_job, job_id=job_id, source_url=str(request.youtube_url)
    )
    return VisionJobResponse(
        job_id=job_id,
        status="queued",
        detail="YOLOv11 job queued.",
    )


@router.get("/jobs/{job_id}", response_model=VisionJobResponse)
async def get_job_status(job_id: str) -> VisionJobResponse:
    status = _service.get_status(job_id)
    result = _service.get_result(job_id)
    if not result:
        return VisionJobResponse(
            job_id=job_id,
            status="unknown",
            detail="Job not found.",
        )
    return VisionJobResponse(
        job_id=job_id,
        status=status,
        detail="OK",
        result=result.get("result") if result else None,
        error=str(result.get("error")) if result and result.get("error") else None,
    )


@router.get("/jobs/{job_id}/stream")
async def stream_job_status(job_id: str) -> StreamingResponse:
    async def event_stream():
        snapshot = _service.get_snapshot(job_id)
        if not snapshot:
            yield f"data: {dumps({'status': 'unknown', 'detail': 'Job not found.'})}\n\n"
            return
        last_event_id = -1
        while True:
            snapshot = _service.get_snapshot(job_id) or {}
            current_event_id = int(snapshot.get("event_id", 0))
            payload = dict(snapshot)
            if current_event_id == last_event_id:
                payload["heartbeat"] = True
            else:
                payload["heartbeat"] = False
                last_event_id = current_event_id
            yield f"data: {dumps(payload)}\n\n"
            if snapshot.get("status") in {"done", "error"}:
                break
            await sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> dict:
    stopped = _service.stop_job(job_id)
    if not stopped:
        return {"status": "not_found", "job_id": job_id}
    return {"status": "stopped", "job_id": job_id}
