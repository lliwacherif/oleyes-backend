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
from app.models.scene_context import SceneContext
from app.services.vision_engine.yolo26_service import Yolo26Service

router = APIRouter(prefix="/vision")
_service = Yolo26Service()


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


class VisionRtspRequest(BaseModel):
    rtsp_url: str = Field(..., min_length=1, description="RTSP stream URL")
    scene_context: str | None = Field(
        default=None,
        description="Optional scene description to ground the AI analysis",
    )


class VisionRtmpRequest(BaseModel):
    stream_key: str = Field(
        ..., min_length=1,
        description="MediaMTX stream key (e.g. 'cam1')",
    )
    scene_context: str | None = Field(
        default=None,
        description="Optional scene description to ground the AI analysis",
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
            select(SceneContext).where(SceneContext.user_id == current_user.id)
        )
        sc = result.scalar_one_or_none()
        if sc and sc.refined_text:
            scene_context = sc.refined_text

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
        detail="YOLO26 job queued.",
    )


@router.post("/detect-rtsp", response_model=VisionJobResponse)
async def start_rtsp_detection(
    request: VisionRtspRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> VisionJobResponse:
    scene_context = request.scene_context
    if not scene_context:
        result = await db.execute(
            select(SceneContext).where(SceneContext.user_id == current_user.id)
        )
        sc = result.scalar_one_or_none()
        if sc and sc.refined_text:
            scene_context = sc.refined_text

    job_id = str(uuid4())
    _service.register_job(
        job_id=job_id,
        source_url=request.rtsp_url,
        scene_context=scene_context,
    )
    background_tasks.add_task(
        _service.start_job, job_id=job_id, source_url=request.rtsp_url
    )
    return VisionJobResponse(
        job_id=job_id,
        status="queued",
        detail="RTSP detection job queued.",
    )


@router.post("/detect-rtmp", response_model=VisionJobResponse)
async def start_rtmp_detection(
    request: VisionRtmpRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> VisionJobResponse:
    scene_context = request.scene_context
    if not scene_context:
        result = await db.execute(
            select(SceneContext).where(SceneContext.user_id == current_user.id)
        )
        sc = result.scalar_one_or_none()
        if sc and sc.refined_text:
            scene_context = sc.refined_text

    rtmp_url = f"rtmp://localhost:1935/live/{request.stream_key}"
    job_id = str(uuid4())
    _service.register_job(
        job_id=job_id,
        source_url=rtmp_url,
        scene_context=scene_context,
        stream_protocol="RTMP",
    )
    background_tasks.add_task(
        _service.start_job, job_id=job_id, source_url=rtmp_url
    )
    return VisionJobResponse(
        job_id=job_id,
        status="queued",
        detail=f"RTMP detection job queued (stream_key={request.stream_key}).",
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


@router.post("/stop-all")
async def stop_all_jobs() -> dict:
    count = _service.stop_all_running()
    return {"status": "ok", "stopped": count}
