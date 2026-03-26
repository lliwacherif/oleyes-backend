import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user, get_current_user_from_query_token
from app.models.user import User
from app.models.camera import Camera

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")

_LIVE_FPS = 15
_MAX_CONSECUTIVE_DROPS = 30
_CONNECT_TIMEOUT_S = 10
_RTMP_RECONNECT_DELAY = 2.0
_RTMP_MAX_RECONNECTS = 5
_JPEG_QUALITY = 65
_PREVIEW_MAX_WIDTH = 960
_STREAM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mjpeg")


class StreamProtocol(str, Enum):
    RTSP = "RTSP"
    RTMP = "RTMP"


class CameraCreateRequest(BaseModel):
    name: str = Field(..., max_length=255)
    rtsp_url: str = Field("", description="RTSP URL (required for RTSP cameras)")
    stream_protocol: StreamProtocol = Field(StreamProtocol.RTSP)
    stream_key: str | None = Field(
        None,
        max_length=255,
        description="MediaMTX stream key, e.g. 'cam1' (required for RTMP cameras)",
    )
    is_active: bool = Field(False)


class CameraUpdateRequest(BaseModel):
    name: str | None = Field(None, max_length=255)
    rtsp_url: str | None = Field(None)
    stream_protocol: StreamProtocol | None = None
    stream_key: str | None = Field(None, max_length=255)
    is_active: bool | None = None


class CameraResponse(BaseModel):
    id: str
    user_id: str
    name: str
    rtsp_url: str
    stream_protocol: str
    stream_key: str | None
    effective_url: str
    is_active: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


def _to_response(cam: Camera) -> CameraResponse:
    return CameraResponse(
        id=str(cam.id),
        user_id=str(cam.user_id),
        name=cam.name,
        rtsp_url=cam.rtsp_url,
        stream_protocol=cam.stream_protocol,
        stream_key=cam.stream_key,
        effective_url=cam.effective_url(),
        is_active=cam.is_active,
        created_at=cam.created_at.isoformat(),
        updated_at=cam.updated_at.isoformat(),
    )


def _validate_create(req: CameraCreateRequest) -> None:
    if req.stream_protocol == StreamProtocol.RTMP:
        if not req.stream_key:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="stream_key is required for RTMP cameras.",
            )
    else:
        if not req.rtsp_url:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="rtsp_url is required for RTSP cameras.",
            )


@router.get("/", response_model=list[CameraResponse])
async def get_cameras(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[CameraResponse]:
    result = await db.execute(
        select(Camera)
        .where(Camera.user_id == current_user.id)
        .order_by(Camera.created_at)
    )
    cameras = result.scalars().all()
    return [_to_response(cam) for cam in cameras]


@router.post("/", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    request: CameraCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CameraResponse:
    _validate_create(request)
    cam = Camera(
        user_id=current_user.id,
        name=request.name,
        rtsp_url=request.rtsp_url,
        stream_protocol=request.stream_protocol.value,
        stream_key=request.stream_key,
        is_active=request.is_active,
    )
    db.add(cam)
    await db.flush()
    return _to_response(cam)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    request: CameraUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CameraResponse:
    result = await db.execute(
        select(Camera).where(
            Camera.id == camera_id,
            Camera.user_id == current_user.id,
        )
    )
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found.",
        )

    if request.name is not None:
        cam.name = request.name
    if request.rtsp_url is not None:
        cam.rtsp_url = request.rtsp_url
    if request.stream_protocol is not None:
        cam.stream_protocol = request.stream_protocol.value
    if request.stream_key is not None:
        cam.stream_key = request.stream_key
    if request.is_active is not None:
        cam.is_active = request.is_active

    await db.flush()
    return _to_response(cam)


@router.delete("/{camera_id}", status_code=status.HTTP_200_OK)
async def delete_camera(
    camera_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        select(Camera).where(
            Camera.id == camera_id,
            Camera.user_id == current_user.id,
        )
    )
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found.",
        )

    await db.delete(cam)
    await db.flush()
    return {"status": "deleted", "camera_id": camera_id}


# ---------------------------------------------------------------------------
# FrameGrabber -- drains the capture buffer in a tight loop so the
# MJPEG generator always gets the latest frame with zero stale lag.
# ---------------------------------------------------------------------------

class _FrameGrabber:
    """Daemon thread that continuously reads frames from a VideoCapture,
    keeping only the most recent one.  For RTMP streams it also handles
    reconnection internally.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        url: str,
        *,
        is_rtmp: bool = False,
    ) -> None:
        self._cap = cap
        self._url = url
        self._is_rtmp = is_rtmp
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        consecutive_fails = 0
        reconnects = 0

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                consecutive_fails += 1
                if consecutive_fails < _MAX_CONSECUTIVE_DROPS:
                    continue

                if not self._is_rtmp:
                    break

                self._cap.release()
                if reconnects >= _RTMP_MAX_RECONNECTS:
                    logger.warning("grabber_rtmp_max_reconnects url=%s", self._url)
                    break
                reconnects += 1
                logger.info(
                    "grabber_rtmp_reconnecting attempt=%d/%d url=%s",
                    reconnects, _RTMP_MAX_RECONNECTS, self._url,
                )
                time.sleep(_RTMP_RECONNECT_DELAY)
                self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
                if not self._cap.isOpened():
                    self._cap.release()
                    logger.warning("grabber_rtmp_reconnect_failed url=%s", self._url)
                    continue
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_fails = 0
                logger.info("grabber_rtmp_reconnected url=%s", self._url)
                continue

            consecutive_fails = 0
            reconnects = 0
            with self._lock:
                self._frame = frame

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3)
        try:
            self._cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray) -> bytes | None:
    """Downscale if needed and JPEG-encode at preview quality."""
    h, w = frame.shape[:2]
    if w > _PREVIEW_MAX_WIDTH:
        scale = _PREVIEW_MAX_WIDTH / w
        frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok or jpeg is None:
        return None
    return jpeg.tobytes()


# ---------------------------------------------------------------------------
# Live MJPEG streaming
# ---------------------------------------------------------------------------

def _open_capture(url: str) -> cv2.VideoCapture | None:
    """Open an RTSP or RTMP stream via FFmpeg."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        logger.warning("camera_open_failed url=%s", url)
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("camera_connected url=%s", url)
    return cap


@router.get("/{camera_id}/live")
async def camera_live(
    camera_id: str,
    current_user: User = Depends(get_current_user_from_query_token),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream live MJPEG from the camera. Auth via ?token=JWT.

    Uses a FrameGrabber thread to always serve the latest frame,
    eliminating OpenCV buffer lag for both RTSP and RTMP.
    """
    result = await db.execute(
        select(Camera).where(
            Camera.id == camera_id,
            Camera.user_id == current_user.id,
        )
    )
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found.",
        )

    stream_url = cam.effective_url()
    is_rtmp = cam.stream_protocol == "RTMP"

    loop = asyncio.get_event_loop()
    try:
        cap = await asyncio.wait_for(
            loop.run_in_executor(_STREAM_POOL, _open_capture, stream_url),
            timeout=_CONNECT_TIMEOUT_S + 2,
        )
    except asyncio.TimeoutError:
        logger.warning("camera_timeout id=%s url=%s", camera_id, stream_url)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera connection timed out.",
        )

    if cap is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to camera stream.",
        )

    grabber = _FrameGrabber(cap, stream_url, is_rtmp=is_rtmp)

    return StreamingResponse(
        _mjpeg_generator(grabber),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_generator(grabber: _FrameGrabber):
    """Yield MJPEG frames from a FrameGrabber (always latest frame)."""
    interval = 1.0 / _LIVE_FPS
    empty_cycles = 0

    try:
        while empty_cycles < _MAX_CONSECUTIVE_DROPS:
            frame = grabber.get_frame()
            if frame is None:
                empty_cycles += 1
                time.sleep(0.03)
                continue
            empty_cycles = 0

            data = _encode_frame(frame)
            if data is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                + data
                + b"\r\n"
            )
            time.sleep(interval)
    finally:
        grabber.stop()
