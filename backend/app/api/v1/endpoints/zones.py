import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.camera import Camera
from app.models.zone import Zone

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/zones")


class ZonePoint(BaseModel):
    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")


class ZoneCreateRequest(BaseModel):
    camera_id: str
    name: str = Field(..., max_length=100)
    points: list[ZonePoint] = Field(
        ..., min_length=3,
        description="Polygon vertices (minimum 3 points)",
    )
    color: str = Field("#FF0000", max_length=20)
    instruction: str = Field("", max_length=500, description="Alert rule, e.g. 'Alert if any person enters this area'")


class ZoneUpdateRequest(BaseModel):
    name: str | None = Field(None, max_length=100)
    points: list[ZonePoint] | None = Field(None, min_length=3)
    color: str | None = Field(None, max_length=20)
    instruction: str | None = Field(None, max_length=500)


class ZoneResponse(BaseModel):
    id: str
    camera_id: str
    name: str
    points: list[ZonePoint]
    color: str
    instruction: str
    created_at: str
    updated_at: str


def _to_response(z: Zone) -> ZoneResponse:
    pts = json.loads(z.points)
    return ZoneResponse(
        id=str(z.id),
        camera_id=str(z.camera_id),
        name=z.name,
        points=[ZonePoint(x=p[0], y=p[1]) for p in pts],
        color=z.color,
        instruction=z.instruction or "",
        created_at=z.created_at.isoformat(),
        updated_at=z.updated_at.isoformat(),
    )


async def _verify_camera_owner(
    camera_id: str, user: User, db: AsyncSession
) -> Camera:
    result = await db.execute(
        select(Camera).where(Camera.id == camera_id, Camera.user_id == user.id)
    )
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found.")
    return cam


@router.get("/camera/{camera_id}", response_model=list[ZoneResponse])
async def get_zones(
    camera_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ZoneResponse]:
    await _verify_camera_owner(camera_id, current_user, db)
    result = await db.execute(
        select(Zone).where(Zone.camera_id == camera_id).order_by(Zone.created_at)
    )
    return [_to_response(z) for z in result.scalars().all()]


@router.post("/", response_model=ZoneResponse, status_code=status.HTTP_201_CREATED)
async def create_zone(
    request: ZoneCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ZoneResponse:
    await _verify_camera_owner(request.camera_id, current_user, db)
    pts_json = json.dumps([[p.x, p.y] for p in request.points])
    zone = Zone(
        camera_id=request.camera_id,
        name=request.name,
        points=pts_json,
        color=request.color,
        instruction=request.instruction,
    )
    db.add(zone)
    await db.flush()
    return _to_response(zone)


@router.put("/{zone_id}", response_model=ZoneResponse)
async def update_zone(
    zone_id: str,
    request: ZoneUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ZoneResponse:
    result = await db.execute(select(Zone).where(Zone.id == zone_id))
    zone = result.scalar_one_or_none()
    if not zone:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Zone not found.")
    await _verify_camera_owner(str(zone.camera_id), current_user, db)

    if request.name is not None:
        zone.name = request.name
    if request.points is not None:
        zone.points = json.dumps([[p.x, p.y] for p in request.points])
    if request.color is not None:
        zone.color = request.color
    if request.instruction is not None:
        zone.instruction = request.instruction

    await db.flush()
    return _to_response(zone)


@router.delete("/{zone_id}", status_code=status.HTTP_200_OK)
async def delete_zone(
    zone_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(select(Zone).where(Zone.id == zone_id))
    zone = result.scalar_one_or_none()
    if not zone:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Zone not found.")
    await _verify_camera_owner(str(zone.camera_id), current_user, db)

    await db.delete(zone)
    await db.flush()
    return {"status": "deleted", "zone_id": zone_id}
