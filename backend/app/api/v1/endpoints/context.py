from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.user_context import UserContext

router = APIRouter(prefix="/context")


class SecurityPriorities(BaseModel):
    theft_detection: bool = False
    suspicious_behavior_detection: bool = False
    loitering_detection: bool = False
    employee_monitoring: bool = False
    customer_behavior_analytics: bool = False


class ContextRequest(BaseModel):
    business_type: str = Field(
        ...,
        max_length=100,
        description="Step 1: e.g. 'Retail', 'Supermarket', 'Warehouse'",
    )
    business_name: str = Field(..., max_length=200)
    short_description: str = Field("", max_length=500)
    number_of_locations: str = Field("", max_length=100)
    estimated_cameras: str = Field("", max_length=100)
    business_size: str = Field("", max_length=50)
    camera_type: str = Field("", max_length=100)
    security_priorities: SecurityPriorities = Field(default_factory=SecurityPriorities)


class ContextResponse(BaseModel):
    id: str
    user_id: str
    business_type: str | None
    business_name: str | None
    short_description: str | None
    number_of_locations: str | None
    estimated_cameras: str | None
    business_size: str | None
    camera_type: str | None
    security_priorities: SecurityPriorities
    context_text: str
    environment_type: str | None
    created_at: str
    updated_at: str


def _apply_request(ctx: UserContext, req: ContextRequest) -> None:
    ctx.business_type = req.business_type
    ctx.business_name = req.business_name
    ctx.short_description = req.short_description
    ctx.number_of_locations = req.number_of_locations
    ctx.estimated_number_of_cameras = req.estimated_cameras
    ctx.business_size = req.business_size
    ctx.camera_type = req.camera_type
    sp = req.security_priorities
    ctx.theft_detection = sp.theft_detection
    ctx.suspicious_behavior_detection = sp.suspicious_behavior_detection
    ctx.loitering_detection = sp.loitering_detection
    ctx.employee_monitoring = sp.employee_monitoring
    ctx.customer_behavior_analytics = sp.customer_behavior_analytics
    ctx.rebuild_context_text()


def _to_response(ctx: UserContext) -> ContextResponse:
    return ContextResponse(
        id=str(ctx.id),
        user_id=str(ctx.user_id),
        business_type=ctx.business_type,
        business_name=ctx.business_name,
        short_description=ctx.short_description,
        number_of_locations=ctx.number_of_locations,
        estimated_cameras=ctx.estimated_number_of_cameras,
        business_size=ctx.business_size,
        camera_type=ctx.camera_type,
        security_priorities=SecurityPriorities(
            theft_detection=ctx.theft_detection,
            suspicious_behavior_detection=ctx.suspicious_behavior_detection,
            loitering_detection=ctx.loitering_detection,
            employee_monitoring=ctx.employee_monitoring,
            customer_behavior_analytics=ctx.customer_behavior_analytics,
        ),
        context_text=ctx.context_text,
        environment_type=ctx.environment_type,
        created_at=ctx.created_at.isoformat(),
        updated_at=ctx.updated_at.isoformat(),
    )


@router.post("/", response_model=ContextResponse, status_code=status.HTTP_201_CREATED)
async def create_context(
    request: ContextRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ContextResponse:
    result = await db.execute(
        select(UserContext).where(UserContext.user_id == current_user.id)
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Context already exists. Use PUT to update.",
        )

    ctx = UserContext(user_id=current_user.id)
    _apply_request(ctx, request)
    db.add(ctx)
    await db.flush()

    return _to_response(ctx)


@router.get("/", response_model=ContextResponse)
async def get_context(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ContextResponse:
    result = await db.execute(
        select(UserContext).where(UserContext.user_id == current_user.id)
    )
    ctx = result.scalar_one_or_none()
    if not ctx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No context found. Create one first.",
        )

    return _to_response(ctx)


@router.put("/", response_model=ContextResponse)
async def update_context(
    request: ContextRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ContextResponse:
    result = await db.execute(
        select(UserContext).where(UserContext.user_id == current_user.id)
    )
    ctx = result.scalar_one_or_none()
    if not ctx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No context found. Create one first with POST.",
        )

    _apply_request(ctx, request)
    await db.flush()

    return _to_response(ctx)


@router.delete("/", status_code=status.HTTP_200_OK)
async def delete_context(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        select(UserContext).where(UserContext.user_id == current_user.id)
    )
    ctx = result.scalar_one_or_none()
    if not ctx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No context found to delete.",
        )

    await db.delete(ctx)
    await db.flush()

    return {"status": "deleted", "user_id": str(current_user.id)}
