import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.user_context import UserContext
from app.models.scene_context import SceneContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context")

_REFINE_PROMPT = (
    "You are configuring an AI video-surveillance system. "
    "Based on the business information below, write EXACTLY 2-3 sentences that describe:\n"
    "1) What the physical scene looks like (environment, objects normally present).\n"
    "2) What constitutes NORMAL behaviour in this setting.\n"
    "3) What should be considered SUSPICIOUS.\n\n"
    "Business info:\n{raw}\n\n"
    "Write ONLY the 2-3 sentence context. No bullet points, no headings."
)


class SecurityPriorities(BaseModel):
    theft_detection: bool = False
    fire_detection: bool = False
    person_fall_detection: bool = False
    violence_detection: bool = False
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
    refined_context: str | None
    environment_type: str | None
    created_at: str
    updated_at: str


async def _refine_context(raw_data: dict) -> str:
    """Send raw context to the LLM and get a 2-3 sentence scene description."""
    priorities = []
    if raw_data.get("theft_detection"):
        priorities.append("theft detection")
    if raw_data.get("fire_detection"):
        priorities.append("fire detection")
    if raw_data.get("person_fall_detection"):
        priorities.append("person fall detection")
    if raw_data.get("violence_detection"):
        priorities.append("violence detection")
    if raw_data.get("customer_behavior_analytics"):
        priorities.append("customer analytics")

    raw_text = (
        f"Type: {raw_data.get('business_type', 'N/A')}\n"
        f"Name: {raw_data.get('business_name', 'N/A')}\n"
        f"Description: {raw_data.get('short_description', 'N/A')}\n"
        f"Size: {raw_data.get('business_size', 'N/A')}\n"
        f"Camera type: {raw_data.get('camera_type', 'N/A')}\n"
        f"Monitoring priorities: {', '.join(priorities) or 'general surveillance'}"
    )

    prompt = _REFINE_PROMPT.format(raw=raw_text)

    async with httpx.AsyncClient(
        base_url=config.SCALWAY_BASE_URL,
        headers={"Authorization": f"Bearer {config.SCALWAY_API_KEY}"},
        timeout=config.SCALWAY_TIMEOUT,
    ) as client:
        r = await client.post("/chat/completions", json={
            "model": config.SCALWAY_ANALYSIS_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.4,
        })
        r.raise_for_status()
        return (
            r.json().get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )


async def _save_refined_context(
    db: AsyncSession, user_id, raw_data: dict
) -> str:
    """Generate refined context via AI and persist it."""
    import json
    raw_json = json.dumps(raw_data, ensure_ascii=False)

    try:
        refined_text = await _refine_context(raw_data)
        logger.info("scene_context_refined user=%s text=%s", user_id, refined_text[:80])
    except Exception as exc:
        logger.warning("scene_context_refine_failed user=%s error=%s", user_id, exc)
        refined_text = ""

    result = await db.execute(
        select(SceneContext).where(SceneContext.user_id == user_id)
    )
    sc = result.scalar_one_or_none()
    if sc:
        sc.refined_text = refined_text
        sc.raw_input = raw_json
    else:
        sc = SceneContext(
            user_id=user_id,
            refined_text=refined_text,
            raw_input=raw_json,
        )
        db.add(sc)
    await db.flush()
    return refined_text


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
    ctx.fire_detection = sp.fire_detection
    ctx.person_fall_detection = sp.person_fall_detection
    ctx.violence_detection = sp.violence_detection
    ctx.customer_behavior_analytics = sp.customer_behavior_analytics
    ctx.rebuild_context_text()


def _to_response(ctx: UserContext, refined: str | None = None) -> ContextResponse:
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
            fire_detection=ctx.fire_detection,
            person_fall_detection=ctx.person_fall_detection,
            violence_detection=ctx.violence_detection,
            customer_behavior_analytics=ctx.customer_behavior_analytics,
        ),
        context_text=ctx.context_text,
        refined_context=refined,
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

    refined = await _save_refined_context(db, current_user.id, {
        "business_type": request.business_type,
        "business_name": request.business_name,
        "short_description": request.short_description,
        "business_size": request.business_size,
        "camera_type": request.camera_type,
        "theft_detection": request.security_priorities.theft_detection,
        "fire_detection": request.security_priorities.fire_detection,
        "person_fall_detection": request.security_priorities.person_fall_detection,
        "violence_detection": request.security_priorities.violence_detection,
        "customer_behavior_analytics": request.security_priorities.customer_behavior_analytics,
    })

    return _to_response(ctx, refined)


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

    sc_result = await db.execute(
        select(SceneContext).where(SceneContext.user_id == current_user.id)
    )
    sc = sc_result.scalar_one_or_none()

    return _to_response(ctx, sc.refined_text if sc else None)


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

    refined = await _save_refined_context(db, current_user.id, {
        "business_type": request.business_type,
        "business_name": request.business_name,
        "short_description": request.short_description,
        "business_size": request.business_size,
        "camera_type": request.camera_type,
        "theft_detection": request.security_priorities.theft_detection,
        "fire_detection": request.security_priorities.fire_detection,
        "person_fall_detection": request.security_priorities.person_fall_detection,
        "violence_detection": request.security_priorities.violence_detection,
        "customer_behavior_analytics": request.security_priorities.customer_behavior_analytics,
    })

    return _to_response(ctx, refined)


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
