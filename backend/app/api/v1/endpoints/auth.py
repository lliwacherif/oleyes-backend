"""
Authentication endpoints: signup, login, token refresh, current user.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.models.user_context import UserContext

logger = logging.getLogger("oleyes.auth")
router = APIRouter(prefix="/auth")


# ── Pydantic schemas ─────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleSignInRequest(BaseModel):
    email: EmailStr


class RefreshRequest(BaseModel):
    refresh_token: str


class GoogleSignInResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserContextInfo(BaseModel):
    has_context: bool
    context_data: dict | None = None


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_active: bool
    created_at: str
    context: UserContextInfo | None = None

    model_config = {"from_attributes": True}


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    request: SignupRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Create a new user account and return access + refresh tokens."""

    # Check if email already taken
    existing = await db.execute(
        select(User).where(User.email == request.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered.",
        )

    # Check if username already taken
    existing = await db.execute(
        select(User).where(User.username == request.username)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken.",
        )

    # Create user
    user = User(
        email=request.email,
        username=request.username,
        hashed_password=hash_password(request.password),
    )
    db.add(user)
    await db.flush()  # assigns the id

    logger.info("New user registered: %s (%s)", user.username, user.email)

    # Generate tokens
    access_token = create_access_token(subject=str(user.id))
    refresh_token = create_refresh_token(subject=str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate with email + password and return tokens."""

    result = await db.execute(
        select(User).where(User.email == request.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated.",
        )

    logger.info("User logged in: %s", user.email)

    access_token = create_access_token(subject=str(user.id))
    refresh_token = create_refresh_token(subject=str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/google", response_model=GoogleSignInResponse)
async def google_sign_in(
    request: GoogleSignInRequest,
    db: AsyncSession = Depends(get_db),
) -> GoogleSignInResponse:
    """Google Sign-In: look up user by email, return tokens if found."""

    result = await db.execute(
        select(User).where(User.email == request.email)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with this email.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated.",
        )

    logger.info("Google sign-in: %s", user.email)

    access_token = create_access_token(subject=str(user.id))
    refresh_token = create_refresh_token(subject=str(user.id))

    return GoogleSignInResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Exchange a valid refresh token for a new access token pair."""

    try:
        payload = decode_token(request.refresh_token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a refresh token.",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload invalid.",
        )

    # Verify user still exists and is active
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )

    # Issue new token pair
    access_token = create_access_token(subject=str(user.id))
    new_refresh_token = create_refresh_token(subject=str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


@router.get("/me", response_model=UserResponse)
async def me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    result = await db.execute(
        select(UserContext).where(UserContext.user_id == current_user.id)
    )
    ctx = result.scalar_one_or_none()

    context_info = None
    if ctx:
        context_data = {
            "business_type": ctx.business_type,
            "business_name": ctx.business_name,
            "short_description": ctx.short_description,
            "number_of_locations": ctx.number_of_locations,
            "estimated_cameras": ctx.estimated_number_of_cameras,
            "business_size": ctx.business_size,
            "camera_type": ctx.camera_type,
            "security_priorities": {
                "theft_detection": ctx.theft_detection,
                "fire_detection": ctx.fire_detection,
                "person_fall_detection": ctx.person_fall_detection,
                "violence_detection": ctx.violence_detection,
                "customer_behavior_analytics": ctx.customer_behavior_analytics,
            },
            "environment_type": ctx.environment_type,
        }
        context_info = UserContextInfo(
            has_context=True,
            context_data=context_data,
        )
    else:
        context_info = UserContextInfo(has_context=False)

    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
        context=context_info,
    )


@router.delete("/me", status_code=status.HTTP_200_OK)
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Permanently delete the authenticated user's account and all related data."""
    await db.delete(current_user)
    await db.flush()
    logger.info("User deleted: %s (%s)", current_user.username, current_user.email)
    return {"status": "deleted", "user_id": str(current_user.id)}


# ── Profile update schemas ────────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    username: str | None = Field(None, min_length=3, max_length=100)
    email: EmailStr | None = None


class UpdatePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=128)


# ── Profile update endpoints ──────────────────────────────────────────

@router.put("/me", response_model=UserResponse)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Update the authenticated user's username and/or email."""
    if request.username is None and request.email is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one field to update (username or email).",
        )

    if request.email and request.email != current_user.email:
        existing = await db.execute(
            select(User).where(User.email == request.email)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already taken.",
            )
        current_user.email = request.email

    if request.username and request.username != current_user.username:
        existing = await db.execute(
            select(User).where(User.username == request.username)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already taken.",
            )
        current_user.username = request.username

    await db.flush()
    logger.info("Profile updated: %s (%s)", current_user.username, current_user.email)

    result = await db.execute(
        select(UserContext).where(UserContext.user_id == current_user.id)
    )
    ctx = result.scalar_one_or_none()
    context_info = UserContextInfo(has_context=False)
    if ctx:
        context_info = UserContextInfo(
            has_context=True,
            context_data={
                "business_type": ctx.business_type,
                "business_name": ctx.business_name,
                "short_description": ctx.short_description,
                "number_of_locations": ctx.number_of_locations,
                "estimated_cameras": ctx.estimated_number_of_cameras,
                "business_size": ctx.business_size,
                "camera_type": ctx.camera_type,
                "security_priorities": {
                    "theft_detection": ctx.theft_detection,
                    "fire_detection": ctx.fire_detection,
                    "person_fall_detection": ctx.person_fall_detection,
                    "violence_detection": ctx.violence_detection,
                    "customer_behavior_analytics": ctx.customer_behavior_analytics,
                },
                "environment_type": ctx.environment_type,
            },
        )

    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
        context=context_info,
    )


@router.put("/me/password", status_code=status.HTTP_200_OK)
async def change_password(
    request: UpdatePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Change the authenticated user's password."""
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    current_user.hashed_password = hash_password(request.new_password)
    await db.flush()
    logger.info("Password changed: %s", current_user.email)
    return {"status": "password_changed"}
