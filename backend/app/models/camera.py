import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    rtsp_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    stream_protocol: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="RTSP",
        server_default="RTSP",
    )
    stream_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def effective_url(self) -> str:
        """Return the actual URL to connect to.

        For RTMP cameras the URL is built from the stream_key pointing
        at the MediaMTX instance (configured via RTMP_BASE_URL env var).
        For RTSP cameras the stored rtsp_url is returned as-is.
        """
        if self.stream_protocol == "RTMP" and self.stream_key:
            from app.core import config
            base = config.RTMP_BASE_URL.rstrip("/")
            return f"{base}/{self.stream_key}"
        return self.rtsp_url

    def __repr__(self) -> str:
        return f"<Camera {self.name} proto={self.stream_protocol} user_id={self.user_id}>"
