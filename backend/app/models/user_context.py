import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class UserContext(Base):
    __tablename__ = "user_contexts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    business_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    business_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    short_description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    number_of_locations: Mapped[str | None] = mapped_column(String(100), nullable=True)
    estimated_number_of_cameras: Mapped[str | None] = mapped_column(String(100), nullable=True)
    business_size: Mapped[str | None] = mapped_column(String(50), nullable=True)
    camera_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    theft_detection: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fire_detection: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    person_fall_detection: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    violence_detection: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    customer_behavior_analytics: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    context_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    environment_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

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

    def rebuild_context_text(self) -> None:
        data = {
            "business_type": self.business_type,
            "business_name": self.business_name,
            "short_description": self.short_description,
            "number_of_locations": self.number_of_locations,
            "estimated_number_of_cameras": self.estimated_number_of_cameras,
            "business_size": self.business_size,
            "camera_type": self.camera_type,
            "theft_detection": self.theft_detection,
            "fire_detection": self.fire_detection,
            "person_fall_detection": self.person_fall_detection,
            "violence_detection": self.violence_detection,
            "customer_behavior_analytics": self.customer_behavior_analytics,
        }
        self.context_text = json.dumps(data)
        self.environment_type = self.business_type

    def __repr__(self) -> str:
        return f"<UserContext user_id={self.user_id}>"
