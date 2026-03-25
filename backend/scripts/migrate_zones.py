"""
Create zones table for ROI monitoring.

Run:  python scripts/migrate_zones.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from app.core.database import engine


async def migrate():
    ddl = """
    CREATE TABLE IF NOT EXISTS zones (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
        name VARCHAR(100) NOT NULL,
        points TEXT NOT NULL,
        color VARCHAR(20) NOT NULL DEFAULT '#FF0000',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """
    idx = "CREATE INDEX IF NOT EXISTS ix_zones_camera_id ON zones (camera_id)"

    async with engine.begin() as conn:
        await conn.execute(text(ddl))
        print("  OK created zones table")
        await conn.execute(text(idx))
        print("  OK created index on camera_id")

    print("\nDone. Restart the backend.")


if __name__ == "__main__":
    asyncio.run(migrate())
