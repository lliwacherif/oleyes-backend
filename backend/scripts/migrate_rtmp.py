"""
Add RTMP columns to cameras table.

New columns:
  stream_protocol  VARCHAR(10) NOT NULL DEFAULT 'RTSP'
  stream_key       VARCHAR(255) NULL

Run:  python scripts/migrate_rtmp.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from app.core.database import engine


async def migrate():
    columns = [
        (
            "stream_protocol",
            "ALTER TABLE cameras ADD COLUMN stream_protocol VARCHAR(10) NOT NULL DEFAULT 'RTSP'",
        ),
        (
            "stream_key",
            "ALTER TABLE cameras ADD COLUMN stream_key VARCHAR(255) NULL",
        ),
    ]

    async with engine.begin() as conn:
        for col_name, ddl in columns:
            try:
                await conn.execute(text(ddl))
                print(f"  OK added {col_name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print(f"  - {col_name} (already exists, skipped)")
                else:
                    raise

    print("\nDone. Restart the backend.")


if __name__ == "__main__":
    asyncio.run(migrate())
