"""
Add instruction column to zones table.

Run:  python scripts/migrate_zone_instruction.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from app.core.database import engine


async def migrate():
    ddl = "ALTER TABLE zones ADD COLUMN instruction TEXT NOT NULL DEFAULT ''"

    async with engine.begin() as conn:
        try:
            await conn.execute(text(ddl))
            print("  OK added instruction column")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("  - instruction column already exists, skipped")
            else:
                raise

    print("\nDone. Restart the backend.")


if __name__ == "__main__":
    asyncio.run(migrate())
