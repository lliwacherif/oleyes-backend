"""
Migrate security priority columns in user_contexts table.

Old columns → New columns:
  suspicious_behavior_detection → fire_detection
  loitering_detection           → person_fall_detection
  employee_monitoring           → violence_detection

Run:  python scripts/migrate_priorities.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from app.core.database import engine


async def migrate():
    renames = [
        ("suspicious_behavior_detection", "fire_detection"),
        ("loitering_detection", "person_fall_detection"),
        ("employee_monitoring", "violence_detection"),
    ]

    async with engine.begin() as conn:
        for old, new in renames:
            try:
                await conn.execute(
                    text(f'ALTER TABLE user_contexts RENAME COLUMN "{old}" TO "{new}"')
                )
                print(f"  OK {old} -> {new}")
            except Exception as e:
                if "does not exist" in str(e) or "already exists" in str(e):
                    print(f"  - {old} -> {new} (skipped: {e})")
                else:
                    raise

    print("\nDone. Restart the backend.")


if __name__ == "__main__":
    asyncio.run(migrate())
