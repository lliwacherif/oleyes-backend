"""
Create all ORM tables in the oleyes database.

Usage:
    python scripts/create_tables.py
"""

import asyncio
import sys
import os

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.models  # noqa: F401 — registers models with Base.metadata
from app.core.database import engine, Base


async def main() -> None:
    sys.stdout.buffer.write(b"Creating tables...\n")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    sys.stdout.buffer.write(b"Done. Tables created successfully.\n")


if __name__ == "__main__":
    asyncio.run(main())
