"""
Create the 'oleyes' PostgreSQL database if it doesn't already exist.

Usage:
    python scripts/create_db.py

Reads DATABASE_URL from ../.env (or falls back to the default).
Connects to the 'postgres' maintenance DB to issue CREATE DATABASE.
"""

import os
import sys
from urllib.parse import urlparse

import psycopg
from dotenv import load_dotenv

# Load .env from the backend root
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/oleyes",
)

# ── Parse the URL ────────────────────────────────────────────────────
# Strip the "+asyncpg" driver part
clean_url = DATABASE_URL.replace("+asyncpg", "")
parsed = urlparse(clean_url)

DB_NAME = parsed.path.lstrip("/")        # e.g. "oleyes"
DB_USER = parsed.username or "postgres"
DB_PASS = parsed.password or "postgres"
DB_HOST = parsed.hostname or "localhost"
DB_PORT = parsed.port or 5432


def main() -> None:
    print(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT} as '{DB_USER}' ...")

    # Connect to the default 'postgres' database
    conninfo = f"host={DB_HOST} port={DB_PORT} dbname=postgres user={DB_USER} password={DB_PASS}"
    conn = psycopg.connect(conninfo, autocommit=True)

    # Check if the database already exists
    cur = conn.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s;",
        (DB_NAME,),
    )
    exists = cur.fetchone()

    if exists:
        print(f"Database '{DB_NAME}' already exists. Nothing to do.")
    else:
        # CREATE DATABASE cannot run inside a transaction — autocommit handles this
        conn.execute(f'CREATE DATABASE "{DB_NAME}"')
        print(f"Database '{DB_NAME}' created successfully.")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except psycopg.OperationalError as e:
        print(f"\nERROR: Could not connect to PostgreSQL.\n{e}")
        print("Make sure PostgreSQL is running and the credentials in .env are correct.")
        sys.exit(1)
