import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import auth, cameras, connect, context, llm, stream, vision
from app.core.database import init_db, close_db
import app.models  # noqa: F401 — register ORM models with Base.metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Quiet down noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # ── startup ──
    await init_db()
    logging.getLogger("oleyes").info("Database connected.")
    yield
    # ── shutdown ──
    await close_db()
    logging.getLogger("oleyes").info("Database disconnected.")


app = FastAPI(
    title="Smart AI CCTV Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok"}


app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
app.include_router(cameras.router, prefix="/api/v1", tags=["cameras"])
app.include_router(connect.router, prefix="/api/v1", tags=["connect"])
app.include_router(context.router, prefix="/api/v1", tags=["context"])
app.include_router(llm.router, prefix="/api/v1", tags=["llm"])
app.include_router(stream.router, prefix="/api/v1", tags=["stream"])
app.include_router(vision.router, prefix="/api/v1", tags=["vision"])
