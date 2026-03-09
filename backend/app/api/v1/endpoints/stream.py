from fastapi import APIRouter

router = APIRouter(prefix="/stream")


@router.get("/health")
async def stream_health() -> dict:
    return {"status": "ok"}
