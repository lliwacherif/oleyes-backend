from fastapi import APIRouter

router = APIRouter(prefix="/connect")


@router.get("/health")
async def connect_health() -> dict:
    return {"status": "ok"}
