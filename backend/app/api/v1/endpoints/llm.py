import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core import config
from app.services.llm_engine import ScalewayClient

logger = logging.getLogger("oleyes.llm")
router = APIRouter(prefix="/llm")


class LLMMessage(BaseModel):
    role: str = Field(..., examples=["system", "user", "assistant"])
    content: str


class LLMChatRequest(BaseModel):
    messages: list[LLMMessage]
    scene_context: str | None = Field(
        default=None,
        description="Optional scene description to ground the AI analysis (e.g. 'Supermarket, busy hour, primarily food aisles')",
    )
    max_tokens: int | None = 512
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    presence_penalty: float | None = 0.0


class LLMChatResponse(BaseModel):
    model: str
    content: str


SCENE_CONTEXT_TEMPLATE = (
    "\n\nCONTEXT INFORMATION:\n"
    "The user has provided the following context for the scene being observed:\n"
    "\"{scene_context}\"\n"
    "Use this context to better understand the environment, potential risks, "
    "and normal behaviors associated with this setting."
)


def _inject_scene_context(
    messages: list[dict], scene_context: str
) -> list[dict]:
    """Append scene-context block to the first system message, or prepend a
    new system message if none exists."""
    context_block = SCENE_CONTEXT_TEMPLATE.format(scene_context=scene_context)

    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] += context_block
            return messages

    # No system message found – prepend one with the context
    messages.insert(0, {"role": "system", "content": context_block.strip()})
    return messages


@router.post("/chat", response_model=LLMChatResponse)
async def create_chat(request: LLMChatRequest) -> LLMChatResponse:
    logger.info("💬 Chat request received  |  messages: %d", len(request.messages))

    messages = [msg.model_dump() for msg in request.messages]

    if request.scene_context:
        logger.info("🎬 Scene context provided: \"%s\"", request.scene_context)
        messages = _inject_scene_context(messages, request.scene_context)
        logger.info("✅ Scene context injected into system prompt")
    else:
        logger.info("⚠️  No scene context provided — using default prompt only")

    client = ScalewayClient()
    try:
        logger.info("🚀 Sending %d message(s) to LLM...", len(messages))
        content = await client.chat(
            messages=messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            presence_penalty=request.presence_penalty or 0.0,
        )
        logger.info("🤖 LLM response received  |  length: %d chars", len(content))
    finally:
        await client.close()
    return LLMChatResponse(model=config.SCALWAY_MODEL or "unknown", content=content)
