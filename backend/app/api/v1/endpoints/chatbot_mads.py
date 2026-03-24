import logging
import re
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot-mads")

_MAX_DOC_CHARS = 45_000

_SYSTEM_PROMPT = (
    "**Role:** You are the MADS CRM In-App Support Assistant. "
    "Your job is to help Commercials and Poseurs use the mobile app.\n\n"
    "**Tone:** Helpful, concise, and direct. Your users are often on mobile "
    "devices in the field, so use short sentences and bulleted lists for "
    "step-by-step instructions.\n\n"
    "**Rules:**\n"
    "1. **Strictly adhere to the provided documentation.** Do not invent "
    "features, buttons, or workflows.\n"
    "2. **Be specific with navigation.** Tell users exactly what to tap "
    "(e.g., \"Tap the '+ New Client' button\").\n"
    "3. **Handle missing info gracefully.** If the answer is not in the "
    "provided text, do not guess. Tell the user: \"I don't have that "
    "information. Please go to Profile > Contact us for support.\"\n"
    "4. **Never use internal reasoning, thinking tags, or chain-of-thought.** "
    "Answer the user directly and immediately.\n"
    "5. **Always respond in the same language the user writes in.**"
)

_PDF_PATH = Path(__file__).resolve().parents[4] / "chatbot_files" / "MADS.pdf"


def _load_pdf_text() -> str:
    """Extract text from the MADS PDF at import time."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(_PDF_PATH))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        logger.info("mads_pdf_loaded pages=%d chars=%d", len(reader.pages), len(text))
        return text
    except ImportError:
        logger.warning("pypdf not installed, trying PyPDF2")
    try:
        from PyPDF2 import PdfReader as PdfReader2

        reader = PdfReader2(str(_PDF_PATH))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        logger.info("mads_pdf_loaded pages=%d chars=%d", len(reader.pages), len(text))
        return text
    except ImportError:
        logger.warning("PyPDF2 not installed either")

    if _PDF_PATH.exists():
        raw = _PDF_PATH.read_bytes()
        text = raw.decode("utf-8", errors="ignore")
        logger.info("mads_pdf_raw_fallback chars=%d", len(text))
        return text
    logger.error("mads_pdf_not_found path=%s", _PDF_PATH)
    return ""


def _compress_doc(text: str) -> str:
    """Reduce whitespace and page markers to save tokens."""
    text = re.sub(r"-- \d+ of \d+ --", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    if len(text) > _MAX_DOC_CHARS:
        text = text[:_MAX_DOC_CHARS] + "\n\n[Document truncated]"
        logger.warning("mads_doc_truncated to %d chars", _MAX_DOC_CHARS)
    return text


_DOC_CONTEXT = _compress_doc(_load_pdf_text())
logger.info("mads_doc_final_size chars=%d", len(_DOC_CONTEXT))


def _build_system_message() -> str:
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"--- START OF MADS CRM DOCUMENTATION ---\n\n"
        f"{_DOC_CONTEXT}\n\n"
        f"--- END OF MADS CRM DOCUMENTATION ---"
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the MADS CRM support chatbot and get a reply."""
    system_msg = _build_system_message()

    payload = {
        "model": config.SCALWAY_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": request.message},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 1,
        "presence_penalty": 0,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(
            base_url=config.SCALWAY_BASE_URL,
            headers={"Authorization": f"Bearer {config.SCALWAY_API_KEY}"},
            timeout=90.0,
        ) as client:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        ) or ""

        content = _strip_thinking(content)

        if not content.strip():
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM returned empty response.",
            )

        logger.info("mads_chat tokens_used=%s", data.get("usage", {}))
        return ChatResponse(reply=content.strip())

    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:300]
        logger.error("mads_chat_error status=%s body=%s", exc.response.status_code, body)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API error {exc.response.status_code}: {body}",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM request timed out.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("mads_chat_unexpected error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(exc)[:200]}",
        )


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit despite instructions."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
