from __future__ import annotations

from typing import AsyncGenerator, Iterable

import httpx

from app.core import config


class ScalewayClient:
    def __init__(self) -> None:
        if not config.SCALWAY_API_KEY:
            raise ValueError("SCALWAY_API_KEY is not set")

        self._client = httpx.AsyncClient(
            base_url=config.SCALWAY_BASE_URL,
            headers={"Authorization": f"Bearer {config.SCALWAY_API_KEY}"},
            timeout=config.SCALWAY_TIMEOUT,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
    ) -> str:
        payload = {
            "model": config.SCALWAY_MODEL,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "stream": False,
            "response_format": {"type": "text"},
        }
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    async def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": config.SCALWAY_MODEL,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "stream": True,
            "response_format": {"type": "text"},
        }

        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                yield line[5:].strip()
