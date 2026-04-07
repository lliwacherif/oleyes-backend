from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from app.core import config

logger = logging.getLogger(__name__)

SUPREME_SYSTEM_PROMPT = (
    "You are a top-tier retail security analyst for OLEYES. "
    "You will be provided with a sequence of 3 chronological images showing a person "
    "interacting with an item. Analyze the temporal sequence to determine if shoplifting "
    "or theft is occurring. Look for items being concealed, erratic movements, or "
    "suspicious departure.\n\n"
    "You MUST respond ONLY with a valid JSON object in this exact format:\n"
    '{\n'
    '  "analysis": "A brief 1-sentence explanation of the sequence.",\n'
    '  "theft_detected": true/false,\n'
    '  "confidence_score": 0-100\n'
    '}'
)

SUPREME_USER_PROMPT = (
    "Analyze this 3-frame chronological sequence of a user interacting with an item. "
    "Based on the visual evidence, is a theft occurring? Return the requested JSON."
)


class PixtralClient:
    """Synchronous Pixtral VLM client for multi-image theft analysis.

    Designed to run inside a background thread (same pattern as the existing
    LLM analysis consumer in Yolo26Service).
    """

    def __init__(self) -> None:
        if not config.SCALWAY_API_KEY:
            raise ValueError("SCALWAY_API_KEY is not set")
        self._client = OpenAI(
            base_url=config.SCALWAY_BASE_URL,
            api_key=config.SCALWAY_API_KEY,
        )

    def analyze_frames(self, base64_frames: list[str]) -> dict:
        """Send multiple Base64-encoded JPEG frames to Pixtral for analysis.

        Returns a dict with keys: analysis, theft_detected, confidence_score.
        """
        content: list[dict] = [{"type": "text", "text": SUPREME_USER_PROMPT}]
        for frame_b64 in base64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        response = self._client.chat.completions.create(
            model=config.SCALWAY_VLM_MODEL,
            messages=[
                {"role": "system", "content": SUPREME_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=config.SUPREME_VLM_MAX_TOKENS,
            temperature=config.SUPREME_VLM_TEMPERATURE,
            top_p=1,
            presence_penalty=0,
            stream=False,
        )

        raw = response.choices[0].message.content or ""
        logger.info("pixtral_raw_response length=%d", len(raw))
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Extract the JSON object from the VLM output, tolerating markdown fences."""
        fallback = {
            "analysis": "VLM response could not be parsed.",
            "theft_detected": False,
            "confidence_score": 0,
        }

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return _validate(obj)
        except (json.JSONDecodeError, TypeError):
            pass

        fence = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", raw, re.DOTALL)
        if fence:
            try:
                obj = json.loads(fence.group(1))
                if isinstance(obj, dict):
                    return _validate(obj)
            except (json.JSONDecodeError, TypeError):
                pass

        # Brute-force: find first balanced { ... }
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        if isinstance(obj, dict):
                            return _validate(obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    start = -1

        logger.warning("pixtral_parse_failed raw=%s", raw[:300])
        return fallback


def _validate(obj: dict) -> dict:
    """Ensure required keys exist with correct types."""
    return {
        "analysis": str(obj.get("analysis", "")),
        "theft_detected": bool(obj.get("theft_detected", False)),
        "confidence_score": int(obj.get("confidence_score", 0)),
    }
