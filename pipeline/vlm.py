"""Validated VLM scene understanding with an explicit fallback."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import requests

LOGGER = logging.getLogger(__name__)


def _validate(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    caption = payload.get("caption")
    exciting = payload.get("exciting")
    reason = payload.get("reason")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("VLM caption must be a non-empty string")
    if not isinstance(exciting, bool):
        raise ValueError("VLM exciting must be a boolean")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("VLM reason must be a non-empty string")
    return {"caption": caption.strip(), "exciting": exciting, "reason": reason.strip()}


def fallback_understanding(objects: Dict[str, int] | None) -> Dict[str, Any]:
    labels = sorted(objects or {}, key=lambda label: (-objects[label], label))
    subject = ", ".join(labels[:3]) if labels else "the scene"
    return {
        "caption": f"Scene showing {subject}.",
        "exciting": False,
        "reason": "No VLM was configured; this is a deterministic fallback.",
    }


def understand_frame(
    frame_path: str,
    endpoint: Optional[str] = None,
    model: str = "llava",
    retries: int = 2,
    fallback_objects: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible vision endpoint and strictly validate JSON."""

    if not endpoint:
        return fallback_understanding(fallback_objects)
    image = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
    prompt = (
        'Return only JSON with exactly these fields: {"caption": string, '
        '"exciting": boolean, "reason": string}. Caption the image in one sentence '
        "and decide whether something exciting or important is happening."
    )
    request = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                ],
            }
        ],
    }
    last_error: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            response = requests.post(endpoint, json=request, timeout=90)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _validate(json.loads(content))
        except (OSError, KeyError, TypeError, ValueError, requests.RequestException) as exc:
            last_error = exc
            LOGGER.warning("VLM response invalid; retrying: %s", exc)
    LOGGER.error("VLM failed after retries: %s", last_error)
    return fallback_understanding(fallback_objects)
