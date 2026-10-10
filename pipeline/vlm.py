"""Validated VLM scene understanding with an explicit fallback."""

from __future__ import annotations

import base64
import json
import logging
import os
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


def _parse_json(content: str) -> Dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return _validate(json.loads(content))


def fallback_understanding(
    objects: Optional[Dict[str, int]],
) -> Dict[str, Any]:
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
    model: str = "gemini-2.5-flash",
    retries: int = 2,
    fallback_objects: Optional[Dict[str, int]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Gemini or an OpenAI-compatible vision endpoint."""

    gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
    use_gemini = bool(gemini_key and not endpoint)
    if not endpoint and not use_gemini:
        return fallback_understanding(fallback_objects)
    image = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
    prompt = (
        'Return only JSON with exactly these fields: {"caption": string, '
        '"exciting": boolean, "reason": string}. Caption the image in one sentence '
        "and decide whether something exciting or important is happening."
    )
    if use_gemini:
        endpoint = (
            endpoint
            or f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        request = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image}},
                ]
            }],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        headers = {"x-goog-api-key": gemini_key or ""}
    else:
        request = {
            "model": model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }},
                ],
            }],
        }
        headers = {}
    last_error: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=request, timeout=90)
            response.raise_for_status()
            payload = response.json()
            if use_gemini:
                content = payload["candidates"][0]["content"]["parts"][0]["text"]
            else:
                content = payload["choices"][0]["message"]["content"]
            return _parse_json(content)
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in (401, 403, 404):
                if status_code == 404:
                    detail = (
                        "The configured Gemini model is unavailable to this project. "
                        "Check GEMINI_MODEL and the project's model access."
                    )
                else:
                    detail = "Check GEMINI_API_KEY and the Google project's access."
                LOGGER.error(
                    "VLM request rejected (HTTP %s). %s Using local fallback.",
                    status_code, detail
                )
                break
            LOGGER.warning("VLM request failed; retrying: %s", exc)
        except (OSError, KeyError, TypeError, ValueError, requests.RequestException) as exc:
            last_error = exc
            LOGGER.warning("VLM response invalid; retrying: %s", exc)
    if last_error and not (
        isinstance(last_error, requests.HTTPError)
        and last_error.response is not None
        and last_error.response.status_code in (401, 403, 404)
    ):
        LOGGER.error("VLM failed after retries: %s", last_error)
    return fallback_understanding(fallback_objects)
