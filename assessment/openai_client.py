"""Minimal OpenAI API client for structured assessment output."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests


class OpenAIClientError(RuntimeError):
    """Raised when OpenAI API calls fail or return invalid content."""


@dataclass(slots=True)
class OpenAIClient:
    api_key: str
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = 600
    max_retries: int = 3
    base_url: str = "https://api.openai.com/v1"

    def generate_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path] | None = None) -> dict:
        """Request a JSON response and parse it into a dict."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self._chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, image_paths=image_paths)
                return self._extract_json(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise OpenAIClientError(f"OpenAI request failed after retries: {last_error}")

    def _chat_completion(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path] | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        
        if image_paths:
            for path in image_paths:
                if not path.exists():
                    continue
                encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
                mime_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
                })

        body = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        response = requests.post(
            url,
            headers=headers,
            json={**body, "response_format": {"type": "json_object"}},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            # Some OpenAI-compatible providers reject response_format=json_object.
            response = requests.post(url, headers=headers, json=body, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise OpenAIClientError(f"OpenAI HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise OpenAIClientError("OpenAI response did not include choices.")
        message = choices[0].get("message", {})
        content_res = message.get("content")
        if not isinstance(content_res, str) or not content_res.strip():
            raise OpenAIClientError("OpenAI response content was empty.")
        return content_res

    @staticmethod
    def _extract_json(text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise OpenAIClientError("No JSON object found in model output.")
        parsed = json.loads(stripped[start : end + 1])
        if not isinstance(parsed, dict):
            raise OpenAIClientError("Model output JSON root must be an object.")
        return parsed

