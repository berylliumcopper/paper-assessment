"""Minimal Gemini API client for structured assessment output."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests


class GeminiClientError(RuntimeError):
    """Raised when Gemini API calls fail or return invalid content."""


@dataclass(slots=True)
class GeminiClient:
    api_key: str
    model: str = "gemini-1.5-flash"
    timeout_seconds: int = 600
    max_retries: int = 3
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def generate_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path] | None = None) -> dict:
        """Request a JSON response and parse it into a dict."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self._generate_content(system_prompt=system_prompt, user_prompt=user_prompt, image_paths=image_paths)
                return self._extract_json(payload)
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise GeminiClientError(f"Gemini request failed after retries: {last_error}")

    def _generate_content(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path] | None = None) -> str:
        # Gemini expects system instructions separately or as part of the prompt.
        # Using the v1beta generateContent endpoint.
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
        }
        
        parts = [{"text": f"{system_prompt}\n\n{user_prompt}"}]
        
        if image_paths:
            for path in image_paths:
                if not path.exists():
                    continue
                encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
                mime_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded
                    }
                })

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=self.timeout_seconds,
        )
        
        if response.status_code >= 400:
            raise GeminiClientError(f"Gemini HTTP {response.status_code}: {response.text[:500]}")
        
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise GeminiClientError("Gemini response did not include candidates.")
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise GeminiClientError("Gemini response content parts were empty.")
            
        text = parts[0].get("text")
        if not isinstance(text, str) or not text.strip():
            raise GeminiClientError("Gemini response text was empty.")
            
        return text

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
            raise GeminiClientError("No JSON object found in model output.")
        parsed = json.loads(stripped[start : end + 1])
        if not isinstance(parsed, dict):
            raise GeminiClientError("Model output JSON root must be an object.")
        return parsed
