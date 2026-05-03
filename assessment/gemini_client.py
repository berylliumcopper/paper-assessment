"""Minimal Gemini API client for structured assessment output."""

from __future__ import annotations

import base64
import json
import time
import threading
from dataclasses import dataclass
from pathlib import Path

import requests


class GeminiClientError(RuntimeError):
    """Raised when Gemini API calls fail or return invalid content."""


@dataclass
class GeminiClient:
    api_key: str
    model: str = "gemini-1.5-flash"
    timeout_seconds: int = 600
    max_retries: int = 5
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    _last_request_time: float = 0.0
    _min_interval: float = 13.0  # minimum seconds between requests (free tier: 5 RPM)

    def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between API requests to stay within RPM limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            print(f"    [rate-limit] waiting {wait:.1f}s before next Gemini request...", flush=True)
            time.sleep(wait)
        self._last_request_time = time.time()

    def generate_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path] | None = None) -> dict:
        """Request a JSON response and parse it into a dict."""
        prompt_chars = len(system_prompt) + len(user_prompt)
        n_images = len(image_paths) if image_paths else 0
        print(
            f"    [api] calling {self.model} | prompt ~{prompt_chars:,} chars"
            + (f" + {n_images} images" if n_images else "")
            + f" | timeout {self.timeout_seconds}s",
            flush=True,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit_wait()
                stop_event = threading.Event()
                spinner = threading.Thread(
                    target=self._spinner, args=(stop_event, attempt), daemon=True
                )
                spinner.start()
                try:
                    start = time.perf_counter()
                    payload = self._generate_content(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_paths=image_paths,
                    )
                    elapsed_s = time.perf_counter() - start
                finally:
                    stop_event.set()
                    spinner.join(timeout=1)

                result = self._extract_json(payload)
                print(f"    [api] ✓ response received in {elapsed_s:.1f}s", flush=True)
                return result
            except GeminiClientError as exc:
                last_error = exc
                err_msg = str(exc)
                if "429" in err_msg:
                    backoff = min(15 * attempt, 60)
                    print(
                        f"    [api] ⚠ rate limited (429), backing off {backoff}s "
                        f"(attempt {attempt}/{self.max_retries})...",
                        flush=True,
                    )
                    time.sleep(backoff)
                elif attempt == self.max_retries:
                    break
                else:
                    time.sleep(min(2**attempt, 8))
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise GeminiClientError(f"Gemini request failed after retries: {last_error}")

    @staticmethod
    def _spinner(stop_event: threading.Event, attempt: int) -> None:
        """Print elapsed time every 10 seconds so the user knows we're alive."""
        start = time.perf_counter()
        while not stop_event.is_set():
            stop_event.wait(10)
            if not stop_event.is_set():
                elapsed = time.perf_counter() - start
                print(f"    [api] ... waiting ({elapsed:.0f}s elapsed)", flush=True)

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

        # Print token usage if available
        usage = data.get("usageMetadata", {})
        if usage:
            prompt_tokens = usage.get("promptTokenCount", "?")
            completion_tokens = usage.get("candidatesTokenCount", "?")
            print(
                f"    [api] tokens: {prompt_tokens} in → {completion_tokens} out",
                flush=True,
            )

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
