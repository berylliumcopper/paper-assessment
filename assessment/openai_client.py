"""Minimal OpenAI API client for structured assessment output."""

from __future__ import annotations

import base64
import json
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import requests


class OpenAIClientError(RuntimeError):
    """Raised when OpenAI API calls fail or return invalid content."""


@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = 600
    max_retries: int = 5
    base_url: str = "https://api.openai.com/v1"
    _last_request_time: float = 0.0
    _min_interval: float = 2.0  # seconds between requests

    def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between API requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
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
        use_images = image_paths  # may be set to None if model rejects images
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit_wait()
                # Start a background spinner so the user sees activity
                stop_event = threading.Event()
                spinner = threading.Thread(
                    target=self._spinner, args=(stop_event, attempt), daemon=True
                )
                spinner.start()
                try:
                    start = time.perf_counter()
                    payload = self._chat_completion(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_paths=use_images,
                    )
                    elapsed = time.perf_counter() - start
                finally:
                    stop_event.set()
                    spinner.join(timeout=1)

                result = self._extract_json(payload)
                print(f"    [api] ✓ response received in {elapsed:.1f}s", flush=True)
                return result
            except OpenAIClientError as exc:
                last_error = exc
                err_msg = str(exc)
                if "image_url" in err_msg and "400" in err_msg and use_images:
                    # Model doesn't support vision — retry without images
                    print(
                        f"    [api] ⚠ model doesn't support images, retrying text-only...",
                        flush=True,
                    )
                    use_images = None
                    continue
                elif "429" in err_msg:
                    backoff = min(15 * attempt, 60)
                    print(
                        f"    [api] ⚠ rate limited (429), backing off {backoff}s "
                        f"(attempt {attempt}/{self.max_retries})...",
                        flush=True,
                    )
                    time.sleep(backoff)
                elif "504" in err_msg or "502" in err_msg or "503" in err_msg:
                    backoff = min(10 * attempt, 30)
                    print(
                        f"    [api] ⚠ gateway error, retrying in {backoff}s "
                        f"(attempt {attempt}/{self.max_retries})...",
                        flush=True,
                    )
                    time.sleep(backoff)
                elif attempt == self.max_retries:
                    break
                else:
                    time.sleep(min(2**attempt, 8))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise OpenAIClientError(f"OpenAI request failed after retries: {last_error}")

    @staticmethod
    def _spinner(stop_event: threading.Event, attempt: int) -> None:
        """Print elapsed time every 10 seconds so the user knows we're alive."""
        start = time.perf_counter()
        while not stop_event.is_set():
            stop_event.wait(10)
            if not stop_event.is_set():
                elapsed = time.perf_counter() - start
                print(f"    [api] ... waiting ({elapsed:.0f}s elapsed)", flush=True)

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
        
        # Print token usage if available
        usage = data.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")
            print(
                f"    [api] tokens: {prompt_tokens} in → {completion_tokens} out",
                flush=True,
            )
        
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
