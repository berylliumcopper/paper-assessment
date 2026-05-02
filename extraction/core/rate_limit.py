"""Simple per-domain throttling with jitter and backoff."""

from __future__ import annotations

import random
import time
from urllib.parse import urlparse


class RateLimiter:
    def __init__(self, base_delay_seconds: float, jitter_seconds: float) -> None:
        self.base_delay_seconds = max(0.0, base_delay_seconds)
        self.jitter_seconds = max(0.0, jitter_seconds)
        self._last_call_by_domain: dict[str, float] = {}

    @staticmethod
    def domain_from_url(url: str) -> str:
        return urlparse(url).netloc.lower()

    def wait_for_url(self, url: str) -> None:
        self.wait_for_domain(self.domain_from_url(url))

    def wait_for_domain(self, domain: str) -> None:
        now = time.monotonic()
        last = self._last_call_by_domain.get(domain)
        delay = self.base_delay_seconds + random.uniform(0, self.jitter_seconds)
        if last is not None:
            elapsed = now - last
            remaining = delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call_by_domain[domain] = time.monotonic()

    def backoff(self, multiplier: float = 2.0) -> None:
        extra = self.base_delay_seconds * max(1.0, multiplier)
        time.sleep(extra + random.uniform(0, self.jitter_seconds))
