"""Playwright-backed browser client for authenticated extraction."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import BrowserContext, Error, sync_playwright

from extraction.config.defaults import DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT
from extraction.core.rate_limit import RateLimiter


@dataclass(slots=True)
class PageSnapshot:
    url: str
    html: str
    status_code: int | None


class BrowserClient:
    def __init__(
        self,
        profile_dir: Path,
        headless: bool,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        rate_limiter: RateLimiter | None = None,
        browser_channel: str = "chromium",
        real_browser_mode: bool = False,
    ) -> None:
        self.profile_dir = profile_dir
        self.headless = headless
        self.timeout_ms = timeout_seconds * 1000
        self.rate_limiter = rate_limiter
        self.browser_channel = browser_channel
        self.real_browser_mode = real_browser_mode
        self._playwright = None
        self._context: BrowserContext | None = None

    @staticmethod
    def _rand_ms(low: int, high: int) -> int:
        return random.randint(low, high)

    def _human_pause(self, page, low_ms: int = 200, high_ms: int = 1200) -> None:
        page.wait_for_timeout(self._rand_ms(low_ms, high_ms))

    def _humanize_page(self, page) -> None:
        """Add lightweight randomized interactions to mimic real browsing rhythm."""
        try:
            width, height = 1440, 980
            for _ in range(random.randint(1, 3)):
                x = random.randint(80, width - 80)
                y = random.randint(80, height - 80)
                steps = random.randint(8, 22)
                page.mouse.move(x, y, steps=steps)
                self._human_pause(page, 80, 280)
            if random.random() < 0.7:
                page.mouse.wheel(0, random.randint(120, 500))
                self._human_pause(page, 120, 320)
            if random.random() < 0.45:
                page.mouse.wheel(0, -random.randint(80, 300))
                self._human_pause(page, 80, 220)
        except Exception:  # noqa: BLE001
            # Humanization is best-effort and should never block extraction.
            return

    def __enter__(self) -> "BrowserClient":
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        launch_kwargs = {}
        if self.browser_channel != "chromium":
            launch_kwargs["channel"] = self.browser_channel
        context_kwargs = {
            "user_data_dir": str(self.profile_dir),
            "headless": self.headless,
            "viewport": {"width": 1440, "height": 980},
            "locale": "en-US",
            "timezone_id": "Asia/Shanghai",
            **launch_kwargs,
        }
        if self.real_browser_mode:
            # Keep browser defaults as close as possible to human browsing.
            self._context = self._playwright.chromium.launch_persistent_context(
                user_agent=DEFAULT_USER_AGENT,
                **context_kwargs,
            )
        else:
            self._context = self._playwright.chromium.launch_persistent_context(
                user_agent=DEFAULT_USER_AGENT,
                ignore_default_args=["--enable-automation"],
                args=["--disable-blink-features=AutomationControlled"],
                **context_kwargs,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._context is not None:
            self._context.close()
        if self._playwright is not None:
            self._playwright.stop()

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("BrowserClient not initialized. Use as a context manager.")
        return self._context

    def fetch_page(self, url: str, wait_until: str = "domcontentloaded") -> PageSnapshot:
        if self.rate_limiter is not None:
            self.rate_limiter.wait_for_url(url)
        page = self.context.new_page()
        try:
            if not self.real_browser_mode:
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self._human_pause(page, 180, 650)
            response = page.goto(url, wait_until=wait_until, timeout=self.timeout_ms)
            self._humanize_page(page)
            self._human_pause(page, 600, 2200)
            html = page.content()
            current_url = page.url
            status_code = None if response is None else response.status
            return PageSnapshot(url=current_url, html=html, status_code=status_code)
        finally:
            page.close()

    def download_binary(self, url: str) -> bytes:
        if self.rate_limiter is not None:
            self.rate_limiter.wait_for_url(url)
        response = self.context.request.get(url, timeout=self.timeout_ms)
        status = response.status
        if status in {403, 429} and self.rate_limiter is not None:
            self.rate_limiter.backoff()
        if status >= 400:
            raise Error(f"Failed to download {url}: HTTP {status}")
        return response.body()

    def download_binary_via_page(self, url: str) -> bytes:
        if self.rate_limiter is not None:
            self.rate_limiter.wait_for_url(url)
        page = self.context.new_page()
        try:
            if not self.real_browser_mode:
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self._human_pause(page, 120, 500)
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            if response is None:
                raise Error(f"Failed to download {url}: no response")
            self._human_pause(page, 250, 900)
            status = response.status
            if status in {403, 429} and self.rate_limiter is not None:
                self.rate_limiter.backoff()
            if status >= 400:
                raise Error(f"Failed to download {url}: HTTP {status}")
            return response.body()
        finally:
            page.close()

    def download_binary_via_download_event(self, url: str, referer_url: str | None = None) -> bytes:
        """Trigger a browser-native download event in current session context."""
        if self.rate_limiter is not None:
            self.rate_limiter.wait_for_url(url)
        page = self.context.new_page()
        try:
            if not self.real_browser_mode:
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            if referer_url:
                page.goto(referer_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                self._humanize_page(page)
                self._human_pause(page, 400, 1300)
            with page.expect_download(timeout=self.timeout_ms) as download_info:
                page.evaluate(
                    """(u) => {
                        const a = document.createElement('a');
                        a.href = u;
                        a.rel = 'noopener noreferrer';
                        a.target = '_self';
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                    }""",
                    url,
                )
            self._human_pause(page, 200, 700)
            download = download_info.value
            download_path = download.path()
            if download_path is None:
                raise Error(f"Failed to download {url}: no download path")
            return Path(download_path).read_bytes()
        finally:
            page.close()

    def interactive_login(
        self,
        url: str,
        wait_seconds: int = 90,
        detect_challenge: bool = True,
    ) -> bool:
        """Open a page for manual login and optionally detect challenge resolution."""
        page = self.context.new_page()
        challenge_phrases = (
            "verifying you are human",
            "just a moment",
            "security service to protect against malicious bots",
        )
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            if self.headless:
                page.wait_for_timeout(wait_seconds * 1000)
                return True

            deadline = time.time() + max(10, wait_seconds)
            challenge_seen = False
            while time.time() < deadline:
                body_text = page.inner_text("body").lower()
                in_challenge = any(phrase in body_text for phrase in challenge_phrases)
                if in_challenge:
                    challenge_seen = True
                elif challenge_seen:
                    return True
                page.wait_for_timeout(2500)

            return not challenge_seen
        finally:
            page.close()
