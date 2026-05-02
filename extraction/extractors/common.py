"""Common helper functions for publisher extractors."""

from __future__ import annotations

import re
import json
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from extraction.core.models import DownloadedAsset


def is_challenge_page(soup: BeautifulSoup, status_code: int | None = None) -> bool:
    if status_code in {403, 429}:
        return True
    title = title_from_soup(soup)
    if title and any(phrase in title.lower() for phrase in ["just a moment", "attention required", "security measure"]):
        return True
    body_text = soup.get_text(" ", strip=True).lower()
    challenge_phrases = [
        "verifying you are human",
        "security service to protect against malicious bots",
        "checking your browser before accessing",
        "please wait while we verify",
    ]
    return any(phrase in body_text for phrase in challenge_phrases)

def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return parsed.netloc.replace(".", "_")
    raw = "_".join(parts[-3:])
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_").lower() or "article"


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def text_from_selectors(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        blocks = soup.select(selector)
        if blocks:
            paragraphs: list[str] = []
            for block in blocks:
                text = block.get_text(" ", strip=True)
                if text:
                    paragraphs.append(text)
            if paragraphs:
                return "\n\n".join(paragraphs)
    return ""


def title_from_soup(soup: BeautifulSoup) -> str | None:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    meta = soup.select_one('meta[property="og:title"]')
    if meta and meta.get("content"):
        return str(meta["content"]).strip()
    return None


def abstract_from_soup(soup: BeautifulSoup) -> str | None:
    candidates = [
        'meta[name="dc.description"]',
        'meta[name="description"]',
        'meta[property="og:description"]',
    ]
    for selector in candidates:
        el = soup.select_one(selector)
        if el and el.get("content"):
            value = str(el["content"]).strip()
            if value:
                return value
    return None


def normalize_figure_url(base_url: str, raw: str) -> str:
    return urljoin(base_url, raw)


def infer_extension(url: str) -> str:
    lowered = url.lower()
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"):
        if ext in lowered:
            return ext
    return ".bin"


def build_figure_asset(url: str, content: bytes, index: int) -> DownloadedAsset:
    ext = infer_extension(url)
    filename = f"figure_{index:03d}{ext}"
    return DownloadedAsset(filename=filename, source_url=url, content=content)


def jsonld_blocks(soup: BeautifulSoup) -> list[dict]:
    blocks: list[dict] = []
    for node in soup.select('script[type="application/ld+json"]'):
        raw = node.string or node.get_text()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(parsed, list):
            blocks.extend([item for item in parsed if isinstance(item, dict)])
        elif isinstance(parsed, dict):
            blocks.append(parsed)
    return blocks


def authors_from_jsonld(soup: BeautifulSoup) -> list[str]:
    names: list[str] = []
    for block in jsonld_blocks(soup):
        author = block.get("author")
        if isinstance(author, list):
            for entry in author:
                if isinstance(entry, dict) and entry.get("name"):
                    names.append(str(entry["name"]).strip())
        elif isinstance(author, dict) and author.get("name"):
            names.append(str(author["name"]).strip())
    return [n for n in dict.fromkeys(names) if n]


def abstract_from_jsonld(soup: BeautifulSoup) -> str | None:
    for block in jsonld_blocks(soup):
        value = block.get("description") or block.get("abstract")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
