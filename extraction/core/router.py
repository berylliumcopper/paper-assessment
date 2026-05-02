"""Route a resolved URL to a supported source extractor."""

from __future__ import annotations

from urllib.parse import urlparse

SUPPORTED_SOURCES = {"arxiv", "nature", "science", "aps"}


def detect_source(url: str) -> str:
    host = urlparse(url).netloc.lower()

    if "arxiv.org" in host:
        return "arxiv"
    if "nature.com" in host:
        return "nature"
    if "science.org" in host or host.endswith("sciencemag.org"):
        return "science"
    if "aps.org" in host or host.endswith("journals.aps.org"):
        return "aps"

    return "unknown"
