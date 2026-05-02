"""Data models shared across the extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ResolvedTarget:
    raw_input: str
    is_doi: bool
    doi: str | None
    resolved_url: str
    source: str


@dataclass(slots=True)
class DownloadedAsset:
    filename: str
    source_url: str
    content: bytes


@dataclass(slots=True)
class ExtractionAttempt:
    name: str
    success: bool
    message: str


@dataclass(slots=True)
class ExtractionOutcome:
    source: str
    resolved_url: str
    doi: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    article_markdown: str | None = None
    figures: list[DownloadedAsset] = field(default_factory=list)
    supplementary_files: list[DownloadedAsset] = field(default_factory=list)
    pdf: DownloadedAsset | None = None
    attempts: list[ExtractionAttempt] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    anti_bot_blocked: bool = False
    access_limited: bool = False
    access_warning: str | None = None
