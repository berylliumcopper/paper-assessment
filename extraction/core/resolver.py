"""Resolve DOI or URL input into a normalized target URL."""

from __future__ import annotations

import re

import requests

from extraction.core.models import ResolvedTarget
from extraction.core.router import detect_source

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
DOI_IN_URL_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


def looks_like_doi(value: str) -> bool:
    return bool(DOI_PATTERN.match(value.strip()))


def normalize_doi(value: str) -> str:
    doi = value.strip()
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    doi = doi.removeprefix("doi:")
    return doi.strip()


def extract_doi_from_text(value: str) -> str | None:
    match = DOI_IN_URL_PATTERN.search(value)
    if not match:
        return None
    doi = normalize_doi(match.group(1))
    return doi if looks_like_doi(doi) else None


def resolve_target(raw_input: str, timeout_seconds: int = 30) -> ResolvedTarget:
    candidate = raw_input.strip()
    doi = normalize_doi(candidate)
    is_doi = looks_like_doi(doi)

    if is_doi:
        doi_url = f"https://doi.org/{doi}"
        response = requests.get(doi_url, timeout=timeout_seconds, allow_redirects=True)
        response.raise_for_status()
        resolved_url = response.url
        source = detect_source(resolved_url)
        return ResolvedTarget(
            raw_input=raw_input,
            is_doi=True,
            doi=doi,
            resolved_url=resolved_url,
            source=source,
        )

    resolved_url = candidate
    source = detect_source(resolved_url)
    return ResolvedTarget(
        raw_input=raw_input,
        is_doi=False,
        doi=extract_doi_from_text(candidate),
        resolved_url=resolved_url,
        source=source,
    )
