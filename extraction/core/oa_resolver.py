"""Open-access resolution via Crossref and Unpaywall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class OAResolution:
    doi: str
    status: str
    source: str | None
    landing_url: str | None
    oa_url: str | None
    message: str | None


def _crossref_lookup(doi: str, timeout_seconds: int) -> dict[str, Any] | None:
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url, timeout=timeout_seconds)
    if response.status_code >= 400:
        return None
    payload = response.json()
    message = payload.get("message")
    if isinstance(message, dict):
        return message
    return None


def _crossref_candidate(message: dict[str, Any]) -> str | None:
    links = message.get("link")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            candidate = link.get("URL")
            if isinstance(candidate, str) and candidate:
                return candidate
    resource = message.get("resource")
    if isinstance(resource, dict):
        primary = resource.get("primary")
        if isinstance(primary, dict):
            url = primary.get("URL")
            if isinstance(url, str) and url:
                return url
    return None


def _unpaywall_lookup(doi: str, email: str, timeout_seconds: int) -> dict[str, Any] | None:
    url = f"https://api.unpaywall.org/v2/{doi}"
    response = requests.get(url, params={"email": email}, timeout=timeout_seconds)
    if response.status_code >= 400:
        return None
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    return None


def _unpaywall_candidate(payload: dict[str, Any]) -> str | None:
    best = payload.get("best_oa_location")
    if isinstance(best, dict):
        pdf_url = best.get("url_for_pdf")
        if isinstance(pdf_url, str) and pdf_url:
            return pdf_url
        url = best.get("url")
        if isinstance(url, str) and url:
            return url

    locations = payload.get("oa_locations")
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            pdf_url = location.get("url_for_pdf")
            if isinstance(pdf_url, str) and pdf_url:
                return pdf_url
            url = location.get("url")
            if isinstance(url, str) and url:
                return url
    return None


def resolve_open_access(
    doi: str,
    timeout_seconds: int = 30,
    unpaywall_email: str | None = None,
) -> OAResolution:
    crossref = _crossref_lookup(doi, timeout_seconds=timeout_seconds)
    crossref_landing = None if crossref is None else crossref.get("URL")
    if not isinstance(crossref_landing, str):
        crossref_landing = None

    if unpaywall_email:
        unpaywall = _unpaywall_lookup(doi, email=unpaywall_email, timeout_seconds=timeout_seconds)
        if unpaywall is not None:
            is_oa = bool(unpaywall.get("is_oa"))
            if is_oa:
                candidate = _unpaywall_candidate(unpaywall)
                if candidate:
                    return OAResolution(
                        doi=doi,
                        status="open_access_found",
                        source="unpaywall",
                        landing_url=crossref_landing,
                        oa_url=candidate,
                        message="Open-access URL found via Unpaywall.",
                    )

    if crossref is not None:
        candidate = _crossref_candidate(crossref)
        if candidate:
            return OAResolution(
                doi=doi,
                status="candidate_found",
                source="crossref",
                landing_url=crossref_landing,
                oa_url=candidate,
                message="Crossref candidate URL found.",
            )

    return OAResolution(
        doi=doi,
        status="not_found",
        source=None,
        landing_url=crossref_landing,
        oa_url=None,
        message="No open-access candidate found from Crossref/Unpaywall.",
    )
