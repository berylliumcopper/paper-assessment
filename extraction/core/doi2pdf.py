"""DOI-to-PDF backup pipeline using multiple strategies.

Provides a fallback when browser-based publisher extraction is blocked by
anti-bot checks or access restrictions.  Strategies (tried in order):

    1. OpenAlex OA URL lookup (legit, no Sci-Hub)
    2. Enhanced CrossRef PDF-specific URL detection
    3. Multi-mirror Sci-Hub fallback (configurable mirror list)

Usage::

    from extraction.core.doi2pdf import try_doi2pdf_fallback

    result = try_doi2pdf_fallback("10.1038/s41586-026-10420-y")
    if result.success:
        with open("paper.pdf", "wb") as f:
            f.write(result.pdf_content)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from extraction.core.models import DownloadedAsset, ExtractionAttempt, ExtractionOutcome

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Sci-Hub mirrors (discoverable via sci-hub.se which redirects).
# ---------------------------------------------------------------------------
DEFAULT_SCI_HUB_MIRRORS: list[str] = [
    "https://sci-hub.se",
    "https://sci-hub.ru",
    "https://sci-hub.ee",
    "https://sci-hub.st",
]

# ---------------------------------------------------------------------------
# Browser-like headers to avoid trivial blocking at mirror sites.
# ---------------------------------------------------------------------------
_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_PDF_HEADERS: dict[str, str] = dict(_DEFAULT_HEADERS, Accept="application/pdf,*/*;q=0.8")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Doi2PdfResult:
    """Result of a doi2pdf fallback attempt."""

    success: bool
    pdf_content: bytes | None
    source: str  # e.g. "doi2pdf_library", "crossref", "sci-hub.se"
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_pdf_content(content: bytes) -> bool:
    """Check whether *content* starts with the PDF magic bytes."""
    return content[:4] == b"%PDF"


def _fetch_pdf(url: str, timeout: int, headers: dict[str, str] | None = None) -> bytes | None:
    """Fetch *url* and return raw bytes if the response is a PDF, else ``None``."""
    resp = requests.get(url, timeout=timeout, headers=headers or _PDF_HEADERS, allow_redirects=True)
    if resp.status_code >= 400:
        _logger.debug("HTTP %s for %s", resp.status_code, url)
        return None
    raw = resp.content
    if not _is_pdf_content(raw):
        _logger.debug("Not a PDF response from %s (content-type: %s)", url, resp.headers.get("content-type", ""))
        return None
    return raw


def _extract_pdf_url_from_html(html: str, base_url: str) -> str | None:
    """Try to locate a PDF URL inside a Sci-Hub result page."""
    soup = BeautifulSoup(html, "html.parser")

    # Pattern 1: <embed type="application/pdf" src="...">
    embed = soup.select_one("embed[type='application/pdf']")
    if embed and embed.get("src"):
        return _abs_url(str(embed["src"]), base_url)

    # Pattern 2: <iframe id="pdf" src="...">
    for selector in ("iframe#pdf", "iframe#iframe-pdf", "iframe[src*='.pdf']"):
        node = soup.select_one(selector)
        if node and node.get("src"):
            return _abs_url(str(node["src"]), base_url)

    # Pattern 3: <a> download links that look like PDFs
    for link in soup.select('a[href*=".pdf"], a[href*="downloads/"]'):
        href = link.get("href")
        if href:
            return _abs_url(str(href), base_url)

    # Pattern 4: object / embed without explicit type
    for tag in ("object", "embed"):
        node = soup.select_one(f"{tag}[data*='.pdf'], {tag}[src*='.pdf']")
        if node:
            candidate = node.get("data") or node.get("src")
            if candidate:
                return _abs_url(str(candidate), base_url)

    return None


def _abs_url(href: str, base: str) -> str:
    """Resolve a potentially relative *href* against *base*."""
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        match = re.match(r"(https?://[^/]+)", base)
        if match:
            return f"{match.group(1)}{href}"
    return href


# ---------------------------------------------------------------------------
# Strategy 1 — OpenAlex OA URL lookup
# ---------------------------------------------------------------------------

OPENALEX_TIMEOUT: int = 15


def _try_openalex(doi: str, timeout: int = OPENALEX_TIMEOUT) -> Doi2PdfResult | None:
    """Check OpenAlex for an open-access PDF link for *doi*.

    OpenAlex (https://openalex.org/) indexes OA URLs and is the same source
    the ``doi2pdf`` pip package uses — but without the bug where Sci-Hub is
    skipped when no OA URL exists.
    """
    try:
        resp = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            timeout=timeout,
            headers={"User-Agent": "PaperAssessment-Doi2Pdf/1.0"},
        )
        if resp.status_code >= 400:
            return None

        data = resp.json()
        if not isinstance(data, dict):
            return None

        oa_url = data.get("open_access", {}).get("oa_url")
        if oa_url and isinstance(oa_url, str):
            _logger.debug("OpenAlex OA URL found: %s", oa_url)
            pdf_data = _fetch_pdf(oa_url, timeout=timeout)
            if pdf_data is not None:
                return Doi2PdfResult(
                    success=True,
                    pdf_content=pdf_data,
                    source="openalex",
                    message=f"PDF from OpenAlex OA URL: {oa_url}",
                )
    except (requests.RequestException, ValueError, TypeError) as exc:
        _logger.debug("OpenAlex lookup error: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Strategy 2 — Enhanced CrossRef PDF detection
# ---------------------------------------------------------------------------

CROSSREF_TIMEOUT: int = 15


def _try_crossref_pdf(doi: str, timeout: int = CROSSREF_TIMEOUT) -> Doi2PdfResult | None:
    """Check CrossRef for a direct PDF link (content-type = application/pdf).

    The existing OA resolution only checks the first link; this function
    iterates all links and looks specifically for PDF content types.
    """
    try:
        resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=timeout,
            headers={"User-Agent": "PaperAssessment-Doi2Pdf/1.0"},
        )
        if resp.status_code >= 400:
            return None

        message = resp.json().get("message")
        if not isinstance(message, dict):
            return None

        links = message.get("link")
        if isinstance(links, list):
            for link in links:
                if not isinstance(link, dict):
                    continue
                ct = link.get("content-type", "")
                url = link.get("URL")
                if not isinstance(url, str) or not url:
                    continue
                is_pdf = "pdf" in ct.lower() or url.lower().endswith(".pdf")
                if not is_pdf:
                    continue
                _logger.debug("CrossRef PDF link found: %s (type=%s)", url, ct)
                pdf_data = _fetch_pdf(url, timeout=timeout)
                if pdf_data is not None:
                    return Doi2PdfResult(
                        success=True,
                        pdf_content=pdf_data,
                        source="crossref",
                        message=f"PDF from CrossRef content-type link: {url}",
                    )

    except (requests.RequestException, ValueError, TypeError) as exc:
        _logger.debug("CrossRef PDF lookup error: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Strategy 3 — Multi-mirror Sci-Hub (our custom implementation)
# ---------------------------------------------------------------------------

SCI_HUB_TIMEOUT_PER_MIRROR: int = 30


def _try_sci_hub(
    doi: str,
    mirrors: list[str],
    timeout: int = SCI_HUB_TIMEOUT_PER_MIRROR,
) -> Doi2PdfResult | None:
    """Try to download a PDF for *doi* via Sci-Hub mirrors.

    Returns the first successful result, or ``None`` if all mirrors fail.
    """
    for mirror in mirrors:
        url = f"{mirror.rstrip('/')}/{doi}"
        _logger.debug("Trying Sci-Hub: %s", url)
        try:
            resp = requests.get(url, timeout=timeout, headers=_DEFAULT_HEADERS, allow_redirects=True)
            if resp.status_code >= 400:
                _logger.debug("Sci-Hub %s returned HTTP %s", url, resp.status_code)
                continue

            content_type = resp.headers.get("content-type", "")
            raw = resp.content

            if "application/pdf" in content_type or _is_pdf_content(raw):
                _logger.info("Got direct PDF from %s", url)
                return Doi2PdfResult(
                    success=True,
                    pdf_content=raw,
                    source=mirror,
                    message=f"Direct PDF from Sci-Hub mirror {mirror}",
                )

            pdf_inner = _extract_pdf_url_from_html(raw.decode("utf-8", errors="replace"), url)
            if pdf_inner:
                _logger.debug("Resolved PDF URL from Sci-Hub HTML: %s", pdf_inner)
                pdf_data = _fetch_pdf(pdf_inner, timeout=timeout)
                if pdf_data is not None:
                    return Doi2PdfResult(
                        success=True,
                        pdf_content=pdf_data,
                        source=mirror,
                        message=f"PDF via {mirror} (inner URL)",
                    )

        except requests.RequestException as exc:
            _logger.debug("Sci-Hub %s error: %s", url, exc)
            continue

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def try_doi2pdf_fallback(
    doi: str,
    timeout_seconds: int = 30,
    sci_hub_mirrors: list[str] | None = None,
) -> Doi2PdfResult:
    """Try to download a PDF for *doi* using backup strategies.

    Strategies are tried in order; the first successful result is returned.

    Parameters
    ----------
    doi:
        The DOI to resolve (e.g. ``"10.1038/s41586-026-10420-y"``).
    timeout_seconds:
        Global timeout hint passed to individual strategies.
    sci_hub_mirrors:
        List of Sci-Hub base URLs.  Defaults to :data:`DEFAULT_SCI_HUB_MIRRORS`.

    Returns
    -------
    Doi2PdfResult
    """
    mirrors = sci_hub_mirrors or DEFAULT_SCI_HUB_MIRRORS

    # Strategy 1 — OpenAlex OA URL (legit, no Sci-Hub)
    _logger.info("doi2pdf: trying OpenAlex OA URL for %s", doi)
    result = _try_openalex(doi, timeout=min(timeout_seconds, OPENALEX_TIMEOUT))
    if result is not None and result.success:
        return result

    # Strategy 2 — CrossRef PDF links (fast, legit)
    _logger.info("doi2pdf: trying CrossRef PDF for %s", doi)
    result = _try_crossref_pdf(doi, timeout=min(timeout_seconds, CROSSREF_TIMEOUT))
    if result is not None and result.success:
        return result

    # Strategy 3 — Custom multi-mirror Sci-Hub
    _logger.info("doi2pdf: trying custom Sci-Hub for %s", doi)
    result = _try_sci_hub(doi, mirrors, timeout=min(timeout_seconds, SCI_HUB_TIMEOUT_PER_MIRROR))
    if result is not None and result.success:
        return result

    return Doi2PdfResult(
        success=False,
        pdf_content=None,
        source="",
        message="All doi2pdf strategies exhausted — no PDF found.",
    )


def apply_doi2pdf_fallback(
    outcome: ExtractionOutcome,
    doi: str,
    timeout_seconds: int = 30,
    sci_hub_mirrors: list[str] | None = None,
) -> bool:
    """Run :func:`try_doi2pdf_fallback` and update *outcome* in place.

    Returns ``True`` if a PDF was obtained and attached to the outcome.
    """
    result = try_doi2pdf_fallback(doi, timeout_seconds=timeout_seconds, sci_hub_mirrors=sci_hub_mirrors)

    if result.success and result.pdf_content is not None:
        outcome.pdf = DownloadedAsset(
            filename="article.pdf",
            source_url=f"https://doi.org/{doi}",
            content=result.pdf_content,
        )
        outcome.attempts.append(
            ExtractionAttempt(
                name=f"doi2pdf_{result.source}",
                success=True,
                message=result.message,
            )
        )
        outcome.metadata["doi2pdf_fallback_source"] = result.source
        outcome.metadata["doi2pdf_fallback_message"] = result.message

        if outcome.anti_bot_blocked:
            outcome.anti_bot_blocked = False
            outcome.access_limited = False
            outcome.access_warning = None
            outcome.errors = [e for e in outcome.errors if "anti-bot" not in e.lower()]

        _logger.info("doi2pdf fallback succeeded: %s", result.message)
        return True

    outcome.attempts.append(
        ExtractionAttempt(
            name="doi2pdf_fallback",
            success=False,
            message=result.message,
        )
    )
    outcome.errors.append(f"doi2pdf fallback failed: {result.message}")
    _logger.info("doi2pdf fallback failed for %s", doi)
    return False
