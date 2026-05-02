"""arXiv extractor implementation."""

from __future__ import annotations

import requests

from extraction.core.models import ExtractionAttempt, ExtractionOutcome, ResolvedTarget
from extraction.core.rate_limit import RateLimiter
from extraction.extractors.common import (
    build_figure_asset,
    is_challenge_page,
    normalize_figure_url,
    slug_from_url,
    soup_from_html,
    text_from_selectors,
    title_from_soup,
)


def _abs_url(url: str) -> str:
    if "/pdf/" in url:
        return url.replace("/pdf/", "/abs/").replace(".pdf", "")
    return url


def _pdf_url(url: str) -> str:
    if "/abs/" in url:
        article_id = url.split("/abs/", maxsplit=1)[-1].strip("/")
        return f"https://arxiv.org/pdf/{article_id}.pdf"
    if url.endswith(".pdf"):
        return url
    return url.rstrip("/") + ".pdf"


def extract(
    target: ResolvedTarget,
    rate_limiter: RateLimiter,
    mode: str,
    timeout_seconds: int,
) -> ExtractionOutcome:
    resolved = _abs_url(target.resolved_url)
    outcome = ExtractionOutcome(source="arxiv", resolved_url=resolved, doi=target.doi)

    if mode in {"both", "structured"}:
        try:
            rate_limiter.wait_for_url(resolved)
            response = requests.get(resolved, timeout=timeout_seconds)
            response.raise_for_status()
            soup = soup_from_html(response.text)

            if is_challenge_page(soup, response.status_code):
                outcome.anti_bot_blocked = True
                outcome.errors.append("Blocked by anti-bot challenge page.")
                outcome.attempts.append(ExtractionAttempt(name="structured", success=False, message="Anti-bot challenge detected."))
                return outcome

            title = title_from_soup(soup) or slug_from_url(resolved)
            abstract_node = soup.select_one("blockquote.abstract")
            abstract = abstract_node.get_text(" ", strip=True) if abstract_node else None
            body_text = text_from_selectors(
                soup,
                selectors=["blockquote.abstract", "div#content", "main"],
            )
            authors = [a.get_text(strip=True) for a in soup.select("div.authors a")]
            outcome.metadata.update(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "source_url": resolved,
                }
            )
            if body_text:
                md_parts = [f"# {title}"]
                if abstract:
                    md_parts.extend(["", "## Abstract", abstract])
                if authors:
                    md_parts.extend(["", "## Authors", ", ".join(authors)])
                md_parts.extend(["", "## Extracted Content", body_text])
                outcome.article_markdown = "\n".join(md_parts).strip()

            images = [normalize_figure_url(resolved, img["src"]) for img in soup.select("img[src]")[:10]]
            for idx, image_url in enumerate(images, start=1):
                rate_limiter.wait_for_url(image_url)
                image_response = requests.get(image_url, timeout=timeout_seconds)
                if image_response.status_code < 400:
                    outcome.figures.append(build_figure_asset(image_url, image_response.content, idx))

            structured_ok = bool(outcome.article_markdown)
            message = "Structured extraction completed." if structured_ok else "No structured content matched configured selectors."
            outcome.attempts.append(ExtractionAttempt(name="structured", success=structured_ok, message=message))
        except Exception as exc:  # noqa: BLE001
            outcome.errors.append(f"Structured extraction failed: {exc}")
            outcome.attempts.append(ExtractionAttempt(name="structured", success=False, message=str(exc)))

    if mode in {"both", "pdf"}:
        try:
            pdf_url = _pdf_url(resolved)
            rate_limiter.wait_for_url(pdf_url)
            pdf_response = requests.get(pdf_url, timeout=timeout_seconds)
            pdf_response.raise_for_status()
            outcome.pdf = build_figure_asset(pdf_url, pdf_response.content, 0)
            outcome.pdf.filename = "article.pdf"
            outcome.attempts.append(ExtractionAttempt(name="pdf", success=True, message="PDF download completed."))
        except Exception as exc:  # noqa: BLE001
            outcome.errors.append(f"PDF download failed: {exc}")
            outcome.attempts.append(ExtractionAttempt(name="pdf", success=False, message=str(exc)))

    return outcome
