"""Science extractor implementation."""

from __future__ import annotations

from urllib.parse import urljoin

from extraction.core.browser import BrowserClient
from extraction.core.models import ExtractionAttempt, ExtractionOutcome, ResolvedTarget
from extraction.extractors.common import (
    abstract_from_jsonld,
    authors_from_jsonld,
    build_figure_asset,
    is_challenge_page,
    normalize_figure_url,
    soup_from_html,
    text_from_selectors,
    title_from_soup,
)


def _doi_from_url(url: str) -> str | None:
    marker = "/doi/"
    if marker not in url:
        return None
    return url.split(marker, maxsplit=1)[-1].split("?", maxsplit=1)[0].strip("/")


def _find_pdf_url(base_url: str, html: str, doi: str | None) -> str | None:
    soup = soup_from_html(html)
    selectors = [
        'a[href*="/doi/pdf/"]',
        'a[href*="/doi/epdf/"]',
        'a[href*=".pdf"]',
        'a[title*="PDF"]',
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node and node.get("href"):
            return urljoin(base_url, str(node["href"]))
    for node in soup.select("a[href]"):
        label = node.get_text(" ", strip=True).lower()
        href = str(node.get("href", ""))
        if "pdf" in label or "pdf" in href.lower():
            return urljoin(base_url, href)
    if doi:
        return f"https://www.science.org/doi/pdf/{doi}"
    return None


def extract(
    target: ResolvedTarget,
    browser: BrowserClient,
    mode: str,
) -> ExtractionOutcome:
    outcome = ExtractionOutcome(source="science", resolved_url=target.resolved_url, doi=target.doi)
    snapshot = browser.fetch_page(target.resolved_url)
    soup = soup_from_html(snapshot.html)

    if is_challenge_page(soup, snapshot.status_code):
        outcome.anti_bot_blocked = True
        outcome.errors.append("Blocked by anti-bot challenge page.")
        outcome.attempts.append(ExtractionAttempt(name="structured", success=False, message="Anti-bot challenge detected."))
        outcome.attempts.append(ExtractionAttempt(name="pdf", success=False, message="Anti-bot challenge detected."))
        return outcome

    title = title_from_soup(soup)
    authors = [a.get_text(" ", strip=True) for a in soup.select(".article-header__authors a, .name, [property='author']")]
    authors.extend(authors_from_jsonld(soup))
    abstract_meta = (
        soup.select_one('meta[name="dc.Description"]')
        or soup.select_one('meta[name="description"]')
        or soup.select_one('meta[property="og:description"]')
    )
    abstract = abstract_meta.get("content").strip() if abstract_meta and abstract_meta.get("content") else None
    if not abstract:
        abstract = abstract_from_jsonld(soup)
    outcome.metadata.update(
        {
            "title": title,
            "authors": list(dict.fromkeys(authors)),
            "abstract": abstract,
            "source_url": snapshot.url,
            "status_code": snapshot.status_code,
        }
    )

    if mode in {"both", "structured"}:
        try:
            text = text_from_selectors(
                soup,
                selectors=[
                    "article.news-article-content p",
                    "div.article-text p",
                    "section.article-section p",
                    "div.article__body p",
                    "section.article__section p",
                    "article p",
                    "main p",
                ],
            )
            if text:
                heading = title or "Science Article"
                sections = [f"# {heading}"]
                if abstract:
                    sections.extend(["", "## Abstract", abstract])
                sections.extend(["", "## Extracted Content", text])
                outcome.article_markdown = "\n".join(sections)

            image_urls = [
                normalize_figure_url(snapshot.url, str(img["src"]))
                for img in soup.select("figure img[src], img.figure__image[src]")
            ]
            for idx, figure_url in enumerate(list(dict.fromkeys(image_urls))[:15], start=1):
                try:
                    payload = browser.download_binary(figure_url)
                    outcome.figures.append(build_figure_asset(figure_url, payload, idx))
                except Exception:  # noqa: BLE001
                    continue

            structured_ok = bool(outcome.article_markdown)
            message = "Structured extraction completed." if structured_ok else "No structured content matched configured selectors."
            outcome.attempts.append(ExtractionAttempt(name="structured", success=structured_ok, message=message))
        except Exception as exc:  # noqa: BLE001
            outcome.errors.append(f"Structured extraction failed: {exc}")
            outcome.attempts.append(ExtractionAttempt(name="structured", success=False, message=str(exc)))

    if mode in {"both", "pdf"}:
        try:
            doi = target.doi or _doi_from_url(snapshot.url)
            pdf_url = _find_pdf_url(snapshot.url, snapshot.html, doi)
            if not pdf_url:
                raise ValueError("Could not find PDF link on page.")
            try:
                payload = browser.download_binary(pdf_url)
            except Exception:
                payload = browser.download_binary_via_page(pdf_url)
            outcome.pdf = build_figure_asset(pdf_url, payload, 0)
            outcome.pdf.filename = "article.pdf"
            outcome.attempts.append(ExtractionAttempt(name="pdf", success=True, message="PDF download completed."))
        except Exception as exc:  # noqa: BLE001
            outcome.errors.append(f"PDF download failed: {exc}")
            outcome.attempts.append(ExtractionAttempt(name="pdf", success=False, message=str(exc)))

    return outcome
