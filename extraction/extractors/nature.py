"""Nature-family extractor implementation."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlparse

from extraction.core.browser import BrowserClient
from extraction.core.models import DownloadedAsset, ExtractionAttempt, ExtractionOutcome, ResolvedTarget
from extraction.extractors.common import (
    build_figure_asset,
    is_challenge_page,
    normalize_figure_url,
    soup_from_html,
    text_from_selectors,
    title_from_soup,
)


def _collect_pdf_url(base_url: str, html: str) -> str | None:
    soup = soup_from_html(html)
    for selector in ['a[href*=".pdf"]', 'a[data-track-action="download pdf"]', 'a[href*="/pdf"]']:
        node = soup.select_one(selector)
        if node and node.get("href"):
            return urljoin(base_url, str(node["href"]))
    # Fallback: try appending .pdf to the article URL (works for many Nature journals).
    pdf_candidate = base_url.rstrip("/") + ".pdf"
    return pdf_candidate


def _supplementary_filename(url: str, index: int) -> str:
    parsed = urlparse(url)
    candidate = Path(parsed.path).name.strip()
    if candidate:
        return f"supp_{index:03d}_{candidate}"
    return f"supp_{index:03d}.bin"


def _collect_supplementary_urls(base_url: str, html: str) -> list[str]:
    soup = soup_from_html(html)
    urls: list[str] = []
    
    # Strategy 1: Look for links inside sections that are likely to contain supplementary/data info
    # Nature often uses specific classes or data-test attributes for these sections
    container_selectors = [
        "div.c-article-supplementary-information",
        "section#supplementary-information",
        "div.c-article-data-availability",
        "section#data-availability",
        "div.c-article-section", # Generic section, we'll check its heading
        "div.c-article-main-column",
        "div.c-article-sidebar",
        "div.c-article-section__content",
        "div.c-article-header",
        "div.c-article-body",
        "figure",
    ]
    
    for selector in container_selectors:
        for container in soup.select(selector):
            # If it's a generic section, check if the heading suggests it's supplementary/data
            heading = container.select_one("h2, h3, h4")
            if not heading:
                # Check preceding sibling for heading if not inside
                prev = container.find_previous_sibling(["h2", "h3", "h4"])
                heading = prev if prev else None
            
            heading_text = heading.get_text(" ", strip=True).lower() if heading else ""
            is_relevant_section = any(key in heading_text for key in ["supplementary", "data availability", "source data", "additional information", "extended data"])
            
            # If it's a known specific container or a relevant section, grab all download-like links
            if is_relevant_section or any(k in selector for k in ["supplementary", "data-availability"]):
                for node in container.select("a[href]"):
                    href = str(node.get("href", ""))
                    if not href:
                        continue
                    href_l = href.lower()
                    # Capture anything that looks like a file download or external reference
                    if any(key in href_l for key in ["/articles/s", "/extref/", "download", "supp", "data", "media", "figshare", "zenodo", ".xlsx", ".xls", ".pdf", ".zip"]):
                        urls.append(urljoin(base_url, href))
            elif selector in {"figure", "div.c-article-body"}:
                # Figure blocks often carry "Source Data" links where labels live outside the anchor text.
                for node in container.select("a[href]"):
                    href = str(node.get("href", ""))
                    if not href:
                        continue
                    text_l = node.get_text(" ", strip=True).lower()
                    href_l = href.lower()
                    if any(key in href_l for key in ["/articles/s", "/extref/", "download", ".xlsx", ".xls", ".pdf", ".zip"]):
                        urls.append(urljoin(base_url, href))
                        continue
                    if any(key in text_l for key in ["source data", "download xlsx", "download pdf", "supplementary"]):
                        urls.append(urljoin(base_url, href))

    # Strategy 2: Broad scan with existing keyword logic as fallback
    for node in soup.select("a[href]"):
        href = str(node.get("href", ""))
        if not href:
            continue
        text = node.get_text(" ", strip=True).lower()
        href_l = href.lower()
        
        # Check parent/ancestor text for context (the "box" mentioned by the user)
        parent_text = ""
        # Check up to 3 levels of parents for context
        curr = node.parent
        for _ in range(3):
            if not curr:
                break
            parent_text += " " + curr.get_text(" ", strip=True).lower()
            curr = curr.parent
        prev_heading = node.find_previous(["h2", "h3", "h4", "strong"])
        heading_text = prev_heading.get_text(" ", strip=True).lower() if prev_heading else ""

        is_material_text = any(
            key in text or key in parent_text or key in heading_text
            for key in [
                "supplementary",
                "supplemental",
                "supp info",
                "additional file",
                "source data",
                "extended data",
                "download xlsx",
                "download pdf",
            ]
        )
        is_material_url = any(key in href_l for key in ["/extref/", "supplementary-information", "suppinfo", "source-data", "media-", "/articles/s", "download", ".xlsx", ".xls", ".pdf", ".zip"])

        if is_material_text or is_material_url:
            urls.append(urljoin(base_url, href))
            
    return list(dict.fromkeys(urls))


def extract(
    target: ResolvedTarget,
    browser: BrowserClient,
    mode: str,
) -> ExtractionOutcome:
    outcome = ExtractionOutcome(source="nature", resolved_url=target.resolved_url, doi=target.doi)
    snapshot = browser.fetch_page(target.resolved_url)
    soup = soup_from_html(snapshot.html)

    if is_challenge_page(soup, snapshot.status_code):
        outcome.anti_bot_blocked = True
        outcome.errors.append("Blocked by anti-bot challenge page.")
        outcome.attempts.append(ExtractionAttempt(name="structured", success=False, message="Anti-bot challenge detected."))
        outcome.attempts.append(ExtractionAttempt(name="pdf", success=False, message="Anti-bot challenge detected."))
        return outcome

    title = title_from_soup(soup)
    abstract_node = soup.select_one("div.c-article-section__content p")
    abstract = abstract_node.get_text(" ", strip=True) if abstract_node else None
    authors = [a.get_text(" ", strip=True) for a in soup.select('a[data-test="author-name"]')]
    outcome.metadata.update(
        {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "source_url": snapshot.url,
            "status_code": snapshot.status_code,
        }
    )

    try:
        supplementary_urls = _collect_supplementary_urls(snapshot.url, snapshot.html)
        for idx, supp_url in enumerate(supplementary_urls, start=1):
            try:
                try:
                    payload = browser.download_binary_via_download_event(supp_url, referer_url=snapshot.url)
                except Exception:
                    try:
                        payload = browser.download_binary(supp_url)
                    except Exception:
                        payload = browser.download_binary_via_page(supp_url)
            except Exception:
                continue
            outcome.supplementary_files.append(
                DownloadedAsset(
                    filename=_supplementary_filename(supp_url, idx),
                    source_url=supp_url,
                    content=payload,
                )
            )
        supp_ok = bool(outcome.supplementary_files) or not supplementary_urls
        message = (
            f"Supplementary extraction completed ({len(outcome.supplementary_files)} files)."
            if supp_ok
            else "Supplementary links found but all downloads failed."
        )
        outcome.attempts.append(ExtractionAttempt(name="supplementary", success=supp_ok, message=message))
    except Exception as exc:  # noqa: BLE001
        outcome.errors.append(f"Supplementary extraction failed: {exc}")
        outcome.attempts.append(ExtractionAttempt(name="supplementary", success=False, message=str(exc)))

    if mode in {"both", "structured"}:
        try:
            text = text_from_selectors(
                soup,
                selectors=[
                    "div.c-article-body p",
                    "article p",
                    "main p",
                ],
            )
            if text and title:
                outcome.article_markdown = f"# {title}\n\n## Extracted Content\n\n{text}"

            image_urls: list[str] = []
            for img in soup.select("figure img[src]"):
                image_urls.append(normalize_figure_url(snapshot.url, str(img["src"])))
            deduped = list(dict.fromkeys(image_urls))[:15]
            for idx, figure_url in enumerate(deduped, start=1):
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
            pdf_url = _collect_pdf_url(snapshot.url, snapshot.html)
            if not pdf_url:
                raise ValueError("Could not find PDF link on page.")
            try:
                pdf_payload = browser.download_binary_via_download_event(pdf_url, referer_url=snapshot.url)
            except Exception:
                try:
                    pdf_payload = browser.download_binary(pdf_url)
                except Exception:
                    pdf_payload = browser.download_binary_via_page(pdf_url)
            outcome.pdf = build_figure_asset(pdf_url, pdf_payload, 0)
            outcome.pdf.filename = "article.pdf"
            outcome.attempts.append(ExtractionAttempt(name="pdf", success=True, message="PDF download completed."))
        except Exception as exc:  # noqa: BLE001
            outcome.errors.append(f"PDF download failed: {exc}")
            outcome.attempts.append(ExtractionAttempt(name="pdf", success=False, message=str(exc)))

    return outcome
