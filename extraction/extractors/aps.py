"""APS extractor implementation."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlparse

from extraction.core.browser import BrowserClient
from extraction.core.models import DownloadedAsset, ExtractionAttempt, ExtractionOutcome, ResolvedTarget
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

APS_CHALLENGE_RETRIES = 3


def _doi_from_url(url: str) -> str | None:
    marker = "/10."
    if marker in url:
        return "10." + url.split(marker, maxsplit=1)[-1].split("?", maxsplit=1)[0].strip("/")
    return None


def _pdf_url(base_url: str, html: str, doi: str | None) -> str | None:
    soup = soup_from_html(html)
    for selector in [
        'a[href*="/pdf/"]',
        'a[href*=".pdf"]',
        'a[title*="PDF"]',
        'a[aria-label*="PDF"]',
    ]:
        node = soup.select_one(selector)
        if node and node.get("href"):
            return urljoin(base_url, str(node["href"]))
    normalized = base_url.replace("/abstract/", "/pdf/").replace("/full/", "/pdf/")
    if normalized != base_url and "/pdf/" in normalized:
        return normalized
    if doi:
        if "/prl/" in base_url:
            return f"https://journals.aps.org/prl/pdf/{doi}"
        if "/pra/" in base_url:
            return f"https://journals.aps.org/pra/pdf/{doi}"
        if "/prb/" in base_url:
            return f"https://journals.aps.org/prb/pdf/{doi}"
        if "/prc/" in base_url:
            return f"https://journals.aps.org/prc/pdf/{doi}"
        if "/prd/" in base_url:
            return f"https://journals.aps.org/prd/pdf/{doi}"
        if "/pre/" in base_url:
            return f"https://journals.aps.org/pre/pdf/{doi}"
        if "/prx/" in base_url:
            return f"https://journals.aps.org/prx/pdf/{doi}"
        return f"https://link.aps.org/doi/{doi}"
    return None


def _supplementary_filename(url: str, index: int) -> str:
    parsed = urlparse(url)
    candidate = Path(parsed.path).name.strip()
    if candidate:
        return f"supp_{index:03d}_{candidate}"
    return f"supp_{index:03d}.bin"


def _supplementary_urls(base_url: str, html: str) -> list[str]:
    soup = soup_from_html(html)
    urls: list[str] = []
    for node in soup.select("a[href]"):
        href = str(node.get("href", ""))
        if not href:
            continue
        text = node.get_text(" ", strip=True).lower()
        href_l = href.lower()
        if any(key in text for key in ["supplemental", "supplementary", "supplemental material", "supplementary material"]):
            urls.append(urljoin(base_url, href))
            continue
        if any(key in href_l for key in ["/supplemental/", "suppl", "supplemental", "media/"]):
            urls.append(urljoin(base_url, href))
    return list(dict.fromkeys(urls))


def _fetch_aps_snapshot_with_retries(browser: BrowserClient, url: str):
    snapshot = browser.fetch_page(url)
    soup = soup_from_html(snapshot.html)
    if not is_challenge_page(soup, snapshot.status_code):
        return snapshot, soup, 0

    alternate_url = url.replace("/abstract/", "/full/") if "/abstract/" in url else url
    warmup_urls = ["https://journals.aps.org/", "https://link.aps.org/"]
    blocked_attempts = 1

    for attempt in range(1, APS_CHALLENGE_RETRIES + 1):
        if browser.rate_limiter is not None:
            browser.rate_limiter.backoff(multiplier=1.2 + attempt * 0.8)
        try:
            browser.fetch_page(warmup_urls[(attempt - 1) % len(warmup_urls)])
        except Exception:  # noqa: BLE001
            pass

        candidate_url = alternate_url if attempt % 2 == 1 else url
        snapshot = browser.fetch_page(candidate_url)
        soup = soup_from_html(snapshot.html)
        if not is_challenge_page(soup, snapshot.status_code):
            return snapshot, soup, blocked_attempts
        blocked_attempts += 1

    return snapshot, soup, blocked_attempts


def extract(
    target: ResolvedTarget,
    browser: BrowserClient,
    mode: str,
) -> ExtractionOutcome:
    outcome = ExtractionOutcome(source="aps", resolved_url=target.resolved_url, doi=target.doi)
    snapshot, soup, blocked_attempts = _fetch_aps_snapshot_with_retries(browser, target.resolved_url)

    if is_challenge_page(soup, snapshot.status_code):
        outcome.anti_bot_blocked = True
        outcome.errors.append(
            f"Blocked by anti-bot challenge page after {blocked_attempts} attempt(s)."
        )
        outcome.attempts.append(
            ExtractionAttempt(
                name="challenge_recovery",
                success=False,
                message=f"Still blocked after {blocked_attempts} attempt(s).",
            )
        )
        outcome.attempts.append(ExtractionAttempt(name="structured", success=False, message="Anti-bot challenge detected."))
        outcome.attempts.append(ExtractionAttempt(name="pdf", success=False, message="Anti-bot challenge detected."))
        return outcome
    if blocked_attempts > 0:
        outcome.attempts.append(
            ExtractionAttempt(
                name="challenge_recovery",
                success=True,
                message=f"Challenge cleared after {blocked_attempts} blocked attempt(s).",
            )
        )

    title = title_from_soup(soup)
    abstract_node = soup.select_one("div#abstract p, section.abstract p")
    abstract = abstract_node.get_text(" ", strip=True) if abstract_node else None
    if not abstract:
        abstract_meta = (
            soup.select_one('meta[name="dc.Description"]')
            or soup.select_one('meta[name="description"]')
            or soup.select_one('meta[property="og:description"]')
        )
        abstract = abstract_meta.get("content").strip() if abstract_meta and abstract_meta.get("content") else None
    if not abstract:
        abstract = abstract_from_jsonld(soup)
    authors = [a.get_text(" ", strip=True) for a in soup.select(".authors a, a[href*='/author/']")]
    authors.extend(authors_from_jsonld(soup))
    outcome.metadata.update(
        {
            "title": title,
            "authors": list(dict.fromkeys(authors)),
            "abstract": abstract,
            "source_url": snapshot.url,
            "status_code": snapshot.status_code,
        }
    )

    try:
        supplementary_urls = _supplementary_urls(snapshot.url, snapshot.html)
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
                    "article.article p",
                    "div.content__body p",
                    "div.article p",
                    "div.content p",
                    "article p",
                    "section.article p",
                    "main p",
                ],
            )
            if text:
                heading = title or "APS Article"
                sections = [f"# {heading}"]
                if abstract:
                    sections.extend(["", "## Abstract", abstract])
                sections.extend(["", "## Extracted Content", text])
                outcome.article_markdown = "\n".join(sections)

            image_urls = [
                normalize_figure_url(snapshot.url, str(img["src"]))
                for img in soup.select("figure img[src], img.figure[src]")
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
            pdf_url = _pdf_url(snapshot.url, snapshot.html, doi)
            if not pdf_url:
                raise ValueError("Could not find PDF link on page.")
            try:
                payload = browser.download_binary_via_download_event(pdf_url, referer_url=snapshot.url)
            except Exception:
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
