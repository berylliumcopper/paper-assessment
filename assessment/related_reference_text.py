"""Fast text-only extraction for related-work PDFs (not for target/supplementary; use marker there)."""

from __future__ import annotations

import re
import time
from pathlib import Path

from extraction.extractors.common import slug_from_url

# Per shortlisted related paper (independent of target-article size)
RELATED_PDF_EXTRACT_MAX_CHARS: int = 50_000


def extract_related_pdf_plain_text(
    path: Path,
    *,
    max_pages: int = 50,
    max_chars: int = RELATED_PDF_EXTRACT_MAX_CHARS,
) -> str:
    """
    Extract machine text from a PDF as quickly as possible (no layout/OCR/marker).
    Tries PyMuPDF (if installed) first, then pypdf. Caps pages and characters for speed and prompt size.
    """
    t0 = time.perf_counter()
    text: str | None = None
    if max_pages < 1:
        max_pages = 1
    if max_chars < 1:
        return ""

    try:
        import fitz  # type: ignore  # PyMuPDF  # noqa: I001
    except Exception:  # noqa: BLE001
        fitz = None  # type: ignore[assignment]

    if fitz is not None:
        try:
            doc = fitz.open(str(path))  # type: ignore[union-attr]
            n = min(len(doc), max_pages)
            parts: list[str] = []
            for i in range(n):
                page = doc[i]
                parts.append(page.get_text() or "")
            text = "\n\n".join(parts).strip()
        except Exception:  # noqa: BLE001
            text = None

    if not text or not text.strip():
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:  # noqa: BLE001
            return f"[pypdf not available; could not read {path.name}]"

        try:
            reader = PdfReader(str(path))
            n = min(len(reader.pages), max_pages)
            parts = []
            for i in range(n):
                t = reader.pages[i].extract_text() or ""
                if t.strip():
                    parts.append(t.strip())
            text = "\n\n".join(parts).strip()
        except Exception as exc:  # noqa: BLE001
            return f"[PDF read failed: {exc}]"

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[TRUNCATED at {max_chars} chars; total time {time.perf_counter() - t0:.1f}s]"

    return text


def _candidate_urls_for_slug(paper: dict) -> list[str]:
    """Order matches attempt_related_pdf_extraction (doi, pdf_url, url, then arxiv fallback)."""
    out: list[str] = []
    for key in ("doi", "pdf_url", "url"):
        t = paper.get(key)
        if not isinstance(t, str) or not t.strip():
            continue
        raw = t.strip()
        if key == "doi" and not raw.lower().startswith("http"):
            raw = re.sub(r"^doi:\s*", "", raw, flags=re.IGNORECASE)
            out.append(f"https://doi.org/{raw}")
        else:
            out.append(raw)
    # Also check arXiv-based slug as fallback.
    arxiv_id = paper.get("arxiv_id")
    if isinstance(arxiv_id, str) and arxiv_id.strip():
        out.append(f"https://arxiv.org/abs/{arxiv_id.strip()}")
    return out


def resolve_downloaded_related_pdf_path(paper: dict, output_dir: Path) -> Path | None:
    """
    After extraction.main --mode pdf, article.pdf lives at output_dir/<slug>/article.pdf
    with slug = slug_from_url(extraction target).

    Prefers the ``reference_pdf_path`` key set by
    :func:`attempt_related_pdf_extraction` during the download (most reliable).
    Falls back to slug matching for PDFs placed by other means.
    """
    # Direct recorded path (set during download — most reliable).
    stored = paper.get("reference_pdf_path")
    if isinstance(stored, str) and stored.strip():
        p = Path(stored)
        if p.is_file():
            return p

    # Slug-based fallback.
    for u in _candidate_urls_for_slug(paper):
        slug = slug_from_url(u)
        p = output_dir / slug / "article.pdf"
        if p.is_file():
            return p
        if output_dir.is_dir():
            for d in output_dir.iterdir():
                if d.is_dir() and d.name.startswith(slug):
                    candidate = d / "article.pdf"
                    if candidate.is_file():
                        return candidate
    return None


def enrich_papers_with_fast_fulltext(
    papers: list[dict],
    output_dir: Path,
    *,
    max_papers: int,
    max_pages: int,
    max_chars: int,
) -> None:
    """
    Mutate each paper dict: set reference_fulltext_excerpt, reference_fulltext_status,
    reference_pdf_path (if any).

    Extracts text from any related-paper PDF already on disk (downloaded by
    ``attempt_related_pdf_extraction`` in a previous step).  Papers without a local
    PDF are silently skipped — no budget is consumed because the download step has
    already run.
    """
    if max_papers <= 0:
        for p in papers:
            p.setdefault("reference_fulltext_excerpt", "")
            p["reference_fulltext_status"] = "disabled"
        return

    total = 0
    count_pdf_downloaded = 0
    count_extracted = 0
    count_extraction_failed = 0

    for p in papers:
        target = p.get("doi") or p.get("pdf_url") or p.get("url")
        if not isinstance(target, str) or not target.strip():
            p.setdefault("reference_fulltext_excerpt", "")
            p["reference_fulltext_status"] = "no_target_url"
            continue

        total += 1
        pdf = resolve_downloaded_related_pdf_path(p, output_dir)
        if pdf is None or not pdf.is_file():
            # Try direct arXiv PDF download as a fallback (freely accessible).
            arxiv_id = p.get("arxiv_id", "")
            arxiv_downloaded = False
            if arxiv_id:
                arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
                try:
                    import requests as _req
                    r_arxiv = _req.get(arxiv_pdf_url, timeout=30, stream=True)
                    if r_arxiv.status_code == 200:
                        from extraction.extractors.common import slug_from_url
                        arxiv_slug = slug_from_url(f"https://arxiv.org/abs/{arxiv_id}")
                        pdf_dir = output_dir / arxiv_slug
                        pdf_dir.mkdir(parents=True, exist_ok=True)
                        pdf_path = pdf_dir / "article.pdf"
                        with open(pdf_path, "wb") as f:
                            for chunk in r_arxiv.iter_content(chunk_size=65536):
                                f.write(chunk)
                        pdf = pdf_path
                        p["reference_pdf_path"] = str(pdf_path.resolve())
                        arxiv_downloaded = True
                        print(
                            f"[info] related_fulltext: downloaded arXiv PDF {arxiv_pdf_url} -> {pdf_path}",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"[info] related_fulltext: arXiv download failed for {arxiv_id}: {exc}",
                        flush=True,
                    )

            if not arxiv_downloaded:
                p.setdefault("reference_fulltext_excerpt", "")
                p["reference_fulltext_status"] = "no_local_pdf"
                p["reference_pdf_path"] = ""
                continue

        count_pdf_downloaded += 1
        p["reference_pdf_path"] = str(pdf.resolve())
        t0 = time.perf_counter()
        text = extract_related_pdf_plain_text(
            pdf, max_pages=max_pages, max_chars=max_chars
        )
        dt = time.perf_counter() - t0
        p["reference_fulltext_excerpt"] = text
        p["reference_fulltext_status"] = f"ok_chars={len(text)}_s={dt:.1f}"

        # Detect if extraction actually produced useful text
        if text and not text.startswith("[") and len(text.strip()) > 50:
            count_extracted += 1
        elif text.startswith("[PDF read") or text.startswith("[pypdf not"):
            count_extraction_failed += 1
        else:
            count_extraction_failed += 1

    print(
        f"[info] total {total} papers, {count_pdf_downloaded} pdf downloaded, "
        f"{count_extracted} extracted full text, "
        f"{count_extraction_failed} has pdf but extraction failed",
        flush=True,
    )
