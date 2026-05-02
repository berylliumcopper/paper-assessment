"""Related-paper search utilities for assessment workflow."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote_plus

import requests

from extraction.core.doi2pdf import try_doi2pdf_fallback
from extraction.extractors.common import slug_from_url


def _http_get(
    url: str,
    *,
    timeout_seconds: int,
) -> requests.Response | None:
    """
    GET with connect/read split and one retry. Returns None on failure instead of raising,
    so Crossref / arXiv / Semantic Scholar cannot crash the whole related-work stage.
    """
    read = max(20, min(int(timeout_seconds), 120))
    connect = min(10, read)
    for _ in range(2):
        try:
            return requests.get(url, timeout=(connect, read))
        except requests.RequestException:  # noqa: BLE001
            continue
    return None


@dataclass(slots=True)
class RelatedPaper:
    source: str
    title: str
    url: str
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    authors: list[str] | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None


def _normalize_title(raw: str | None) -> str:
    """Strip HTML/LaTeX formatting and normalize whitespace for dedup comparison."""
    if not raw:
        return ""
    import re
    t = raw
    # Remove HTML tags
    t = re.sub(r"<[^>]+>", "", t)
    # Remove LaTeX display math $$...$$ (keep inner content)
    t = re.sub(r"\$\$(.*?)\$\$", r"\1", t)
    # Remove LaTeX inline math $...$ (keep inner content)
    t = re.sub(r"\$(.*?)\$", r"\1", t)
    # Remove backslash escapes
    t = t.replace("\\", " ")
    # Remove underscores and carets (LaTeX sub/superscript markers)
    t = t.replace("_", "").replace("^", "")
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _dedupe(records: list[RelatedPaper], exclude_doi: str | None = None) -> list[RelatedPaper]:
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    unique: list[RelatedPaper] = []
    
    # Normalize exclude_doi if provided
    normalized_exclude_doi = exclude_doi.strip().lower() if exclude_doi else None
    
    for item in records:
        doi_key = (item.doi or "").strip().lower()
        norm = _normalize_title(item.title)

        # DOI-based dedup
        if doi_key:
            if doi_key in seen_doi:
                continue
            seen_doi.add(doi_key)
            # Also register the normalized title so a no-DOI duplicate is caught.
            if norm:
                seen_title.add(norm)
        else:
            # No DOI — deduplicate by normalized title.
            if norm and norm in seen_title:
                continue
            if norm:
                seen_title.add(norm)
            
        # Check if this is the original paper by DOI
        if normalized_exclude_doi and item.doi and item.doi.strip().lower() == normalized_exclude_doi:
            continue
            
        unique.append(item)
    return unique


def search_related_papers(*, query: str, timeout_seconds: int = 30, per_source_limit: int = 5, use_web_search: bool = False, exclude_doi: str | None = None) -> dict:
    """Search Crossref, arXiv, and Semantic Scholar for related work."""
    crossref = _search_crossref(query=query, timeout_seconds=timeout_seconds, limit=per_source_limit)
    arxiv = _search_arxiv(query=query, timeout_seconds=timeout_seconds, limit=per_source_limit)
    semsch = _search_semantic_scholar(query=query, timeout_seconds=timeout_seconds, limit=per_source_limit)
    
    web_results: list[RelatedPaper] = []
    if use_web_search:
        web_results = _search_web(query=query, limit=per_source_limit)

    merged = _dedupe(crossref + arxiv + semsch + web_results, exclude_doi=exclude_doi)
    return {
        "query": query,
        "counts": {
            "crossref": len(crossref),
            "arxiv": len(arxiv),
            "semantic_scholar": len(semsch),
            "web_search": len(web_results),
            "merged_unique": len(merged),
        },
        "papers": [asdict(item) for item in merged],
    }


def _search_web(*, query: str, limit: int) -> list[RelatedPaper]:
    """Use an LLM-based web search to find related papers."""
    # This is a placeholder for a real web search implementation.
    # In this environment, we can't directly call the WebSearch tool from code,
    # but we can simulate the intent or provide a hook for the CLI to pass results.
    # For now, we'll return an empty list and expect the CLI to handle the tool call if needed,
    # or we can use a mock implementation if we want to show how it would look.
    return []


def search_related_papers_with_web(
    *, 
    query: str, 
    timeout_seconds: int = 30, 
    per_source_limit: int = 5, 
    web_results: list[dict] | None = None,
    exclude_doi: str | None = None
) -> dict:
    """Search academic sources and optionally merge provided web search results."""
    crossref = _search_crossref(query=query, timeout_seconds=timeout_seconds, limit=per_source_limit)
    arxiv = _search_arxiv(query=query, timeout_seconds=timeout_seconds, limit=per_source_limit)
    semsch = _search_semantic_scholar(query=query, timeout_seconds=timeout_seconds, limit=per_source_limit)
    
    web_papers: list[RelatedPaper] = []
    if web_results:
        for res in web_results:
            web_papers.append(
                RelatedPaper(
                    source="web_search",
                    title=res.get("title", "Unknown Title"),
                    url=res.get("url", ""),
                    year=res.get("year"),
                    doi=res.get("doi"),
                    abstract=res.get("abstract"),
                    authors=res.get("authors"),
                )
            )

    merged = _dedupe(crossref + arxiv + semsch + web_papers, exclude_doi=exclude_doi)
    return {
        "query": query,
        "counts": {
            "crossref": len(crossref),
            "arxiv": len(arxiv),
            "semantic_scholar": len(semsch),
            "web_search": len(web_papers),
            "merged_unique": len(merged),
        },
        "papers": [asdict(item) for item in merged],
    }


def _search_crossref(*, query: str, timeout_seconds: int, limit: int) -> list[RelatedPaper]:
    url = f"https://api.crossref.org/works?query.bibliographic={quote_plus(query)}&rows={limit}"
    response = _http_get(url, timeout_seconds=timeout_seconds)
    if response is None or response.status_code >= 400:
        return []
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return []
    items = payload.get("message", {}).get("items", [])
    out: list[RelatedPaper] = []
    for item in items:
        title_list = item.get("title") or []
        title = title_list[0].strip() if title_list else ""
        if not title:
            continue
        doi = item.get("DOI")
        doi_url = f"https://doi.org/{doi}" if isinstance(doi, str) and doi else ""
        year = None
        issued = item.get("issued", {}).get("date-parts", [])
        if issued and isinstance(issued, list) and issued[0]:
            year = issued[0][0]
        authors = []
        for author in item.get("author", []) or []:
            given = author.get("given", "")
            family = author.get("family", "")
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)
        out.append(
            RelatedPaper(
                source="crossref",
                title=title,
                url=doi_url or item.get("URL", ""),
                year=year if isinstance(year, int) else None,
                doi=doi if isinstance(doi, str) else None,
                abstract=item.get("abstract"),
                authors=authors or None,
            )
        )
    return out


def _search_arxiv(*, query: str, timeout_seconds: int, limit: int) -> list[RelatedPaper]:
    # arXiv public API requests are spaced by at least 5 seconds.  Insert a small
    # delay before each call to avoid HTTP 429.
    time.sleep(5.1)
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{quote_plus(query)}&start=0&max_results={limit}"
    )
    response = _http_get(url, timeout_seconds=timeout_seconds)
    if response is None:
        print(
            "[warn] arXiv request failed (timeout or network error); no arXiv search results for this query.",
            file=sys.stderr,
        )
        return []
    if response.status_code >= 400:
        print(
            f"[warn] arXiv returned HTTP {response.status_code}; no arXiv search results.",
            file=sys.stderr,
        )
        return []
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        print(
            "[warn] arXiv response was not valid Atom XML; no arXiv search results.",
            file=sys.stderr,
        )
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: list[RelatedPaper] = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        if not title:
            continue
        link = ""
        pdf_url = None
        for node in entry.findall("a:link", ns):
            href = node.attrib.get("href", "")
            rel = node.attrib.get("rel", "")
            node_type = node.attrib.get("type", "")
            if rel == "alternate" and href:
                link = href
            if node_type == "application/pdf" and href:
                pdf_url = href
        published = entry.findtext("a:published", default="", namespaces=ns)
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        authors = []
        for author_node in entry.findall("a:author", ns):
            name = (author_node.findtext("a:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)
        arxiv_id = ""
        if link:
            arxiv_id = link.rsplit("/", maxsplit=1)[-1]
        out.append(
            RelatedPaper(
                source="arxiv",
                title=title,
                url=link,
                year=year,
                abstract=summary or None,
                authors=authors or None,
                arxiv_id=arxiv_id or None,
                pdf_url=pdf_url,
            )
        )
    return out


def _search_semantic_scholar(*, query: str, timeout_seconds: int, limit: int) -> list[RelatedPaper]:
    fields = "title,year,url,abstract,authors,externalIds"
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={quote_plus(query)}&limit={limit}&fields={quote_plus(fields)}"
    )
    response = _http_get(url, timeout_seconds=timeout_seconds)
    if response is None or response.status_code >= 400:
        return []
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return []
    out: list[RelatedPaper] = []
    for item in payload.get("data", []):
        title = item.get("title", "")
        if not isinstance(title, str) or not title.strip():
            continue
        external_ids = item.get("externalIds", {}) or {}
        doi = external_ids.get("DOI")
        authors = [author.get("name", "").strip() for author in item.get("authors", []) if author.get("name")]
        out.append(
            RelatedPaper(
                source="semantic_scholar",
                title=title.strip(),
                url=item.get("url", ""),
                year=item.get("year") if isinstance(item.get("year"), int) else None,
                doi=doi if isinstance(doi, str) else None,
                abstract=item.get("abstract"),
                authors=authors or None,
            )
        )
    return out


def search_citations_by_doi(
    *, doi: str, timeout_seconds: int = 30, max_fetch: int = 400, arxiv_id: str = ""
) -> list[RelatedPaper]:
    """
    Fetch papers that cite the given paper using Semantic Scholar's citation graph API.

    Accepts either a DOI or an arXiv ID (or both). Uses whichever is available; DOI is preferred.
    Fetches up to ``max_fetch`` citing papers (default 200) with minimal fields (title, year,
    authors, DOI) — no abstracts, to keep the call fast. The caller should then run an AI
    pre-filter to reduce this list to a relevant shortlist.
    """
    if not doi and not arxiv_id:
        return []
    fields = "title,year,url,authors,externalIds"
    out: list[RelatedPaper] = []

    # Build the Semantic Scholar paper identifier.
    if doi and doi.strip():
        clean = doi.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")
        paper_id = f"DOI:{quote_plus(clean)}"
    elif arxiv_id and arxiv_id.strip():
        clean = arxiv_id.strip()
        paper_id = f"ArXiv:{quote_plus(clean)}"
    else:
        return []

    offset = 0
    batch_size = 100
    while offset < max_fetch:
        take = min(batch_size, max_fetch - offset)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
            f"?limit={take}&offset={offset}&fields={quote_plus(fields)}"
        )
        response = _http_get(url, timeout_seconds=timeout_seconds)
        if response is None or response.status_code >= 400:
            break
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            break
        items = payload.get("data", [])
        if not items:
            break
        for item in items:
            citing = item.get("citingPaper", {})
            if not citing:
                continue
            title = citing.get("title", "")
            if not isinstance(title, str) or not title.strip():
                continue
            external_ids = citing.get("externalIds", {}) or {}
            citing_doi = external_ids.get("DOI")
            authors = [
                author.get("name", "").strip()
                for author in citing.get("authors", [])
                if author.get("name")
            ]
            out.append(
                RelatedPaper(
                    source="semantic_scholar_citation",
                    title=title.strip(),
                    url=citing.get("url", ""),
                    year=citing.get("year") if isinstance(citing.get("year"), int) else None,
                    doi=citing_doi if isinstance(citing_doi, str) else None,
                    abstract=None,  # intentionally omitted in bulk fetch
                    authors=authors or None,
                )
            )
        offset += len(items)
    return out


def _resolve_arxiv_from_doi(doi: str, title: str | None = None) -> str | None:
    """Find an arXiv version of a DOI-based paper.

    Strategy:
        1. Direct ``10.48550/arXiv.XXXX`` DOI — construct URL directly.
        2. OpenAlex — check explicit ``ids.arxiv`` field; also extracts
           author names for the next step.
        3. CrossRef — check links, relations, alternative-ids.
        4. arXiv API author + title search — uses author last names from
           OpenAlex together with key title words.  This is much more
           reliable than title-only search because author names are stable
           across publisher and arXiv versions.

    Returns an arXiv abstract URL (e.g. ``https://arxiv.org/abs/xxxx.xxxxx``)
    or ``None``.
    """
    # --- 1. Direct arXiv DOI ---
    m_direct = re.match(r"10\.48550/arXiv\.(\d{4}\.\d{4,5}(v\d+)?)", doi)
    if m_direct:
        return f"https://arxiv.org/abs/{m_direct.group(1)}"

    # --- 2. OpenAlex ---
    openalex_title = ""
    openalex_authors: list[str] = []
    try:
        resp = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            timeout=10,
            headers={"User-Agent": "PaperAssessment/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            # Check explicit arxiv field
            ids = data.get("ids") or {}
            arxiv = ids.get("arxiv")
            if isinstance(arxiv, str) and arxiv.strip():
                return arxiv.strip()
            # Save metadata for arXiv search
            openalex_title = (data.get("title") or "").strip()
            for authship in data.get("authorships") or []:
                author = (authship.get("author") or {})
                name = author.get("display_name") or ""
                if name.strip():
                    openalex_authors.append(name.strip())
    except Exception:  # noqa: BLE001
        pass

    # --- 3. CrossRef ---
    try:
        xr = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=10,
            headers={"User-Agent": "PaperAssessment/1.0"},
        )
        if xr.status_code == 200:
            xmsg = xr.json().get("message", {})
            if isinstance(xmsg, dict):
                arxiv_id: str | None = None
                for link_rec in xmsg.get("link", []) or []:
                    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", str(link_rec.get("url", "")), re.I)
                    if m:
                        arxiv_id = m.group(1)
                        break
                if not arxiv_id:
                    for rel_rec in xmsg.get("relation", {}).get("cites", []) or []:
                        rid = rel_rec.get("id", "")
                        if rid.startswith("arxiv:"):
                            arxiv_id = rid.replace("arxiv:", "")
                            break
                if not arxiv_id:
                    for aid in xmsg.get("alternative-id", []) or []:
                        if re.match(r"\d{4}\.\d{4,5}(v\d+)?$", str(aid)):
                            arxiv_id = str(aid)
                            break
                if arxiv_id:
                    return f"https://arxiv.org/abs/{arxiv_id}"
    except Exception:  # noqa: BLE001
        pass

    # --- 4. arXiv API trial-and-error search ---
    #   Two-attempt strategy:
    #     1) 5 authors + 2 title words
    #     2) 3 authors + 1 title word (relaxed)
    #   Each result is scored (70% author overlap + 30% title overlap)
    #   and the best is selected.  Results are printed to aid tuning.
    if openalex_authors:
        try:
            _ns = {"atom": "http://www.w3.org/2005/Atom"}
            _stop_words = {"the", "and", "for", "of", "in", "to", "a", "an",
                           "is", "on", "by", "with", "from", "at", "its",
                           "via", "their", "that", "are", "was", "been"}

            search_title = title or openalex_title

            # Build author last-name terms.
            author_terms: list[str] = []
            for name in openalex_authors:
                parts = name.split()
                if parts:
                    author_terms.append(f"au:{parts[-1]}")

            # Build title terms (up to 4 words for small-author queries).
            title_terms: list[str] = []
            if search_title:
                for word in search_title.split():
                    w = word.strip(",.():;\"'-").lower()
                    if len(w) > 3 and w not in _stop_words:
                        title_terms.append(f"ti:{w}")
                        if len(title_terms) >= 4:
                            break

            def _score_entry(
                entry, target_authors: list[str], target_title: str,
            ) -> tuple[float, str | None, str, int, int, int]:
                """Score an arXiv entry against the target paper.

                Scoring weights:
                  * Author set overlap (same last names anywhere)   — 35 %
                  * Author position match (first 3 + last author)   — 35 %
                  * Title word overlap                              — 30 %

                Returns (score, arxiv_url, entry_title, common_authors,
                         common_title_words, position_matches).
                """
                e_id = entry.find("atom:id", _ns)
                e_url = e_id.text.strip() if e_id is not None and e_id.text else None

                e_title_el = entry.find("atom:title", _ns)
                e_title = e_title_el.text.strip() if e_title_el is not None and e_title_el.text else ""

                e_authors: list[str] = []
                for ae in entry.findall("atom:author", _ns):
                    ne = ae.find("atom:name", _ns)
                    if ne is not None and ne.text:
                        e_authors.append(ne.text.strip())

                # 1) Author set overlap (last names anywhere).
                target_last_set = {n.split()[-1].lower() for n in target_authors if n.split()}
                entry_last_set = {n.split()[-1].lower() for n in e_authors if n.split()}
                common_authors = len(target_last_set & entry_last_set)
                author_set_score = common_authors / max(len(target_last_set), 1) if target_last_set else 0.0

                # 2) Author position match (first 3 + last 1).
                target_last_names = [n.split()[-1].lower() for n in target_authors if n.split()]
                entry_last_names = [n.split()[-1].lower() for n in e_authors if n.split()]
                pos_checks = []
                for i in range(min(3, len(target_last_names), len(entry_last_names))):
                    pos_checks.append(target_last_names[i] == entry_last_names[i])
                if len(target_last_names) > 0 and len(entry_last_names) > 0:
                    pos_checks.append(target_last_names[-1] == entry_last_names[-1])
                position_matches = sum(1 for c in pos_checks if c)
                pos_score = position_matches / max(len(pos_checks), 1)

                # 3) Title word overlap.
                tt = target_title.lower()
                et = e_title.lower()
                target_tw = {w.strip(",.():;\"'-").lower() for w in tt.split()
                             if len(w) > 3 and w.lower() not in _stop_words}
                entry_tw = {w.strip(",.():;\"'-").lower() for w in et.split()
                            if len(w) > 3 and w.lower() not in _stop_words}
                common_words = len(target_tw & entry_tw)
                title_score = common_words / max(len(target_tw), 1) if target_tw else 0.0

                # Combine: 0.35 set + 0.35 position + 0.30 title
                # For papers with <=2 authors, author-matching is less reliable
                # so tilt heavily toward title overlap.
                n_target = len(target_authors)
                if n_target <= 2:
                    aw, pw, tw = 0.0, 0.20, 0.80
                else:
                    aw, pw, tw = 0.35, 0.35, 0.30
                combined = aw * author_set_score + pw * pos_score + tw * title_score
                return (combined, e_url, e_title, common_authors, common_words, position_matches)

            # Build query variants based on author count.
            #   Many authors (>2): 1 try of (5 authors + 2 title words),
            #                      3 tries of (3 authors + different title words)
            #   Few authors (<=2): use ALL authors + remaining title keywords
            #                      (total query components = 4).
            query_variants: list[tuple[str, str]] = []
            n_authors = len(author_terms)
            if n_authors <= 2:
                # Small author count: use all authors + fill with title words.
                n_title = max(0, 4 - n_authors)
                for offset in range(3):
                    tw = title_terms[offset:offset + n_title] if title_terms else []
                    if author_terms and tw:
                        q = " AND ".join(author_terms + tw)
                        query_variants.append((f"{n_authors} authors + {n_title} title words", q))
                    elif author_terms:
                        q = " AND ".join(author_terms)
                        query_variants.append((f"{n_authors} authors only", q))
            else:
                # Standard: 5 authors + 2 title words, then 3 authors + varied title words.
                if len(author_terms) >= 5 and len(title_terms) >= 2:
                    q = " AND ".join(author_terms[:5] + title_terms)
                    query_variants.append(("5 authors + 2 title words", q))
                if len(author_terms) >= 3 and title_terms:
                    q = " AND ".join(author_terms[:3] + title_terms[:1])
                    query_variants.append(("3 authors + 1 title word", q))
                if len(author_terms) >= 3 and len(title_terms) >= 2:
                    q = " AND ".join(author_terms[:3] + title_terms[1:2])
                    query_variants.append(("3 authors + next title word", q))
                if len(author_terms) >= 3 and len(title_terms) >= 3:
                    q = " AND ".join(author_terms[:3] + title_terms[2:3])
                    query_variants.append(("3 authors + third title word", q))

            for label, query in query_variants:
                resp = requests.get(
                    f"http://export.arxiv.org/api/query?search_query={requests.utils.quote(query)}&max_results=10",
                    timeout=10,
                    headers={"User-Agent": "PaperAssessment/1.0"},
                )
                if resp.status_code != 200:
                    continue

                root = ET.fromstring(resp.content)
                entries = list(root.findall("atom:entry", _ns))

                if not entries:
                    continue

                scored: list[tuple[float, str | None, str, int, int, int]] = [
                    _score_entry(e, openalex_authors, search_title) for e in entries
                ]
                scored.sort(key=lambda x: x[0], reverse=True)

                best_score, best_url, best_title, best_auth, best_title_n, best_pos = scored[0]
                if best_score >= 0.6 and best_url:
                    return best_url
        except Exception:  # noqa: BLE001
            pass

    return None


def attempt_related_pdf_extraction(
    *,
    papers: list[dict],
    output_dir: Path,
    max_downloads: int,
    headless: bool = False,
) -> list[dict]:
    """Try downloading PDFs for a subset of related records via existing extraction CLI."""
    attempts: list[dict] = []
    if max_downloads <= 0:
        return attempts
    paper_count = 0
    count_extraction_main = 0
    count_doi2pdf = 0
    count_arxiv = 0
    count_failed = 0
    for paper in papers:
        if paper_count >= max_downloads:
            break
        # Collect candidate targets: DOI / URL first, then arXiv ID as fallback.
        candidates: list[str] = []
        for key in ("doi", "pdf_url", "url"):
            v = paper.get(key)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
        arxiv_id = paper.get("arxiv_id")
        if isinstance(arxiv_id, str) and arxiv_id.strip():
            candidates.append(f"https://arxiv.org/abs/{arxiv_id.strip()}")

        if not candidates:
            continue

        paper_count += 1
        success = False
        source_label = ""
        paper_timeout = 600  # 10 minutes per paper
        paper_start = time.perf_counter()
        for target in candidates:
            if time.perf_counter() - paper_start > paper_timeout:
                attempts.append({
                    "target": target,
                    "return_code": -1,
                    "stdout_tail": "",
                    "stderr_tail": "TIMEOUT: per-paper budget of 600s exceeded.",
                })
                print(
                    f"  [timeout] skipping candidate {target} — paper budget exceeded",
                    flush=True,
                )
                break
            command = [
                sys.executable,
                "-m",
                "extraction.main",
                "--input",
                target,
                "--mode",
                "pdf",
                "--output-dir",
                str(output_dir),
            ]
            if headless:
                command.append("--headless")
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, check=False, timeout=600,
                )
            except subprocess.TimeoutExpired:
                attempts.append({
                    "target": target,
                    "return_code": -1,
                    "stdout_tail": "",
                    "stderr_tail": "TIMEOUT: subprocess exceeded 600s.",
                })
                print(
                    f"  [timeout] candidate {target} timed out after 600s",
                    flush=True,
                )
                continue
            attempts.append(
                {
                    "target": target,
                    "return_code": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
            if completed.returncode == 0:
                # Verify the PDF actually exists — extraction.main can return 0
                # even when the PDF download was blocked.
                slug = slug_from_url(target)
                pdf_path = output_dir / slug / "article.pdf"
                if pdf_path.is_file():
                    success = True
                    src = "extraction.main"
                    if "/arxiv.org/" in target.lower() or "arxiv" in completed.stdout.lower():
                        src = "arxiv"
                    source_label = src
                    paper["reference_pdf_path"] = str(pdf_path.resolve())
                    break

            # If the subprocess (browser-based extraction) failed and the
            # target is a DOI, try the doi2pdf fallback directly.
            target_stripped = target.strip()
            if (target_stripped.startswith("10.") or "doi.org/" in target_stripped):
                doi = target_stripped
                if "doi.org/" in doi:
                    doi = doi.split("doi.org/", maxsplit=1)[-1].split("?", maxsplit=1)[0]
                doi = doi.rstrip("/")
                result = try_doi2pdf_fallback(doi, timeout_seconds=30)
                if result.success and result.pdf_content:
                    pdf_dir = output_dir / slug_from_url(f"https://doi.org/{doi}")
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    pdf_path = pdf_dir / "article.pdf"
                    pdf_path.write_bytes(result.pdf_content)
                    paper["reference_pdf_path"] = str(pdf_path.resolve())
                    success = True
                    source_label = f"doi2pdf:{result.source}"
                    break

        # If all candidates failed and the paper has a DOI, try to resolve
        # an arXiv version via CrossRef (the pre-populated arxiv_id might be
        # missing from the search metadata).
        if not success:
            doi: str | None = None
            for c in candidates:
                cs = c.strip()
                if cs.startswith("10."):
                    doi = cs
                    break
                m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s?]+)", cs)
                if m:
                    doi = m.group(1).rstrip("/")
                    break
            if doi:
                arxiv_url = _resolve_arxiv_from_doi(doi, title=paper.get("title"))
                if arxiv_url:
                    # Download the arXiv PDF directly (freely accessible).
                    arxiv_pdf_url = arxiv_url.replace("/abs/", "/pdf/").rstrip("/") + ".pdf"
                    try:
                        r_arxiv = requests.get(arxiv_pdf_url, timeout=30, stream=True)
                        if r_arxiv.status_code == 200:
                            arxiv_slug = slug_from_url(arxiv_url)
                            pdf_dir = output_dir / arxiv_slug
                            pdf_dir.mkdir(parents=True, exist_ok=True)
                            pdf_path = pdf_dir / "article.pdf"
                            with open(pdf_path, "wb") as f:
                                for chunk in r_arxiv.iter_content(chunk_size=65536):
                                    f.write(chunk)
                            paper["reference_pdf_path"] = str(pdf_path.resolve())
                            success = True
                            source_label = "arxiv (CrossRef fallback)"
                    except Exception:  # noqa: BLE001
                        pass

        title = (paper.get("title") or (candidates[0] if candidates else "?"))[:60]
        if success:
            if source_label.startswith("doi2pdf"):
                count_doi2pdf += 1
            elif "arxiv" in source_label.lower():
                count_arxiv += 1
            else:
                count_extraction_main += 1
            print(f"[info] related_pdf_download: {title!r} — {source_label}", flush=True)
        else:
            count_failed += 1
            print(f"[info] related_pdf_download: {title!r} — all sources failed", flush=True)

    total_downloaded = count_extraction_main + count_doi2pdf + count_arxiv
    parts = []
    if count_extraction_main:
        parts.append(f"{count_extraction_main} from extraction.main")
    if count_doi2pdf:
        parts.append(f"{count_doi2pdf} from doi2pdf")
    if count_arxiv:
        parts.append(f"{count_arxiv} from arxiv")
    src_summary = ", ".join(parts) if parts else "0 downloaded"
    print(
        f"[info] total {paper_count} papers, {total_downloaded} downloaded ({src_summary}), {count_failed} failed",
        flush=True,
    )
    return attempts

