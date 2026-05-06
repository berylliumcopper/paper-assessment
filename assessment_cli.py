"""Single-paper AI assessment workflow CLI."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse
import requests

# Reconfigure stdout/stderr to handle Unicode on Windows (gbk codec issues).
_stdout_bio = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stdout = _stdout_bio
_stderr_bio = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = _stderr_bio

from assessment.api_settings import DEFAULT_API_SETTINGS_PATH, load_api_settings
from assessment.related_reference_text import RELATED_PDF_EXTRACT_MAX_CHARS
from assessment.openai_client import OpenAIClient
from assessment.gemini_client import GeminiClient
from assessment.paper_reader import build_paper_context, prepare_supplementary_for_assessment
from assessment.source_data_availability import (
    MAX_SOURCE_MATERIAL_BYTES,
    expand_github_material_download_urls,
    expand_zenodo_material_download_urls,
    expand_figshare_material_download_urls,
    expand_dryad_material_download_urls,
    expand_osf_material_download_urls,
    resolve_repository_api,
    http_get_with_size_cap,
    load_combined_article_markdown,
    scan_availability_statement,
    write_audit_file,
)
from assessment.related_work import (
    RelatedPaper,
    _dedupe,
    attempt_related_pdf_extraction,
    search_related_papers,
    search_related_papers_with_web,
)
from dataclasses import asdict
from assessment.related_reference_text import enrich_papers_with_fast_fulltext
from assessment.report_writer import write_assessment_outputs, write_related_work_report
from assessment.rubric import SYSTEM_PROMPT, build_user_prompt, normalize_assessment
from assessment.figure_table_assessment import run_figure_table_assessment
from extraction.config.defaults import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_JITTER_SECONDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEOUT_SECONDS,
    PROFILE_DIR,
)
from extraction.core.browser import BrowserClient
from extraction.core.doi2pdf import DEFAULT_SCI_HUB_MIRRORS, apply_doi2pdf_fallback
from extraction.core.models import ExtractionOutcome
from extraction.core.rate_limit import RateLimiter
from extraction.core.resolver import resolve_target
from extraction.main import (
    _extract_for_target,
    _extract_oa_pdf_only,
    _load_unpaywall_email,
    _mark_access_limited_if_needed,
    _oa_first_target,
)
from extraction.output.writer import write_outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-paper AI assessment workflow.")
    parser.add_argument("target", nargs="?", help="One target paper: local PDF path OR DOI/URL.")
    parser.add_argument("--input", default="", help="One target paper: local PDF path OR DOI/URL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory root.")
    parser.add_argument(
        "--openai-model",
        default="",
        help="Optional model override (otherwise uses API config file or OPENAI_MODEL).",
    )
    parser.add_argument(
        "--provider",
        default="",
        choices=["openai", "gemini"],
        help="Force a specific API provider protocol (openai or gemini).",
    )
    parser.add_argument(
        "--api-config",
        default=str(DEFAULT_API_SETTINGS_PATH),
        help="Path to JSON file containing api_key/base_url/model.",
    )
    parser.add_argument("--mock-assessment", action="store_true", help="Generate a local mock assessment without API calls.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Network timeout for extraction/search.")
    parser.add_argument("--api-timeout-seconds", type=int, default=600, help="Read timeout for LLM API calls.")
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS, help="Base delay for extraction.")
    parser.add_argument("--jitter-seconds", type=float, default=DEFAULT_JITTER_SECONDS, help="Jitter for extraction.")
    parser.add_argument("--headed", action="store_false", dest="headless", help="Run browser in headed mode (default is headless).")
    parser.add_argument("--headless", action="store_true", dest="headless", help="Run browser in headless mode (default).")
    parser.set_defaults(headless=True)
    parser.add_argument("--browser-channel", default="chromium", choices=["chromium", "chrome", "msedge"])
    parser.add_argument("--real-browser-mode", action="store_true")
    parser.add_argument("--mode", default="both", choices=["structured", "pdf", "both"], help="Extraction mode for DOI/URL input.")
    parser.add_argument("--unpaywall-email", default="", help="Optional Unpaywall email.")
    parser.add_argument(
        "--convert-script",
        default=r"extraction\convert_pdf.py",
        help="Path to PDF conversion script.",
    )
    parser.add_argument("--skip-convert", action="store_true", help="Skip PDF-to-markdown conversion.")
    parser.add_argument("--skip-related-search", action="store_true", help="Skip related-work search.")
    parser.add_argument("--no-web-search", action="store_false", dest="use_web_search", help="Disable web search for related papers.")
    parser.set_defaults(use_web_search=True)
    parser.add_argument(
        "--ai-search-queries",
        type=int,
        default=7,
        help="Use AI to generate N search queries from paper content (0 disables, default 7).",
    )
    parser.add_argument(
        "--related-download-count",
        type=int,
        default=20,
        help="Attempt PDF extraction for up to N related papers (0 disables).",
    )
    parser.add_argument(
        "--related-fulltext-max-pages",
        type=int,
        default=50,
        help="Max pages to read per related PDF (plain-text; PyMuPDF/pypdf; not Marker).",
    )
    parser.add_argument(
        "--related-fulltext-chars",
        type=int,
        default=RELATED_PDF_EXTRACT_MAX_CHARS,
        help="Max characters of plain text per shortlisted related PDF (independent of target-article size).",
    )
    parser.add_argument(
        "--no-related-fulltext",
        action="store_true",
        help="Do not read main text for related PDFs; related reports use abstract only.",
    )
    parser.add_argument(
        "--related-pdf-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for storing downloaded related-paper PDFs (default: extraction/output_data).",
    )
    parser.add_argument("--trace-dir-name", default="ai_traces", help="Directory name for saved prompts/responses.")
    parser.add_argument(
        "--sci-hub-mirrors",
        nargs="*",
        default=[],
        help=(
            "Space-separated Sci-Hub mirror URLs for the doi2pdf fallback. "
            "If omitted, built-in defaults are used."
        ),
    )
    parser.add_argument(
        "--no-doi2pdf",
        action="store_true",
        help="Disable the doi2pdf fallback when extraction is blocked.",
    )
    return parser


def main() -> int:
    started_at = time.perf_counter()
    args = build_parser().parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    api_settings = load_api_settings(Path(args.api_config))
    api_key = api_settings.get("api_key", "").strip()
    api_base_url = api_settings.get("base_url", "").strip()
    api_model = args.openai_model.strip() if args.openai_model.strip() else api_settings.get("model", "").strip()
    
    # Determine provider
    provider_hint = args.provider.strip().lower()
    if provider_hint:
        is_gemini = provider_hint == "gemini"
    else:
        # Auto-detect if not explicitly forced
        is_gemini = "gemini" in api_model.lower() or "google" in api_base_url.lower()
    
    if not api_base_url:
        api_base_url = "https://generativelanguage.googleapis.com/v1beta" if is_gemini else "https://api.openai.com/v1"
    if not api_model:
        api_model = "gemini-1.5-flash" if is_gemini else "gpt-4o-mini"

    print(f"[config] provider={'gemini' if is_gemini else 'openai'} api_base_url={api_base_url} model={api_model} mock={args.mock_assessment}")
    print(f"[config] api_key_present={bool(api_key)} output_dir={output_root}")
    if not args.mock_assessment and not api_key:
        print(
            f"[error] API key missing. Set OPENAI_API_KEY or fill {Path(args.api_config)}.",
            file=sys.stderr,
        )
        return 2

    limiter = RateLimiter(base_delay_seconds=args.delay_seconds, jitter_seconds=args.jitter_seconds)
    target = (args.input or args.target or "").strip()
    if not target:
        print("[error] Missing target paper. Use positional TARGET or --input.", file=sys.stderr)
        return 2
    source_pdf_path: Path | None = None
    print(f"[stage] prepare_input target={target}")
    prepare_started = time.perf_counter()
    try:
        if _looks_like_local_pdf(target):
            print("[stage] input_type=local_pdf")
            source_pdf_path = Path(target).resolve()
            article_dir, metadata = _prepare_from_local_pdf(target=target, output_root=output_root)
        else:
            print("[stage] input_type=remote_doi_or_url")
            article_dir, metadata = _prepare_from_remote_target(
                target=target,
                output_root=output_root,
                limiter=limiter,
                args=args,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] failed to prepare paper artifacts: {exc}", file=sys.stderr)
        return 1
    print(f"[ok] prepare_input completed in {time.perf_counter() - prepare_started:.1f}s article_dir={article_dir}")
    # Pipeline for supplementary data (when applicable):
    # 1) convert main PDF -> article markdown (separate step below)
    # 2) discover/collect material links from article markdown, download into supplementary/
    # 3) convert supplementary PDF/Excel to readable text/CSV (marker + table export)
    # 4) build_paper_context includes these readable snippets in the final assessment prompt

    print("[stage] convert_pdf")
    convert_started = time.perf_counter()
    conversion_note = _maybe_convert_pdf(
        article_dir=article_dir,
        convert_script=Path(args.convert_script),
        skip_convert=args.skip_convert,
        source_pdf_path=source_pdf_path,
    )
    print(f"[ok] convert_pdf completed in {time.perf_counter() - convert_started:.1f}s result={conversion_note}")
    print("[stage] download_declared_materials")
    materials_started = time.perf_counter()
    materials_note = _attempt_download_declared_materials(
        article_dir=article_dir,
        resolved_url=str(metadata.get("resolved_url", "")),
        timeout_seconds=args.timeout_seconds,
        headless=args.headless,
        browser_channel=args.browser_channel,
        real_browser_mode=args.real_browser_mode,
    )
    print(f"[ok] download_declared_materials completed in {time.perf_counter() - materials_started:.1f}s {materials_note}")

    print("[stage] convert_supplementary_to_readable")
    readables_started = time.perf_counter()
    readables_note = prepare_supplementary_for_assessment(article_dir=article_dir)
    print(f"[ok] convert_supplementary_to_readable completed in {time.perf_counter() - readables_started:.1f}s {readables_note}")

    # --- Defaults for variables that may be set by the related-search block ---
    paper_category = "A"
    theory_comp_exp_distribution = [0.33, 0.33, 0.34]
    target_abstract = ""
    target_problems: list = []

    related_payload = {"query": "", "counts": {}, "papers": [], "notes": ["Related search skipped."]}
    if not args.skip_related_search:
        print("[stage] related_search")
        related_started = time.perf_counter()
        
        # Build list of search queries
        # We'll initialize with the metadata title, but AI will likely override it
        base_query = _build_related_query(metadata=metadata, fallback=article_dir.name, article_dir=article_dir)
        search_queries = []
        
        if args.ai_search_queries > 0:
            print(f"[stage] ai_query_generation count={args.ai_search_queries}")
            try:
                ai_search_payload = _generate_ai_search_queries(
                    article_dir=article_dir,
                    count=args.ai_search_queries,
                    api_key=api_key,
                    api_model=api_model,
                    api_base_url=api_base_url,
                    is_gemini=is_gemini,
                    timeout_seconds=args.api_timeout_seconds
                )
                ai_queries = ai_search_payload.get("queries", [])
                target_abstract = ai_search_payload.get("target_abstract", "")
                target_problems = ai_search_payload.get("target_problems", [])
                extracted_title = ai_search_payload.get("paper_title", "")
                paper_category = str(ai_search_payload.get("paper_category", "A"))
                theory_comp_exp_distribution = ai_search_payload.get("theory_comp_exp_distribution", [0.33, 0.33, 0.34])
                
                if extracted_title:
                    print(f"[info] ai_extracted_title='{extracted_title}'")
                    search_queries.append(extracted_title)
                    # Update metadata for filtering and report
                    metadata["title"] = extracted_title
                else:
                    search_queries.append(base_query)

                search_queries.extend(ai_queries)
                # Dedupe queries
                search_queries = list(dict.fromkeys(search_queries))
            except Exception as exc:
                print(f"[warn] AI query generation failed: {exc}")
                search_queries = [base_query]
                target_abstract = ""
                target_problems = []
                paper_category = "A"
                theory_comp_exp_distribution = [0.33, 0.33, 0.34]
        else:
            search_queries = [base_query]
            target_abstract = ""
            target_problems = []
            paper_category = "A"
            theory_comp_exp_distribution = [0.33, 0.33, 0.34]

        all_papers = []
        all_counts = {"crossref": 0, "arxiv": 0, "semantic_scholar": 0, "web_search": 0, "merged_unique": 0}
        
        num_queries = len(search_queries)
        target_doi = metadata.get("doi")
        for idx, q in enumerate(search_queries):
            print(f"[info] [{idx+1}/{num_queries}] searching_query='{q}'")
            if args.use_web_search:
                payload = search_related_papers_with_web(
                    query=q, 
                    timeout_seconds=args.timeout_seconds, 
                    per_source_limit=8,
                    web_results=[],
                    exclude_doi=target_doi
                )
            else:
                payload = search_related_papers(
                    query=q, 
                    timeout_seconds=args.timeout_seconds, 
                    per_source_limit=8,
                    exclude_doi=target_doi
                )
            
            all_papers.extend(payload.get("papers", []))
            for k in all_counts:
                if k != "merged_unique":
                    all_counts[k] += payload.get("counts", {}).get(k, 0)

        # Citation search: find papers that cite the target paper (critical comments, follow-ups).
        # Two-stage: fetch many (minimal info), then AI-filter to ~20.
        target_id_source = "doi"
        target_id_for_citation = target_doi

        # Prefer arXiv ID over metadata DOI for arXiv papers - metadata DOI may be from a cited paper.
        ru = (metadata.get("resolved_url") or "").strip()
        arxiv_m = re.match(r"(?:https?://arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?", ru)
        if arxiv_m:
            target_id_for_citation = arxiv_m.group(1)
            target_id_source = "arxiv"
            print(f"[info] citation_search: using arXiv ID from resolved_url: {target_id_for_citation}", flush=True)
        elif not target_id_for_citation:
            # Fallback 1: extract arXiv ID from resolved_url.
            ru = (metadata.get("resolved_url") or "").strip()
            m = re.match(r"(?:https?://arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?", ru)
            if m:
                target_id_for_citation = m.group(1)
                target_id_source = "arxiv"
        if not target_id_for_citation:
            # Fallback 2: search the converted article markdown for a DOI.
            for md_rel in ("converted/article.md", "article.md"):
                md_path = article_dir / md_rel
                if md_path.is_file():
                    try:
                        body = md_path.read_text(encoding="utf-8", errors="ignore")
                        doi_found = re.search(r"(10\.\d{4,}/[^\s<>\"'()\[\]]+)", body)
                        if doi_found:
                            target_id_for_citation = doi_found.group(1).rstrip(".,;")
                            target_id_source = "doi_from_markdown"
                            print(f"[info] citation_search: found DOI in {md_rel}: {target_id_for_citation}", flush=True)
                            break
                    except OSError:
                        pass
        if target_id_for_citation:
            print(f"[info] searching_citations_via_{target_id_source}={target_id_for_citation}")
            from assessment.related_work import search_citations_by_doi
            kwargs = {"doi": target_id_for_citation, "timeout_seconds": args.timeout_seconds}
            if target_id_source == "arxiv":
                kwargs = {"arxiv_id": target_id_for_citation, "timeout_seconds": args.timeout_seconds}
            raw_citing = search_citations_by_doi(**kwargs)
            if raw_citing:
                raw_dicts = [asdict(p) for p in raw_citing]
                print(f"[info] citation_search: fetched {len(raw_dicts)} raw citing papers, running AI filter", flush=True)
                if not args.mock_assessment:
                    filtered_citing = _filter_citations_with_ai(
                        target_title=metadata.get("title", ""),
                        target_abstract=target_abstract,
                        target_problems=target_problems,
                        citing_papers=raw_dicts,
                        api_key=api_key,
                        api_model=api_model,
                        api_base_url=api_base_url,
                        is_gemini=is_gemini,
                        timeout_seconds=args.api_timeout_seconds,
                        target_count=20,
                    )
                else:
                    # Mock mode: just take the most recent 20.
                    raw_dicts.sort(key=lambda p: -(p.get("year") or 0))
                    filtered_citing = raw_dicts[:20]
                all_papers.extend(filtered_citing)
                all_counts["cited_by"] = all_counts.get("cited_by", 0) + len(filtered_citing)
                print(f"[info] citation_search: {len(filtered_citing)}/{len(raw_dicts)} passing AI filter", flush=True)
            else:
                print("[info] citation_search: no citing papers found or DOI not resolvable", flush=True)
        else:
            print("[info] citation_search: skipped (no DOI, arXiv ID, or markdown DOI found)", flush=True)

        # Final deduplication of all papers from all queries
        from assessment.related_work import RelatedPaper, _dedupe
        # Convert dicts back to RelatedPaper objects for _dedupe
        paper_objs = []
        for p in all_papers:
            paper_objs.append(RelatedPaper(**{k: v for k, v in p.items() if k in RelatedPaper.__slots__}))
        
        unique_papers = _dedupe(paper_objs, exclude_doi=target_doi)
        
        # Step 2.5: AI-powered rough filtering
        print(f"[stage] ai_filtering_results total={len(unique_papers)}")
        filtered_papers = _filter_papers_with_ai(
            target_metadata=metadata,
            target_abstract=target_abstract,
            target_problems=target_problems,
            papers=[asdict(p) for p in unique_papers],
            api_key=api_key,
            api_model=api_model,
            api_base_url=api_base_url,
            is_gemini=is_gemini,
            timeout_seconds=args.api_timeout_seconds
        )
        print(f"[ok] ai_filtering completed: {len(unique_papers)} -> {len(filtered_papers)} papers")

        # Stabilize shortlist: sort by relevance (descending) and cap at 20 to reduce run-to-run variation.
        filtered_papers.sort(key=lambda x: -x.get("relevance_score", 0))
        MAX_SHORTLIST = 20
        if len(filtered_papers) > MAX_SHORTLIST:
            excess = len(filtered_papers) - MAX_SHORTLIST
            filtered_papers = filtered_papers[:MAX_SHORTLIST]
            print(f"[info] shortlist capped to {MAX_SHORTLIST} (removed {excess} lowest-scored papers)", flush=True)

        unique_paper_dicts: list[dict] = [asdict(p) for p in unique_papers]
        not_selected_hits: list[dict] = _not_in_shortlist_candidates(unique_paper_dicts, filtered_papers)

        related_payload = {
            "query": " | ".join(search_queries),
            "counts": {
                **all_counts,
                "merged_unique": len(unique_papers),
                "filtered": len(filtered_papers),
                "not_shortlisted": len(not_selected_hits),
            },
            "papers": filtered_papers,
            "not_selected_search_hits": not_selected_hits,
            "notes": [],
        }

        if not related_payload.get("papers"):
            related_payload["notes"].append("No related papers were found from configured sources.")
        
        download_attempts = attempt_related_pdf_extraction(
            papers=related_payload.get("papers", []),
            output_dir=Path(args.related_pdf_dir),
            max_downloads=max(0, args.related_download_count),
            headless=args.headless,
        )
        related_payload["related_pdf_extraction_attempts"] = download_attempts

        # Fast plain-text main body for shortlisted papers (Marker stays for target/supplementary only).
        if (
            not args.no_related_fulltext
            and max(0, args.related_download_count) > 0
            and related_payload.get("papers")
        ):
            print(
                f"[stage] related_fulltext_extract (pypdf/PyMuPDF, max {args.related_fulltext_max_pages} p / "
                f"{args.related_fulltext_chars} chars per paper)"
            )
            _ft_start = time.perf_counter()
            enrich_papers_with_fast_fulltext(
                related_payload["papers"],
                Path(args.related_pdf_dir),
                max_papers=max(0, args.related_download_count),
                max_pages=args.related_fulltext_max_pages,
                max_chars=args.related_fulltext_chars,
            )
            print(
                f"[ok] related_fulltext_extract completed in {time.perf_counter() - _ft_start:.1f}s"
            )
        # Step 3: shortlist — one LLM call per batch (scratch + scores + target-focused narrative vs target)
        related_payload["reports"] = _generate_related_work_reports(
            papers=related_payload["papers"],
            target_title=str(metadata.get("title", "")),
            target_abstract=target_abstract or "",
            target_problems=target_problems or [],
            api_key=api_key,
            api_model=api_model,
            api_base_url=api_base_url,
            is_gemini=is_gemini,
            timeout_seconds=args.api_timeout_seconds,
        )
        for p in related_payload.get("papers", []):
            ky = _paper_stem_key(p)
            for r in related_payload.get("reports", []):
                if _paper_stem_key(r) == ky:
                    p["target_focused_narrative"] = r.get("target_focused_narrative", "")
                    break

        nsh_list = list(related_payload.get("not_selected_search_hits") or [])
        if nsh_list:
            if not args.mock_assessment and api_key:
                print(
                    f"[stage] not_selected_brief_abstracts (one batch, one sentence out; "
                    f"count={len(nsh_list)})"
                )
                related_payload["not_selected_search_hits"] = _enrich_not_selected_brief_abstracts(
                    [dict(p) for p in nsh_list],
                    str(metadata.get("title", "")),
                    api_key=api_key,
                    api_model=api_model,
                    api_base_url=api_base_url,
                    is_gemini=is_gemini,
                    timeout_seconds=args.api_timeout_seconds,
                )
            else:
                heur: list[dict] = []
                for p in nsh_list:
                    d = dict(p)
                    t0 = (d.get("title") or "")[:120]
                    a0 = (d.get("abstract") or "")[:400]
                    d["brief_abstract"] = f"{t0} — {a0}…" if a0 else t0
                    heur.append(d)
                related_payload["not_selected_search_hits"] = heur

        counts = related_payload.get("counts", {})
        print(
            "[ok] related_search completed in "
            f"{time.perf_counter() - related_started:.1f}s "
            f"crossref={counts.get('crossref', 0)} arxiv={counts.get('arxiv', 0)} "
            f"semantic_scholar={counts.get('semantic_scholar', 0)} "
            f"web_search={counts.get('web_search', 0)} "
            f"merged={counts.get('merged_unique', 0)}"
        )

        # Print basic information for each related paper found
        papers = related_payload.get("papers", [])
        if papers:
            print("\n[info] related_papers_found:")
            for i, paper in enumerate(papers, 1):
                title = paper.get("title", "Unknown Title")
                year = paper.get("year")
                source = paper.get("source", "unknown")
                year_str = f" ({year})" if year else ""
                print(f"  {i}. [{source.upper()}] {title}{year_str}")
            print("")

        print("[stage] summarize_related_work")
        summary_started = time.perf_counter()
        _print_related_work_summary(related_payload)
        print(f"[ok] summarize_related_work completed in {time.perf_counter() - summary_started:.1f}s")

        # Write related work report + JSON now (does not depend on final assessment).
        rw_metadata = {"model": api_model, "api_base_url": api_base_url}
        write_related_work_report(article_dir, related_payload, rw_metadata)
        print("[info] related_work report written before figure/table assessment", flush=True)
    else:
        print("[stage] related_search skipped by flag")

    print("[stage] data_file_analysis")
    data_file_analysis_started = time.perf_counter()
    data_file_analysis = None
    supp_dir = article_dir / "supplementary"
    if supp_dir.is_dir() and any(supp_dir.iterdir()) and not args.mock_assessment:
        data_file_analysis = _analyze_data_files_with_ai(
            article_dir=article_dir,
            api_key=api_key,
            api_model=api_model,
            api_base_url=api_base_url,
            is_gemini=is_gemini,
            timeout_seconds=args.api_timeout_seconds,
        )
        print(
            f"[info] data_file_analysis: overview_len={len(data_file_analysis.get('overview_block',''))} "
            f"snippets={len(data_file_analysis.get('snippets',[]))}",
            flush=True,
        )
    else:
        data_file_analysis = {"overview": "", "overview_block": "", "snippets": [], "selected_files": []}
        if args.mock_assessment:
            print("[info] data_file_analysis: skipped (mock assessment)", flush=True)
        else:
            print("[info] data_file_analysis: skipped (no supplementary directory)", flush=True)
    print(f"[ok] data_file_analysis completed in {time.perf_counter() - data_file_analysis_started:.1f}s")

    # --- Enrich data_file_analysis with oversized file info from audit ---
    _supp_dir = article_dir / "supplementary"
    _audit_p = _supp_dir / "_source_data_audit.json"
    if data_file_analysis and _audit_p.is_file():
        try:
            import json as _json
            _audit = _json.loads(_audit_p.read_text(encoding="utf-8", errors="ignore"))
            _oversized = [ev for ev in (_audit.get("url_events") or []) if ev.get("outcome") in ("skipped_oversized", "skipped_oversized_stream")]
            if _oversized:
                _note = "\n\n### Oversized Files (confirmed to exist, not auto-downloaded)\n\n"
                for _ev in _oversized:
                    _fn = _ev.get("inferred_filename") or "?"
                    _sz = _ev.get("declared_size_bytes")
                    _sz_str = f"{_sz / (1024*1024):.1f} MiB" if _sz else "unknown size"
                    _note += f"- `{_fn}` ({_sz_str})\n"
                _note += "\nThese files are **confirmed to exist** at the repository and were skipped only due to pipeline download limits. Include them in openness assessment using their sizes.\n"
                _existing = data_file_analysis.get("overview_block", "")
                data_file_analysis["overview_block"] = (_existing + "\n\n" + _note) if _existing else _note
                print(f"[info] data_file_analysis: enriched with {len(_oversized)} oversized file(s)", flush=True)
        except Exception as _exc:
            print(f"[info] data_file_analysis: could not add oversized info: {_exc}", flush=True)

    # --- Determine assessment mix from paper category ---
    CATEGORY_COUNTS = {
        "A": (3, 0, 0),
        "B": (2, 1, 0),
        "C": (1, 2, 0),
        "D": (0, 3, 0),
        "E": (2, 0, 1),
        "F": (1, 0, 2),
        "G": (0, 2, 1),
        "H": (0, 1, 2),
        "I": (1, 1, 1),
    }
    ft_count, deriv_count, pipe_count = CATEGORY_COUNTS.get(paper_category, (3, 0, 0))
    print(
        f"[info] paper_category={paper_category} distribution={theory_comp_exp_distribution} "
        f"ft_count={ft_count} deriv_count={deriv_count} pipe_count={pipe_count}",
        flush=True,
    )

    # --- Figure/table assessment (ft_count times) ---
    print("[stage] figure_table_assessment")
    ft_assessment_started = time.perf_counter()
    ft_assessments: list[dict] = []
    if ft_count > 0 and not args.mock_assessment:
        for i in range(ft_count):
            print(f"[info] figure_table_assessment run {i+1}/{ft_count}", flush=True)
            ft_result = run_figure_table_assessment(
                article_dir=article_dir,
                data_file_analysis=data_file_analysis or {},
                api_key=api_key,
                api_model=api_model,
                api_base_url=api_base_url,
                is_gemini=is_gemini,
                timeout_seconds=args.api_timeout_seconds,
            )
            ft_assessments.append(ft_result)
    elif args.mock_assessment:
        ft_assessments.append({
            "figure_table_analysis": "Skipped (mock assessment).",
            "analysis_block": "",
            "files_for_final_step": [],
            "simplified_data_block": "",
        })
        print("[info] figure_table_assessment: skipped (mock assessment)", flush=True)
    # Combine multiple FT assessment analysis blocks
    ft_assessment_blocks: list[str] = []
    ft_final_data_blocks: list[str] = []
    for ft in ft_assessments:
        blk = ft.get("analysis_block", "")
        if blk:
            ft_assessment_blocks.append(blk)
        fd = ft.get("simplified_data_block", "")
        if fd:
            ft_final_data_blocks.append(fd)
    # Use the last FT assessment for output writing
    ft_assessment_for_output = ft_assessments[-1] if ft_assessments else {
        "figure_table_analysis": "No FT assessment run.",
        "analysis_block": "",
        "files_for_final_step": [],
        "simplified_data_block": "",
    }
    print(
        f"[ok] figure_table_assessment completed in "
        f"{time.perf_counter() - ft_assessment_started:.1f}s "
        f"runs={len(ft_assessments)}",
    )

    # Write figure/table assessment output files (JSON trace + markdown report) — global sequential index
    for ft_idx, ft in enumerate(ft_assessments):
        _write_figure_table_assessment_outputs(article_dir, ft, api_model, round_index=ft_idx + 1)

    # --- Derivation assessment (deriv_count times) ---
    from assessment.derivation_assessment import run_derivation_assessment  # noqa: PLC0415
    print("[stage] derivation_assessment")
    deriv_assessment_started = time.perf_counter()
    deriv_assessments: list[dict] = []
    if deriv_count > 0 and not args.mock_assessment:
        for i in range(deriv_count):
            print(f"[info] derivation_assessment run {i+1}/{deriv_count}", flush=True)
            deriv_result = run_derivation_assessment(
                article_dir=article_dir,
                api_key=api_key,
                api_model=api_model,
                api_base_url=api_base_url,
                is_gemini=is_gemini,
                timeout_seconds=args.api_timeout_seconds,
            )
            deriv_assessments.append(deriv_result)
    else:
        print(
            f"[info] derivation_assessment: skipped (count={deriv_count}, "
            f"mock={args.mock_assessment})",
            flush=True,
        )
    deriv_assessment_blocks: list[str] = []
    deriv_final_blocks: list[str] = []
    for da in deriv_assessments:
        blk = da.get("analysis_block", "")
        if blk:
            deriv_assessment_blocks.append(blk)
        fd = da.get("simplified_derivation_block", "")
        if fd:
            deriv_final_blocks.append(fd)
    print(
        f"[ok] derivation_assessment completed in "
        f"{time.perf_counter() - deriv_assessment_started:.1f}s "
        f"runs={len(deriv_assessments)}",
    )
    # Write derivation assessment output files — global sequential index
    for deriv_idx, da in enumerate(deriv_assessments):
        _write_derivation_assessment_outputs(article_dir, da, api_model, round_index=len(ft_assessments) + deriv_idx + 1)

    # --- Pipeline assessment (pipe_count times) ---
    from assessment.pipeline_assessment import run_pipeline_assessment  # noqa: PLC0415
    print("[stage] pipeline_assessment")
    pipeline_assessment_started = time.perf_counter()
    pipeline_assessments: list[dict] = []
    if pipe_count > 0 and not args.mock_assessment:
        for i in range(pipe_count):
            print(f"[info] pipeline_assessment run {i+1}/{pipe_count}", flush=True)
            pipeline_result = run_pipeline_assessment(
                article_dir=article_dir,
                api_key=api_key,
                api_model=api_model,
                api_base_url=api_base_url,
                is_gemini=is_gemini,
                timeout_seconds=args.api_timeout_seconds,
            )
            pipeline_assessments.append(pipeline_result)
    else:
        print(
            f"[info] pipeline_assessment: skipped (count={pipe_count}, "
            f"mock={args.mock_assessment})",
            flush=True,
        )
    pipeline_assessment_blocks: list[str] = []
    pipeline_final_blocks: list[str] = []
    for pa in pipeline_assessments:
        blk = pa.get("analysis_block", "")
        if blk:
            pipeline_assessment_blocks.append(blk)
        fd = pa.get("simplified_pipeline_block", "")
        if fd:
            pipeline_final_blocks.append(fd)
    print(
        f"[ok] pipeline_assessment completed in "
        f"{time.perf_counter() - pipeline_assessment_started:.1f}s "
        f"runs={len(pipeline_assessments)}",
    )
    # Write pipeline assessment output files — global sequential index
    for pipe_idx, pa in enumerate(pipeline_assessments):
        _write_pipeline_assessment_outputs(article_dir, pa, api_model, round_index=len(ft_assessments) + len(deriv_assessments) + pipe_idx + 1)

    print("[stage] build_context_and_prompt")
    prompt_started = time.perf_counter()
    context = build_paper_context(article_dir=article_dir, metadata=metadata)

    # Merge AI data-file analysis into the context (overview text only)
    if data_file_analysis:
        overview_text = data_file_analysis.get("overview", "").strip()
        if overview_text:
            context.data_file_analysis_overview = (
                "## AI-Screened Supplementary Data Overview\n\n"
                f"{overview_text}"
            )

    # Merge figure/table assessment blocks into the context
    if ft_assessment_blocks:
        context.figure_table_assessment_block = "\n\n".join(ft_assessment_blocks)
    if ft_final_data_blocks:
        context.figure_table_final_data_block = "\n\n".join(ft_final_data_blocks)

    # Merge derivation assessment blocks into the context
    if deriv_assessment_blocks:
        context.derivation_assessment_block = "\n\n".join(deriv_assessment_blocks)
    if deriv_final_blocks:
        context.derivation_assessment_final_block = "\n\n".join(deriv_final_blocks)

    # Merge pipeline assessment blocks into the context
    if pipeline_assessment_blocks:
        context.pipeline_assessment_block = "\n\n".join(pipeline_assessment_blocks)
    if pipeline_final_blocks:
        context.pipeline_assessment_final_block = "\n\n".join(pipeline_final_blocks)

    user_prompt = build_user_prompt(
        paper_context_block=context.to_prompt_block(),
        related_work_payload=related_payload,
        theory_comp_exp_distribution=theory_comp_exp_distribution,
    )
    print(f"[ok] build_context_and_prompt completed in {time.perf_counter() - prompt_started:.1f}s prompt_chars={len(user_prompt)}")

    print("[stage] run_assessment_model")
    assess_started = time.perf_counter()
    if args.mock_assessment:
        assessment_raw = _build_mock_assessment(metadata=metadata)
        print("[info] using mock assessment payload")
    else:
        if is_gemini:
            client = GeminiClient(
                api_key=api_key,
                model=api_model,
                timeout_seconds=args.api_timeout_seconds,
                base_url=api_base_url,
            )
        else:
            client = OpenAIClient(
                api_key=api_key,
                model=api_model,
                timeout_seconds=args.api_timeout_seconds,
                base_url=api_base_url,
            )
        assessment_raw = client.generate_json(
            system_prompt=SYSTEM_PROMPT, 
            user_prompt=user_prompt,
            image_paths=context.image_paths
        )
    print(f"[ok] run_assessment_model completed in {time.perf_counter() - assess_started:.1f}s")

    print("[stage] normalize_and_write_outputs")
    write_started = time.perf_counter()
    assessment_payload = normalize_assessment(assessment_raw, theory_comp_exp_distribution=theory_comp_exp_distribution)

    trace_dir = article_dir / args.trace_dir_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "assessment_prompt.txt").write_text(user_prompt, encoding="utf-8")
    (trace_dir / "assessment_raw.json").write_text(
        json.dumps(assessment_raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_assessment_outputs(
        article_dir=article_dir,
        assessment_payload=assessment_payload,
        run_metadata={
            "input": target,
            "model": api_model,
            "api_base_url": api_base_url,
            "mock_assessment": args.mock_assessment,
            "conversion_note": conversion_note,
            "materials_note": materials_note,
            "readables_note": readables_note,
            "paper_category": paper_category,
            "theory_comp_exp_distribution": theory_comp_exp_distribution,
        },
    )
    print(
        f"[ok] normalize_and_write_outputs completed in {time.perf_counter() - write_started:.1f}s "
        f"trace_dir={trace_dir}"
    )
    elapsed_total = time.perf_counter() - started_at
    _print_run_summary(article_dir, elapsed_total)
    return 0


def _print_run_summary(article_dir: Path, elapsed_total: float) -> None:
    """Print a clear summary of all output files at the end of a run."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ✓ ASSESSMENT COMPLETE — {elapsed_total:.0f}s total")
    print(f"  Output folder: {article_dir.resolve()}")
    print(sep)

    # Describe key output files
    file_descriptions = {
        "assessment.md": "📄 Main assessment report",
        "assessment.json": "📊 Assessment data (JSON)",
        "related_work_report.md": "🔗 Related work analysis",
        "related_work.json": "🔗 Related work data (JSON)",
        "article.pdf": "📕 Original paper PDF",
    }
    # Dynamic: figure/table/derivation/pipeline reports
    for f in sorted(article_dir.iterdir()):
        if f.name.startswith("figure_table_assessment_report"):
            file_descriptions[f.name] = "🖼️  Figure/table analysis"
        elif f.name.startswith("derivation_assessment_report"):
            file_descriptions[f.name] = "🧮 Derivation analysis"
        elif f.name.startswith("pipeline_assessment_report"):
            file_descriptions[f.name] = "⚙️  Pipeline analysis"

    print()
    found_any = False
    for fname, desc in file_descriptions.items():
        fpath = article_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  {desc:40s} {fname} ({size_kb:.1f} KB)")
            found_any = True
    if not found_any:
        print("  (no output files found)")

    # Count related PDFs
    related_pdfs = list(article_dir.glob("related_pdfs/*.pdf"))
    if related_pdfs:
        print(f"\n  📚 Related paper PDFs saved: {len(related_pdfs)} files")
        for p in related_pdfs[:5]:
            print(f"     - {p.name}")
        if len(related_pdfs) > 5:
            print(f"     ... and {len(related_pdfs) - 5} more")

    print(f"\n{sep}\n")


def _prepare_from_remote_target(*, target: str, output_root: Path, limiter: RateLimiter, args) -> tuple[Path, dict]:
    resolved = resolve_target(target, timeout_seconds=args.timeout_seconds)
    unpaywall_email = _load_unpaywall_email(args.unpaywall_email)
    resolved, oa_result = _oa_first_target(
        target=resolved,
        timeout_seconds=args.timeout_seconds,
        unpaywall_email=unpaywall_email,
    )

    # If no OA URL was found, try to find an arXiv version via CrossRef before
    # falling through to the browser (which may hang on paywalled publisher pages).
    if oa_result is not None and not oa_result.oa_url:
        doi = resolved.doi
        if doi:
            try:
                xr = requests.get(
                    f"https://api.crossref.org/works/{doi}",
                    timeout=10,
                    headers={"User-Agent": "PaperAssessment/1.0"},
                )
                if xr.status_code == 200:
                    xmsg = xr.json().get("message", {})
                    arxiv_id = None
                    for link_rec in xmsg.get("link", []) or []:
                        m_a = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", str(link_rec.get("url", "")), re.I)
                        if m_a:
                            arxiv_id = m_a.group(1)
                            break
                    if not arxiv_id:
                        for rel_rec in xmsg.get("relation", {}).get("cites", []) or []:
                            rid = rel_rec.get("id", "")
                            if rid.startswith("arxiv:"):
                                arxiv_id = rid.replace("arxiv:", "")
                    if not arxiv_id:
                        for aid in (xmsg.get("alternative-id", []) or []):
                            if re.match(r"\d{4}\.\d{4,5}(v\d+)?$", str(aid)):
                                arxiv_id = str(aid)
                                break
                    if arxiv_id:
                        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                        print(f"[info] arXiv fallback found for DOI {doi}: {arxiv_url}", flush=True)
                        from extraction.extractors.common import detect_source
                        resolved = type(resolved)(
                            raw_input=resolved.raw_input,
                            is_doi=resolved.is_doi,
                            doi=resolved.doi,
                            resolved_url=arxiv_url,
                            source=detect_source(arxiv_url),
                        )
            except Exception as exc:
                print(f"[info] arXiv fallback lookup failed for {doi}: {exc}", flush=True)

    with BrowserClient(
        profile_dir=PROFILE_DIR,
        headless=args.headless,
        timeout_seconds=args.timeout_seconds,
        rate_limiter=limiter,
        browser_channel=args.browser_channel,
        real_browser_mode=args.real_browser_mode,
    ) as browser:
        if resolved.resolved_url.lower().endswith(".pdf"):
            outcome = _extract_oa_pdf_only(target=resolved, browser=browser, oa_result=oa_result)
        else:
            outcome = _extract_for_target(
                target=resolved,
                browser=browser,
                limiter=limiter,
                mode=args.mode,
                timeout=args.timeout_seconds,
            )
            if oa_result is not None:
                outcome.metadata.update(
                    {
                        "oa_resolution_source": oa_result.source,
                        "oa_resolution_status": oa_result.status,
                        "oa_resolution_message": oa_result.message,
                        "oa_landing_url": oa_result.landing_url,
                    }
                )
    _mark_access_limited_if_needed(outcome)

    # Try doi2pdf fallback when browser extraction is blocked or
    # access-limited and we have a DOI to work with.
    doi_for_fallback = outcome.doi or resolved.doi
    if not args.no_doi2pdf and doi_for_fallback and (outcome.anti_bot_blocked or outcome.access_limited):
        print(f"[doi2pdf] Trying fallback for DOI {doi_for_fallback} ...")
        sci_hub_mirrors = args.sci_hub_mirrors or None
        if apply_doi2pdf_fallback(
            outcome,
            doi=doi_for_fallback,
            timeout_seconds=args.timeout_seconds,
            sci_hub_mirrors=sci_hub_mirrors,
        ):
            print(f"[doi2pdf] PDF downloaded via {outcome.metadata.get('doi2pdf_fallback_source', '?')}")
        else:
            print(f"[doi2pdf] Fallback did not recover a PDF.")

    destination = write_outcome(output_root=output_root, outcome=outcome)
    return destination, _metadata_from_outcome(outcome, target)


def _prepare_from_local_pdf(*, target: str, output_root: Path) -> tuple[Path, dict]:
    import shutil
    pdf_path = Path(target).resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"Local PDF not found: {pdf_path}")
    # Create a per-paper subfolder under output_root based on the PDF filename
    paper_slug = pdf_path.stem.replace(" ", "_")
    article_dir = output_root / paper_slug
    article_dir.mkdir(parents=True, exist_ok=True)
    # Copy the PDF into the output folder as article.pdf (if not already there)
    dest_pdf = article_dir / "article.pdf"
    if not dest_pdf.exists() or dest_pdf.resolve() != pdf_path:
        shutil.copy2(pdf_path, dest_pdf)
        print(f"[info] copied PDF to {dest_pdf}")
    metadata = {
        "title": pdf_path.stem,
        "source": "local_pdf",
        "source_pdf": str(pdf_path),
        "resolved_url": str(pdf_path),
        "doi": None,
    }
    (article_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    run_log = {
        "input_type": "local_pdf",
        "has_pdf": True,
        "has_markdown": False,
        "errors": [],
    }
    (article_dir / "run_log.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    return article_dir, metadata


def _maybe_convert_pdf(*, article_dir: Path, convert_script: Path, skip_convert: bool, source_pdf_path: Path | None = None) -> str:
    if skip_convert:
        return "Conversion skipped by flag."
    pdf_path = article_dir / "article.pdf"
    if not pdf_path.exists() and source_pdf_path is not None and source_pdf_path.exists():
        pdf_path = source_pdf_path
    if not pdf_path.exists():
        return "No article.pdf found; conversion skipped."
    if not convert_script.exists():
        return f"Conversion script not found at {convert_script}; skipped."
    converted_dir = article_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    existing_conversion_note = _existing_converted_assets_note(converted_dir=converted_dir)
    if existing_conversion_note is not None:
        return existing_conversion_note
    command = [
        sys.executable,
        str(convert_script),
        str(pdf_path),
        str(converted_dir),
        "--name",
        "article",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (article_dir / "conversion.log").write_text(
        f"return_code: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode == 0:
        return "PDF converted to markdown/images successfully."

    # --- Fallback: use PyMuPDF for lightweight text extraction ---
    print("[info] Marker conversion failed, trying PyMuPDF fallback...", flush=True)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages_text.append(f"<!-- Page {page_num + 1} -->\n{text}")
        doc.close()

        if pages_text:
            md_content = "\n\n---\n\n".join(pages_text)
            md_path = converted_dir / "article.md"
            md_path.write_text(md_content, encoding="utf-8")
            print(f"[ok] PyMuPDF extracted text from {len(pages_text)} pages", flush=True)

            # Also extract images from the PDF
            doc = fitz.open(str(pdf_path))
            img_dir = converted_dir / "article"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_count = 0
            for page_num in range(len(doc)):
                for img_info in doc[page_num].get_images(full=True):
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n >= 5:  # CMYK: convert to RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_path = img_dir / f"page{page_num + 1}_img{img_count + 1}.png"
                        pix.save(str(img_path))
                        img_count += 1
                    except Exception:
                        pass
            doc.close()
            if img_count:
                print(f"[info] extracted {img_count} images from PDF", flush=True)

            return f"PDF converted via PyMuPDF fallback ({len(pages_text)} pages, {img_count} images)."
        else:
            return "PyMuPDF fallback: no text content found in PDF."
    except ImportError:
        print("[warn] PyMuPDF not installed. Run: pip install PyMuPDF", flush=True)
        return f"PDF conversion failed (Marker rc={completed.returncode}, PyMuPDF not available)."
    except Exception as exc:
        return f"PDF conversion failed (Marker rc={completed.returncode}, PyMuPDF error: {exc})."


def _existing_converted_assets_note(*, converted_dir: Path) -> str | None:
    md_path = converted_dir / "article.md"
    image_root = converted_dir / "article"
    if not md_path.exists():
        return None
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    image_files: list[Path] = []
    if image_root.exists():
        image_files.extend([p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions])
    image_files.extend([p for p in converted_dir.glob("*") if p.is_file() and p.suffix.lower() in image_extensions])
    unique_paths = {p.resolve() for p in image_files}
    if not unique_paths:
        return None
    return (
        "Conversion skipped: existing converted markdown/images detected "
        f"(article.md + {len(unique_paths)} image files)."
    )


def _metadata_from_outcome(outcome: ExtractionOutcome, fallback_input: str) -> dict:
    metadata = dict(outcome.metadata)
    metadata.setdefault("source", outcome.source)
    metadata.setdefault("resolved_url", outcome.resolved_url)
    metadata.setdefault("doi", outcome.doi)
    metadata.setdefault("title", metadata.get("title") or _title_from_url_or_text(outcome.resolved_url, fallback_input))
    return metadata


def _title_from_url_or_text(resolved_url: str, fallback: str) -> str:
    parsed = urlparse(resolved_url)
    if parsed.path:
        tail = parsed.path.rsplit("/", maxsplit=1)[-1]
        cleaned = re.sub(r"[_-]+", " ", tail)
        cleaned = re.sub(r"\.pdf$", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned
    return fallback


def _build_related_query(*, metadata: dict, fallback: str, article_dir: Path | None = None) -> str:
    title = metadata.get("title")
    
    # If the title is a generic filename, try to extract it from the markdown
    is_generic = title is None or title.lower() in {"original", "article", "paper", "manuscript", "document"}
    if is_generic and article_dir:
        md_title = _extract_title_from_markdown(article_dir)
        if md_title:
            return md_title

    if isinstance(title, str) and title.strip():
        return title.strip()
    doi = metadata.get("doi")
    if isinstance(doi, str) and doi.strip():
        return doi.strip()
    return fallback


def _extract_title_from_markdown(article_dir: Path) -> str | None:
    """Try to find the title in the converted markdown file."""
    # Check both root and converted/ directory
    paths = [
        article_dir / "article.md",
        article_dir / "converted" / "article.md"
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            # Usually the title is the first non-empty line or the first # header
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            for line in lines[:10]: # Check first 10 lines
                # Skip markdown image tags or very short lines
                if line.startswith("![") or len(line) < 10:
                    continue
                # Remove markdown header prefix if present
                cleaned = re.sub(r"^#+\s*", "", line).strip()
                if cleaned:
                    print(f"[info] extracted_title_from_md='{cleaned}'")
                    return cleaned
        except Exception:
            continue
    return None


def _generate_ai_search_queries(
    *, 
    article_dir: Path, 
    count: int, 
    api_key: str, 
    api_model: str, 
    api_base_url: str, 
    is_gemini: bool, 
    timeout_seconds: int
) -> dict:
    """Use AI to analyze the paper content and generate N diverse search queries, abstract, and key problems."""
    md_path = article_dir / "article.md"
    if not md_path.exists():
        md_path = article_dir / "converted" / "article.md"
    
    if not md_path.exists():
        return {"queries": [], "target_abstract": "", "target_problems": []}

    content = md_path.read_text(encoding="utf-8", errors="ignore")
    # Use first 8000 chars for context
    snippet = content[:8000]
    
    system_prompt = "You are a research assistant. Analyze the paper snippet and provide diverse, specific search queries, a concise abstract, and key scientific problems addressed."
    user_prompt = (
        f"Based on this paper snippet, generate {count} distinct search queries. "
        "Each query should focus on a different aspect: one for the core method, one for the specific problem domain, "
        "one for potential alternative explanations, and one for similar experimental setups. "
        "Also provide the actual title of the paper (ignoring journal names like 'Nature Physics'), "
        "a concise 1-paragraph abstract of the paper, and a list of 5-8 key scientific problems or questions it addresses.\n\n"
        "Additionally, answer:\n"
        "1. What is the distribution (experimental / theoretical / computational) of this paper's "
        "true contribution? Return three floats that sum to 1.0 (e.g., [0.7, 0.25, 0.05]). "
        "Definitions:\n"
        "   - **Experimental** (first float): Advances in measurement technique, sample synthesis, data collection, or experimental design, and the discussions related to or derived from the experimental data."
        "Using standard computer programs to process raw data (plotting, curve fitting, error analysis) is part of experimental work — it does NOT count as computational. "
        "Even a custom Python script that simply processes and plots measured data is still experimental, not computational.\n"
        "   - **Theoretical** (second float): Advances in analytical derivations, formalisms, theoretical models, or first-principles frameworks. "
        "CRITICAL: Most experimental papers contain some theoretical content — a model to fit data, standard approximations to interpret measurements, textbook formulas adapted to the system. "
        "This interpretive theory is part of the experimental analysis and does NOT constitute a theoretical contribution. "
        "Only count genuinely novel theoretical frameworks, new formalisms, or non-trivial derivations that would have standalone value even without the experimental data. "
        "If the theory section merely applies known models to explain observations, treat it as experimental, not theoretical.\n"
        "   - **Computational/pipeline** (third float): Advances in genuinely new algorithms, software architectures, computational pipelines, or numerical methods that have standalone value as a tool, and the discussions related to or derived from the new tool or pipeline. "
        "This means the algorithm/code itself is the novel contribution. "
        "If the paper simply uses existing computational tools (even with significant parameter tuning or non-trivial implementation) to study a new physical system, the advance is theoretical or experimental, not computational.\n"
        "Do NOT count words/pages. Judge what truly contributes to the conclusion and what is the true advance point.\n"
        "2. Which category best describes this paper? Return a single letter A-I as defined below.\n"
        "   IMPORTANT: Distinguish carefully between three qualitatively different kinds of computational work:\n"
        "   (i) Using standard computer programs (e.g., spreadsheet, Python scripts) to **process experimental data** — this is part of experimental work, NOT a pipeline advance.\n"
        "   (ii) Using already well-established computational pipelines/simulation codes (e.g., VASP, LAMMPS, standard DFT codes) to compute a new system — this is **theoretical/numerical simulation**, NOT a pipeline advance.\n"
        "   (iii) Developing genuinely **new software, algorithms, architectures, or computational pipelines** — THIS is a computational/pipeline advance.\n"
        "   \n"
        "   Categories:\n"
        "   (A) Pure experimental (3+0+0): All key advances are experimental measurements or materials synthesis. Any computation is just routine data processing (type i above). Any theoretical content (models, approximations, data-fitting formulas) merely serves to interpret the experimental data — it has no standalone theoretical value. Do NOT categorize interpretive theory as a theoretical contribution.\n"
        "   (B) Majorly experimental but with many derivations (2+1+0): Primarily an experimental paper, but also includes non-trivial analytical derivations or theoretical models. The theories should have their standalone value and not just be the copy of previous works.\n"
        "   (C) Theoretical, numerical results also important (1+2+0): The main contribution is theoretical (new formalism, model, or analytical derivation), but numerical simulations (type ii above) produce figures that are also central to the paper's conclusions.\n"
        "   (D) Theoretical, only formulas important (0+3+0): Pure theory paper. All key advances are analytical derivations, formulas, or theoretical frameworks. Any figures are schematic or plot the derived formulas. The paper should contain no numerical calculations or only straightforward and simple numerical results.\n"
        "   (E) Experimental with novel data pipeline (2+0+1): Primarily an experimental paper, but a genuinely new software/algorithmic pipeline (type iii above) is a key part of the contribution.\n"
        "   (F) Mostly pipeline/software with figures (1+0+2): The main contribution is a new computational method, algorithm, or software tool (type iii above). Figures primarily demonstrate the tool's output or benchmarks.\n"
        "   (G) Mostly theoretical with important pipeline (0+2+1): The main contribution is theoretical, but a new computational implementation or algorithm (type iii above) is also essential for the results. The computational contribution should not just be a routine application of existing tools, but have its own standalone value.\n"
        "   (H) Mostly pipeline with theoretical derivations (0+1+2): The main contribution is a new computational pipeline, but non-trivial theoretical derivation underpins or justifies the algorithm. The theoretical contribution should be significant and have its own standalone value.\n"
        "   (I) All three equally important (1+1+1): The paper makes comparable advances in experimental technique or numerical findings, theoretical formulation, and computational methodology. Either one of the three aspects has its own standalone value and is not just a routine application of existing tools. \n"
        "   The numbers in parentheses are (figure_table, derivation, pipeline) assessment counts.\n\n"
        "Return a JSON object with these keys: "
        "'queries' (list of strings), 'paper_title' (string), 'target_abstract' (string), "
        "'target_problems' (list of strings), "
        "'theory_comp_exp_distribution' (list of 3 floats), "
        "'paper_category' (string, one letter A-I).\n\n"
        f"Paper Snippet:\n{snippet}"
    )

    if is_gemini:
        client = GeminiClient(
            api_key=api_key,
            model=api_model,
            timeout_seconds=timeout_seconds,
            base_url=api_base_url,
        )
    else:
        client = OpenAIClient(
            api_key=api_key,
            model=api_model,
            timeout_seconds=timeout_seconds,
            base_url=api_base_url,
        )
    
    result = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
    return {
        "queries": [str(q) for q in result.get("queries", [])][:count],
        "paper_title": str(result.get("paper_title", "")),
        "target_abstract": str(result.get("target_abstract", "")),
        "target_problems": [str(p) for p in result.get("target_problems", [])],
        "theory_comp_exp_distribution": result.get("theory_comp_exp_distribution", [0.33, 0.33, 0.34]),
        "paper_category": str(result.get("paper_category", "A")),
    }


def _filter_papers_with_ai(
    *,
    target_metadata: dict,
    target_abstract: str,
    target_problems: list[str],
    papers: list[dict],
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int
) -> list[dict]:
    """Use AI to filter and categorize irrelevant papers based on title, abstract, and technical overlap."""
    if not papers:
        return []

    target_title = target_metadata.get("title", "Unknown")
    problems_str = "\n".join(f"- {p}" for p in target_problems)
    
    paper_list = ""
    for idx, p in enumerate(papers):
        title = p.get("title", "No Title")
        abstract = (p.get("abstract") or "No abstract")[:1000]
        year = p.get("year", "")
        source = p.get("source", "")
        year_str = f" ({year})" if year else ""
        paper_list += (
            f"--- Paper {idx} ---\n"
            f"Title: {title}{year_str} [{source}]\n"
            f"Abstract: {abstract}\n\n"
        )
        
    system_prompt = (
        "You are a research assistant selecting papers that are useful for critically assessing a target paper. "
        "Your task is to identify papers that help evaluate the target's claims, methods, or significance — "
        "not just papers that are broadly in the same field. "
        "Be precise and consistent: apply the same criteria to every paper."
    )
    user_prompt = (
        f"Target Paper Title: {target_title}\n"
        f"Target Paper Abstract: {target_abstract}\n"
        f"Target Paper Key Problems:\n{problems_str}\n\n"
        "CRITICAL: If any of the search results below is the SAME paper as the target paper (even if it's from a different source like arXiv vs a journal), you MUST exclude it.\n\n"
        "From the following search results, select approximately **12-16** papers that would be most "
        "**useful for assessing** the target paper. A paper is useful if it helps answer questions like: "
        "Are the methods sound? Are the claims supported? Are there alternative explanations? "
        "Is this result consistent with other work?\n\n"
        "Inclusion criteria (any ONE is sufficient):\n"
        "1. PROBLEM OVERLAP: Same physical system, phenomenon, or research question.\n"
        "2. TECHNICAL/METHOD OVERLAP: Uses similar techniques, measurement methods, or theoretical frameworks.\n"
        "3. COMPARATIVE VALUE: Provides baseline results, competing claims, or directly comparable data.\n"
        "4. FOUNDATIONAL BACKGROUND: Provides critical parameters, data, or theory that the target paper builds upon.\n\n"
        "Exclusion rules (apply strictly):\n"
        "- Remove papers that are only tangentially related (same broad field but different specific problem).\n"
        "- Remove papers older than ~15 years unless they are foundational to the target's specific subfield.\n"
        "- Remove papers that are purely general background (e.g., a textbook-style review of the entire field).\n"
        "- When in doubt about a borderline paper, err on the side of INCLUDING it.\n\n"
        "Tiebreaker rules (for equally relevant papers):\n"
        "- Prefer more recent papers over older ones.\n"
        "- Prefer papers with direct methodological overlap over purely topical overlap.\n\n"
        "Scoring guidelines:\n"
        "- 9-10: Directly addresses the same problem with overlapping methods; critical for assessment.\n"
        "- 6-8: Same research area with useful methods or comparative data.\n"
        "- 4-5: Background information or tangentially related; include only if space permits.\n"
        "- 1-3: Minimally relevant; do not include.\n\n"
        f"{paper_list}"
        "Return a JSON object with a 'results' key containing a list of objects for ONLY the selected papers. "
        "Each object must have:\n"
        "  - 'index' (int): the paper's index from above.\n"
        "  - 'relevance_score' (int 1-10): following the scoring guidelines above.\n"
        "  - 'topic_tag' (string): a 1-3 word category (e.g. 'Neural Networks', 'Quantum Dynamics').\n"
        "  - 'relevance_note' (string): one short sentence explaining why this paper is relevant to the target.\n"
        "Sort the results by relevance_score descending (most relevant first)."
    )

    try:
        if is_gemini:
            client = GeminiClient(api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url)
        else:
            client = OpenAIClient(api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url)
        
        res = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        results = res.get("results", [])
        
        filtered = []
        for r in results:
            idx = r.get("index")
            if isinstance(idx, int) and 0 <= idx < len(papers):
                p = dict(papers[idx])
                p["relevance_score"] = r.get("relevance_score", 5)
                p["topic_tag"] = r.get("topic_tag", "General")
                p["relevance_note"] = str(r.get("relevance_note", ""))
                filtered.append(p)
        return filtered
    except Exception as exc:
        print(f"[warn] AI filtering failed: {exc}")
        return papers


def _filter_citations_with_ai(
    *,
    target_title: str,
    target_abstract: str,
    target_problems: list[str],
    citing_papers: list[dict],
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int,
    target_count: int = 20,
) -> list[dict]:
    """
    Given a large list of papers that cite the target (titles + years only), use AI
    to select the ~``target_count`` most relevant ones for assessing the target paper.
    Returns the selected subset with relevance_score and topic_tag added.
    """
    if not citing_papers:
        return []

    problems_str = "\n".join(f"- {p}" for p in target_problems)

    paper_lines: list[str] = []
    for idx, p in enumerate(citing_papers):
        title = p.get("title", "Unknown")
        year = p.get("year", "")
        authors = p.get("authors")
        author_str = ""
        if isinstance(authors, list) and authors:
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
        year_str = f" ({year})" if year else ""
        paper_lines.append(f"[{idx}] {title}{year_str} — {author_str}")

    system_prompt = (
        "You are a research assistant identifying relevant citing papers for a scientific assessment. "
        "From a large pool of papers that cite the target, select the most useful ones for evaluating "
        "the target paper's claims, methods, and significance."
    )
    user_prompt = (
        f"Target Paper Title: {target_title}\n"
        f"Target Paper Abstract: {target_abstract}\n"
        f"Target Paper Key Problems:\n{problems_str}\n\n"
        "Below is a list of papers that cite the target. Select approximately "
        f"**{target_count}** papers that are most valuable for critically assessing the target paper.\n\n"
        "Prioritize papers that:\n"
        "- Provide critical comments, corrections, or conflicting results\n"
        "- Extend or test the target's claims with new experiments or data\n"
        "- Use the target's methods in a different context\n"
        "- Discuss the target in a review or comparison\n\n"
        "Since only titles, years, and authors are available, use your scientific judgment to infer relevance. "
        "Prefer more recent papers when in doubt.\n\n"
        f"{chr(10).join(paper_lines)}\n\n"
        "Return JSON with a 'results' key containing a list of objects for ONLY the selected papers. "
        "Each object must have:\n"
        "  - 'index' (int): the paper's index from the list above.\n"
        "  - 'relevance_score' (int 1-10): how relevant this paper is for assessing the target.\n"
        "  - 'topic_tag' (string): a 1-3 word category.\n"
        "  - 'relevance_note' (string): one sentence explaining the expected relevance.\n"
        "Sort by relevance_score descending."
    )

    try:
        if is_gemini:
            from assessment.gemini_client import GeminiClient
            client: Any = GeminiClient(
                api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
            )
        else:
            from assessment.openai_client import OpenAIClient
            client = OpenAIClient(
                api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
            )
        res = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        results = res.get("results", [])
    except Exception as exc:
        print(f"[warn] citation AI pre-filter failed: {exc}", flush=True)
        # Fallback: take first N by recency.
        results = [{"index": i, "relevance_score": 5, "topic_tag": "Citation"} for i in range(min(len(citing_papers), target_count))]

    selected: list[dict] = []
    for r in results:
        idx = r.get("index")
        if isinstance(idx, int) and 0 <= idx < len(citing_papers):
            p = dict(citing_papers[idx])
            p["relevance_score"] = r.get("relevance_score", 5)
            p["topic_tag"] = r.get("topic_tag", "Citation")
            p["relevance_note"] = str(r.get("relevance_note", ""))
            selected.append(p)
    return selected


DATA_FILE_ANALYSIS_SYSTEM_PROMPT = """You are a research data analyst. Your task is to examine a file tree of supplementary materials downloaded for a scientific paper and determine:

1. **OVERVIEW**: What kind of data is present overall (raw measurements, processed tables, figure-source data, code, etc.).

2. **IMPORTANT FILES**: Identify files that are likely **source data for main-text figures or key claims**. These files should be forwarded to the final assessment. Rules:
   - If many files in a folder are similar (e.g. per-sample measurement files), pick only **one representative example**.
   - Prefer files with descriptive names (e.g. "Fig2_data.csv", "main_results.txt"). Files that directly generate figures should be included.
   - For spreadsheets (xls/xlsx), note the sheet names provided.
   - Exclude README, info files, and code if they don't contain primary data.
   - For each selected file, state why it matters (e.g. "likely contains data for Figure 3").

Return valid JSON with this exact structure:
{
  "overview": "string — 1-3 paragraphs describing the data overall",
  "selected_files": [
    {
      "path": "string — relative path to the file",
      "reason": "string — why this file is important for the paper assessment",
      "is_example_of_group": false,
      "group_description": "string or null — if this file represents a group of similar files, describe the group"
    }
  ]
}"""


# Content extraction helpers (borrowed from data_screener)
_DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".xml",
                  ".dat", ".log", ".out", ".h5", ".hdf5", ".fits", ".nc",
                  ".npy", ".npz", ".mat"}
_PREVIEW_READ_CAP = 64 * 1024
_PREVIEW_MAX_CHARS = 8_000


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.1f} MiB"


def _preview_text(path: Path, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    try:
        to_read = min(path.stat().st_size, _PREVIEW_READ_CAP)
        raw = path.read_bytes()[:to_read]
        text = raw.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + "\n[... truncated ...]"
        return text
    except OSError:
        return "(unable to read)"


def _preview_csv(path: Path, max_lines: int = 8) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head: list[str] = []
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                head.append(line.rstrip("\n\r"))
        return "\n".join(head) if head else "(empty)"
    except OSError:
        return "(unable to read)"


def _resolve_data_path(supplementary_dir: Path, file_path_str: str) -> Path | None:
    """Locate a file on disk by its relative path in the tree."""
    candidate = supplementary_dir / file_path_str
    if candidate.is_file():
        return candidate
    for extracted_dir in supplementary_dir.glob("_extracted/*"):
        candidate2 = extracted_dir / file_path_str
        if candidate2.is_file():
            return candidate2
    candidate3 = supplementary_dir / "_converted_spreadsheets" / file_path_str
    if candidate3.is_file():
        return candidate3
    fname = Path(file_path_str).name
    for found in sorted(supplementary_dir.rglob(fname)):
        if found.is_file():
            return found
    return None


def _extract_file_content_preview(
    disk_path: Path,
    supplementary_dir: Path,
    max_chars: int = 24_000,
) -> str:
    """Extract a preview of a data file, handling xls/xlsx conversion."""
    suffix = disk_path.suffix.lower()
    sz_str = _human_size(disk_path.stat().st_size)

    if suffix in _DATA_SUFFIXES | {".md", ".csv", ".tsv"}:
        if disk_path.stat().st_size < 64 * 1024:
            content = _preview_text(disk_path, max_chars=max_chars)
        else:
            if suffix in {".csv", ".tsv"}:
                content = _preview_csv(disk_path, max_lines=20)
            else:
                content = _preview_text(disk_path, max_chars=_PREVIEW_MAX_CHARS)
            content += f"\n\n[File is {sz_str}; showing preview only]"
        return content

    if suffix in {".xls", ".xlsx"}:
        base = disk_path.stem
        csv_dir = supplementary_dir / "_converted_spreadsheets"
        csv_candidates = sorted(csv_dir.glob(f"{base}__*.csv")) if csv_dir.is_dir() else []
        if csv_candidates:
            csv_path = csv_candidates[0]
            if csv_path.stat().st_size < 64 * 1024:
                content = csv_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            else:
                content = _preview_csv(csv_path, max_lines=20)
                content += f"\n\n[File is {_human_size(csv_path.stat().st_size)}; showing preview only]"
        else:
            from assessment.paper_reader import _extract_excel_text_snippet
            content, _ = _extract_excel_text_snippet(disk_path, max_chars=8_000)
        return content

    return f"[Binary file, {sz_str}]"


def _analyze_data_files_with_ai(
    *,
    article_dir: Path,
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int,
) -> dict:
    """
    Call the AI to analyze the supplementary file tree and identify important data files.

    Returns dict with:
      - overview (str): AI-written overview of the downloaded data
      - selected_files (list[dict]): validated selected files with path/reason/is_example/group_description
      - overview_block (str): formatted markdown block for the assessment prompt
      - snippets (list[str]): extracted content from selected files
    """
    from assessment.paper_reader import build_supplementary_file_tree, format_tree_for_ai_prompt

    tree = build_supplementary_file_tree(article_dir)
    if not tree:
        return {"overview": "No supplementary data files were downloaded.", "selected_files": [],
                "overview_block": "", "snippets": []}

    tree_prompt = format_tree_for_ai_prompt(tree)

    user_prompt = (
        "Analyze the following file tree of supplementary materials for a scientific paper.\n\n"
        f"{tree_prompt}\n\n"
        "Identify which files contain data likely used for main-text figures or key claims. "
        "Return JSON with 'overview' (what the data contains) and 'selected_files' "
        "(files to include in the final assessment, with reasons)."
    )

    try:
        if is_gemini:
            from assessment.gemini_client import GeminiClient
            client: Any = GeminiClient(
                api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
            )
        else:
            from assessment.openai_client import OpenAIClient
            client = OpenAIClient(
                api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
            )
        result = client.generate_json(
            system_prompt=DATA_FILE_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        print(f"[warn] data file AI analysis failed: {exc}", flush=True)
        return {"overview": f"AI analysis failed: {exc}", "selected_files": [],
                "overview_block": "", "snippets": []}

    if not isinstance(result, dict):
        return {"overview": "AI returned non-dict response.", "selected_files": [],
                "overview_block": "", "snippets": []}

    overview = str(result.get("overview", ""))
    selected = result.get("selected_files", [])
    if not isinstance(selected, list):
        selected = []

    # Validate selected files and extract content
    supplementary_dir = article_dir / "supplementary"
    validated: list[dict] = []
    all_snippets: list[str] = []
    overview_lines: list[str] = []

    overview_lines.append("## AI-Screened Supplementary Data Overview")
    if overview:
        overview_lines.append("")
        overview_lines.append(overview)
    overview_lines.append("")
    overview_lines.append("### Selected data files for detailed review")

    from assessment.paper_reader import MAX_DATA_ANALYSIS_SNIPPET_TOTAL_CHARS

    # Pre-count how many selected files actually exist on disk to allocate per-file budget.
    on_disk_count = 0
    for sf in selected:
        if not isinstance(sf, dict):
            continue
        rel_path = sf.get("path", "")
        if not rel_path:
            continue
        disk_path = _resolve_data_path(supplementary_dir, rel_path)
        if disk_path is not None and disk_path.is_file():
            on_disk_count += 1
    per_file_budget = max(4_000, MAX_DATA_ANALYSIS_SNIPPET_TOTAL_CHARS // max(on_disk_count, 1))

    for sf in selected:
        if not isinstance(sf, dict):
            continue
        rel_path = sf.get("path", "")
        if not rel_path:
            continue
        reason = str(sf.get("reason", ""))
        is_example = bool(sf.get("is_example_of_group", False))
        group_desc = str(sf.get("group_description") or "")

        disk_path = _resolve_data_path(supplementary_dir, rel_path)
        if disk_path is None or not disk_path.is_file():
            overview_lines.append(f"- `{rel_path}` — {reason}  [NOT FOUND ON DISK]")
            continue

        sz_str = _human_size(disk_path.stat().st_size)
        example_tag = " (example)" if is_example else ""
        overview_lines.append(f"- `{rel_path}` ({sz_str}) — {reason}{example_tag}")
        if group_desc:
            overview_lines.append(f"  - Group: {group_desc}")

        content = _extract_file_content_preview(disk_path, supplementary_dir, max_chars=per_file_budget)
        if content.strip():
            header = f"#### `{rel_path}`"
            all_snippets.append(f"{header}\n{content}")

        validated.append({
            "path": rel_path,
            "reason": reason,
            "is_example_of_group": is_example,
            "group_description": group_desc,
        })

    overview_block = "\n".join(overview_lines)

    # Cap total data-file analysis snippets to stay within model context limit.
    if all_snippets:
        total_chars = sum(len(s) for s in all_snippets)
        if total_chars > MAX_DATA_ANALYSIS_SNIPPET_TOTAL_CHARS:
            budget = MAX_DATA_ANALYSIS_SNIPPET_TOTAL_CHARS
            capped: list[str] = []
            for s in all_snippets:
                if len(s) <= budget:
                    capped.append(s)
                    budget -= len(s)
                else:
                    capped.append(s[:budget])
                    capped.append(
                        f"\n\n[... data-file analysis snippet budget reached at "
                        f"{MAX_DATA_ANALYSIS_SNIPPET_TOTAL_CHARS} chars; further snippets omitted]"
                    )
                    break
            all_snippets = capped
            print(
                f"[info] data_file_analysis: truncated snippets from {total_chars} to "
                f"{MAX_DATA_ANALYSIS_SNIPPET_TOTAL_CHARS} chars",
                flush=True,
            )

    n_validated = len(validated)
    n_selected = len(selected)
    if n_selected > 0:
        print(
            f"[info] data_file_analysis: {n_validated}/{n_selected} files validated on disk, "
            f"{len(all_snippets)} with content previews",
            flush=True,
        )
    else:
        print("[info] data_file_analysis: no supplementary data files to analyze (AI found none)", flush=True)

    return {
        "overview": overview,
        "selected_files": validated,
        "overview_block": overview_block,
        "snippets": all_snippets,
    }


def _looks_like_local_pdf(value: str) -> bool:
    path = Path(value)
    if path.suffix.lower() == ".pdf" and path.exists():
        return True
    return False


def _slug_from_file_name(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_").lower()
    return cleaned or "local_pdf_article"


def _print_related_work_summary(related_payload: dict) -> None:
    """Print a concise summary of the related papers found."""
    papers = related_payload.get("papers", [])
    if not papers:
        print("[info] No related papers to summarize.")
        return

    print("-" * 40)
    print(f"RELATED WORK SUMMARY (Total: {len(papers)})")
    print("-" * 40)
    
    # Group by source
    by_source: dict[str, int] = {}
    for p in papers:
        src = p.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    
    src_line = ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items()))
    print(f"Sources: {src_line}")
    
    # Find newest and oldest
    years = [p.get("year") for p in papers if isinstance(p.get("year"), int)]
    if years:
        print(f"Date Range: {min(years)} - {max(years)}")
    
    # List top 5 titles as a highlight
    print("\nHighlights:")
    for p in papers[:5]:
        title = p.get("title", "Unknown Title")
        if len(title) > 80:
            title = title[:77] + "..."
        print(f"  • {title}")
    
    if len(papers) > 5:
        print(f"  ... and {len(papers) - 5} more.")
    print("-" * 40)


def _coerce_relevance_int(val: object, default: int) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _normalize_one_related_report(
    r: object,
    *,
    paper: dict,
) -> dict:
    """
    Coerce a single model output to {summary, relevance_score, should_read_full, target_focused_narrative}.
    Models sometimes return plain strings (one summary per list slot) or partial dicts.
    """
    default_rel = _coerce_relevance_int(paper.get("relevance_score", 0), 0)
    if r is None:
        return {
            "summary": "",
            "relevance_score": default_rel,
            "should_read_full": False,
            "target_focused_narrative": "",
        }
    if isinstance(r, str):
        s = r.strip()
        return {
            "summary": s,
            "relevance_score": default_rel,
            "should_read_full": False,
            "target_focused_narrative": s,
        }
    if isinstance(r, dict):
        summ = r.get("summary", "")
        if not isinstance(summ, str):
            summ = str(summ) if summ is not None else ""
        tfn = (
            r.get("target_focused_narrative")
            or r.get("narrative")
            or r.get("target_focused")
            or ""
        )
        if not isinstance(tfn, str):
            tfn = str(tfn) if tfn is not None else ""
        tfn = tfn.strip()
        if not tfn:
            tfn = summ.strip()
        srf = r.get("should_read_full", False)
        if isinstance(srf, str):
            srf = srf.lower() in ("true", "yes", "1")
        return {
            "summary": summ.strip(),
            "relevance_score": _coerce_relevance_int(r.get("relevance_score"), default_rel),
            "should_read_full": bool(srf),
            "target_focused_narrative": tfn,
        }
    s = str(r).strip() if r is not None else ""
    return {
        "summary": s,
        "relevance_score": default_rel,
        "should_read_full": False,
        "target_focused_narrative": s,
    }


def _coerce_to_report_list(reports_value: object) -> list[object]:
    if reports_value is None:
        return []
    if isinstance(reports_value, list):
        return list(reports_value)
    return [reports_value]


def _align_and_normalize_batch_reports(
    raw_reports: list[object], batch: list[dict]
) -> list[dict]:
    """Length-align to batch, then normalize each item so `.get` is never used on a str."""
    n = len(batch)
    if n == 0:
        return []
    if len(raw_reports) < n:
        raw_list = list(raw_reports) + [None] * (n - len(raw_reports))
    elif len(raw_reports) > n:
        raw_list = raw_reports[:n]
    else:
        raw_list = list(raw_reports)
    return [_normalize_one_related_report(raw_list[i], paper=batch[i]) for i in range(n)]


def _paper_stem_key(p: dict) -> str:
    """Identity for matching the same work across shortlist, reports, and search results."""
    doi = (p.get("doi") or "").strip().lower()
    u = (p.get("url") or "").strip().lower()
    pdfu = (p.get("pdf_url") or "").strip().lower()
    t = re.sub(r"\s+", " ", (p.get("title") or "").strip().lower())[:200]
    return f"{doi}|{u}|{pdfu}|{t}"


def _not_in_shortlist_candidates(
    unique_paper_dicts: list[dict], shortlist: list[dict]
) -> list[dict]:
    want = {_paper_stem_key(p) for p in shortlist if isinstance(p, dict)}
    return [p for p in unique_paper_dicts if _paper_stem_key(p) not in want]


def _batch_papers_merged_target_related(sorted_papers: list[dict]) -> list[list[dict]]:
    """
    Batch shortlist for one LLM call: target block + per-paper context + long JSON,
    so limits are stricter than the old pre-merge scratch-only batching.
    """
    has_main = any(
        isinstance(p.get("reference_fulltext_excerpt"), str) and p.get("reference_fulltext_excerpt", "").strip()
        for p in sorted_papers
    )
    if has_main:
        per_batch_max = 150_000
        max_papers = 6
    else:
        per_batch_max = 50_000
        max_papers = 8
    subbatches: list[list[dict]] = []
    cur: list[dict] = []
    ch = 0
    for p in sorted_papers:
        ex = p.get("reference_fulltext_excerpt") or ""
        ex = ex if isinstance(ex, str) else ""
        ab = p.get("abstract") or ""
        piece = 2_500 + len(ab) + min(len(ex), RELATED_PDF_EXTRACT_MAX_CHARS)
        if cur and (ch + piece > per_batch_max or len(cur) >= max_papers):
            subbatches.append(cur)
            cur, ch = [], 0
        cur.append(p)
        ch += piece
    if cur:
        subbatches.append(cur)
    return subbatches


def _build_target_paper_block_for_rel(
    target_title: str,
    target_abstract: str,
    target_problems: list[str],
) -> str:
    pl = "\n".join(f"- {x}" for x in (target_problems or [])) or "- (none given)"
    return (
        f"## Target paper (relate every candidate below to THIS work)\n"
        f"**Title:** {target_title or '(unknown)'}\n\n"
        f"**Abstract:**\n{target_abstract or '(not provided; use the candidate list only where needed)'}\n\n"
        f"**Target problems / focus:**\n{pl}\n"
    )


def _generate_related_work_reports(
    *,
    papers: list[dict],
    target_title: str,
    target_abstract: str,
    target_problems: list[str],
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int,
) -> list[dict]:
    """
    One `generate_json` per batch: for each shortlist item, a scratch summary, scores, and
    a target-focused 1-2-paragraph narrative (all in a single call with the target block in prompt).
    """
    if not papers:
        return []

    sorted_papers = sorted(
        papers,
        key=lambda x: (x.get("topic_tag", "General"), -x.get("relevance_score", 0)),
    )
    target_block = _build_target_paper_block_for_rel(target_title, target_abstract, target_problems)
    batches = _batch_papers_merged_target_related(sorted_papers)
    nbatch = len(batches)
    print(
        f"[stage] shortlist_merged_rel_reports (scratch+target_narrative in one call) "
        f"papers={len(papers)} batches={nbatch}"
    )

    reports: list[dict] = []
    for idx, batch in enumerate(batches, 1):
        topics = sorted({p.get("topic_tag", "General") for p in batch})
        print(f"[info] [{idx}/{nbatch}] merged_batch topics={', '.join(topics)} size={len(batch)}")

        paper_blocks: list[str] = []
        for j, p in enumerate(batch):
            ex = p.get("reference_fulltext_excerpt")
            ex_s = (ex or "") if isinstance(ex, str) else ""
            if len(ex_s) > RELATED_PDF_EXTRACT_MAX_CHARS:
                ex_s = ex_s[:RELATED_PDF_EXTRACT_MAX_CHARS] + "\n[… excerpt truncated …]\n"
            ab = (p.get("abstract") or "")[:4000]
            paper_blocks.append(
                f"### CANDIDATE {j}\n"
                f"Title: {p.get('title', 'Unknown')}\n"
                f"Topic / tag: {p.get('topic_tag', 'General')}\n"
                f"Abstract (online / metadata):\n{ab}\n\n"
                f"Main-text excerpt (if any; plain text from shortlist PDF, not layout-perfect):\n"
                f"{ex_s.strip() if ex_s.strip() else '(none; rely on abstract for this item.)'}"
            )
        u_tail = (
            f"\n\nFor each CANDIDATE 0..{len(batch) - 1} in order, the model must return one object in `reports` with:\n"
            "- `summary` (string): one paragraph — relevance to the **target** above, key methods/results, support or tension. "
            "If main-text excerpt is present, lean on it over the abstract.\n"
            "- `relevance_score` (int, 1-10): 9-10=directly overlapping problem and methods, 6-8=useful comparison, "
            "4-5=background context, 1-3=marginal. Use the original score from filtering as a reference.\n"
            "- `should_read_full` (bool): true only if the PDF is genuinely needed to verify a critical claim (method details, exact numbers, control conditions) that the abstract/excerpt cannot confirm.\n"
            "- `target_focused_narrative` (string): 1-2 paragraphs stressing **only** what matters for comparison with the **target** (methods, regime, results, limits). "
            "Do not recapitulate the whole field.\n\n"
            "Return JSON: { \"reports\": [ ... ] } with the same count and order as the candidates; each object has "
            "summary, relevance_score, should_read_full, target_focused_narrative."
        )
        user_prompt = target_block + "\n" + "\n\n".join(paper_blocks) + u_tail
        sp = (
            "You are assisting a single-paper scientific assessment. A TARGET is given first; shortlist candidates follow. "
            f"For each candidate you produce a short summary plus a dense target-focused narrative (topics: {', '.join(topics)}). "
            "Be consistent: assign relevance_score by how directly the candidate informs evaluation of the target's claims. "
            "Output valid JSON only per the user message."
        )
        try:
            if is_gemini:
                client: GeminiClient | OpenAIClient = GeminiClient(
                    api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
                )
            else:
                client = OpenAIClient(
                    api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
                )
            res = client.generate_json(system_prompt=sp, user_prompt=user_prompt)
            if not isinstance(res, dict):
                res = {}
            raw_reports = _coerce_to_report_list(res.get("reports"))
            norm = _align_and_normalize_batch_reports(raw_reports, batch)
            for p, part in zip(batch, norm):
                reports.append({**p, **part})
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] shortlist_merged batch {idx} failed: {exc}")
            for p in batch:
                reports.append(
                    {
                        **p,
                        "summary": "Failed to generate summary.",
                        "relevance_score": p.get("relevance_score", 0),
                        "should_read_full": False,
                        "target_focused_narrative": "",
                    }
                )
    return reports


def _abstract_to_single_paragraph(*, text: str, max_chars: int) -> str:
    """At most one paragraph: first block if split by blank lines, then whitespace-collapsed, capped."""
    t = (text or "").strip()
    if not t:
        return "No abstract in metadata."
    part = re.split(r"\n\s*\n", t, maxsplit=1)[0].strip()
    part = re.sub(r"[\n\r\t]+", " ", part)
    part = re.sub(r"  +", " ", part)
    if len(part) > max_chars:
        part = part[: max_chars - 1] + "…"
    return part


def _enrich_not_selected_brief_abstracts(
    candidates: list[dict],
    target_title: str,
    *,
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int,
) -> list[dict]:
    """
    One-sentence line per not-shortlisted hit (model output), from at most one paragraph
    of metadata abstract per item in the user prompt. Single `generate_json` for the whole list.
    When API fails or mock, uses a local one-line heuristic.
    """
    _in_para = 2000
    out: list[dict] = []
    for p in candidates:
        row = dict(p)
        row["brief_abstract"] = ""
        out.append(row)
    if not out:
        return out
    if is_gemini:
        client2: GeminiClient | OpenAIClient = GeminiClient(
            api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
        )
    else:
        client2 = OpenAIClient(
            api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url
        )
    blocks: list[str] = []
    for m, c in enumerate(out):
        t = c.get("title", "No title")
        a = _abstract_to_single_paragraph(
            text=str(c.get("abstract") or ""),
            max_chars=_in_para,
        )
        y = c.get("year", "")
        src = c.get("source", "")
        # One block per paper: a single short paragraph in the user message
        blocks.append(
            f"[{m}] Source: {src}; year: {y}. Title: {t}. "
            f"Abstract (at most one paragraph): {a}"
        )
    big = (
        f"Target paper under assessment: «{target_title}».\n"
        "The following are search results **not** on the shortlist. For each, the abstract above is the only evidence.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nReturn JSON: {{ 'briefs': [ {{ 'm': 0, 'one_sentence': '...' }}, ... ] }} with one entry for every m from 0 to "
        f"{len(out) - 1} in order. "
        "Each `one_sentence` must be a **single** sentence, factual, only what is supported by that item’s abstract, no new claims."
    )
    try:
        rj = client2.generate_json(
            system_prompt="You condense one paragraph of metadata per work into one sentence. Do not invent content beyond the given text.",
            user_prompt=big,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] not_selected brief generation failed: {exc}")
        for row in out:
            ab = _abstract_to_single_paragraph(text=str(row.get("abstract") or ""), max_chars=500)
            t = (row.get("title") or "")[:120]
            row["brief_abstract"] = f"{t}: {ab}" if ab and ab != "No abstract in metadata." else t
        return out
    if not isinstance(rj, dict):
        rj = {}
    briefs = rj.get("briefs", rj.get("items", []))
    by_m: dict[int, str] = {}
    for b in briefs:
        if not isinstance(b, dict):
            continue
        m = b.get("m", b.get("index"))
        txt = (
            (b.get("one_sentence") or b.get("one_or_two_sentences") or b.get("text") or "")
            .strip()
        )
        if isinstance(m, int) and txt:
            by_m[m] = txt
    for m, row in enumerate(out):
        if m in by_m:
            row["brief_abstract"] = by_m[m]
        else:
            ab = (row.get("abstract") or "")[:500].strip()
            row["brief_abstract"] = ((row.get("title") or "") + (" — " + ab if ab else ""))[:600]
    return out


def _write_figure_table_assessment_outputs(article_dir: Path, ft_assessment: dict, model_name: str, round_index: int) -> None:
    """Write figure/table assessment results as JSON trace + markdown report.
    ``round_index`` is the global sequential run number for this paper (1, 2, 3...)."""
    from assessment.report_writer import _pipeline_header  # noqa: PLC0415

    suffix = f"_{round_index}"

    # Raw JSON trace
    (article_dir / f"figure_table_assessment{suffix}.json").write_text(
        json.dumps(ft_assessment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Formatted markdown report
    analysis = ft_assessment.get("figure_table_analysis", "")
    files_for_final = ft_assessment.get("files_for_final_step", [])
    simplified_data = ft_assessment.get("simplified_data_block", "")

    lines: list[str] = [
        f"# Figure/Table Data Assessment Report — {_pipeline_header()} model={model_name}",
        "",
        "",
    ]

    # Truncation status banner
    any_truncated = ft_assessment.get("any_data_truncated", False)
    total_files = ft_assessment.get("total_files_selected", 0)
    clipped_count = ft_assessment.get("clipped_files_count", 0)
    threshold_B = ft_assessment.get("uniform_clip_threshold_B", 0)
    if any_truncated:
        lines.append("!! DATA TRUNCATION OCCURRED !!")
        lines.append("")
        lines.append(
            f"- Total files selected by AI: {total_files}\n"
            f"- Uniform clip threshold B: {threshold_B:,} chars\n"
            f"- Files clipped: {clipped_count}\n"
            f"- Files fully included: {total_files - clipped_count}\n"
        )
        lines.append(
            "See `[!! TRUNCATED ... !!]` or `[!! DATA CLIPPED ... !!]` markers in the "
            "source data file contents below. The AI was explicitly warned about this "
            "and instructed not to mistake truncation for incomplete paper data."
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Per-Panel / Per-Table Analysis")
    lines.append("")
    lines.append(analysis or "(No analysis was produced.)")
    lines.append("")
    if files_for_final:
        lines.append("## Files Forwarded to Final Assessment")
        lines.append("")
        for entry in files_for_final:
            path = entry.get("path", "?")
            reason = entry.get("reason", "")
            lines.append(f"- `{path}` — {reason}")
        lines.append("")
    if simplified_data:
        lines.append("## Simplified Source Data for Final Review")
        lines.append("")
        lines.append(simplified_data)
        lines.append("")

    (article_dir / f"figure_table_assessment_report{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[info] figure_table_assessment: wrote report{suffix} ({len(analysis)} chars analysis, "
        f"{len(files_for_final)} files flagged for final step)",
        flush=True,
    )


def _write_derivation_assessment_outputs(article_dir: Path, deriv_assessment: dict, model_name: str, round_index: int) -> None:
    """Write derivation assessment results as JSON trace + markdown report.
    ``round_index`` is the global sequential run number for this paper (1, 2, 3...)."""
    from assessment.report_writer import _pipeline_header  # noqa: PLC0415

    suffix = f"_{round_index}"

    # Raw JSON trace
    json_path = article_dir / f"derivation_assessment{suffix}.json"
    json_path.write_text(
        json.dumps(deriv_assessment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Formatted markdown report
    analysis = deriv_assessment.get("derivation_analysis", "")
    files_for_final = deriv_assessment.get("files_for_final_step", [])
    simplified_block = deriv_assessment.get("simplified_derivation_block", "")

    lines: list[str] = [
        f"# Derivation Assessment Report — {_pipeline_header()} model={model_name}",
        "",
    ]
    lines.append("## Per-Step Derivation Analysis")
    lines.append("")
    lines.append(analysis or "(No analysis was produced.)")
    lines.append("")
    if files_for_final:
        lines.append("## Files Forwarded to Final Assessment")
        lines.append("")
        for entry in files_for_final:
            path = entry.get("path", "?")
            reason = entry.get("reason", "")
            lines.append(f"- `{path}` — {reason}")
        lines.append("")
    if simplified_block:
        lines.append("## Simplified Derivation Block for Final Review")
        lines.append("")
        lines.append(simplified_block)
        lines.append("")

    report_path = article_dir / f"derivation_assessment_report{suffix}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[info] derivation_assessment: wrote report{suffix} ({len(analysis)} chars analysis, "
        f"{len(files_for_final)} files flagged for final step)",
        flush=True,
    )


def _write_pipeline_assessment_outputs(article_dir: Path, pipeline_assessment: dict, model_name: str, round_index: int) -> None:
    """Write pipeline assessment results as JSON trace + markdown report.
    ``round_index`` is the global sequential run number for this paper (1, 2, 3...)."""
    from assessment.report_writer import _pipeline_header  # noqa: PLC0415

    suffix = f"_{round_index}"

    # Raw JSON trace
    json_path = article_dir / f"pipeline_assessment{suffix}.json"
    json_path.write_text(
        json.dumps(pipeline_assessment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Formatted markdown report
    analysis = pipeline_assessment.get("pipeline_analysis", "")
    files_for_final = pipeline_assessment.get("files_for_final_step", [])
    simplified_block = pipeline_assessment.get("simplified_pipeline_block", "")

    pipeline_tier = pipeline_assessment.get("pipeline_tier", "3")
    lines: list[str] = [
        f"# Pipeline/Software Assessment Report — {_pipeline_header()} model={model_name}",
        "",
        f"**Pipeline tier:** {pipeline_tier}",
        "",
    ]
    lines.append("## Analysis")
    lines.append("")
    lines.append(analysis or "(No analysis was produced.)")
    lines.append("")
    if files_for_final:
        lines.append("## Files Forwarded to Final Assessment")
        lines.append("")
        for entry in files_for_final:
            path = entry.get("path", "?")
            reason = entry.get("reason", "")
            lines.append(f"- `{path}` — {reason}")
        lines.append("")
    if simplified_block:
        lines.append("## Simplified Pipeline Block for Final Review")
        lines.append("")
        lines.append(simplified_block)
        lines.append("")

    report_path = article_dir / f"pipeline_assessment_report{suffix}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[info] pipeline_assessment: wrote report{suffix} ({len(analysis)} chars analysis, "
        f"{len(files_for_final)} files flagged for final step, tier={pipeline_tier})",
        flush=True,
    )


def _build_mock_assessment(*, metadata: dict) -> dict:
    title = metadata.get("title", "Unknown paper")
    return {
        "paper_understanding": {
            "methods": f"Mock summary of methods for {title}.",
            "results": "Mock summary of main results.",
            "conclusions": "Mock summary of conclusions.",
        },
        "related_work_summary": "Mock mode: related work analysis was not generated by LLM.",
        "q1_derivations": [],
        "q2_experimental_and_processing_methods": [],
        "q3_explanation_uniqueness": {
            "is_unique_explanation": None,
            "analysis": "Mock mode placeholder.",
            "alternative_explanations": [],
            "evidence": [],
        },
        "q4_data_and_figure_integrity": {
            "issues_found": [],
            "processing_artifact_risks": [],
            "analysis": "Mock mode placeholder.",
            "evidence": [],
        },
        "q5_replications_and_related_systems": {
            "replication_status": "unknown",
            "related_system_works": [],
            "analysis": "Mock mode placeholder.",
            "evidence": [],
        },
        "q6_data_code_openness": {
            "data_open_sourced": None,
            "code_open_sourced": None,
            "reproducibility_what_is_shared": "Not evaluated (mock).",
            "details": "Mock mode placeholder.",
            "evidence": [],
        },
        "q7_problem_importance": {
            "problem_importance_assessment": "Mock mode placeholder.",
            "importance_level": "niche_or_low_impact",
            "analysis": "Mock mode placeholder.",
            "evidence": [],
        },
        "q8_technical_method_advance": {
            "method_novelty_assessment": "Mock mode placeholder.",
            "advance_level": "routine_application",
            "analysis": "Mock mode placeholder.",
            "evidence": [],
        },
        "q9_area_change": {
            "area_change_assessment": "Mock mode placeholder.",
            "change_level": "incremental",
            "analysis": "Mock mode placeholder.",
            "evidence": [],
        },
        "q10_community_impact": {
            "community_impact_assessment": "Mock mode placeholder.",
            "impact_type": "minimal",
            "textbook_or_commercial_potential": "Not evaluated (mock).",
            "analysis": "Mock mode placeholder.",
            "evidence": [],
        },
        "q11_expansion_potential": {
            "expansion_potential_assessment": "Mock mode placeholder.",
            "follow_up_level": "niche",
            "analysis": "Mock mode placeholder.",
            "evidence": [],
        },
        "final_reasoning_comment": "Mock mode was used for smoke testing; no scientific judgment was produced.",
        "scores": {
            "reliability": {
                "derivation_rigor": 50,
                "experimental_validity": 50,
                "evidence_uniqueness": 50,
                "data_integrity": 50,
                "replication_support": 50,
                "openness": 50,
            },
            "novelty_impact": {
                "problem_importance": 50,
                "technical_method_advance": 50,
                "area_change": 50,
                "community_impact": 50,
                "expansion_potential": 50,
            },
        },
        "confidence": 0,
        "limitations": ["Mock mode output only."],
        "citation_traces": [],
    }


def _clean_url(raw: str) -> str:
    """Strip trailing punctuation/non-URL chars from regex-extracted URLs."""
    u = raw.strip()
    # Unescape common LaTeX escapes that Marker may preserve in markdown URLs.
    u = u.replace("\\_", "_").replace("\\&", "&").replace("\\%", "%").replace("\\#", "#")
    u = u.rstrip(".,;:!?)\"'>$")
    u = u.strip()
    return u


def _looks_like_direct_supplementary_asset(href: str) -> bool:
    """Heuristic: URL points at a file-like supplementary asset, without hardcoding a specific paper or publisher home URL."""
    low = href.lower()
    if any(seg in low for seg in (".pdf", ".xlsx", ".xls", ".zip", ".tar.gz", ".tgz", ".csv", ".tar")):
        return True
    if "moesm" in low:
        return True
    if "download-xlsx" in low or "download.pdf" in low or "/mediaobjects/" in low or "/esm/" in low:
        return True
    return False


def _material_filename_suffix_from_url(url: str, idx: int) -> tuple[str, str]:
    """Best-effort (filename, suffix) from URL path, including .tar.gz."""
    path = urlparse(url).path or ""
    raw_name = Path(path).name.strip() or f"declared_material_{idx}.bin"
    low = raw_name.lower()
    for ext in (".tar.gz", ".tgz", ".zip", ".pdf", ".xlsx", ".xls", ".csv", ".tsv", ".json", ".tar"):
        if low.endswith(ext):
            return raw_name, ext
    return raw_name, Path(raw_name).suffix.lower()


def _clean_doi_path_fragment(fragment: str) -> str:
    t = fragment.strip()
    for stop in (")", "]", ">", "'", '"', "`", "<"):
        if stop in t:
            t = t[: t.index(stop)]
    t = t.split()[0] if t else t
    return t.rstrip(".,;).]\"").strip()


def _extract_doi_landing_url_from_text(text: str) -> str | None:
    """
    Get a single http(s) URL suitable for GETting article HTML, from common DOI patterns
    in converted article markdown (doi.org, dx.doi.org, and publisher /doi/... paths).
    """
    if not text.strip():
        return None

    # 1) Explicit doi.org (and dx.doi.org) links; stop at delimiters.
    m = re.search(
        r"(?i)https?://(?:dx\.)?doi\.org/(\S+)",
        text,
    )
    if m:
        return f"https://doi.org/{_clean_doi_path_fragment(m.group(1))}"

    # 2) Publisher pages: https://.../.../doi/<doi>
    for m2 in re.finditer(
        r'(?i)https?://[^\s<>\[\]()\]"\'#]+/doi/[:/]?(10\.\S+)',
        text,
    ):
        u = f"https://doi.org/{_clean_doi_path_fragment(m2.group(1))}"
        if re.match(r"^https://doi.org/10\.\d{4,9}/", u):
            return u

    # 3) Host/path without scheme, e.g. `science.org/doi/10.1126/...` (common in PDF->md footers)
    m3b = re.search(
        r"(?i)\b((?:[a-z0-9-]+\.)+[a-z]+)/doi/[:/]?(10\.\d{4,9}/\S+?)(?=[\s<>\])\"',]|$)",
        text,
    )
    if m3b:
        return f"https://doi.org/{_clean_doi_path_fragment(m3b.group(2))}"

    # 4) Line with "doi" and bare 10. .../...  (e.g. footers)
    for line in text.splitlines():
        if re.search(r"(?i)\bdoi\b", line):
            m4 = re.search(r"(\b10\.\d{4,9}/[A-Z0-9._-]+)", line, re.I)
            if m4:
                return f"https://doi.org/{_clean_doi_path_fragment(m4.group(1))}"

    return None


def _article_landing_url_for_materials(*, article_dir: Path, resolved_url: str) -> str:
    """
    Return an HTTP landing-page URL suitable for fetching supplementary link lists.

    - If ``resolved_url`` is already an HTTP URL, return it (stripping ``.pdf`` suffix).
    - If ``resolved_url`` is a file path, attempt to extract a DOI from the converted
      article markdown and construct the landing URL from it.
    - If the markdown fallback fails, return ``resolved_url`` unchanged.
    """
    base = (urldefrag(resolved_url)[0] or resolved_url).strip()
    if base.startswith("http://") or base.startswith("https://"):
        # Strip .pdf suffix to get the HTML landing page.
        lower = base.lower()
        if lower.endswith(".pdf"):
            pdf_stripped = base[: -len(".pdf")]
            # Only use the stripped version if it's non-empty and different.
            if pdf_stripped and pdf_stripped != base:
                print(f"[info] material_landing: stripped .pdf from resolved_url")
                base = pdf_stripped
        return base
    # Non-HTTP (file path): extract DOI from article markdown.
    for rel in ("converted/article.md", "article.md"):
        p = article_dir / rel
        if not p.is_file():
            continue
        u = _extract_doi_landing_url_from_text(p.read_text(encoding="utf-8", errors="ignore"))
        if u:
            print(f"[info] material_landing_from_doi_artifact=resolved path={p.as_posix()}")
            return u
    return base


def _fetch_material_hrefs_from_landing_page(*, landing_url: str, timeout_seconds: int) -> tuple[list[str], str]:
    """
    Fetch the article HTML and collect supplementary/source-data style links.

    This matches what standalone tests do when they GET the live page: converted article.md from PDFs
    usually does not contain the raw publisher hrefs, so markdown-only collection often finds nothing.
    """
    if not (landing_url.startswith("http://") or landing_url.startswith("https://")):
        return [], "skipped_not_http"
    try:
        response = requests.get(
            landing_url,
            timeout=timeout_seconds,
            allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
    except OSError as exc:  # noqa: BLE001
        return [], f"request_failed:{exc}"

    if response.status_code >= 400:
        return [], f"http_{response.status_code}"

    final = response.url or landing_url
    host = urlparse(final).netloc.lower()

    if "nature.com" in host:
        from extraction.extractors.nature import (  # noqa: WPS433
            _collect_supplementary_urls,
        )

        return _collect_supplementary_urls(final, response.text), "nature_article_html"

    from bs4 import BeautifulSoup  # noqa: WPS433

    soup = BeautifulSoup(response.text, "html.parser")
    generic: list[str] = []
    for node in soup.select("a[href]"):
        href = str(node.get("href", "")).strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        absu = urljoin(final, href)
        low = absu.lower()
        if _looks_like_direct_supplementary_asset(low) or any(
            k in low
            for k in (
                "supplementary",
                "supplemental",
                "source data",
                "data availability",
                "github.com",
                "figshare",
                "zenodo",
            )
        ):
            generic.append(absu)
    return list(dict.fromkeys(generic)), "generic_html_links"


def _iter_local_main_text_pdf_paths(article_dir: Path) -> list[Path]:
    """
    Paths to the main article PDF when it already exists on disk (so we can skip
    re-downloading the same file into supplementary/).
    """
    seen: set[Path] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        if not p.is_file() or p.suffix.lower() != ".pdf":
            return
        try:
            r = p.resolve()
        except OSError:
            return
        if "supplementary" in r.parts:
            return
        if r in seen:
            return
        seen.add(r)
        out.append(r)

    add(article_dir / "article.pdf")
    meta_path = article_dir / "metadata.json"
    if not meta_path.is_file():
        return out
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    ru = data.get("resolved_url")
    if not isinstance(ru, str) or not ru.strip():
        return out
    low = ru.lower().split(":", 1)[0]
    if low in ("http", "https"):
        return out
    p = Path(ru.strip())
    add(p)
    return out


def _sha256_hex_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str | None:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def _main_text_pdf_digests_for_dedup(article_dir: Path) -> set[str]:
    digests: set[str] = set()
    for p in _iter_local_main_text_pdf_paths(article_dir):
        hx = _sha256_hex_of_file(p)
        if hx:
            digests.add(hx)
    return digests


MATERIALS_DOWNLOAD_STAMP = ".materials_download_fingerprint"


def _urls_from_markdown_for_fingerprint(
    source_text: str, *, canonical_resolved_url: str
) -> list[str]:
    """
    HTTP(S) URLs used for the materials-download skip stamp. Does not use full
    document text so re-converting the main PDF (which rewrites article.md) does
    not invalidate a valid stamp.
    """
    if not (source_text or "").strip():
        return []
    absolute_url_pattern = r"https?://[^\s)\]\"'>]+"
    all_urls: list[str] = list(re.findall(absolute_url_pattern, source_text))
    for raw_target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source_text):
        candidate = raw_target.strip().strip("<>").rstrip(".,;)")
        if not candidate:
            continue
        if candidate.startswith("http://") or candidate.startswith("https://"):
            all_urls.append(candidate)
        elif candidate.startswith("/"):
            all_urls.append(urljoin(canonical_resolved_url, candidate))
    return sorted(set(all_urls))


def _materials_download_input_fingerprint(
    article_dir: Path,
    *,
    canonical_resolved_url: str,
    source_text: str,
    landing_page_urls: list[str] | None = None,
) -> str:
    """Hash of landing URL, optional DOI, markdown URLs, and landing-page link set."""
    h = hashlib.sha256()
    h.update((canonical_resolved_url or "").encode("utf-8", errors="replace"))
    h.update(b"\0")
    meta_path = article_dir / "metadata.json"
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                ru = data.get("resolved_url")
                if isinstance(ru, str) and ru.strip():
                    h.update(b"resolved\0")
                    h.update(ru.encode("utf-8", errors="replace"))
                doi = data.get("doi")
                if isinstance(doi, str) and doi.strip():
                    h.update(b"doi\0")
                    h.update(doi.lower().encode())
        except json.JSONDecodeError:
            pass
    h.update(b"\0urls\0")
    for u in _urls_from_markdown_for_fingerprint(source_text, canonical_resolved_url=canonical_resolved_url):
        h.update(u.encode("utf-8", errors="replace"))
        h.update(b"\1")
    h.update(b"\0landing\0")
    for u in sorted(set(landing_page_urls or [])):
        h.update(u.encode("utf-8", errors="replace"))
        h.update(b"\1")
    return h.hexdigest()


def _write_materials_download_stamp(supplementary_dir: Path, fingerprint: str) -> None:
    try:
        (supplementary_dir / MATERIALS_DOWNLOAD_STAMP).write_text(fingerprint, encoding="utf-8")
    except OSError:
        pass


def _dedupe_pdf_against_main_text(
    data: bytes,
    *,
    main_digests: set[str],
) -> bool:
    """
    If data is byte-identical to a local main-article PDF, return True (caller should not save to supplementary).
    """
    if not data or not main_digests:
        return False
    return hashlib.sha256(data).hexdigest() in main_digests


def _attempt_download_declared_materials(
    *,
    article_dir: Path,
    resolved_url: str,
    timeout_seconds: int,
    headless: bool,
    browser_channel: str,
    real_browser_mode: bool,
) -> str:
    """Try downloading supplementary/source-data/code links declared in markdown."""
    url_events: list[dict[str, Any]] = []
    statement_scan = scan_availability_statement(load_combined_article_markdown(article_dir))
    raw_resolved = (urldefrag(resolved_url)[0] or resolved_url).strip()
    canonical_resolved_url = _article_landing_url_for_materials(article_dir=article_dir, resolved_url=raw_resolved)
    supplementary_dir = article_dir / "supplementary"
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    existing_material_files = [
        p
        for p in supplementary_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".pdf", ".xls", ".xlsx", ".csv", ".tsv", ".zip", ".tar", ".gz"}
    ]

    md_candidates = [article_dir / "article.md", article_dir / "converted" / "article.md"]
    source_text = ""
    for md_path in md_candidates:
        if md_path.exists():
            source_text = md_path.read_text(encoding="utf-8", errors="ignore")
            if source_text.strip():
                break

    has_http_landing = canonical_resolved_url.startswith("http://") or canonical_resolved_url.startswith("https://")
    page_urls, page_source = [], "skipped_not_http"
    if has_http_landing:
        page_urls, page_source = _fetch_material_hrefs_from_landing_page(
            landing_url=canonical_resolved_url, timeout_seconds=timeout_seconds
        )

    download_input_fp = _materials_download_input_fingerprint(
        article_dir,
        canonical_resolved_url=canonical_resolved_url,
        source_text=source_text,
        landing_page_urls=page_urls,
    )
    stamp_path = supplementary_dir / MATERIALS_DOWNLOAD_STAMP
    if stamp_path.is_file() and stamp_path.read_text(encoding="utf-8", errors="ignore").strip() == download_input_fp:
        print(
            "[info] skip download_declared_materials: same link-discovery fingerprint as the last run "
            f"(landing URL + DOI/resolved local path from metadata + markdown URLs + landing-page links; "
            f"stamp: {supplementary_dir.as_posix()}/{MATERIALS_DOWNLOAD_STAMP}).",
            flush=True,
        )
        return (
            "skipped: supplementary download (inputs unchanged; remove "
            f"{MATERIALS_DOWNLOAD_STAMP} under supplementary/ to force a fresh run)"
        )

    def _finalize_stage(result_note: str) -> str:
        write_audit_file(supplementary_dir, statement=statement_scan, url_events=url_events)
        _write_materials_download_stamp(supplementary_dir, download_input_fp)
        return result_note

    if not source_text.strip() and not has_http_landing:
        if existing_material_files:
            return _finalize_stage(
                f"existing_materials_detected={len(existing_material_files)} no_markdown_context"
            )
        return _finalize_stage("no_markdown_context")

    absolute_url_pattern = r"https?://[^\s)\]\"'>]+"
    all_urls: list[str] = []
    if source_text.strip():
        for raw in re.findall(absolute_url_pattern, source_text):
            cleaned = _clean_url(raw)
            if cleaned:
                all_urls.append(cleaned)
        markdown_link_targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", source_text)
        for raw_target in markdown_link_targets:
            candidate = raw_target.strip().strip("<>").rstrip(".,;)")
            if not candidate:
                continue
            if candidate.startswith("http://") or candidate.startswith("https://"):
                all_urls.append(candidate)
            elif candidate.startswith("/"):
                all_urls.append(urljoin(canonical_resolved_url, candidate))
    md_url_count = len(all_urls)
    if page_urls:
        all_urls = list(dict.fromkeys([*all_urls, *page_urls]))
    if has_http_landing:
        print(
            f"[info] material_url_sources=markdown_hits={md_url_count} "
            f"landing_hits={len(page_urls)} landing_source={page_source} merged_raw={len(all_urls)}"
        )
    all_urls = list(dict.fromkeys(all_urls))
    if not all_urls:
        if existing_material_files:
            return _finalize_stage(
                f"existing_materials_detected={len(existing_material_files)} no_urls_found"
            )
        return _finalize_stage("no_urls_found")

    keyword_hits = (
        "supplementary",
        "supplemental",
        "source-data",
        "source_data",
        "dataset",
        "data availability",
        "figshare",
        "zenodo",
        "github.com",
        "code availability",
        "extref",
        "download",
        "attachment",
    )
    candidate_urls: list[str] = []
    preferred_candidate_urls: list[str] = []
    for raw in all_urls:
        u = raw.strip().rstrip(".,;)")
        low = u.lower()
        is_preferred_match = _looks_like_direct_supplementary_asset(low)
        if any(k in low for k in keyword_hits) or is_preferred_match:
            candidate_urls.append(u)
            # Keep the exact successful selection strategy as highest priority.
            if is_preferred_match:
                preferred_candidate_urls.append(u)
    keyword_matched = list(dict.fromkeys(candidate_urls))
    # Expand data-repository links to derive download URLs.
    repo_derived: list[str] = []
    for u in keyword_matched:
        repo_derived.extend(expand_github_material_download_urls(u))
        repo_derived.extend(expand_zenodo_material_download_urls(u))
        repo_derived.extend(expand_figshare_material_download_urls(u))
        repo_derived.extend(expand_dryad_material_download_urls(u))
        repo_derived.extend(expand_osf_material_download_urls(u))
    repo_derived = list(dict.fromkeys(repo_derived))

    # Resolve repository API URLs to actual file download URLs.
    resolved_file_urls: list[str] = []
    for api_url in repo_derived:
        entries = resolve_repository_api(
            api_url=api_url,
            timeout_seconds=timeout_seconds,
        )
        for entry in entries:
            dl = entry.get("url", "")
            if dl:
                resolved_file_urls.append(dl)
    repo_derived = list(dict.fromkeys([*repo_derived, *resolved_file_urls]))

    # Prefer direct publisher file assets, but still try repository derivations.
    if preferred_candidate_urls:
        candidate_urls = list(dict.fromkeys([*preferred_candidate_urls, *repo_derived]))
    else:
        candidate_urls = list(dict.fromkeys([*keyword_matched, *repo_derived]))
    if not candidate_urls:
        if existing_material_files:
            return _finalize_stage(
                f"existing_materials_detected={len(existing_material_files)} no_material_urls_matched"
            )
        return _finalize_stage("no_material_urls_matched")
    max_downloads = 50
    print(
        f"[info] material_link_candidates={len(candidate_urls)} "
        f"(attempts_this_run: up_to_{min(max_downloads, len(candidate_urls))} urls)"
    )

    main_text_pdf_digests = _main_text_pdf_digests_for_dedup(article_dir)
    if main_text_pdf_digests:
        print(
            f"[info] main_text_pdf_dedup: {len(main_text_pdf_digests)} digest(s) from local article PDF(s); "
            f"identical downloads will be skipped (not written to supplementary/)."
        )

    downloaded = 0
    failed = 0
    browser_fallback_downloaded = 0
    skipped_non_material = 0
    skipped_unknown_suffix = 0
    skipped_no_extension = 0
    skipped_duplicate_main_pdf = 0
    skipped_identical_to_existing_file = 0
    saved_filenames: list[str] = []
    download_rate_limiter = RateLimiter(base_delay_seconds=2.0, jitter_seconds=1.0)
    allowed_suffixes = {".pdf", ".xlsx", ".xls", ".zip", ".csv", ".tsv", ".json", ".tar", ".tgz", ".dat", ".log", ".out", ".txt"}
    session_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def _allowed_suffix_for_filename(name: str) -> str | None:
        low = name.lower()
        if low.endswith(".tar.gz"):
            return ".tar.gz"
        suf = Path(name).suffix.lower()
        return suf if suf in allowed_suffixes else None

    browser_client: BrowserClient | None = None
    try:
        for idx, url in enumerate(candidate_urls[:max_downloads], start=1):
            print(f"  [download] [{idx}/{len(candidate_urls[:max_downloads])}] attempting: {url}", flush=True)

            # --- Pre-download suffix check: skip known non-file extensions ---
            url_path = urlparse(url).path.lower()
            unknown_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".ogg", ".flac",
                                ".jpg", ".jpeg", ".png", ".gif", ".svg", ".bmp", ".ico",
                                ".html", ".htm", ".php", ".asp", ".aspx", ".jsp"}
            url_suffix = Path(url_path).suffix
            # Quick heuristic: skip anchor links (#) and URLs with no dot (no file extension)
            if "#" in url and "." not in url_path:
                skipped_no_extension += 1
                fname = url_path.rsplit("/", 1)[-1] if "/" in url_path else url_path
                reason = "no '.' and has '#' (anchor link)"
                print(f"  [download]   -> skipped_no_extension ({reason}, name={fname})", flush=True)
                url_events.append({
                    "url": url,
                    "outcome": "skipped_no_extension",
                    "message": reason,
                    "declared_size_bytes": None,
                    "final_url": url,
                    "inferred_filename": fname,
                    "saved_relative_path": None,
                })
                continue

            if url_suffix in unknown_suffixes:
                skipped_unknown_suffix += 1
                fname = url_path.rsplit("/", 1)[-1] if "/" in url_path else url_path
                print(f"  [download]   -> skipped_unknown_suffix (suffix={url_suffix} name={fname})", flush=True)
                url_events.append(
                    {
                        "url": url,
                        "outcome": "skipped_unknown_suffix",
                        "message": f"URL suffix '{url_suffix}' is not a supported material type.",
                        "declared_size_bytes": None,
                        "final_url": url,
                        "inferred_filename": fname,
                        "saved_relative_path": None,
                    }
                )
                continue

            # --- Determine size threshold based on file type ---
            is_pdf_url = url_suffix == ".pdf" or "pdf" in url_path
            max_bytes_for_url = 100 * 1024 * 1024 if is_pdf_url else 30 * 1024 * 1024

            # Add domain-specific Referer header for repositories that require it.
            request_headers = dict(session_headers)
            ulow = url.lower()

            # --- Zenodo special handling: skip direct HTTP (always 403), go straight to browser ---
            is_zenodo = "zenodo.org" in ulow
            if is_zenodo:
                # Use zenodo_get library which handles Zenodo's API correctly
                zm = re.search(r"files/(.+?)(?:\?|$)", url)  # use original URL, not ulow — preserve case
                zenodo_filename = None
                if zm:
                    from urllib.parse import unquote
                    zenodo_filename = unquote(zm.group(1))
                    # Strip trailing /content from API endpoints
                    if zenodo_filename.lower().endswith("/content"):
                        zenodo_filename = zenodo_filename[:-len("/content")]
                    existing = list(supplementary_dir.rglob(zenodo_filename))
                    if existing:
                        print(f"  [download]   -> zenodo_url (already present: {zenodo_filename})", flush=True)
                        url_events.append({
                            "url": url,
                            "outcome": "skipped_already_present",
                            "message": f"File {zenodo_filename} already exists in supplementary/.",
                            "declared_size_bytes": None,
                            "final_url": url,
                            "inferred_filename": zenodo_filename,
                            "saved_relative_path": zenodo_filename,
                        })
                        continue

                zm2 = re.search(r"zenodo\.org/(?:api/)?records/(\d+)", ulow)
                record_id = zm2.group(1) if zm2 else ""
                if not record_id:
                    print(f"  [download]   -> zenodo_url (could not extract record ID, skipping)", flush=True)
                    url_events.append({
                        "url": url,
                        "outcome": "failed",
                        "message": "Could not extract Zenodo record ID from URL.",
                        "declared_size_bytes": None,
                        "final_url": url,
                        "saved_relative_path": None,
                    })
                    continue

                print(f"  [download]   -> zenodo_url (using zenodo_get library, record={record_id})", flush=True)

                # Require a specific filename — skip record-level URLs (no specific file)
                if not zenodo_filename:
                    print(f"  [download]   -> skipped: no specific filename in URL (record-level URL)", flush=True)
                    url_events.append({"url": url, "outcome": "failed", "message": "No specific filename in Zenodo URL (record-level URL).", "declared_size_bytes": None, "final_url": url, "saved_relative_path": None})
                    continue

                # Size check via Zenodo API record metadata
                _max_z = 100 * 1024 * 1024 if zenodo_filename.lower().endswith(".pdf") or "pdf" in zenodo_filename.lower() else 30 * 1024 * 1024
                _oversized = False
                print(f"  [download]   -> checking size via Zenodo API (threshold={_max_z / (1024*1024):.0f} MiB)...", flush=True)
                try:
                    import httpx as _hx
                    _zr = _hx.get(
                        f"https://zenodo.org/api/records/{record_id}",
                        timeout=15.0,
                    )
                    print(f"  [download]   -> httpx API status={_zr.status_code}", flush=True)
                    if _zr.status_code == 200:
                        _found = False
                        for _zf in (_zr.json().get("files") or []):
                            _api_key = _zf.get("key", "")
                            _api_size = _zf.get("size", 0)
                            print(f"  [download]   ->   API file: key={_api_key!r} size={_api_size} ({_api_size/1024/1024:.1f} MiB)", flush=True)
                            if _api_key == zenodo_filename or (_api_key or "").endswith("/" + zenodo_filename):
                                _found = True
                                if _api_size and _api_size > _max_z:
                                    _zs_str = f"{_api_size / (1024*1024):.1f} MiB"
                                    _zl_str = f"{_max_z / (1024*1024):.0f} MiB"
                                    print(f"  [download]   -> skipped_oversized ({_zs_str} exceeds {_zl_str} limit)", flush=True)
                                    url_events.append({"url": url, "outcome": "skipped_oversized", "message": f"File is {_zs_str}, exceeds {_zl_str} limit.", "declared_size_bytes": _api_size, "final_url": url, "inferred_filename": zenodo_filename, "saved_relative_path": None})
                                    _oversized = True
                                else:
                                    print(f"  [download]   -> size OK ({_api_size/1024/1024:.1f} MiB)", flush=True)
                                break
                        if not _found:
                            # Fallback: check total record size
                            _total = sum((_f.get("size") or 0) for _f in (_zr.json().get("files") or []))
                            _tot_str = f"{_total / (1024*1024):.1f} MiB" if _total > 1024*1024 else f"{_total / 1024:.1f} KiB"
                            print(f"  [download]   -> file {zenodo_filename!r} not in API metadata, total={_tot_str}", flush=True)
                            if _total > _max_z:
                                _zl_str = f"{_max_z / (1024*1024):.0f} MiB"
                                print(f"  [download]   -> skipped (total {_tot_str} exceeds {_zl_str} limit)", flush=True)
                                url_events.append({"url": url, "outcome": "skipped_oversized", "message": f"File not found in API; total record size {_tot_str} exceeds limit.", "declared_size_bytes": None, "final_url": url, "inferred_filename": zenodo_filename, "saved_relative_path": None})
                                _oversized = True
                    else:
                        print(f"  [download]   -> httpx API returned {_zr.status_code}, skipping size check", flush=True)
                except Exception as _exc:
                    print(f"  [download]   -> httpx size check failed: {_exc}", flush=True)
                if _oversized:
                    continue

                try:
                    from zenodo_get import download as zenodo_download
                    zenodo_download(
                        record=record_id,
                        output_dir=str(supplementary_dir),
                        file_glob=zenodo_filename,
                        retry_attempts=3,
                        retry_pause=3.0,
                        timeout=60.0,
                        verbosity=1,
                        exceptions_on_failure=True,
                        max_http_retries=5,
                        backoff_factor=1.0,
                    )
                except Exception as exc:
                    print(f"  [download]   -> zenodo_get failed: {exc}", flush=True)
                    url_events.append({
                        "url": url,
                        "outcome": "failed",
                        "message": f"zenodo_get failed: {exc}",
                        "declared_size_bytes": None,
                        "final_url": url,
                        "saved_relative_path": None,
                    })
                    continue

                # Check if the file was actually downloaded
                if zenodo_filename:
                    downloaded_files = list(supplementary_dir.rglob(zenodo_filename))
                    if downloaded_files:
                        downloaded += 1
                        target = downloaded_files[0]
                        target_path = target.relative_to(supplementary_dir).as_posix()
                        size_str = f"{target.stat().st_size / (1024*1024):.1f} MiB" if target.stat().st_size > 1024*1024 else f"{target.stat().st_size / 1024:.1f} KiB"
                        print(f"  [download]   -> saved {target.name} ({size_str})", flush=True)
                        url_events.append({
                            "url": url,
                            "outcome": "downloaded",
                            "message": None,
                            "declared_size_bytes": target.stat().st_size,
                            "final_url": url,
                            "saved_relative_path": target_path,
                        })
                    else:
                        print(f"  [download]   -> zenodo_get completed but file not found", flush=True)
                        url_events.append({
                            "url": url,
                            "outcome": "failed",
                            "message": "zenodo_get completed but expected file not found on disk.",
                            "declared_size_bytes": None,
                            "final_url": url,
                            "saved_relative_path": None,
                        })
                continue
            else:
                if "figshare.com" in ulow:
                    request_headers["Referer"] = "https://figshare.com/"
                elif "datadryad.org" in ulow:
                    request_headers["Referer"] = "https://datadryad.org/"
                elif "osf.io" in ulow:
                    request_headers["Referer"] = "https://osf.io/"
                download_rate_limiter.wait_for_url(url)
                bou = http_get_with_size_cap(
                    url=url, timeout_seconds=timeout_seconds, session_headers=request_headers,
                    max_bytes=max_bytes_for_url,
                )
                final_url = bou.final_url or url
                payload: bytes | None = bou.data
                used_browser = False

            if bou is not None and bou.outcome in ("skipped_oversized", "skipped_oversized_stream"):
                fname, _ = _material_filename_suffix_from_url(final_url, idx)
                print(f"  [download]   -> {bou.outcome} size={bou.declared_length} name={fname}", flush=True)
                url_events.append(
                    {
                        "url": url,
                        "outcome": bou.outcome,
                        "message": bou.message,
                        "declared_size_bytes": bou.declared_length,
                        "final_url": final_url,
                        "inferred_filename": fname,
                        "saved_relative_path": None,
                    }
                )
                continue

            http_ok = bou is not None and bou.outcome == "downloaded" and bool(payload)
            if not http_ok:
                download_rate_limiter.wait_for_url(url)
                print(f"  [download]   -> http_{bou.outcome}; trying browser fallback...", flush=True)
                try:
                    if browser_client is None:
                        fallback_limiter = RateLimiter(base_delay_seconds=0.3, jitter_seconds=0.2)
                        browser_client = BrowserClient(
                            profile_dir=PROFILE_DIR / "material_download",
                            headless=True,
                            timeout_seconds=timeout_seconds,
                            rate_limiter=fallback_limiter,
                            browser_channel=browser_channel,
                            real_browser_mode=real_browser_mode,
                        )
                        browser_client.__enter__()
                    payload = _download_material_with_browser_session(
                        browser=browser_client,
                        url=url,
                        referer_url=canonical_resolved_url,
                    )
                    used_browser = True
                except Exception as exc:
                    failed += 1
                    print(f"  [download]   -> browser fallback failed: {exc}", flush=True)
                    url_events.append(
                        {
                            "url": url,
                            "outcome": "failed",
                            "message": str(exc),
                            "declared_size_bytes": bou.declared_length if bou else None,
                            "final_url": final_url,
                            "saved_relative_path": None,
                        }
                    )
                    continue

            if used_browser:
                print(f"  [download]   -> browser fallback succeeded ({len(payload)} bytes)", flush=True)

            if bou is not None and not payload:
                failed += 1
                print(f"  [download]   -> empty payload (outcome={bou.outcome})", flush=True)
                url_events.append(
                    {
                        "url": url,
                        "outcome": bou.outcome,
                        "message": bou.message or "empty response body",
                        "declared_size_bytes": bou.declared_length,
                        "final_url": final_url,
                        "saved_relative_path": None,
                    }
                )
                continue

            if bou is not None and len(payload) > MAX_SOURCE_MATERIAL_BYTES:
                fname, _ = _material_filename_suffix_from_url(final_url, idx)
                print(f"  [download]   -> oversized_browser {len(payload)}B name={fname}", flush=True)
                url_events.append(
                    {
                        "url": url,
                        "outcome": "skipped_oversized_browser",
                        "message": f"Downloaded {len(payload)} bytes (> {MAX_SOURCE_MATERIAL_BYTES}); not saved.",
                        "declared_size_bytes": len(payload),
                        "final_url": final_url,
                        "inferred_filename": fname,
                        "saved_relative_path": None,
                    }
                )
                continue

            body_head = payload[:256].lower()
            looks_like_html = body_head.startswith(b"<!doctype html") or body_head.startswith(b"<html")
            if looks_like_html:
                skipped_non_material += 1
                print(f"  [download]   -> blocked_or_html (payload starts with HTML)", flush=True)
                url_events.append(
                    {
                        "url": url,
                        "outcome": "blocked_or_html",
                        "message": bou.message
                        or "Payload looks like HTML (anti-bot, login, or landing page); cannot verify file contents.",
                        "declared_size_bytes": len(payload),
                        "final_url": final_url,
                        "saved_relative_path": None,
                    }
                )
                continue

            raw_name, _hint = _material_filename_suffix_from_url(final_url, idx)
            filename = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw_name) or f"declared_material_{idx}.bin"
            suffix = _allowed_suffix_for_filename(filename)
            if suffix is None:
                if "download-xlsx" in url.lower():
                    filename = f"{filename}.xlsx"
                    suffix = ".xlsx"
                elif "pdf" in url.lower():
                    filename = f"{filename}.pdf"
                    suffix = ".pdf"
                elif filename.lower().endswith(".tar.gz"):
                    suffix = ".tar.gz"
                elif ".zip" in url.lower():
                    filename = f"{filename}.zip"
                    suffix = ".zip"
            allowed_effective = set(allowed_suffixes) | {".tar.gz"}
            if suffix is None or suffix not in allowed_effective:
                skipped_non_material += 1
                print(
                    f"  [download]   -> skipped_not_file (unknown suffix from final_url; name={filename!r})",
                    flush=True,
                )
                url_events.append(
                    {
                        "url": url,
                        "outcome": "skipped_not_file",
                        "message": f"Unrecognized material suffix after download (name={filename!r}).",
                        "declared_size_bytes": len(payload),
                        "final_url": final_url,
                        "saved_relative_path": None,
                    }
                )
                continue

            if suffix == ".pdf" and _dedupe_pdf_against_main_text(payload, main_digests=main_text_pdf_digests):
                skipped_duplicate_main_pdf += 1
                print(f"  [download]   -> skipped_duplicate_main_pdf (SHA matches main article PDF)", flush=True)
                url_events.append(
                    {
                        "url": url,
                        "outcome": "skipped_duplicate_main_pdf",
                        "message": "Byte-identical to local main-article PDF.",
                        "declared_size_bytes": len(payload),
                        "final_url": final_url,
                        "saved_relative_path": None,
                    }
                )
                continue

            target_path = supplementary_dir / filename
            if target_path.exists():
                try:
                    if target_path.is_file() and target_path.read_bytes() == payload:
                        skipped_identical_to_existing_file += 1
                        url_events.append(
                            {
                                "url": url,
                                "outcome": "skipped_identical_existing",
                                "message": "Already present in supplementary/ with identical bytes.",
                                "declared_size_bytes": len(payload),
                                "final_url": final_url,
                                "saved_relative_path": target_path.relative_to(supplementary_dir).as_posix(),
                            }
                        )
                        continue
                except OSError:
                    pass
                stem = target_path.stem
                suf = target_path.suffix
                target_path = supplementary_dir / f"{stem}_{idx}{suf}"
            target_path.write_bytes(payload)
            saved_filenames.append(target_path.name)
            downloaded += 1
            if used_browser:
                browser_fallback_downloaded += 1
            size_str = f"{len(payload) / 1024:.1f} KiB" if len(payload) < 1024 * 1024 else f"{len(payload) / (1024 * 1024):.1f} MiB"
            print(f"  [download]   -> saved {target_path.name} ({size_str})", flush=True)
            url_events.append(
                {
                    "url": url,
                    "outcome": "downloaded",
                    "message": None,
                    "declared_size_bytes": len(payload),
                    "final_url": final_url,
                    "saved_relative_path": target_path.relative_to(supplementary_dir).as_posix(),
                }
            )
    finally:
        if browser_client is not None:
            browser_client.__exit__(None, None, None)

    direct_http = downloaded - browser_fallback_downloaded
    print(
        f"[summary] material_download: wrote {downloaded} new file(s) to {supplementary_dir.as_posix()}/ "
        f"(direct_http={direct_http}, browser_fallback={browser_fallback_downloaded}); "
        f"files_already_in_folder_before_run={len(existing_material_files)}; "
        f"responses_skipped_not_file={skipped_non_material} (e.g. HTML or disallowed type); "
        f"skipped_unknown_suffix={skipped_unknown_suffix} (e.g. mp4, images); "
        f"skipped_no_extension={skipped_no_extension} (webpage links, no file extension); "
        f"skipped_same_as_local_main_pdf={skipped_duplicate_main_pdf}; "
        f"skipped_identical_to_existing_supplementary={skipped_identical_to_existing_file}; "
        f"url_attempts_failed={failed}."
    )
    if saved_filenames:
        preview = saved_filenames[:12]
        extra = f" (+{len(saved_filenames) - len(preview)} more)" if len(saved_filenames) > len(preview) else ""
        print(f"  [saved] {', '.join(preview)}{extra}")

    return _finalize_stage(
        f"existing_materials={len(existing_material_files)} material_urls={len(candidate_urls)} downloaded={downloaded} "
        f"browser_fallback={browser_fallback_downloaded} skipped_non_material={skipped_non_material} "
        f"skipped_same_as_local_main_pdf={skipped_duplicate_main_pdf} "
        f"skipped_identical_to_existing_supplementary={skipped_identical_to_existing_file} failed={failed}"
    )


def _download_material_with_browser_session(*, browser: BrowserClient, url: str, referer_url: str) -> bytes:
    try:
        return browser.download_binary_via_download_event(url, referer_url=referer_url or None)
    except Exception:
        try:
            return browser.download_binary_via_page(url)
        except Exception:
            return browser.download_binary(url)


if __name__ == "__main__":
    raise SystemExit(main())

