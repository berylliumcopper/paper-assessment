"""CLI entrypoint for multi-source article extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from extraction.config.defaults import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_JITTER_SECONDS,
    DEFAULT_MODE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEOUT_SECONDS,
    PROFILE_DIR,
    STATE_DIR,
)
from extraction.core.browser import BrowserClient
from extraction.core.doi2pdf import DEFAULT_SCI_HUB_MIRRORS, apply_doi2pdf_fallback
from extraction.core.models import DownloadedAsset, ExtractionAttempt, ExtractionOutcome
from extraction.core.oa_resolver import resolve_open_access
from extraction.core.rate_limit import RateLimiter
from extraction.core.resolver import looks_like_doi, normalize_doi, resolve_target
from extraction.core.router import detect_source
from extraction.output.writer import write_outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract article content (structured + PDF) from supported sources.")
    parser.add_argument("--input", dest="inputs", nargs="+", required=True, help="Article URL(s) or DOI(s).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for extraction artifacts.")
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["structured", "pdf", "both"],
        help="Extraction mode.",
    )
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS, help="Base delay between requests.")
    parser.add_argument("--jitter-seconds", type=float, default=DEFAULT_JITTER_SECONDS, help="Extra random delay range.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP/browser timeout.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument(
        "--unpaywall-email",
        default="",
        help="Optional email for Unpaywall lookups. Can also come from UNPAYWALL_EMAIL or extraction/.local/local.json.",
    )
    parser.add_argument(
        "--browser-channel",
        default="chromium",
        choices=["chromium", "chrome", "msedge"],
        help="Browser channel. Use chrome/msedge for less bot-like profile if installed.",
    )
    parser.add_argument(
        "--real-browser-mode",
        action="store_true",
        help="Use fewer automation tweaks (recommended when Cloudflare challenge loops).",
    )
    parser.add_argument(
        "--manual-login-url",
        default="",
        help="Optional URL to open for manual sign-in before extraction.",
    )
    parser.add_argument(
        "--manual-login-wait-seconds",
        type=int,
        default=180,
        help="How long to keep manual login/challenge page open.",
    )
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


def _extract_for_target(target, browser, limiter, mode, timeout):
    if target.source == "arxiv":
        from extraction.extractors import arxiv

        return arxiv.extract(target=target, rate_limiter=limiter, mode=mode, timeout_seconds=timeout)
    if target.source == "nature":
        from extraction.extractors import nature

        return nature.extract(target=target, browser=browser, mode=mode)
    if target.source == "science":
        from extraction.extractors import science

        return science.extract(target=target, browser=browser, mode=mode)
    if target.source == "aps":
        from extraction.extractors import aps

        return aps.extract(target=target, browser=browser, mode=mode)
    raise ValueError(f"Unsupported source for URL: {target.resolved_url}")


def _load_unpaywall_email(cli_value: str) -> str | None:
    if cli_value.strip():
        return cli_value.strip()
    env_value = os.getenv("UNPAYWALL_EMAIL", "").strip()
    if env_value:
        return env_value
    local_config = Path("extraction/.local/local.json")
    if local_config.exists():
        try:
            payload = json.loads(local_config.read_text(encoding="utf-8"))
            value = payload.get("unpaywall_email")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:  # noqa: BLE001
            return None
    return None


def _oa_first_target(target, timeout_seconds: int, unpaywall_email: str | None):
    doi_candidate = target.doi or normalize_doi(target.raw_input)
    doi = doi_candidate if looks_like_doi(doi_candidate) else None
    if not doi:
        return target, None
    oa_result = resolve_open_access(
        doi=doi,
        timeout_seconds=timeout_seconds,
        unpaywall_email=unpaywall_email,
    )
    if not oa_result.oa_url:
        return target, oa_result
    oa_url = oa_result.oa_url
    return (
        type(target)(
            raw_input=target.raw_input,
            is_doi=target.is_doi,
            doi=doi,
            resolved_url=oa_url,
            source=detect_source(oa_url),
        ),
        oa_result,
    )


def _extract_oa_pdf_only(target, browser, oa_result) -> ExtractionOutcome:
    outcome = ExtractionOutcome(source="oa_pdf", resolved_url=target.resolved_url, doi=target.doi)
    try:
        try:
            payload = browser.download_binary_via_download_event(
                target.resolved_url,
                referer_url=oa_result.landing_url if oa_result else None,
            )
        except Exception:
            payload = browser.download_binary(target.resolved_url)
        outcome.pdf = DownloadedAsset(filename="article.pdf", source_url=target.resolved_url, content=payload)
        outcome.attempts.append(ExtractionAttempt(name="pdf", success=True, message="OA PDF download completed."))
    except Exception as exc:  # noqa: BLE001
        outcome.attempts.append(ExtractionAttempt(name="pdf", success=False, message=str(exc)))
        outcome.errors.append(f"OA PDF download failed: {exc}")
    if oa_result is not None:
        outcome.metadata.update(
            {
                "oa_resolution_source": oa_result.source,
                "oa_resolution_status": oa_result.status,
                "oa_resolution_message": oa_result.message,
                "oa_landing_url": oa_result.landing_url,
            }
        )
    return outcome


def _mark_access_limited_if_needed(outcome) -> None:
    if outcome.anti_bot_blocked:
        return
    has_any_artifact = (
        bool(outcome.article_markdown)
        or (outcome.pdf is not None)
        or bool(outcome.figures)
        or bool(outcome.supplementary_files)
    )
    if has_any_artifact:
        return
    outcome.access_limited = True
    outcome.access_warning = (
        "Article appears inaccessible from current network/session. "
        "You may need campus/institution access or publisher login."
    )
    outcome.errors.append(outcome.access_warning)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    limiter = RateLimiter(base_delay_seconds=args.delay_seconds, jitter_seconds=args.jitter_seconds)
    unpaywall_email = _load_unpaywall_email(args.unpaywall_email)
    failures = 0

    with BrowserClient(
        profile_dir=PROFILE_DIR,
        headless=args.headless,
        timeout_seconds=args.timeout_seconds,
        rate_limiter=limiter,
        browser_channel=args.browser_channel,
        real_browser_mode=args.real_browser_mode,
    ) as browser:
        if args.manual_login_url:
            print(f"[login] Opening login page: {args.manual_login_url}")
            resolved = browser.interactive_login(
                args.manual_login_url,
                wait_seconds=args.manual_login_wait_seconds,
                detect_challenge=True,
            )
            if not resolved:
                print("[login-warning] Challenge appears unresolved. Try longer wait, visible browser, and real Chrome channel.")

        for raw_target in args.inputs:
            print(f"[start] Resolving {raw_target}")
            try:
                resolved = resolve_target(raw_target, timeout_seconds=args.timeout_seconds)
                resolved, oa_result = _oa_first_target(
                    target=resolved,
                    timeout_seconds=args.timeout_seconds,
                    unpaywall_email=unpaywall_email,
                )
                print(f"[resolved] {resolved.resolved_url} ({resolved.source})")
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
                if (
                    not args.no_doi2pdf
                    and doi_for_fallback
                    and (outcome.anti_bot_blocked or outcome.access_limited)
                ):
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
                if outcome.anti_bot_blocked:
                    print(f"[blocked] Anti-bot challenge detected for {raw_target}. Artifacts saved to {destination}")
                elif outcome.access_limited:
                    print(f"[warning] Article may be inaccessible from current session/network for {raw_target}. Artifacts saved to {destination}")
                else:
                    print(f"[done] Saved to {destination}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[error] {raw_target}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
