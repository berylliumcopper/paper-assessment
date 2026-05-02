"""Detect online data/code availability claims and record fetch outcomes for assessment."""

from __future__ import annotations

import json
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Protocol: do not download entire payloads above this threshold.
MAX_SOURCE_MATERIAL_BYTES = 50 * 1024 * 1024

AUDIT_FILENAME = "_source_data_audit.json"

# Phrases suggesting the authors point readers to online materials (data and/or code).
_AVAILABILITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?is)\b("
        r"source\s+data\b|\bsource\s+code\b|"
        r"data\s+(?:are|is)\s+available|"
        r"code\s+(?:is\s+)?available|"
        r"available\s+at\s+https?://|"
        r"available\s+online|"
        r"publicly\s+available|"
        r"openly\s+available|"
        r"hosted\s+on\s+github|"
        r"deposited\s+in\s+(?:zenodo|figshare|dryad|osf)|"
        r"data\s+repository|"
        r"supplementary\s+data\s+(?:are|is)\s+available|"
        r"extended\s+data\s+(?:are|is)\s+available|"
        r"\bdata\s+availability\b|"
        r"materials?\s+(?:are|is)\s+available\s+at|"
        r"github\.com/[\w.-]+/[\w.-]+|"
        r"zenodo\.org/record/|"
        r"figshare\.com/articles/|"
        r"dryad\.org/|"
        r"osf\.io/"
        r")\b"
    ),
]


def load_combined_article_markdown(article_dir: Path) -> str:
    parts: list[str] = []
    for rel in ("article.md", "converted/article.md"):
        p = article_dir / rel
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


@dataclass
class AvailabilityStatementScan:
    """Whether the article text claims data and/or code are available online."""

    declares_online_data_or_code: bool
    matched_excerpts: list[str] = field(default_factory=list)
    raw_hits: int = 0


def scan_availability_statement(text: str, *, max_excerpts: int = 6, excerpt_pad: int = 120) -> AvailabilityStatementScan:
    if not (text or "").strip():
        return AvailabilityStatementScan(declares_online_data_or_code=False, raw_hits=0)

    excerpts: list[str] = []
    hits = 0
    seen_spans: set[tuple[int, int]] = set()
    for pat in _AVAILABILITY_PATTERNS:
        for m in pat.finditer(text):
            hits += 1
            start, end = m.start(), m.end()
            key = (start, end)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            lo = max(0, start - excerpt_pad)
            hi = min(len(text), end + excerpt_pad)
            snippet = text[lo:hi].replace("\n", " ").strip()
            if len(snippet) > 400:
                snippet = snippet[:397] + "\u2026"
            if snippet and snippet not in excerpts:
                excerpts.append(snippet)
            if len(excerpts) >= max_excerpts:
                return AvailabilityStatementScan(
                    declares_online_data_or_code=True,
                    matched_excerpts=excerpts,
                    raw_hits=hits,
                )

    return AvailabilityStatementScan(
        declares_online_data_or_code=bool(excerpts),
        matched_excerpts=excerpts,
        raw_hits=hits,
    )


def _looks_like_html_payload(content_type: str, body_head: bytes) -> bool:
    ct = (content_type or "").lower()
    if "text/html" in ct:
        return True
    low = body_head[:512].lower().lstrip()
    return low.startswith(b"<!doctype html") or low.startswith(b"<html")


@dataclass
class BoundedDownloadOutcome:
    outcome: str
    data: bytes | None
    declared_length: int | None
    http_status: int | None
    final_url: str | None
    message: str | None = None


def http_get_with_size_cap(
    *,
    url: str,
    timeout_seconds: int,
    session_headers: dict[str, str],
    max_bytes: int = MAX_SOURCE_MATERIAL_BYTES,
) -> BoundedDownloadOutcome:
    """GET with streaming size cap. Uses Content-Length from HEAD when available (HEAD failures are ignored)."""
    import requests  # local import keeps module import-light for paper_reader-only use

    declared: int | None = None
    try:
        head = requests.head(
            url,
            timeout=timeout_seconds,
            allow_redirects=True,
            headers=session_headers,
        )
        try:
            cl = head.headers.get("Content-Length")
            if cl and str(cl).isdigit():
                declared = int(cl)
        except (TypeError, ValueError):
            declared = None
        if head.status_code < 400 and declared is not None and declared > max_bytes:
            return BoundedDownloadOutcome(
                outcome="skipped_oversized",
                data=None,
                declared_length=declared,
                http_status=head.status_code,
                final_url=head.url or url,
                message=f"Content-Length {declared} exceeds cap {max_bytes}.",
            )
    except OSError:
        pass  # Many hosts block HEAD; fall through to capped GET.

    try:
        resp = requests.get(
            url,
            timeout=timeout_seconds,
            allow_redirects=True,
            headers=session_headers,
            stream=True,
        )
    except OSError as exc:
        return BoundedDownloadOutcome(
            outcome="get_failed",
            data=None,
            declared_length=declared,
            http_status=None,
            final_url=None,
            message=str(exc),
        )

    final_url = resp.url or url
    if resp.status_code >= 400:
        return BoundedDownloadOutcome(
            outcome="http_error",
            data=None,
            declared_length=declared,
            http_status=resp.status_code,
            final_url=final_url,
            message=f"HTTP {resp.status_code}",
        )

    ctype = resp.headers.get("content-type") or ""
    try:
        cl2 = resp.headers.get("Content-Length")
        if cl2 and str(cl2).isdigit():
            declared = int(cl2)
    except (TypeError, ValueError):
        pass

    if declared is not None and declared > max_bytes:
        resp.close()
        return BoundedDownloadOutcome(
            outcome="skipped_oversized",
            data=None,
            declared_length=declared,
            http_status=resp.status_code,
            final_url=final_url,
            message=f"Content-Length {declared} exceeds cap {max_bytes}.",
        )

    chunks: list[bytes] = []
    total = 0
    try:
        for piece in resp.iter_content(chunk_size=256 * 1024):
            if not piece:
                continue
            total += len(piece)
            if total > max_bytes:
                resp.close()
                return BoundedDownloadOutcome(
                    outcome="skipped_oversized_stream",
                    data=None,
                    declared_length=declared,
                    http_status=resp.status_code,
                    final_url=final_url,
                    message=f"Stream exceeded {max_bytes} bytes (partial read {total}).",
                )
            chunks.append(piece)
    finally:
        try:
            resp.close()
        except Exception:
            pass

    data = b"".join(chunks)
    head_probe = data[:512]
    if _looks_like_html_payload(ctype, head_probe):
        return BoundedDownloadOutcome(
            outcome="blocked_or_html",
            data=None,
            declared_length=declared,
            http_status=resp.status_code,
            final_url=final_url,
            message="Response appears to be HTML (login wall, anti-bot, or landing page), not a declared file.",
        )

    return BoundedDownloadOutcome(
        outcome="downloaded",
        data=data,
        declared_length=declared if declared is not None else len(data),
        http_status=resp.status_code,
        final_url=final_url,
        message=None,
    )


def _safe_extract_path(base: Path, member_name: str) -> Path | None:
    """Return a path under base for member_name, or None if unsafe (zip-slip)."""
    try:
        target = (base / member_name).resolve()
        base_r = base.resolve()
    except OSError:
        return None
    try:
        target.relative_to(base_r)
    except ValueError:
        return None
    return target


def _copy_stream_capped(src, dest_f, *, max_bytes: int) -> int:
    """Copy up to max_bytes from readable src to dest_f. Returns bytes written."""
    written = 0
    while written < max_bytes:
        chunk = src.read(min(1024 * 1024, max_bytes - written))
        if not chunk:
            break
        dest_f.write(chunk)
        written += len(chunk)
    return written


def extract_archive_if_needed(
    archive_path: Path,
    dest_dir: Path,
    *,
    max_members: int = 500,
    max_total_uncompressed: int = 500 * 1024 * 1024,
) -> tuple[list[str], list[str]]:
    """
    Extract zip or tar.gz/tgz into dest_dir. Returns (member_file_names, notes).
    """
    notes: list[str] = []
    members: list[str] = []
    name_l = archive_path.name.lower()
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    if archive_path.suffix.lower() == ".zip" or zipfile.is_zipfile(archive_path):
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                infos = [i for i in zf.infolist() if not i.is_dir()]
                if len(infos) > max_members:
                    notes.append(f"Zip has {len(infos)} file members; extracting first {max_members} only.")
                    infos = infos[:max_members]
                for info in infos:
                    if total_written >= max_total_uncompressed:
                        notes.append("Stopped zip extraction: total uncompressed cap reached.")
                        break
                    safe = _safe_extract_path(dest_dir, info.filename)
                    if safe is None:
                        notes.append(f"Skipped unsafe path in zip: {info.filename!r}")
                        continue
                    remaining = max_total_uncompressed - total_written
                    take = min(info.file_size, remaining)
                    safe.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, "r") as src, safe.open("wb") as out:
                        n = _copy_stream_capped(src, out, max_bytes=take)
                        total_written += n
                    if take < info.file_size:
                        notes.append(f"Truncated zip member {info.filename} at {take}/{info.file_size} bytes (cap).")
                    members.append(info.filename)
        except zipfile.BadZipFile as exc:
            notes.append(f"Bad zip: {exc}")
            return [], notes
        return members, notes

    if name_l.endswith(".tar.gz") or name_l.endswith(".tgz") or name_l.endswith(".tar"):
        try:
            mode = "r:gz" if (name_l.endswith(".tar.gz") or name_l.endswith(".tgz")) else "r:"
            with tarfile.open(archive_path, mode) as tf:
                file_members = [m for m in tf.getmembers() if m.isfile()]
                if len(file_members) > max_members:
                    notes.append(f"Tar has {len(file_members)} files; extracting first {max_members} only.")
                    file_members = file_members[:max_members]
                for info in file_members:
                    if total_written >= max_total_uncompressed:
                        notes.append("Stopped tar extraction: total uncompressed cap reached.")
                        break
                    safe = _safe_extract_path(dest_dir, info.name)
                    if safe is None:
                        notes.append(f"Skipped unsafe tar path: {info.name!r}")
                        continue
                    remaining = max_total_uncompressed - total_written
                    take = min(info.size, remaining) if info.size is not None else remaining
                    safe.parent.mkdir(parents=True, exist_ok=True)
                    reader = tf.extractfile(info)
                    if reader is None:
                        continue
                    with reader as src, safe.open("wb") as out:
                        n = _copy_stream_capped(src, out, max_bytes=take)
                        total_written += n
                    if info.size is not None and take < info.size:
                        notes.append(f"Truncated tar member {info.name} at {take}/{info.size} bytes (cap).")
                    members.append(info.name)
        except (tarfile.TarError, OSError) as exc:
            notes.append(f"Tar extract failed: {exc}")
            return [], notes
        return members, notes

    notes.append(f"Unsupported archive type: {archive_path.name}")
    return [], notes


def expand_github_material_download_urls(url: str) -> list[str]:
    """
    Derive URLs suitable for binary GET from GitHub links found in article text.

    GitHub ``/tree/...`` and repo home pages respond as HTML to naive GET; this returns
    extra candidates such as ``/archive/{ref}.zip`` or ``raw.githubusercontent.com`` paths.

    The returned list **does not** include the original *url* (callers merge explicitly).
    """
    from urllib.parse import quote

    u = (url or "").strip().rstrip(".,;)>\"'")
    u = u.replace("\\_", "_").replace("\\&", "&").replace("\\%", "%").replace("\\#", "#")
    if "github.com" not in u.lower():
        return []

    out: list[str] = []

    m_blob = re.match(
        r"(?i)^https?://github\.com/([^/]+)/([^/]+?)/blob/([^/]+)/(.+)$",
        u,
    )
    if m_blob:
        owner, repo, ref, path = m_blob.group(1), m_blob.group(2), m_blob.group(3), m_blob.group(4)
        repo = repo.removesuffix(".git")
        raw_path = quote(path, safe="/")
        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{raw_path}"
        out.append(raw)
        return out

    m_tree = re.match(r"(?i)^https?://github\.com/([^/]+)/([^/]+?)/tree/([^/]+)(?:/.*)?$", u)
    if m_tree:
        owner, repo, ref = m_tree.group(1), m_tree.group(2), m_tree.group(3)
        repo = repo.removesuffix(".git")
        zip_url = f"https://github.com/{owner}/{repo}/archive/{ref}.zip"
        out.append(zip_url)
        return out

    m_home = re.match(r"(?i)^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", u)
    if m_home:
        owner, repo = m_home.group(1), m_home.group(2)
        if repo.lower() in {"pull", "issues", "compare", "wiki", "security", "actions"}:
            return out
        for branch in ("main", "master"):
            out.append(f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip")

    return out


def expand_zenodo_material_download_urls(url: str) -> list[str]:
    u = (url or "").strip().rstrip(".,;)>\"'")
    u = u.replace("\\_", "_").replace("\\&", "&")

    m_doi = re.match(r"(?i).*doi\.org/10\.5281/zenodo\.(\d+)", u)
    if m_doi:
        record_id = m_doi.group(1)
        return [f"https://zenodo.org/api/records/{record_id}"]

    m_record = re.match(r"(?i).*zenodo\.org/records?/(\d+)", u)
    if m_record:
        record_id = m_record.group(1)
        return [f"https://zenodo.org/api/records/{record_id}"]

    return []


def expand_figshare_material_download_urls(url: str) -> list[str]:
    u = (url or "").strip().rstrip(".,;)>\"'")
    u = u.replace("\\_", "_").replace("\\&", "&")

    m = re.match(r"(?i).*figshare\.com/articles/(?:[^/]+/)?(\d+)", u)
    if m:
        article_id = m.group(1)
        return [f"https://api.figshare.com/v2/articles/{article_id}"]

    return []


def expand_dryad_material_download_urls(url: str) -> list[str]:
    u = (url or "").strip().rstrip(".,;)>\"'")
    u = u.replace("\\_", "_").replace("\\&", "&")

    m = re.match(r"(?i).*datadryad\.org/.*(doi:10\.\d+/dryad\.\S+)", u)
    if m:
        doi = m.group(1).rstrip(".,;")
        return [f"https://datadryad.org/api/v2/datasets/{doi}/files"]

    return []


def expand_osf_material_download_urls(url: str) -> list[str]:
    u = (url or "").strip().rstrip(".,;)>\"'")
    u = u.replace("\\_", "_").replace("\\&", "&")

    m = re.match(r"(?i).*osf\.io/([a-z0-9]{5})(?:/.*)?$", u)
    if m:
        node_id = m.group(1)
        return [f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/"]

    return []


def resolve_repository_api(*, api_url: str, timeout_seconds: int) -> list[dict]:
    import requests

    out: list[dict] = []
    try:
        resp = requests.get(
            api_url,
            timeout=timeout_seconds,
            headers={"User-Agent": "paper-assessment/0.3.0 (+https://github.com)"},
        )
        if resp.status_code >= 400:
            import sys
            print(f"[warn] repository API returned {resp.status_code} for {api_url}", file=sys.stderr, flush=True)
            return out
        data = resp.json()
    except Exception as exc:
        import sys
        print(f"[warn] repository API request failed: {exc} ({api_url})", file=sys.stderr, flush=True)
        return out

    low = api_url.lower()

    if "zenodo.org/api/records" in low:
        files = data.get("files") if isinstance(data, dict) else []
        if isinstance(files, list):
            for f in files:
                links = f.get("links") if isinstance(f, dict) else {}
                dl = links.get("self") if isinstance(links, dict) else None
                key = f.get("key", "unknown")
                size = f.get("size")
                if dl:
                    out.append({"url": dl, "name": key, "size_bytes": size})
        return out

    if "api.figshare.com/v2/articles" in low:
        files = data.get("files") if isinstance(data, dict) else data if isinstance(data, list) else []
        if isinstance(files, list):
            for f in files:
                dl = f.get("download_url") if isinstance(f, dict) else None
                name = f.get("name", "unknown") if isinstance(f, dict) else "unknown"
                size = f.get("size") if isinstance(f, dict) else None
                if dl:
                    out.append({"url": dl, "name": name, "size_bytes": size})
        return out

    if "datadryad.org/api/v2/datasets" in low and "/files" in low:
        embedded = data.get("_embedded", {}) if isinstance(data, dict) else {}
        files = embedded.get("files", []) if isinstance(embedded, dict) else []
        if isinstance(files, list):
            for f in files:
                dl = f.get("downloadUrl") if isinstance(f, dict) else None
                name = f.get("name", "unknown") if isinstance(f, dict) else "unknown"
                size = f.get("size") if isinstance(f, dict) else None
                if dl:
                    out.append({"url": dl, "name": name, "size_bytes": size})
        return out

    if "api.osf.io/v2/nodes" in low and "/files/osfstorage" in low:
        embedded = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        if isinstance(embedded, list):
            for item in embedded:
                attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
                links = item.get("links", {}) if isinstance(item, dict) else {}
                dl = links.get("download") if isinstance(links, dict) else None
                name = attrs.get("name", "unknown") if isinstance(attrs, dict) else "unknown"
                size = attrs.get("size") if isinstance(attrs, dict) else None
                if dl:
                    out.append({"url": dl, "name": name, "size_bytes": size})
        return out

    return out


def find_info_files_in_member_list(members: list[str]) -> list[str]:
    out: list[str] = []
    for m in members:
        base = Path(m).name.lower()
        if base in {"readme", "readme.md", "readme.txt"}:
            out.append(m)
        elif "info" in base and (
            base.endswith((".txt", ".md", ".json", ".yaml", ".yml")) or "." not in base
        ):
            out.append(m)
    return sorted(set(out))


def write_audit_file(
    supplementary_dir: Path,
    *,
    statement: AvailabilityStatementScan,
    url_events: list[dict[str, Any]],
) -> Path:
    payload: dict[str, Any] = {
        "protocol_version": 1,
        "text_states_online_availability": statement.declares_online_data_or_code,
        "statement_evidence_excerpts": statement.matched_excerpts,
        "statement_pattern_hits": statement.raw_hits,
        "url_events": url_events,
        "max_download_bytes": MAX_SOURCE_MATERIAL_BYTES,
    }
    path = supplementary_dir / AUDIT_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_audit_file(supplementary_dir: Path) -> dict[str, Any] | None:
    p = supplementary_dir / AUDIT_FILENAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return None


def format_audit_for_prompt(audit: dict[str, Any] | None, *, statement_rescan: AvailabilityStatementScan) -> str:
    """Human-readable block for the assessment model."""
    lines: list[str] = []
    lines.append("## Source data and code \u2014 system audit (follow this for dimension 6)")
    lines.append("")
    lines.append(
        "**Rule (1):** If the article text contains no explicit statement that source data and/or code "
        "is available online, treat **data** as **not open-sourced** (`data_open_sourced` should be false or null "
        "with clear justification), regardless of incidental repository URLs elsewhere."
    )
    lines.append("")
    decl = statement_rescan.declares_online_data_or_code
    if audit and isinstance(audit.get("text_states_online_availability"), bool):
        decl = audit["text_states_online_availability"]
    lines.append(f"- **Text declares online data/code:** {decl}")
    if statement_rescan.matched_excerpts:
        lines.append("- **Fresh scan excerpts (article markdown):**")
        for ex in statement_rescan.matched_excerpts[:8]:
            lines.append(f"  - {ex}")
    ev = (audit or {}).get("statement_evidence_excerpts") or []
    if isinstance(ev, list) and ev and not statement_rescan.matched_excerpts:
        lines.append("- **Stored audit excerpts:**")
        for ex in ev[:8]:
            if isinstance(ex, str):
                lines.append(f"  - {ex}")
    lines.append("")
    lines.append(
        "**GitHub handling:** ``/tree/{ref}/\u2026`` links are expanded to ``\u2026/archive/{ref}.zip`` for download; "
        "``/blob/{ref}/path`` links become ``raw.githubusercontent.com`` file URLs. "
        "The URL log may list both the original link and these derived URLs."
    )
    lines.append("")
    lines.append(
        "**Repository URLs (Zenodo, Figshare, Dryad, OSF):** Links to these repositories are resolved via their "
        "public APIs to obtain direct file download URLs. The URL log shows the API endpoints queried and "
        "the resolved file URLs."
    )
    lines.append("")
    lines.append(
        "**Rule (2):** When a link was present but the payload could not be retrieved or appears to be HTML "
        "(anti-bot / login), state explicitly that a link exists but **you cannot verify** whether it hosts "
        "the true source data."
    )
    lines.append(
        "**Rule (2b):** Files over 50MB were not fully downloaded; use declared sizes and any listed archive "
        "contents from the audit."
    )
    lines.append("")

    # --- Prominent section for files known to exist but not downloaded ---
    events = (audit or {}).get("url_events") or []
    oversized: list[dict] = []
    if isinstance(events, list):
        for ev_ in events:
            if ev_.get("outcome") in ("skipped_oversized", "skipped_oversized_stream"):
                oversized.append(ev_)
    if oversized:
        lines.append("### KNOWN TO EXIST \u2014 NOT AUTO-DOWNLOADED (oversized)")
        lines.append("")
        lines.append(
            "The following files were identified at their repository but could not be auto-downloaded "
            "due to size limits. **They count as evidence of data sharing** \u2014 they exist and are "
            "publicly available. Their large sizes typically indicate complete raw or processed data, if this file size is common for the problem they study. "
            "Treat them as contributing to openness, not as missing data."
        )
        lines.append("")
        for ev_ in oversized[:10]:
            fname = ev_.get("inferred_filename") or ev_.get("message", "?")
            sz = ev_.get("declared_size_bytes")
            sz_str = f"{sz / (1024*1024):.1f} MiB" if sz else "unknown size"
            url_ = ev_.get("url", "")
            lines.append(f"- `{fname}` ({sz_str}) \u2014 {url_}")
        lines.append("")

    if isinstance(events, list) and events:
        lines.append("### URL fetch log (most recent materials stage)")
        for ev_ in events[:40]:
            if not isinstance(ev_, dict):
                continue
            u = ev_.get("url", "")
            oc = ev_.get("outcome", "")
            msg = ev_.get("message") or ""
            sz = ev_.get("declared_size_bytes")
            rel = ev_.get("saved_relative_path") or ""
            fname = ev_.get("inferred_filename") or ""
            extra = f" saved=`{rel}`" if rel else ""
            name_part = f" name=`{fname}`" if fname and not rel else ""
            size_part = f" declared_size={sz}" if sz is not None else ""
            lines.append(f"- **{oc}**{name_part}{size_part}{extra}: {u}")
            if msg:
                lines.append(f"  - {msg}")
        if len(events) > 40:
            lines.append(f"- \u2026 ({len(events) - 40} more events omitted)")
    else:
        lines.append("### URL fetch log")
        lines.append("- (No fetch events recorded \u2014 materials download may have been skipped or found no URLs.)")
    lines.append("")
    return "\n".join(lines)
