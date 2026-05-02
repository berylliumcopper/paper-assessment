"""AI-assisted screening of supplementary data files for final assessment.

After downloading and extracting supplementary materials, this module:
  1. Builds a tree of all non-PDF data files (name + size).
  2. Calls an AI session to identify the most important files for assessment.
  3. Returns a screening result (overview + selected file paths) used by
     ``build_paper_context`` to include only targeted data in the final prompt.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal data-file kinds (suffixes considered "data" rather than support).
# ---------------------------------------------------------------------------
_DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".xml",
                  ".dat", ".log", ".out", ".h5", ".hdf5", ".fits", ".nc",
                  ".npy", ".npz", ".mat"}

# Max bytes we read for a preview of a single text-like file.
_PREVIEW_READ_CAP = 64 * 1024       # 64 KiB raw read
_PREVIEW_MAX_CHARS = 8_000

SCREENING_RESULT_FILENAME = "_data_screening_result.json"


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------


@dataclass
class FileNode:
    """A single entry in the file tree."""
    rel_path: str
    size_bytes: int
    is_archive_member: bool = False


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.1f} MiB"


def _collect_data_files(supplementary_dir: Path) -> list[FileNode]:
    """
    Walk the supplementary directory and collect every non-PDF, non-internal
    data file (including archive member files under ``_extracted/``, and
    converted spreadsheets under ``_converted_spreadsheets/``).
    """
    SKIP_NAMES = {
        SCREENING_RESULT_FILENAME,
        "_source_data_audit.json",
        ".materials_download_fingerprint",
        "__members.json",
        ".extraction_stamp.json",
    }
    SKIP_DIR_PARTS = {"_readable"}  # Marker output for supplementary PDFs

    files: list[FileNode] = []
    for p in sorted(supplementary_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name in SKIP_NAMES:
            continue
        if p.name.startswith("."):
            continue
        rel = p.relative_to(supplementary_dir).as_posix()
        parts = p.relative_to(supplementary_dir).parts
        if any(part in SKIP_DIR_PARTS for part in parts):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        is_extracted = "_extracted" in parts
        files.append(FileNode(rel_path=rel, size_bytes=sz, is_archive_member=is_extracted))
    return files


def format_file_tree(files: list[FileNode], *, max_lines: int = 120) -> str:
    """
    Format the collected files as an indented tree (text), grouped by top-level
    directory for readability.  Archives are displayed under their source
    archive name.
    """
    # Separate "loose" files (in supplementary/ root or _converted_spreadsheets)
    # from archive-extracted files (under _extracted/<archive>/...).
    loose: list[FileNode] = []
    archive_groups: dict[str, list[FileNode]] = {}
    for f in files:
        if f.is_archive_member:
            parts = Path(f.rel_path).parts
            # parts[0] = "_extracted", parts[1] = archive stem
            if len(parts) >= 2:
                key = parts[1]
                archive_groups.setdefault(key, []).append(
                    FileNode(rel_path="/".join(parts[2:]) if len(parts) > 2 else parts[1],
                             size_bytes=f.size_bytes, is_archive_member=True)
                )
            else:
                loose.append(f)
        else:
            loose.append(f)

    lines: list[str] = []
    lines.append("=== Data file tree (non-PDF) ===")

    if loose:
        lines.append("\n--- Loose / converted files ---")
        for f in loose:
            lines.append(f"  {f.rel_path}  ({_human_size(f.size_bytes)})")

    for archive_stem, members in sorted(archive_groups.items()):
        lines.append(f"\n--- From archive: {archive_stem} ---")
        # Sort by path for tree-like ordering
        members.sort(key=lambda x: x.rel_path)
        for f in members:
            lines.append(f"  {f.rel_path}  ({_human_size(f.size_bytes)})")

    total_sz = sum(f.size_bytes for f in files)
    lines.append(f"\nTotal: {len(files)} file(s), {_human_size(total_sz)}")
    return "\n".join(lines[:max_lines])


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------


def _preview_text(path: Path, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    """Return the first ``max_chars`` characters of a text file."""
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
    """Return the first N lines of a CSV-like file."""
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


# ---------------------------------------------------------------------------
# AI screening prompt
# ---------------------------------------------------------------------------


def _build_screening_prompt(article_dir: Path, tree_text: str) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the screening AI session."""

    # Gather paper context to help the AI understand what the paper is about.
    md_parts: list[str] = []
    for rel in ("converted/article.md", "article.md"):
        p = article_dir / rel
        if p.is_file():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                # Take first ~4000 chars as paper summary
                md_parts.append(txt[:4000])
            except OSError:
                pass
    paper_snippet = "\n\n".join(md_parts)[:4000]

    system_prompt = (
        "You are a scientific data curator. Your task is to examine a set of "
        "supplementary data files released alongside a scientific paper, and "
        "identify which files contain the most important data for evaluating "
        "the paper's claims.\n\n"
        "You will be given:\n"
        "1. A snippet of the paper (title, abstract, and first part of content)\n"
        "2. A tree of all non-PDF data files with their names and sizes\n\n"
        "Your job:\n"
        "- Summarize what kind of data was released and its overall structure (overview).\n"
        "- Select the files that most likely contain: (a) data used to generate "
        "main-text figures, (b) primary/processed experimental results, "
        "(c) critical parameters or input files needed for understanding the analysis.\n"
        "- If a folder contains many similar files from repeated experiments, "
        "select at most ONE representative file as an example.\n"
        "- Prioritize: raw/processed numerical data > metadata > peripheral info.\n"
        "- Return structured JSON (see below).\n\n"
        "Return JSON with these keys:\n"
        "- overview (str): 1-2 paragraph description of what the downloaded data "
        "contains and its scientific relevance to the paper.\n"
        "- selected_files (list of objects): each with:\n"
        "    - path (str): the exact relative path shown in the tree.\n"
        "    - reason (str): why this file is important (e.g., 'contains Hall "
        "viscosity values used in Figure 3').\n"
        "    - is_example (bool): true only if this file was chosen as a "
        "representative from a group of similar files.\n"
        "- notes (str): any additional observations about data completeness, "
        "missing context, etc."
    )

    user_prompt = (
        f"## Paper snippet (first ~4000 chars)\n{paper_snippet}\n\n"
        f"## Data file tree\n{tree_text}\n\n"
        "Analyze the data files above. Return the JSON object with `overview`, "
        "`selected_files`, and `notes` as described."
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_data_file_tree(supplementary_dir: Path) -> str:
    """Build a human-readable tree of all non-PDF data files."""
    if not supplementary_dir.is_dir():
        return "(no supplementary directory)"
    files = _collect_data_files(supplementary_dir)
    if not files:
        return "(no data files found)"
    return format_file_tree(files)


def run_data_screening(
    *,
    article_dir: Path,
    supp_tree_text: str,
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """
    Call the AI to screen supplementary data files and return:
      - overview (str)
      - selected_files (list[dict])
      - notes (str)
    """
    system_prompt, user_prompt = _build_screening_prompt(article_dir, supp_tree_text)

    if is_gemini:
        from assessment.gemini_client import GeminiClient
        client = GeminiClient(
            api_key=api_key,
            model=api_model,
            timeout_seconds=timeout_seconds,
            base_url=api_base_url,
        )
    else:
        from assessment.openai_client import OpenAIClient
        client = OpenAIClient(
            api_key=api_key,
            model=api_model,
            timeout_seconds=timeout_seconds,
            base_url=api_base_url,
        )

    result = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
    if not isinstance(result, dict):
        raise TypeError(f"AI screening returned non-dict: {type(result)}")

    # Normalize keys.
    overview = str(result.get("overview", result.get("summary", "")))
    selected_files = result.get("selected_files", result.get("files", []))
    notes = str(result.get("notes", ""))
    if not isinstance(selected_files, list):
        selected_files = []

    return {
        "overview": overview,
        "selected_files": selected_files,
        "notes": notes,
    }


def read_screening_result(supplementary_dir: Path) -> dict[str, Any] | None:
    """Read a previously saved screening result from disk."""
    p = supplementary_dir / SCREENING_RESULT_FILENAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return None


def write_screening_result(supplementary_dir: Path, result: dict[str, Any]) -> Path:
    """Persist the screening result to disk."""
    p = supplementary_dir / SCREENING_RESULT_FILENAME
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def format_screened_data_for_prompt(
    screening_result: dict[str, Any] | None,
    *,
    article_dir: Path,
    supplementary_dir: Path,
) -> tuple[str, list[str]]:
    """
    Given the AI screening result, produce:
      - overview_block (str): markdown section suitable for the assessment prompt
      - selected_snippets (list[str]): content of the selected files

    This is called by ``build_paper_context`` to produce the final prompt block.
    """
    if not screening_result:
        return "", []

    overview = str(screening_result.get("overview", ""))
    notes = str(screening_result.get("notes", ""))
    selected = screening_result.get("selected_files", [])
    if not isinstance(selected, list):
        selected = []

    lines: list[str] = []
    lines.append("## AI-Screened Supplementary Data Overview")
    if overview:
        lines.append("")
        lines.append(overview)
    if notes:
        lines.append("")
        lines.append("**Screening notes:** " + notes)

    # Build the selected-files section.
    selected_lines: list[str] = []
    selected_lines.append("")
    selected_lines.append("### Selected data files for detailed review")

    snippets: list[str] = []
    for entry in selected:
        if not isinstance(entry, dict):
            continue
        file_path_str = str(entry.get("path", ""))
        reason = str(entry.get("reason", ""))
        is_example = bool(entry.get("is_example", False))

        # Resolve to an actual file on disk.
        disk_path = _resolve_data_path(supplementary_dir, file_path_str)
        if disk_path is None or not disk_path.is_file():
            selected_lines.append(
                f"- `{file_path_str}` — {reason} "
                f"{'(example)' if is_example else ''}  [NOT FOUND ON DISK]"
            )
            continue

        sz_str = _human_size(disk_path.stat().st_size)
        selected_lines.append(
            f"- `{file_path_str}` ({sz_str}) — {reason} "
            f"{'(example)' if is_example else ''}"
        )

        # Read content / preview.
        suffix = disk_path.suffix.lower()
        content = ""
        if suffix in _DATA_SUFFIXES | {".md", ".csv", ".tsv"}:
            if disk_path.stat().st_size < 64 * 1024:
                content = _preview_text(disk_path, max_chars=24_000)
            else:
                # Slightly larger → first lines only
                if suffix in {".csv", ".tsv"}:
                    content = _preview_csv(disk_path, max_lines=20)
                else:
                    content = _preview_text(disk_path, max_chars=_PREVIEW_MAX_CHARS)
                content += f"\n\n[File is {sz_str}; showing preview only]"
        elif suffix in {".xls", ".xlsx"}:
            # Use the pre-converted CSV if available.
            base = disk_path.stem
            csv_dir = supplementary_dir / "_converted_spreadsheets"
            csv_candidates = sorted(csv_dir.glob(f"{base}__*.csv")) if csv_dir.is_dir() else []
            if csv_candidates:
                # Take the first CSV as representative.
                csv_path = csv_candidates[0]
                if csv_path.stat().st_size < 64 * 1024:
                    content = csv_path.read_text(encoding="utf-8", errors="replace")[:24_000]
                else:
                    content = _preview_csv(csv_path, max_lines=20)
                    content += f"\n\n[File is {_human_size(csv_path.stat().st_size)}; showing preview only]"
            else:
                # Fallback to pandas/openpyxl extraction
                from assessment.paper_reader import _extract_excel_text_snippet
                content, _ = _extract_excel_text_snippet(disk_path, max_chars=8_000)
        else:
            # Binary / unknown — note the size only.
            content = f"[Binary file, {sz_str}]"

        if content.strip():
            header = f"#### `{file_path_str}`"
            snippets.append(f"{header}\n{content}")

    lines.extend(selected_lines)
    lines.append("")

    return "\n".join(lines), snippets


def _resolve_data_path(supplementary_dir: Path, file_path_str: str) -> Path | None:
    """
    Try to locate a file on disk by its relative path (as reported in the
    tree and selected_files).  Search order:
      1. Direct path under supplementary_dir
      2. Under _extracted/<archive>/...
      3. Under _converted_spreadsheets/...
      4. Under _extracted/ (any subdirectory matching the tail)
    """
    # Direct
    candidate = supplementary_dir / file_path_str
    if candidate.is_file():
        return candidate

    # Under _extracted (archive member paths are stored without _extracted prefix)
    for extracted_dir in supplementary_dir.glob("_extracted/*"):
        candidate2 = extracted_dir / file_path_str
        if candidate2.is_file():
            return candidate2

    # Under _converted_spreadsheets
    candidate3 = supplementary_dir / "_converted_spreadsheets" / file_path_str
    if candidate3.is_file():
        return candidate3

    # Try matching just the filename in _extracted/
    fname = Path(file_path_str).name
    for found in sorted(supplementary_dir.rglob(fname)):
        if found.is_file():
            return found

    return None
