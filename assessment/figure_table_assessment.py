"""AI assessment of figures/tables in the main text against source data files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --- Constants ---

# Total data content chars for this step (~60k tokens × 4 chars/token).
FIGURE_TABLE_DATA_MAX_CHARS = 240_000
# Simplified data block for the final assessment (~10k tokens × 4 chars/token).
FINAL_DATA_MAX_CHARS = 40_000

SYSTEM_PROMPT = """You are a rigorous scientific data auditor. Your task is to examine the main-text figures and tables of a paper, together with the source data files, to assess data integrity, reproducibility, and consistency.

!! IMPORTANT -- SCOPE !!
Focus ONLY on **main-text figures and tables**. Ignore "Extended Data" figures, "Supplementary" figures, "Appendix" figures, and any figures labeled as additional/supporting — these are out of scope and should NOT be analyzed.

!! IMPORTANT -- PER-PANEL ANALYSIS REQUIRED !!
Each **figure panel** that contains data must be analyzed **separately**. Do NOT group panels together even if they appear related. Every panel gets its own set of answers to the 5 questions below. The only exception is when multiple panels literally show the same data with just different parameter values (e.g., same experiment repeated at different temperatures) — in those cases you may group them with a clear note that parameters differ.

!! IMPORTANT -- DATA TRUNCATION NOTICE !!
Some data files or file contents may have been **truncated** to fit within the model's context window. When this happens, an explicit warning like `[!! TRUNCATED: ... !!]` or `[!! DATA CLIPPED: ... !!]` appears. If you see such a warning for a file, **you CANNOT assume the shown data is complete**. Distinguish carefully between:
- "The file's provided content is incomplete" (due to truncation in this pipeline)
- "The paper's original data is incomplete" (an actual assessment finding)

Do NOT report that data is missing or incomplete if the truncation warning is present -- that is a pipeline artifact, not a finding about the paper.

For each figure panel that contains data (and each table), answer the following questions **separately per panel**, with as much detail as possible:

1. **Experimental methods or theoretical pipelines**: Describe the specific experimental technique, measurement setup, or theoretical pipeline that produced the data shown. Name the instrument, method, or algorithm (e.g., "pump-probe Kerr spectroscopy at 800 nm", "DFT with PBE functional", "scanning SQUID microscopy"). State the key parameters (temperature, field, energy, doping level, etc.). Do NOT just repeat the panel label — extract the actual methodological details from the text, caption, and visible axes.
**Note on paper type:**
- For **experimental** papers: This means the measurement instrument, sample preparation, and data-acquisition setup.
- For **theoretical** papers: This means the numerical simulation setup (DFT parameters, MD conditions, model Hamiltonian exact-diagonalization basis, Monte Carlo sweeps, etc.). "Source data files" may contain simulation parameters or output.
- For **computational** papers: This means the algorithm's inputs, the benchmark/test systems, and the evaluation protocol. "Source data files" may contain benchmark results or performance logs.

2. **Data reasonableness (methodological)**: Does the data look like it was **truly collected by the indicated method**? Assess whether the data values, noise levels, statistical properties, and scales are physically plausible for the claimed measurement technique. **Do NOT summarize what the figure shows** — do not restate trends, do not describe the data content. Instead, judge only whether the raw character of the data (range, scatter, features, units) is consistent with the stated methodology. Flag if the data looks too clean, too noisy, or has impossible symmetries or values for the claimed method. A correct answer here is a brief methodological judgment (e.g., "plausible noise levels for this technique", "values too clean for a real measurement"), not a summary of the plotted results.
**Note on paper type:**
- For **experimental** papers: Assess noise levels, statistical scatter, instrument resolution limits, and whether the data range is physically expected.
- For **theoretical/simulation** papers: Assess whether numerical convergence was achieved (e.g., k-point density, basis-set size, time step), whether lattice/cell sizes are adequate, and whether the numerical precision/behavior is consistent with the claimed method.
- For **computational/benchmark** papers: Assess whether benchmark measurements (runtime, memory, accuracy) are reproducible, whether performance metrics are reasonable for the claimed hardware, and whether the numbers are too round or suspiciously clean.

3. **Data provenance**: Trace the full processing chain from raw data to this plot. What was the original measurement, calculation, or simulation that generated the data? What processing steps were applied (background subtraction, normalization, binning, curve fitting, Fourier transform, integration, statistical aggregation, etc.)? What is this panel's **relationship to other figures** — does it derive from data shown in another panel (e.g., a fit to data from Fig. X, a Fourier transform of the data in Fig. Y, an integrated quantity from Fig. Z), is it the same data shown differently (e.g., different axis, zoomed region), or is it an independent measurement unrelated to other panels? Name the specific source figure/panel for every dependency. Also note which source data file(s) correspond to this panel (filename and sheet name if applicable).
**Note on paper type:**
- For **experimental** papers: Trace from raw instrument output → cleaned/normalized → binned/fitted → final plot. Identify whether intermediate processing steps are fully described.
- For **theoretical** papers: Trace from model parameters → simulation execution → post-processing (e.g., Fourier transform of correlation functions, integration over k-space) → final plot. Check if the raw simulation data is available.
- For **computational** papers: Trace from input data → algorithm execution → output processing → benchmark metrics → final comparison plot. Check if intermediate metrics are explained.

4. **Cross-figure and source-file consistency**: Check consistency on two fronts:
   - **Figure vs. source file**: Compare the data values quoted in the panel (or visible in the plot) against the numbers in the provided source data files. Spot-check key data points — do the source file numbers match what is plotted? You should also pick multiple individual data points in different regimes and check whether they are consistent with each other. If source data files are not available, state that explicitly.
   - **Figure vs. figure**: Go back to the **relationships you described in question 3** (e.g., "this panel is derived from Fig. X panel Y", "these data are the same as in Fig. Z but zoomed"). For EACH such claimed relationship, verify it explicitly: check whether the actual numbers, trends, or data points are consistent between the two figures. You should check the correspondance between ranges, trends, and smoothness of data points. You should also pick multiple individual data points in different regimes and check whether they are consistent with each other. Note that this question is still valid even if no source data files available. If question 3 identified no inter-figure relationships, state that explicitly.

5. **Claim-plot consistency**: What does the paper claim this panel shows in its caption and in-text description? List the specific claim(s). Then examine the actual plotted data — axes, ranges, units, trends, error bars, and statistics — and judge whether each claim is actually supported by what is plotted. Are there any symmetries, estimation of parameters, patterns, or other features that is claimed by the paper but does not present in the data? Flag any mismatch, exaggeration, or unsupported inference. If the claim is fully supported, say so.

6. **Consistency with explanation**: Consider the explanation described in the paper in a broader context. You should not limit to the signatures stated in the paper, but use your own reasoning to infer what the data should look like if the explanations are true. Are there any symmetries, estimation of parameters, patterns, or other features that the claims naturally require but does not present in the data? Compare the actual data to your inferred expectations and flag any discrepancies. If the claim is fully supported by the data, say so.

After your per-panel analysis, add a **summary section** at the end covering:
- **Overall openness**: Examine all the source data files that were provided and evaluate the overall open-sourcing extent of the paper's data. Classify based on the following levels. Be **conservative** — when in doubt, classify lower:
   - (0) **No source data**: No source data files were found at all.
   - (A) **Figure-level exports only**: The data files contain only the values needed to replot the figures. The files do not contain substantially more information than what is shown in the plots.
   - (B) **Processed data (more than the plot, but not raw)**: The data files not only contain the full figure-level values, but many data files also contain clearly more information than the plots — e.g., additional measured quantities, clearly extended measurement ranges, results from medium processing steps not covered by the figures, or broader conditions that go significantly beyond what the figures shows. The data files need to contain both almost-complete figure-level data and substantial quantities of additional information not covered by the figures to qualify as (B).
   - (C) **Complete raw data**: The data files contain unprocessed instrument output, individual-trial data, or code + inputs sufficient to regenerate the analysis from scratch.
   Note that different data files may have different levels of openness. Write down clearly if different figures have different levels of openness, especially if some data files or figure panels does not fit the overall openness level (like source data found for most figures but missing for one figure). Your classification should be the average over all the data files or all the figure panels. State your overall classification and justify it.

 **WARNING: score inflation risk**: The plots in the extended data figures, appendix figures, and supplementary should not be considered as additional source data, and thus has no effect on the openness level. Only consider the data files to be source data.

**OVERSIZED FILES NOT ON DISK:** If the "Source Data Files" section below lists files that are not present on disk (an "Oversized Files" notice with filenames and sizes), those files were identified at the paper's repository but could not be auto-downloaded due to size limits. They are **confirmed to exist** at the repository — this is NOT the same as missing data. Include them in your openness assessment using their size as a guide: small files likely contain figure-level exports, multi-MB files likely contain processed data, large files likely contain complete raw data. Do not penalize openness for pipeline download limits.

 **NOTE**: If different parts or figures have different levels of openness, state which parts or figures are in each level of openness clearly.

Indicate whether any data files **must still be read by the final assessment** (only if files remain unusual or unclear after this analysis). This is optional — most files should not need this.

Return valid JSON with this exact structure:
{
  "figure_table_analysis": "string — per-panel/per-table analysis covering all 6 questions, followed by an overall openness evaluation and inventory of unreferenced data files. Markdown format.",
  "files_for_final_step": [
    {
      "path": "string — relative path to the file that must be forwarded to the final assessment",
      "reason": "string — why this file still needs final review"
    }
  ]
}"""


# --- Local helpers (duplicated from assessment_cli to avoid circular imports) ---

_DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".xml",
                  ".dat", ".log", ".out", ".h5", ".hdf5", ".fits", ".nc",
                  ".npy", ".npz", ".mat"}
_PREVIEW_READ_CAP = 64 * 1024


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.1f} MiB"


def _preview_text(path: Path, max_chars: int) -> tuple[str, bool]:
    """Read text up to ``max_chars``. Returns (content, was_truncated)."""
    try:
        to_read = min(path.stat().st_size, _PREVIEW_READ_CAP)
        raw = path.read_bytes()[:to_read]
        text = raw.decode("utf-8", errors="replace")
        total_chars = len(text)
        if total_chars > max_chars:
            return (
                f"{text[:max_chars]}"
                f"\n[!! TRUNCATED: file has {total_chars:,} chars but only {max_chars:,} shown !!]",
                True,
            )
        return text, False
    except OSError:
        return "(unable to read)", False


def _preview_csv(path: Path, max_lines: int) -> tuple[str, bool]:
    """Read CSV up to ``max_lines``. Returns (content, was_truncated)."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head: list[str] = []
            total_lines = 0
            for line in fh:
                total_lines += 1
                if len(head) < max_lines:
                    head.append(line.rstrip("\n\r"))
        content = "\n".join(head) if head else "(empty)"
        if total_lines > max_lines:
            content += (
                f"\n[!! TRUNCATED: file has {total_lines:,} data rows but only {max_lines:,} shown !!]"
            )
            return content, True
        return content, False
    except OSError:
        return "(unable to read)", False


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
    max_chars: int = 240_000,
) -> tuple[str, bool]:
    """Extract a preview of a data file. Returns (content, was_truncated).

    The returned content always includes an explicit [!! TRUNCATED ... !!] marker
    whenever the full file could not be shown.
    """
    suffix = disk_path.suffix.lower()
    sz_str = _human_size(disk_path.stat().st_size)
    truncated = False

    if suffix in _DATA_SUFFIXES | {".md", ".csv", ".tsv"}:
        if disk_path.stat().st_size < 64 * 1024:
            content, truncated = _preview_text(disk_path, max_chars=max_chars)
        else:
            if suffix in {".csv", ".tsv"}:
                content, truncated = _preview_csv(disk_path, max_lines=20)
            else:
                content, _trunc = _preview_text(disk_path, max_chars=8_000)
                truncated = truncated or _trunc
            if not truncated:
                content += (
                    f"\n\n[!! DATA CLIPPED: file is {sz_str} (>64 KiB); "
                    f"showing preview only. The shown content MAY be incomplete. !!]"
                )
                truncated = True
        return content, truncated

    if suffix in {".xls", ".xlsx"}:
        csv_dir = supplementary_dir / "_converted_spreadsheets"
        # Use full filename (including .xlsx extension) to match CSV naming:
        #   "41586_2023_5853_MOESM4_ESM.xlsx__01_Panel_a.csv"
        prefix = disk_path.name  # e.g. "41586_2023_5853_MOESM4_ESM.xlsx"
        csv_candidates = sorted(csv_dir.glob(f"{prefix}__*.csv")) if csv_dir.is_dir() else []
        if csv_candidates:
            # Read all CSV sheets fully. No early clipping — the caller (uniform B
            # algorithm) will handle truncation across all files afterwards.
            parts: list[str] = []
            for csv_path in csv_candidates:
                sheet_label = csv_path.stem.partition("__")[2] or csv_path.stem  # e.g. "01_Panel_a"
                if csv_path.stat().st_size < 64 * 1024:
                    sheet_text = csv_path.read_text(encoding="utf-8", errors="replace")
                else:
                    sheet_text = _preview_csv(csv_path, max_lines=20)[0]
                parts.append(f"--- Sheet: {sheet_label} ---\n{sheet_text}")
            content = "\n\n".join(parts)
            truncated = False  # all data present; clipping happens at call site
        else:
            from assessment.paper_reader import _extract_excel_text_snippet  # noqa: PLC0415
            content, _ = _extract_excel_text_snippet(disk_path, max_chars=8_000)
            truncated = True  # always truncated at 8K
        return content, truncated

    return f"[Binary file, {sz_str}]", False


# --- Main function ---

def run_figure_table_assessment(
    *,
    article_dir: Path,
    data_file_analysis: dict[str, Any],
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """
    Run AI assessment of main-text figures/tables against source data files.

    Returns dict with:
      - figure_table_analysis (str): detailed per-panel/per-table analysis (markdown)
      - analysis_block (str): formatted block for the final assessment prompt
      - files_for_final_step (list[dict]): files still needing final review
      - simplified_data_block (str): minimal data content for final assessment (≤ FINAL_DATA_MAX_CHARS chars)
    """
    # --- Collect main text content ---
    primary_md_path = article_dir / "article.md"
    converted_md_path = article_dir / "converted" / "article.md"

    primary_markdown = ""
    if primary_md_path.exists():
        primary_markdown = primary_md_path.read_text(encoding="utf-8", errors="ignore")

    converted_markdown = ""
    if converted_md_path.exists():
        converted_markdown = converted_md_path.read_text(encoding="utf-8", errors="ignore")

    # --- Collect main-text images (from converted/ only, NOT supplementary _readable) ---
    converted_dir = article_dir / "converted"
    main_text_images: list[Path] = []
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    if converted_dir.exists():
        for p in sorted(converted_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in image_extensions:
                main_text_images.append(p)

    main_text_md = (
        "## Primary Article Markdown\n"
        f"{primary_markdown}\n\n"
        "## Converted PDF Markdown\n"
        f"{converted_markdown}\n\n"
    )

    # --- Build data file content block (~60k token budget = FIGURE_TABLE_DATA_MAX_CHARS chars) ---
    data_overview = data_file_analysis.get("overview_block", "")
    selected_files = data_file_analysis.get("selected_files", [])
    supplementary_dir = article_dir / "supplementary"

    data_lines: list[str] = []
    any_truncated = False

    data_lines.append("### Source Data File Contents")
    data_lines.append("")

    if data_overview:
        data_lines.append(data_overview)
        data_lines.append("")

    # --- Collect all AI-selected files and read their real sizes ---
    # We ignore the pre-existing snippets from the data-analysis step because they
    # are already clipped at a much smaller per-file budget (~4K each). Instead we
    # read each file fresh from disk and allocate the full 240K budget smartly.
    file_entries: list[dict[str, Any]] = []
    for sf in selected_files:
        if not isinstance(sf, dict):
            continue
        rel_path = sf.get("path", "")
        if not rel_path:
            continue
        disk_path = _resolve_data_path(supplementary_dir, rel_path)
        if disk_path is None or not disk_path.is_file():
            continue
        # Read full content (up to 24K per file — enough for ~23KB CSVs)
        content, truncated = _extract_file_content_preview(
            disk_path, supplementary_dir,
        )
        if not content.strip():
            continue
        file_entries.append({
            "rel_path": rel_path,
            "content": content,
            "reason": sf.get("reason", ""),
            "actual_chars": len(content),
        })

    # Initialize clip statistics (updated in the else branch if clipping occurs)
    uniform_clip_threshold_B = 0
    total_files_selected = 0
    clipped_files_count = 0

    if not file_entries:
        data_lines.append("No source data files available for review.")
    else:
        sizes = [e["actual_chars"] for e in file_entries]
        total_files_selected = len(sizes)
        budget = FIGURE_TABLE_DATA_MAX_CHARS
        per_file_overhead = [len(f"#### `{e['rel_path']}`\n") for e in file_entries]

        # Check whether everything fits without any clipping.
        total_raw = sum(sizes) + sum(per_file_overhead)
        if total_raw <= budget:
            # Perfect fit — include every file fully.
            for entry in file_entries:
                block = f"#### `{entry['rel_path']}`\n{entry['content']}"
                data_lines.append(block)
        else:
            # Budget is tight — use a uniform clip threshold B.
            any_truncated = True
            sizes = [e["actual_chars"] for e in file_entries]
            n_files = len(sizes)
            # Per-file overhead = f"#### `{rel}`\n"  (path-dependent)
            per_file_overhead = [len(f"#### `{e['rel_path']}`\n") for e in file_entries]
            trunc_marker_cost = 80  # approximate for "[!! TRUNCATED: ... !!]"

            # Binary search for the largest B where total fits.
            lo, hi = 0, max(sizes)
            best_B = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                total = 0
                for i, s in enumerate(sizes):
                    if s <= mid:
                        total += per_file_overhead[i] + s
                    else:
                        total += per_file_overhead[i] + mid + trunc_marker_cost
                if total <= budget:
                    best_B = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            uniform_clip_threshold_B = max(best_B, 1)  # at least 1 char per file

            clipped_count = 0
            for entry in file_entries:
                if entry["actual_chars"] <= uniform_clip_threshold_B:
                    block = f"#### `{entry['rel_path']}`\n{entry['content']}"
                else:
                    clipped = entry["content"][:uniform_clip_threshold_B]
                    block = (
                        f"#### `{entry['rel_path']}`\n"
                        f"[!! TRUNCATED: file has {entry['actual_chars']:,} chars, "
                        f"showing {uniform_clip_threshold_B:,} chars via uniform clip threshold. "
                        f"The end of this data file is incomplete due to length limitations. !!]\n"
                        f"{clipped}"
                    )
                    clipped_count += 1
                data_lines.append(block)

            clipped_files_count = clipped_count

            data_lines.insert(
                1,
                "\n---\n"
                "!! TRUNCATION WARNING !!\n\n"
                f"Total data content exceeds the {budget:,}-char budget. "
                f"Every file was clipped to at most {uniform_clip_threshold_B:,} chars (uniform threshold). "
                f"{n_files - clipped_count} files fit fully; {clipped_count} were truncated.\n"
                "---\n",
            )

    data_content = "\n\n".join(data_lines) if data_lines else "No source data files available."

    # --- Build user prompt ---
    user_prompt = (
        "Analyze the main-text figures and tables of the following paper, "
        "cross-referencing with the provided source data files.\n\n"
        f"{main_text_md}\n"
        "## Main-text Images (for visual analysis)\n"
        "The images below are provided alongside this text. Refer to them by filename. "
        "NOTE: Some images may be from Extended Data or Supplementary sections. "
        "Ignore those -- only analyze main-text figures (Fig. 1, Fig. 2, etc.) and tables.\n\n"
        "## Source Data Files\n"
        f"{data_content}\n\n"
        "For each figure panel containing data (and each table), answer the questions "
        "described in the system prompt. "
        "Be specific, reference exact figure/table numbers and panel labels. "
        "If no source data files are available, note this in your analysis."
    )

    # --- Call AI ---
    if is_gemini:
        from assessment.gemini_client import GeminiClient  # noqa: PLC0415
        client: Any = GeminiClient(
            api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url,
        )
    else:
        from assessment.openai_client import OpenAIClient  # noqa: PLC0415
        client = OpenAIClient(
            api_key=api_key, model=api_model, timeout_seconds=timeout_seconds, base_url=api_base_url,
        )

    try:
        result = client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=main_text_images,
        )
    except Exception as exc:
        print(f"[warn] figure/table assessment AI call failed: {exc}", flush=True)
        result = {
            "figure_table_analysis": f"Figure/table assessment AI call failed: {exc}",
            "files_for_final_step": [],
        }

    if not isinstance(result, dict):
        result = {
            "figure_table_analysis": "Figure/table assessment returned non-dict response.",
            "files_for_final_step": [],
        }

    analysis_text = str(result.get("figure_table_analysis", ""))
    files_for_final = result.get("files_for_final_step", [])
    if not isinstance(files_for_final, list):
        files_for_final = []

    # --- Build analysis block for final prompt ---
    analysis_block = (
        "## Figure/Table Data Assessment\n"
        "The following is a per-panel/per-table analysis of the main-text figures and tables, "
        "cross-referenced with available source data files.\n\n"
        f"{analysis_text}\n"
    )

    # --- Build simplified data block for final assessment (≤ FINAL_DATA_MAX_CHARS chars) ---
    simplified_lines: list[str] = []
    if files_for_final:
        simplified_lines.append(
            "## Simplified Source Data for Final Review\n"
            "The following data files were flagged by the figure/table assessment as requiring "
            "additional review in the final assessment:\n"
        )
        remaining = FINAL_DATA_MAX_CHARS
        for entry in files_for_final:
            if not isinstance(entry, dict):
                continue
            rel_path = entry.get("path", "")
            reason = entry.get("reason", "")
            if supplementary_dir and supplementary_dir.exists():
                disk_path = _resolve_data_path(supplementary_dir, rel_path)
                # For xlsx files, forward individual CSV sheets from _converted_spreadsheets
                # instead of the whole xlsx (which joins all sheets together).
                if disk_path and disk_path.suffix.lower() in {".xls", ".xlsx"}:
                    csv_dir = supplementary_dir / "_converted_spreadsheets"
                    prefix = disk_path.name
                    csv_candidates = sorted(csv_dir.glob(f"{prefix}__*.csv")) if csv_dir.is_dir() else []
                    if csv_candidates:
                        for csv_path in csv_candidates:
                            sheet_label = csv_path.stem.partition("__")[2] or csv_path.stem
                            csv_rel = csv_path.relative_to(supplementary_dir).as_posix()
                            try:
                                sheet_text = csv_path.read_text(encoding="utf-8", errors="replace")
                            except Exception:
                                sheet_text = "(unreadable)"
                            header = f"#### `{csv_rel}` — {reason} (Sheet: {sheet_label})\n"
                            block = f"{header}{sheet_text}\n"
                            if len(block) <= remaining:
                                simplified_lines.append(block)
                                remaining -= len(block)
                            else:
                                simplified_lines.append(
                                    f"{header}{sheet_text[:remaining]}"
                                    f"\n[!! TRUNCATED: further {len(sheet_text) - remaining:,} chars omitted due to final budget. !!]\n"
                                )
                                remaining = 0
                                break
                        continue  # handled CSV sheets above, skip the generic path
                # Non-xlsx path: read the file directly.
                content = ""
                if disk_path and disk_path.is_file():
                    content, truncated = _extract_file_content_preview(
                        disk_path, supplementary_dir,
                    )
                header = f"#### `{rel_path}` — {reason}\n"
                block = f"{header}{content}\n"
                if len(block) <= remaining:
                    simplified_lines.append(block)
                    remaining -= len(block)
                else:
                    over = len(block) - remaining
                    simplified_lines.append(
                        f"{header}{content[:remaining]}"
                        f"\n[!! TRUNCATED: further {over:,} chars omitted due to final budget. !!]\n"
                    )
                    remaining = 0
                    break
            else:
                simplified_lines.append(f"#### `{rel_path}` — {reason}\n(File not available on disk)\n")
            if remaining <= 0:
                break

    # Add a truncation warning for the simplified block too
    simplified_data_block = "\n".join(simplified_lines)
    if simplified_data_block and any("[!! TRUNCATED" in simplified_data_block for _ in [1]):
        simplified_data_block = (
            "!! TRUNCATION WARNING: Some of the data below was truncated "
            "to fit in this prompt. Files marked with `[!! TRUNCATED ... !!]` are incomplete.\n\n"
            f"{simplified_data_block}"
        )

    print(
        f"[info] figure/table assessment: analysis_len={len(analysis_text)} "
        f"files_for_final={len(files_for_final)} "
        f"simplified_data_len={len(simplified_data_block)} "
        f"any_truncated={any_truncated} "
        f"total_files={total_files_selected} "
        f"clipped_files={clipped_files_count} "
        f"uniform_threshold_B={uniform_clip_threshold_B}",
        flush=True,
    )

    # Store truncation status in the output for downstream reporting
    return {
        "figure_table_analysis": analysis_text,
        "analysis_block": analysis_block,
        "files_for_final_step": files_for_final,
        "simplified_data_block": simplified_data_block,
        "any_data_truncated": any_truncated,
        "total_files_selected": total_files_selected,
        "clipped_files_count": clipped_files_count,
        "uniform_clip_threshold_B": uniform_clip_threshold_B,
    }
