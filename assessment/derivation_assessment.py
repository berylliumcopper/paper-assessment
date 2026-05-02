"""AI assessment of mathematical derivations in a paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a rigorous mathematical physics reviewer. Your task is to examine the mathematical derivations and theoretical analysis in a scientific paper.

!! IMPORTANT -- CONTEXT !!
Derivations in scientific papers serve different purposes depending on paper type:
- In **experimental** papers: Derivations are typically approximations or phenomenological models used to interpret data (e.g., a two-level fit model, a conductivity formula with modified parameters). Assess whether the level of approximation is appropriate for the data quality and whether hidden assumptions could bias the interpretation.
- In **theoretical** papers: Derivations are the core contribution — new formalisms, exact solutions, or controlled approximations. Assess mathematical rigor, completeness, and whether all steps are physically justified.
- In **computational/pipeline** papers: Derivations typically underpin the algorithm (e.g., deriving finite-difference schemes, convergence bounds, or error estimates). Assess whether the numerical implementation correctly reflects the derived formulas.

!! IMPORTANT -- PER-DERIVATION-STEP ANALYSIS REQUIRED !!
Each distinct derivation, approximation, or modeling step must be analyzed **separately**. 
Group related algebraic manipulations that serve the same purpose, but separate different derivational threads.

For each derivation step, answer the following questions **separately**:

1. **Assumptions used** — What explicit/implicit assumptions are made (symmetries, linearity, boundary conditions, continuum vs discrete, perturbative limits, etc.)?
2. **Assumption reasonableness** — Are the assumptions standard, well-justified, and their bounds of validity stated? Or are they ad-hoc, untested, or likely to break under realistic conditions?
3. **Parameters / formulas / models taken** — What specific formulas, parameters, or models are taken as input? Are they from prior work, standard references, or derived within the paper?
4. **Parameter reasonableness** — Are the parameters/models appropriate for the system under study? Are the numerical values physically plausible?
5. **Mathematical correctness** — Is each algebraic or analytical step valid? Check for dimensional consistency, sign errors, missing terms, incorrect integration/differentiation, division by zero, and hidden assumptions that change as the derivation proceeds.
6. **Formula-claim consistency** — Do the derived formulas support what the text claims? Check whether the equations actually imply the conclusions drawn in the text.

After your per-step analysis, add a **summary section** covering:
- **Overall derivation rigor**: Are the derivations logically sound, complete, and properly referenced?
- **Key concerns**: List any suspicious or problematic steps that need further attention.
- **Files for final review**: If any specific source data files or supplementary materials need direct review by the final assessment (because a suspicious result depends on them), list them.

Return valid JSON with this exact structure:
{
  "derivation_analysis": "string — detailed markdown text covering all derivation steps with per-step analysis",
  "analysis_block": "string — formatted block for the final assessment prompt",
  "files_for_final_step": [
    {
      "path": "string — relative path to a file that must be forwarded to the final assessment",
      "reason": "string — why this file needs final review"
    }
  ],
  "simplified_derivation_block": "string — key questionable formulas or derivations for the final review"
}"""


def run_derivation_assessment(
    *,
    article_dir: Path,
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """
    Run AI assessment of mathematical derivations in the paper.

    Returns dict with:
      - derivation_analysis (str): detailed per-step derivation analysis (markdown)
      - analysis_block (str): formatted block for the final assessment prompt
      - files_for_final_step (list[dict]): files still needing final review
      - simplified_derivation_block (str): key questionable formulas/derivations
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

    main_text_md = (
        "## Primary Article Markdown\n"
        f"{primary_markdown}\n\n"
        "## Converted PDF Markdown\n"
        f"{converted_markdown}\n\n"
    )

    # --- Build user prompt ---
    user_prompt = (
        "Analyze the mathematical derivations, theoretical analysis, and modeling in the following paper. "
        "Focus on the derivation steps described in the system prompt.\n\n"
        f"{main_text_md}\n"
        "For each derivation step, answer the questions described in the system prompt. "
        "Be specific, reference exact equation numbers where possible."
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
        )
    except Exception as exc:
        print(f"[warn] derivation assessment AI call failed: {exc}", flush=True)
        result = {
            "derivation_analysis": f"Derivation assessment AI call failed: {exc}",
            "files_for_final_step": [],
            "simplified_derivation_block": "",
        }

    if not isinstance(result, dict):
        result = {
            "derivation_analysis": "Derivation assessment returned non-dict response.",
            "files_for_final_step": [],
            "simplified_derivation_block": "",
        }

    analysis_text = str(result.get("derivation_analysis", ""))
    files_for_final = result.get("files_for_final_step", [])
    if not isinstance(files_for_final, list):
        files_for_final = []

    simplified_derivation = str(result.get("simplified_derivation_block", ""))

    # --- Build analysis block for final prompt ---
    analysis_block = (
        "## Derivation Assessment\n"
        "The following is a per-step analysis of the mathematical derivations, "
        "covering assumptions, correctness, and formula-claim consistency.\n\n"
        f"{analysis_text}\n"
    )

    print(
        f"[info] derivation assessment: analysis_len={len(analysis_text)} "
        f"files_for_final={len(files_for_final)} "
        f"simplified_derivation_len={len(simplified_derivation)}",
        flush=True,
    )

    return {
        "derivation_analysis": analysis_text,
        "analysis_block": analysis_block,
        "files_for_final_step": files_for_final,
        "simplified_derivation_block": simplified_derivation,
    }
