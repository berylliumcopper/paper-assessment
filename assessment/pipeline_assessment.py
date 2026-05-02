"""AI assessment of computational pipelines / software in a paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a rigorous computational science reviewer. Your task is to classify and assess the computational contribution in a scientific paper.

!! IMPORTANT -- TIER CLASSIFICATION FIRST !!
Read the paper and determine which tier best describes its computational content:

**Tier 1 — Routine computation (experimental data processing):**
- Using standard data-processing scripts (Python, MATLAB, Origin) to process experimental data (plotting, fitting, normalizing). This is part of experimental work, not a standalone computational contribution.
- Using well-established simulation codes (VASP, LAMMPS, Quantum ESPRESSO, Wien2k, etc.) to compute properties for a new system. Parameter tuning alone does not make it a pipeline advance.

**Tier 2 — Routine implementation for theory (numerical simulation):**
- Implementing a known theoretical model or numerical method in code for a specific system. If the algorithm itself is standard and the implementation is straightforward, this is a theoretical contribution, not a pipeline advance.
- Running standard simulations to test or validate a theoretical model.

**Tier 3 — Genuine computational/pipeline advance:**
- Developing a **new algorithm, architecture, software framework, or computational pipeline** that is itself a research contribution.
- Combining existing methods in a **novel, non-obvious way** that enables new capabilities.
- Creating a **new software tool, library, or platform** that others would use as a resource.
- Introducing a **new computational technique** (e.g., a new neural network architecture for scientific problems, a new Monte Carlo scheme, a new optimization method).

After classifying the tier, produce output accordingly. **All analysis must happen in this single response — do not leave Tier 1/2 analysis brief or empty:**

**For Tier 1 (routine experimental data processing):**
- State "TIER 1" clearly at the beginning.
- Then perform a **figure/table style analysis** right here. Treat the paper's computational work as experimental data processing and analyze:
  1. **Experimental methods or data pipeline**: Describe the measurement setup or data source, the processing steps (filtering, fitting, normalization, binning), and what the output represents.
  2. **Data reasonableness**: Are the processing steps appropriate for the data type? Could the processing introduce artifacts (smoothing that removes features, background subtraction that creates structure, fitting that over-interprets noise)?
  3. **Data provenance**: Trace the data from raw measurement through all processing steps to the final figures. Name the specific processing applied at each stage.
  4. **Consistency**: Check whether the processed data values are consistent with what would be expected from the described raw measurement and processing pipeline.
  5. **Openness**: Are the raw data and processing scripts shared? Can the data processing be reproduced?

**For Tier 2 (routine theory/numerical implementation):**
- State "TIER 2" clearly at the beginning.
- Then perform a **derivation/numerical analysis** right here. Treat the paper's computational work as theoretical simulation and analyze:
  1. **Theoretical model and numerical method**: Describe the theoretical model, the numerical method used to solve it, and the key parameters.
  2. **Correctness and convergence**: Is the numerical method appropriate? Are convergence criteria reported (k-points, basis sets, time steps, Monte Carlo sweeps)? Are the results numerically stable?
  3. **Implementation fidelity**: Does the numerical implementation faithfully represent the theoretical model? Are approximations clearly stated and justified?
  4. **Formula-output consistency**: Do the numerical results match what the theory predicts? Check specific data points against expected scaling relations or limiting cases.
  5. **Openness**: Are the simulation code and input parameters shared? Can the numerical results be reproduced?

**For Tier 3 (genuine pipeline advance):**
- State "TIER 3" clearly at the beginning.
- Then perform a **full per-module pipeline analysis** as follows.

!! PER-MODULE ANALYSIS (Tier 3 only) !!
Each distinct computational module, algorithm, or software component must be analyzed **separately**.

For each module, answer the following questions **separately**:

1. **Problem to solve** — What computational problem does this module address? What is the input, output, and expected behavior?
2. **Algorithmic/pipeline advance** — What is novel about this module? New architecture, new algorithm, novel combination of existing methods, or a new pipeline/toolchain? Distinguish genuinely new contributions from routine application of existing tools.
3. **Advance reasonableness** — Is it plausible that this method improves upon existing approaches? Is the claimed improvement theoretically justified or just asserted? Consider computational complexity, convergence, accuracy, and stability.
4. **Evidence of advancement** — What evidence is provided that the new method works? Better accuracy (comparison to benchmarks), efficiency (runtime/memory measurements), convergence properties, validation against experiments or known results? Is the evidence sufficient?
5. **Source code openness** — Is the code open-sourced? Where (GitHub, Zenodo, GitLab)? What license? Is there documentation, tests, or example usage? Can the results be reproduced?

After your per-module analysis, add a **summary section** covering:
- **Overall pipeline assessment**: Is the computational contribution sound, well-documented, and reproducible?
- **Key concerns**: List any suspicious or problematic aspects that need further attention.
- **Files for final review**: If any specific source code files, data files, or supplementary materials need direct review by the final assessment, list them.

Return valid JSON with this exact structure:
{
  "pipeline_tier": "string — one of: '1', '2', '3'",
  "pipeline_analysis": "string — detailed markdown text. For Tier 1/2: brief classification + justification. For Tier 3: full per-module analysis.",
  "analysis_block": "string — formatted block for the final assessment prompt",
  "files_for_final_step": [
    {
      "path": "string — relative path to a file that must be forwarded to the final assessment",
      "reason": "string — why this file needs final review"
    }
  ],
  "simplified_pipeline_block": "string — key questionable modules or code snippets for the final review"
}"""


def run_pipeline_assessment(
    *,
    article_dir: Path,
    api_key: str,
    api_model: str,
    api_base_url: str,
    is_gemini: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """
    Run AI classification and (if Tier 3) full assessment of computational pipelines in the paper.

    Returns dict with:
      - pipeline_tier (str): '1', '2', or '3'
      - pipeline_analysis (str): detailed analysis or brief classification note
      - analysis_block (str): formatted block for the final assessment prompt
      - files_for_final_step (list[dict]): files still needing final review
      - simplified_pipeline_block (str): key questionable modules/code snippets
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

    # --- Collect images from converted/ directory for visual context ---
    converted_dir = article_dir / "converted"
    main_text_images: list[Path] = []
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    if converted_dir.exists():
        for p in sorted(converted_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in image_extensions:
                main_text_images.append(p)

    # --- Build user prompt ---
    user_prompt = (
        "Analyze the computational pipelines, algorithms, and software methodology in the following paper. "
        "First classify into Tier 1/2/3 as described in the system prompt, then produce output accordingly.\n\n"
        f"{main_text_md}\n"
        "## Main-text Images (for visual context)\n"
        "The images below are provided alongside this text. Refer to them by filename.\n\n"
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
        print(f"[warn] pipeline assessment AI call failed: {exc}", flush=True)
        result = {
            "pipeline_tier": "3",
            "pipeline_analysis": f"Pipeline assessment AI call failed: {exc}",
            "files_for_final_step": [],
            "simplified_pipeline_block": "",
        }

    if not isinstance(result, dict):
        result = {
            "pipeline_tier": "3",
            "pipeline_analysis": "Pipeline assessment returned non-dict response.",
            "files_for_final_step": [],
            "simplified_pipeline_block": "",
        }

    pipeline_tier = str(result.get("pipeline_tier", "3"))
    analysis_text = str(result.get("pipeline_analysis", ""))
    files_for_final = result.get("files_for_final_step", [])
    if not isinstance(files_for_final, list):
        files_for_final = []
    simplified_pipeline = str(result.get("simplified_pipeline_block", ""))

    # --- Build analysis block for final prompt ---
    if pipeline_tier == "1":
        analysis_block = (
            "## 3rd targeted assessment: Pipeline/Software Assessment\n"
            "(Tier 1 — routine data processing; analysis substituted with figure/table style assessment)\n\n"
            f"{analysis_text}\n"
        )
    elif pipeline_tier == "2":
        analysis_block = (
            "## 3rd targeted assessment: Pipeline/Software Assessment\n"
            "(Tier 2 — routine theory implementation; analysis substituted with derivation style assessment)\n\n"
            f"{analysis_text}\n"
        )
    else:
        analysis_block = (
            "## 3rd targeted assessment: Pipeline/Software Assessment\n"
            "The following is a per-module analysis of the computational pipelines and software "
            "used or developed in this paper.\n\n"
            f"{analysis_text}\n"
        )

    print(
        f"[info] pipeline assessment: tier={pipeline_tier} "
        f"analysis_len={len(analysis_text)} "
        f"files_for_final={len(files_for_final)} "
        f"simplified_pipeline_len={len(simplified_pipeline)}",
        flush=True,
    )

    return {
        "pipeline_tier": pipeline_tier,
        "pipeline_analysis": analysis_text,
        "analysis_block": analysis_block,
        "files_for_final_step": files_for_final,
        "simplified_pipeline_block": simplified_pipeline,
    }
