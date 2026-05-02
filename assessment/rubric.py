"""Prompt contract and normalization for paper assessment."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are a rigorous scientific reviewer.
You must judge claims using evidence and first-principles reasoning.
Do not trust paper statements blindly. Explicitly separate facts, inferences, and uncertainty.

SCORING PHILOSOPHY (global):
- Use a deductive approach: start from strict standards and subtract for each limitation, hidden assumption, or unverified step you identify.
- The numerical scores MUST match the tone of your written analysis: do not give high scores where you have listed major gaps or plausible alternatives.
- Be calibrated: 90+ is exceptional; 50-70 is typical good science with limitations; below 50 indicates serious issues or claims that outrun the evidence.
- The user message contains a separate **Criterion rubric** with detailed guidance per dimension (derivations, methods, explanation uniqueness, integrity, replication, openness) plus novelty/impact assessments (problem importance, technical advance, area change, community impact, expansion potential). Follow that rubric when filling `q1`–`q11` and the two score categories (`scores.reliability` and `scores.novelty_impact`).

!! MULTI-RUN ASSESSMENT CONTEXT !!
The preceding analysis step (step 9 in the pipeline) was run **multiple times** in sequence. The order reflects execution order, not assessment type. The context below contains one or more assessment blocks, each identified by its type header (e.g., `## Figure/Table Data Assessment`, `## Derivation Assessment`, `## Pipeline/Software Assessment`).

**You must combine information from all available blocks** into your final assessment. Follow these rules:
1. If a potential problem is spotted in **only one** assessment block but the reasoning is **robust and well-founded**, it should still be included in your final report — do not discard it just because other blocks did not independently confirm it.
2. If **multiple** assessment blocks independently flag the **same concern**, that concern should be given **extra weight** in your assessment.
3. If two assessment blocks disagree on a point, evaluate both sides using your own judgment rather than blindly averaging.
4. Each assessment block is a **single pass** over a subset of evidence — treat them as expert consultants providing specialized input.

Return valid JSON only."""


CRITERION_RUBRIC_MARKDOWN = """## Criterion rubric (detailed)\n
Apply the following **subsection by subsection** when populating the JSON fields and the corresponding `scores` keys. Do not collapse these into generic praise.\n
\n
### (1) Derivations and theory — `q1_derivations`, `scores.reliability.derivation_rigor`\n
- Enumerate nontrivial derivations, approximations, and idealizations. For each, state the explicit assumptions (symmetries, linearity, neglecting terms, continuum vs discrete, etc.).\n
- **Correctness:** Is each step valid under the stated assumptions? Flag missing steps, dimensional inconsistency, or conflation of sufficient with necessary conditions.\n
- **Assumptions:** Are they standard, bounds-stated, or ad hoc? If the result is sensitive to an assumption that is not tested in the paper, say so.\n
- **Score calibration:** 80+ only if derivations are complete or gaps are minor and clearly flagged. <60 if core steps are hand-waved, wrong under stated limits, or assumptions contradict the experiment.\n
- **WARNING — score inflation risk:** For experimental papers, interpretive derivations (models used to fit or explain data) are NOT evidence of theoretical depth. A well-fitting phenomenological model does NOT make the derivation_rigor score high — it only shows the data processing was adequate. Do NOT inflate this score just because the paper contains equations. The derivation_rigor score should reflect the **standalone theoretical value** of the derivations, not their role in the narrative.\n
- **Type-specific points:**\n
  - *Experimental papers:* Derivations here are typically approximations or phenomenological models used to interpret data (e.g., a two-level fit, Drude model with modifications). Assess whether the level of approximation is appropriate for the data quality. Do not expect the same rigor as a pure theory paper.\n
  - *Theoretical papers:* This is often the core contribution. Assess mathematical completeness, whether boundary conditions and convergence criteria are stated, and whether the derivation truly supports the physical conclusions drawn.\n
  - *Computational/pipeline papers:* Derivations here usually underpin algorithms (finite-difference stencils, convergence bounds, error estimates). Assess whether the derivation is correctly translated into the code and whether numerical approximations are justified.\n
\n
### (2) Experimental / numerical / computational methods — `q2_experimental_and_processing_methods`, `scores.reliability.experimental_validity`\n
- For each main measurement, numerical simulation, or computational analysis chain: describe the setup, instrument/code, calibration, parameters, and processing pipeline.\n
- **Does the setup actually measure / simulate / compute what is claimed?** (Selectivity, confounders, alternative couplings, geometry, field homogeneity, convergence criteria, algorithmic correctness, etc.)\n
- **Uniqueness:** Could another physical mechanism or artifact produce similar-looking results with this pipeline? Cite what would falsify the claim or narrow alternatives. Be creative — do not limit yourself to alternatives the paper considered.\n
- **Score calibration:** 80+ requires clear traceability from raw observable to claim; 50–75 for standard methods with unquantified systematics; <50 for ambiguous setup, missing controls, or processing that can fabricate structure.\n
- **Type-specific points:**\n
  - *Experimental papers:* Assess sample preparation, instrument calibration, baselines, noise, and data processing (filtering, binning, background subtraction, fitting, cuts).\n
  - *Theoretical papers:* Assess analytical/numerical methods — convergence (k-points, basis sets, time steps), numerical stability, error estimates, and whether the numerical method is appropriate for the problem.\n
  - *Computational/pipeline papers:* Assess software architecture, algorithmic choices, implementation correctness, and whether the pipeline's outputs can be validated against known results.\n
\n
### (3) Explanation uniqueness — `q3_explanation_uniqueness`, `scores.reliability.evidence_uniqueness`\n
- Does the data/model/output **compel** the story, or is it **consistent with** the story among others? List concrete alternative mechanisms or interpretations that the same results could support.\n
- Note where the paper uses post-hoc rationalization, cherry-picked ranges, or narrative that runs ahead of the shown evidence.\n
- When the paper rules out alternatives, be careful about whether these are genuinely ruled out by the evidence or merely by the paper's own narrative.\n
- **Score calibration:** 80+ only if alternatives are clearly weaker on the actual evidence shown. 50–75 when several explanations remain viable. <50 if the main claim is one of many equally plausible reads of the data.\n
- **Type-specific points:**\n
  - *Experimental papers:* Could a different physical mechanism or experimental artifact produce similar measurements? Is the claimed phenomenon the simplest explanation?\n
  - *Theoretical papers:* Could a different theoretical formalism or approximation produce the same predicted values? Are the results robust against reasonable changes to the model parameters?\n
  - *Computational/pipeline papers:* Could a different algorithm or implementation produce similar benchmark results? Is the claimed improvement specific to the chosen test cases?\n
\n
### (4) Data and figure integrity — `q4_data_and_figure_integrity`, `scores.reliability.data_integrity`\n
- Check **internal consistency** between text, numbers in tables, figure axes/legends, and (when images are provided) what is visibly plotted. Flag scale mismatches, sign errors, and impossible combinations.\n
- Comment on **artifact risk** (saturating detectors, over-smoothing, aggressive background removal, arbitrary color scales, selection bias in shown cuts, numerical convergence artifacts, rounding errors).\n
- Call out **unnatural** or **too clean** results only when you can tie it to a concrete visual or reported statistic — avoid vague suspicion.\n
- **Processing-pipeline consistency:** Compare the data-processing pipeline described in the text with what the processed data actually look like.\n
- **Score calibration:** 90+ when figures and text align, limitations are visible, and artifact risks are very low; 60-80 when largely consistent with some artifact risks; <50 for contradictions or strong mismatch between claim and what the figure can support.\n
- **Type-specific points:**\n
  - *Experimental papers:* Check measurement noise levels, error bar sizes, detector/calibration limits, and consistency across replicate measurements.\n
  - *Theoretical papers:* Check that numerical results match the derived formulas (e.g., plot a sanity check), that convergence parameters are reported, and that the numerical values are physically plausible.\n
  - *Computational/pipeline papers:* Check that benchmark results and performance metrics are reproducible, that runtime/memory measurements include proper statistics (not single runs), and that comparison baselines are appropriate.\n
\n
### (5) Replications and related systems — `q5_replications_and_related_systems`, `scores.reliability.replication_support`\n
- Separate **direct replication** of this exact result from **consistency** with other materials, similar experiments, alternative calculations, or comparable software tools in the same class of problems.\n
- For very new work, weigh absence of direct replication **lightly** if it is still early; weigh **heavily** whether results conflict with well-established behavior in related systems without a convincing reason.\n
- Use the related-work summaries as context, not as fact — verify claims against the target paper's evidence where possible.\n
- **Score calibration:** 80+ for independent confirmation or very strong triangulation. 50–70 for plausible consistency but no direct replication. <50 for isolated claims that clash with a broad literature or lack any external anchor.\n
- **Type-specific points:**\n
  - *Experimental papers:* Are the results consistent with other experiments on similar materials/systems? Do they contradict well-established measurements?\n
  - *Theoretical papers:* Do the predictions agree with numerical simulations or analytical results from related models? Are there independent calculations that confirm the key results?\n
  - *Computational/pipeline papers:* Has the algorithm/code been validated against existing benchmarks or standard test cases? Do independent implementations produce similar results?\n
\n
### (6) Data, code, and reproducibility (openness) — `q6_data_code_openness`, `scores.reliability.openness`\n
*This is a **distinct** subsection: do not merge its logic into (4). Here you judge what was **shared**, not whether in-paper figures look consistent.*\n
- **Primary source — Figure/Table Assessment:** The preceding **1st targeted assessment: Figure/Table Data Assessment** step examined the available source data files directly and produced an **overall openness level** (0 = no source data, A = figure-level exports only, B = processed data, C = complete raw data). If different parts or figures have different levels of openness, a clarification about which parts or figures are in each level of openness is also provided. Use these information as your **primary evidence** — use it rather than re-examining raw files, and when different levels of openness are present for different runs or different parts or figures, use the average as your final score.\n
- **Secondary source — audit log:** The **Source data and code — system audit** block documents what URLs were reachable. Use it to corroborate the figure/table assessment.\n
- **WARNING — score inflation risk:** The plots in the extended data figures and supplementary are not considered as additional source data. Only consider the data files to be source data.\n
- **Simplified data block:** If a **Simplified Source Data for Final Review** block is present, those specific files were flagged as needing your direct review.\n
- **(1) No online claim → not open-sourced:** If the article text does **not** explicitly say that **source data and/or code** is available **online**, treat **data** as **not open-sourced**.\n
- **(2) If there is a claim:** Cross-reference the figure/table assessment's overall classification against the audit.\n
- **Score calibration (strict rule):** Use the figure/table assessment's **overall openness level**:\n
   - **(0) No source data** or **(A) Figure-level exports only** → score **≤ 49**.\n
   - **(B) Processed data** → score **50–79**.\n
   - **(C) Complete raw data** → score **80+**.\n
- **OVERSIZED FILES — known to exist with confirmed sizes (separate rule from “cannot verify”):**\n
  The audit log may contain a “KNOWN TO EXIST — NOT AUTO-DOWNLOADED” section. This is **not** the same as “link exists but could not be verified” — in this case the file is **confirmed to exist** at the repository, its size is known, and it was only skipped because it exceeded the pipeline’s automatic download size limit. The file is downloadable manually.\n
  **Use the file SIZE as direct evidence of openness level — the size tells you whether the data is plausibly complete:**
  - For oversize cases, the size itself is critical information. Judge openness by comparing the file size to what complete raw data for this type of experiment/simulation would be: if the size is **consistent with full raw data** (instrument output, individual trials, complete simulation snapshots), classify as **level (C)**. If the size is **much smaller than what complete raw data should be** in this area, it is likely processed/summarized data — classify as **level (B)**. 
  - The fact that the paper deposited such large files on a public repository is **strong evidence of data sharing at a high openness level**. Do not penalize for the pipeline download limit.
- **Type-specific points:**\n
  - *Experimental papers:* Source data (raw instrument output, processed measurement tables) is the primary openness concern. Code for data analysis is secondary.\n
  - *Theoretical papers:* Source data is less central; what matters more is whether the derivation details, model parameters, and simulation settings are fully disclosed.\n
  - *Computational/pipeline papers:* **Code availability carries the most weight** — open-sourced code with documentation and tests should score highly even if no experimental data is shared. Closed-source code for a claimed pipeline advance is a major limitation.\n
\n
### (7) Problem importance / relevance — `q7_problem_importance`, `scores.novelty_impact.problem_importance`\n
- Is the problem studied by this paper **important**? Does **many people care about it** in the field or across fields?\n
- Assess the significance of the problem itself, independent of the solution proposed: is it a foundational question, a practical bottleneck, a niche curiosity, or a solved problem being rehashed?\n
- Do NOT blindly believe the paper's own claims of importance. Instead, judge the importance of the problem based on the actual contents and related works.\n
- **Score calibration:** 80+ for problems that are widely recognized as central, timely, or high-stakes; 50–70 for solid but specialized problems that a sub-community cares about; <50 for niche, obscure, or low-impact problems.\n
- **Type-specific points:**\n
  - *Experimental papers:* Is the measured phenomenon or material system of broad interest? Does it address a fundamental open question?\n
  - *Theoretical papers:* Does the theoretical framework solve a long-standing problem or open a new way of thinking about a known issue?\n
  - *Computational/pipeline papers:* Does the new algorithm/tool solve a practical bottleneck that many researchers face? Is it applicable beyond the specific system studied?\n
\n
### (8) Technical/Method advance — `q8_technical_method_advance`, `scores.novelty_impact.technical_method_advance`\n
- How is the technical approach or method different from previous works? Does this work develop a genuinely **new method/pipeline**, **adapt an existing approach to a new system/problem**, or **repeat similar things** that have been done before?\n
- **Score calibration:** 80+ for genuinely new methodology; 50–70 for adaptation of existing methods to new domains with non-trivial modifications; <50 for essentially repeating established work with minimal technical novelty.\n
- **WARNING — theoretical advance vs. interpretive modeling:** Applying a standard theoretical model (e.g., Drude model, Fermi liquid theory, mean-field approximation) to interpret experimental data is NOT a theoretical advance — it is part of the experimental analysis. A theoretical advance requires a new formalism, framework, or non-trivial derivation with standalone value. Be especially careful with experimental papers that cite several equations — many equations does not equal theoretical novelty.\n
- **Type-specific points:**\n
  - *Experimental papers:* Advance is in measurement technique, experimental design, sample synthesis method, or novel combination of existing techniques. Assess whether the new measurement provides genuinely new information or just incremental improvement.\n
  - *Theoretical papers:* Advance is in theoretical framework, formulation, analytical solution, or modeling approach. Assess whether the new theory predicts genuinely new phenomena or merely recasts existing understanding in new language.\n
  - *Computational/pipeline papers:* Advance is in algorithm, software architecture, computational pipeline, or novel combination of methods. Assess whether the code/tool genuinely enables new science or is a routine implementation of known methods.\n
\n
### (9) How this work changes its area — `q9_area_change`, `scores.novelty_impact.area_change`\n
- Does this work open **new possibilities** in its specific area? Would a researcher see this work as redirecting or substantially advancing the direction of the sub-field?\n
- Distinguish between **paradigm-shifting**, **substantial advance**, and **incremental**.\n
- Do NOT blindly believe the paper's own claims. Judge based on actual contents and related works.\n
- **Score calibration:** 80+ for paradigm-shifting or opening entirely new sub-directions; 50–70 for a meaningful advance; <50 for marginal change that does not alter the trajectory of the area.\n
- **Type-specific points:**\n
  - *Experimental papers:* Does the new measurement technique or finding enable a new class of experiments or challenge a dominant paradigm?\n
  - *Theoretical papers:* Does the new theory reframe how the community thinks about a problem? Does it open new directions for calculation?\n
  - *Computational/pipeline papers:* Does the new tool/method enable computational studies that were previously impossible? Does it establish a new benchmark or standard?\n
\n
### (10) Impact on the broader scientific community — `q10_community_impact`, `scores.novelty_impact.community_impact`\n
- Assess the broader importance: does the work provide **theoretical understanding** (principles, mechanisms, frameworks) or **practical application** (tools, datasets, protocols, engineering solutions)?\n
- Does it contain potential **textbook-level findings** or results that might be **commercialized**?\n
- Consider both immediate benefit and long-term significance.\n
- Do NOT blindly believe the paper's own claims.\n
- **Score calibration:** 80+ for textbook-level contributions or clear commercializable potential; 50–70 for useful but contained contributions; <50 for minimal broader impact.\n
- **Type-specific points:**\n
  - *Experimental papers:* Impact depends on whether the measurement/materials are a community resource (e.g., a widely used dataset, a new material platform).\n
  - *Theoretical papers:* Impact depends on whether the theoretical framework is adopted by others (e.g., a new formalism or model that explains diverse phenomena).\n
  - *Computational/pipeline papers:* Impact depends on whether the code/tool is adopted as a community standard, the number of potential users, and the breadth of applicability.\n
\n
### (11) Expansion potential (follow-up research) — `q11_expansion_potential`, `scores.novelty_impact.expansion_potential`\n
- Predict whether many researchers will **expand their research based on this work** or **make use of its results**.\n
- Consider: Does the work provide a new resource, dataset, baseline, method, or finding that others would naturally build upon? Or is it a self-contained result with limited follow-up potential?\n
- Do NOT blindly believe the paper's own claims.\n
- **Score calibration:** 80+ for high potential to spawn many follow-ups; 50–70 for moderate follow-up interest; <50 for niche interest or self-contained result unlikely to generate significant follow-on work.\n
- **Type-specific points:**\n
  - *Experimental papers:* Can others repeat/adapt the measurement for different materials? Is the material system a new platform that many groups will study?\n
  - *Theoretical papers:* Can the theoretical method be applied to other systems or problems? Does it generate testable predictions?\n
  - *Computational/pipeline papers:* Can the code/tool be extended or adapted by others? Is it modular and well-documented enough to build upon?\n
"""


EXPECTED_SCHEMA_HINT = {
    "paper_understanding": {
        "methods": "string",
        "results": "string",
        "conclusions": "string",
    },
    "related_work_summary": "string",
    "q1_derivations": [
        {
            "derivation_name": "string",
            "description": "string",
            "correctness_assessment": "string",
            "assumption_reasonableness": "string",
            "evidence": ["string"],
        }
    ],
    "q2_experimental_and_processing_methods": [
        {
            "method_name": "string",
            "data_processing": "string",
            "properness_assessment": "string",
            "regime_flaws_or_limits": "string",
            "measurement_uniqueness": "string",
            "evidence": ["string"],
        }
    ],
    "q3_explanation_uniqueness": {
        "is_unique_explanation": "boolean_or_null",
        "analysis": "string",
        "alternative_explanations": ["string"],
        "evidence": ["string"],
    },
    "q4_data_and_figure_integrity": {
        "issues_found": ["string"],
        "processing_artifact_risks": ["string"],
        "analysis": "string",
        "evidence": ["string"],
    },
    "q5_replications_and_related_systems": {
        "replication_status": "string",
        "related_system_works": ["string"],
        "analysis": "string",
        "evidence": ["string"],
    },
    "q6_data_code_openness": {
        "data_open_sourced": "boolean_or_null",
        "code_open_sourced": "boolean_or_null",
        "reproducibility_what_is_shared": "string — state clearly: e.g. figure/plot values only, processed tables, or raw/primary + code adequate for full re-analysis",
        "details": "string — include whether openness matches paper claims; call out figure-only deposits",
        "evidence": ["string — cite file sizes, row/column structure, or absence of primary data"],
    },
    "q7_problem_importance": {
        "problem_importance_assessment": "string — assess whether the problem is important and widely cared about",
        "importance_level": "string — one of: central_and_timely, specialized_sub_community, niche_or_low_impact",
        "analysis": "string",
        "evidence": ["string"],
    },
    "q8_technical_method_advance": {
        "method_novelty_assessment": "string — describe how the technical/method approach differs from prior work",
        "advance_level": "string — one of: new_pipeline_or_methodology, adaptation_to_new_domain, routine_application",
        "analysis": "string",
        "evidence": ["string"],
    },
    "q9_area_change": {
        "area_change_assessment": "string — how this work changes or opens possibilities in its specific area",
        "change_level": "string — one of: paradigm_shifting, substantial_advance, incremental",
        "analysis": "string",
        "evidence": ["string"],
    },
    "q10_community_impact": {
        "community_impact_assessment": "string — broader importance for the scientific community",
        "impact_type": "string — one of: theoretical_understanding, practical_application, both, minimal",
        "textbook_or_commercial_potential": "string — describe any textbook-level findings or commercial potential",
        "analysis": "string",
        "evidence": ["string"],
    },
    "q11_expansion_potential": {
        "expansion_potential_assessment": "string — prediction of follow-up research based on this work",
        "follow_up_level": "string — one of: high, moderate, niche",
        "analysis": "string",
        "evidence": ["string"],
    },
    "final_reasoning_comment": "string",
    "scores": {
        "reliability": {
            "derivation_rigor": "0-100 — calibration: Criterion rubric (1)",
            "experimental_validity": "0-100 — calibration: Criterion rubric (2)",
            "evidence_uniqueness": "0-100 — calibration: Criterion rubric (3)",
            "data_integrity": "0-100 — calibration: Criterion rubric (4)",
            "replication_support": "0-100 — calibration: Criterion rubric (5)",
            "openness": "0-100 — calibration: Criterion rubric (6) only; do not reuse (4)",
            "overall": "0-100 — computed automatically as weighted or rounded average of the six reliability dimension scores; the model should not override this",
        },
        "novelty_impact": {
            "problem_importance": "0-100 — calibration: Criterion rubric (7)",
            "technical_method_advance": "0-100 — calibration: Criterion rubric (8)",
            "area_change": "0-100 — calibration: Criterion rubric (9)",
            "community_impact": "0-100 — calibration: Criterion rubric (10)",
            "expansion_potential": "0-100 — calibration: Criterion rubric (11)",
            "overall": "0-100 — computed automatically as rounded average of the five novelty/impact dimension scores; the model should not override this",
        },
        "overall": "0-100 — computed automatically as rounded average of the two category overall scores (reliability.overall and novelty_impact.overall); the model should not override this",
    },
    "confidence": "0-100 integer",
    "limitations": ["string"],
    "citation_traces": [
        {
            "source_id": "string",
            "claim": "string",
            "supporting_excerpt": "string",
        }
    ],
}


# Weight matrix for reliability dimension scoring.
# Rows: dimensions (1-6). Columns: [a_i (exp), b_i (theory), c_i (comp)].
# Order: derivation_rigor, experimental_validity, evidence_uniqueness, data_integrity, replication_support, openness.
RELIABILITY_WEIGHT_MATRIX: list[list[float]] = [
    [0.5, 3.0, 1.0],   # derivation_rigor
    [3.0, 0.5, 1.0],   # experimental_validity
    [2.0, 1.5, 1.5],   # evidence_uniqueness
    [3.0, 1.0, 2.0],   # data_integrity
    [2.0, 1.5, 1.5],   # replication_support
    [1.5, 0.5, 3.0],   # openness
]

RELIABILITY_DIMENSION_KEYS = [
    "derivation_rigor",
    "experimental_validity",
    "evidence_uniqueness",
    "data_integrity",
    "replication_support",
    "openness",
]


def build_user_prompt(
    *,
    paper_context_block: str,
    related_work_payload: dict,
    theory_comp_exp_distribution: list[float] | None = None,
) -> str:
    # Shortlist: target-focused 1-2-paragraph narratives (primary for assessment), plus one-para scratch + scores
    reports = related_work_payload.get("reports", [])
    not_sel = related_work_payload.get("not_selected_search_hits") or []
    report_lines: list[str] = ["## Related work — AI shortlist (relate to the **target** paper)"]
    report_lines.append(
        "The target paper and problem are in **Target Paper Context** above. "
        "The following are search hits that were kept on a shortlist. For each, prefer the **Target-focused narrative**; "
        "the one-paragraph line is a rough first pass for scoring only."
    )
    if reports:
        for r in reports:
            title = r.get("title", "Unknown")
            tfn = (r.get("target_focused_narrative") or "").strip()
            summary = r.get("summary", "No summary.")
            relevance = r.get("relevance_score", 0)
            should_read = " (mark: consider full PDF)" if r.get("should_read_full") else ""
            report_lines.append(f"### {title}{should_read}")
            report_lines.append(f"**Relevance (rough):** {relevance}/10")
            if tfn:
                report_lines.append("**Target-focused narrative (1-2 paragraphs; use this for judgment vs the target):**")
                report_lines.append(tfn)
            report_lines.append("**One-paragraph scratch (secondary):** " + summary)
            report_lines.append("")
    else:
        report_lines.append(f"(No shortlisted related papers. Raw payload: {str(related_work_payload)[:2000]})")
    other_lines: list[str] = [
        "",
        "## Related work — other search candidates (not on the shortlist)",
        "One **sentence** each, from **metadata/one paragraph of abstract only** (for landscape; not a full-paper review).",
    ]
    if not_sel:
        for r in not_sel:
            t = r.get("title", "Unknown")
            b = (r.get("brief_abstract") or r.get("abstract") or "")[:2000]
            y = r.get("year", "")
            src = r.get("source", "")
            yv = f" ({y})" if y else ""
            other_lines.append(f"- **{t}**{yv} [{src}] — {b}")
    else:
        other_lines.append("- (None or all candidates were shortlisted.)")
    report_block = "\n".join(report_lines) + "\n" + "\n".join(other_lines) + "\n"

    return (
        "Assess the target paper with rigorous skepticism and scientific reasoning. "
        "The **Criterion rubric** below is the authoritative, per-dimension guide; follow it section-by-section. "
        "If images (figures, tables, plots) are provided, you MUST use them for checks (4) and, where relevant, (2).\n\n"
        f"{CRITERION_RUBRIC_MARKDOWN}\n"
        "## Response requirements\n"
        "Fill all JSON fields implied by the rubric (`paper_understanding`, `q1`…`q11`, `related_work_summary`, `scores`, `confidence`, `limitations`, `citation_traces`). "
        "For `related_work_summary`, synthesize **both** the AI shortlist (lean on target-focused narratives and how they connect to the target) **and** the not-shortlist one-sentence items as background landscape—make clear which had deeper review. "
        "Include a one-paragraph `final_reasoning_comment`.\n"
        "Scores are divided into two categories: **reliability** (dimensions 1–6: derivations, methods, explanation uniqueness, integrity, replication, openness) and **novelty/impact** (dimensions 7–11: problem importance, technical advance, area change, community impact, expansion potential). "
        "Ensure every `scores` entry is justified by the corresponding `q` block—especially **openness (6)**, which must not be conflated with in-paper figure integrity (4). "
        "For (6), follow the **Source data and code — system audit** block at the end of Target Paper Context (statement scan, URL log, file sizes, archive manifests, and snippets).\n"
        "In the context below you will find three assessment blocks, each labeled by its type "
        "(e.g., **Figure/Table Data Assessment**, **Derivation Assessment**, **Pipeline/Software Assessment**). "
        "Assessment blocks of the same type may appear multiple times (independent runs). "
        "Their order in the context reflects execution sequence, not importance. "
        "Use them as primary evidence for checks (2), (4), and especially (6). "
        "!! MULTI-RUN COMBINATION INSTRUCTION !!\n"
        "Assessment blocks may have been run **multiple times** with the same input, producing independent analyses. "
        "You must combine information from all runs as follows:\n"
        "- If a concern appears in only **one** run but is **well-supported** by specific evidence, include it in your final report.\n"
        "- If a concern appears in **multiple** runs, give it **extra weight**.\n"
        "- Treat each run as an independent expert opinion; do not discard findings just because only one run caught them.\n\n"
        "If a **Simplified Source Data for Final Review** block is present, those data files were flagged specifically "
        "as needing your direct scrutiny — read them carefully."
        "This often happens when the figure/table assessment reports a significant flaw, inconsistency, or data-quality concern, "
        "and the source data entries that support that conclusion are attached in the report. "
        "You should double-check those entries yourself before adopting the assessment's conclusions into your own report.\n\n"
        "(SECTION LENGTH GUIDANCE"
        f"{' BASED ON this paper\'s exp/theory/comp distribution ' + str(theory_comp_exp_distribution) if theory_comp_exp_distribution else ''}: "
        "Do NOT force all sections to equal length. "
        "Let the depth match the paper's true emphasis."
        f"{' Allocate response length accordingly: If experimental proportion is high, write a longer q2 and q4 section. If theory proportion is high, write a longer, more detailed q1_derivations section. If computational proportion is high, write a longer q2 and q6 section.' if theory_comp_exp_distribution else ''}\n\n"
        "Return JSON with this shape (keys must match):\n"
        f"{EXPECTED_SCHEMA_HINT}\n\n"
        "## Target Paper Context\n"
        f"{paper_context_block}\n\n"
        f"{report_block}\n"
    )


def normalize_assessment(payload: dict[str, Any], theory_comp_exp_distribution: list[float] | None = None) -> dict[str, Any]:
    """Best-effort normalization and score clamping.

    If ``theory_comp_exp_distribution`` is provided (a list of 3 floats [exp, theory, comp] summing to 1.0),
    the reliability overall score is computed as a weighted average using RELIABILITY_WEIGHT_MATRIX.
    Otherwise falls back to simple arithmetic average.
    """
    normalized = dict(payload)
    scores = normalized.get("scores")
    if not isinstance(scores, dict):
        scores = {}

    # --- Reliability sub-scores ---
    reliability_keys = [
        "derivation_rigor",
        "experimental_validity",
        "evidence_uniqueness",
        "data_integrity",
        "replication_support",
        "openness",
    ]
    reliability_in = scores.get("reliability")
    if not isinstance(reliability_in, dict):
        reliability_in = scores  # flat fallback for backward compat
    clamped_reliability: dict[str, int] = {}
    for key in reliability_keys:
        value = reliability_in.get(key, 0)
        try:
            ivalue = int(value)
        except Exception:  # noqa: BLE001
            ivalue = 0
        clamped_reliability[key] = max(0, min(100, ivalue))

    rel_values = [clamped_reliability.get(k, 0) for k in RELIABILITY_DIMENSION_KEYS]
    if theory_comp_exp_distribution is not None and len(theory_comp_exp_distribution) == 3:
        # distribution: [exp, theory, comp]
        # weight matrix columns: [a_i (exp), b_i (theory), c_i (comp)]
        # w_i = exp * a_i + theory * b_i + comp * c_i
        raw_weights = [
            theory_comp_exp_distribution[0] * RELIABILITY_WEIGHT_MATRIX[i][0]
            + theory_comp_exp_distribution[1] * RELIABILITY_WEIGHT_MATRIX[i][1]
            + theory_comp_exp_distribution[2] * RELIABILITY_WEIGHT_MATRIX[i][2]
            for i in range(len(RELIABILITY_DIMENSION_KEYS))
        ]
        # Normalize weights to sum to 1.0
        total_weight = sum(raw_weights) or 1.0
        norm_weights = [w / total_weight for w in raw_weights]
        weighted_sum = sum(norm_weights[i] * rel_values[i] for i in range(len(rel_values)))
        clamped_reliability["overall"] = round(weighted_sum)
    else:
        clamped_reliability["overall"] = round(sum(rel_values) / max(len(rel_values), 1))

    # --- Novelty/Impact sub-scores ---
    novelty_keys = [
        "problem_importance",
        "technical_method_advance",
        "area_change",
        "community_impact",
        "expansion_potential",
    ]
    novelty_in = scores.get("novelty_impact")
    if not isinstance(novelty_in, dict):
        novelty_in = scores  # flat fallback for backward compat
    clamped_novelty: dict[str, int] = {}
    for key in novelty_keys:
        value = novelty_in.get(key, 0)
        try:
            ivalue = int(value)
        except Exception:  # noqa: BLE001
            ivalue = 0
        clamped_novelty[key] = max(0, min(100, ivalue))

    nov_values = [clamped_novelty.get(k, 0) for k in novelty_keys]
    clamped_novelty["overall"] = round(sum(nov_values) / max(len(nov_values), 1))

    # --- Top-level overall as average of the two category overalls ---
    cat_overalls = [clamped_reliability["overall"], clamped_novelty["overall"]]
    clamped_overall = round(sum(cat_overalls) / len(cat_overalls))

    clamped_scores: dict[str, Any] = {
        "reliability": clamped_reliability,
        "novelty_impact": clamped_novelty,
        "overall": clamped_overall,
    }
    normalized["scores"] = clamped_scores

    try:
        confidence = int(normalized.get("confidence", 0))
    except Exception:  # noqa: BLE001
        confidence = 0
    normalized["confidence"] = max(0, min(100, confidence))

    for key in [
        "q1_derivations",
        "q2_experimental_and_processing_methods",
        "limitations",
        "citation_traces",
    ]:
        if not isinstance(normalized.get(key), list):
            normalized[key] = []

    for key in [
        "q7_problem_importance",
        "q8_technical_method_advance",
        "q9_area_change",
        "q10_community_impact",
        "q11_expansion_potential",
    ]:
        if not isinstance(normalized.get(key), dict):
            normalized[key] = {
                "analysis": "Not provided — model may not have returned this field.",
                "evidence": [],
            }

    if not isinstance(normalized.get("paper_understanding"), dict):
        normalized["paper_understanding"] = {"methods": "", "results": "", "conclusions": ""}
    if not isinstance(normalized.get("related_work_summary"), str):
        normalized["related_work_summary"] = ""
    if not isinstance(normalized.get("final_reasoning_comment"), str):
        normalized["final_reasoning_comment"] = ""
    return normalized
