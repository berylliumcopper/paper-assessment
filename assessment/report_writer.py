"""Write assessment outputs in JSON and Markdown."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


_PIPELINE_NAME: str | None = None

CATEGORY_LABELS = {
    "A": "Figure/Table only (3x)",
    "B": "Figure/Table (2x), Derivation",
    "C": "Figure/Table, Derivation (2x)",
    "D": "Derivation only (3x)",
    "E": "Figure/Table (2x), Pipeline",
    "F": "Figure/Table, Pipeline (2x)",
    "G": "Derivation (2x), Pipeline",
    "H": "Derivation, Pipeline (2x)",
    "I": "Figure/Table, Derivation, Pipeline",
}


def _pipeline_header() -> str:
    """Return a pipeline-info header line, e.g. ``paper-assessment (v0.4.0) -- model: gemini-2.0-flash``."""
    global _PIPELINE_NAME
    if _PIPELINE_NAME is None:
        try:
            readme = Path(__file__).resolve().parent.parent / "README.md"
            for line in readme.read_text(encoding="utf-8").splitlines():
                m = re.match(r"version\s+(\S+)", line.strip(), re.I)
                if m:
                    _PIPELINE_NAME = f"paper-assessment (v{m.group(1)})"
                    break
        except Exception:  # noqa: BLE001
            _PIPELINE_NAME = "paper-assessment"
        if _PIPELINE_NAME is None:
            _PIPELINE_NAME = "paper-assessment"
    return _PIPELINE_NAME


def write_assessment_outputs(
    *,
    article_dir: Path,
    assessment_payload: dict,
    run_metadata: dict,
) -> None:
    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "assessment.json").write_text(
        json.dumps(assessment_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    enriched_metadata = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        **run_metadata,
    }
    (article_dir / "assessment_run.json").write_text(
        json.dumps(enriched_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (article_dir / "assessment.md").write_text(_render_markdown(assessment_payload, run_metadata), encoding="utf-8")


def write_related_work_report(article_dir: Path, related_work_payload: dict, run_metadata: dict) -> None:
    """Write a detailed report of the related work search and paper summaries plus JSON dump."""
    # Write raw JSON payload
    (article_dir / "related_work.json").write_text(
        json.dumps(related_work_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Write formatted markdown report
    if not (related_work_payload.get("reports") or related_work_payload.get("not_selected_search_hits")):
        return
    lines: list[str] = []
    model = run_metadata.get("model", "unknown")
    lines.append(f"# Related Work Search Report — {_pipeline_header()} model={model}")
    lines.append("")
    lines.append(f"**Search Queries:** {related_work_payload.get('query', 'unknown')}")
    lines.append("")
    
    counts = related_work_payload.get("counts", {})
    lines.append("## Search Statistics")
    lines.append(f"- **Total Unique Papers Found:** {counts.get('merged_unique', 0)}")
    lines.append(f"- **Crossref:** {counts.get('crossref', 0)}")
    lines.append(f"- **arXiv:** {counts.get('arxiv', 0)}")
    lines.append(f"- **Semantic Scholar:** {counts.get('semantic_scholar', 0)}")
    lines.append(f"- **Web Search:** {counts.get('web_search', 0)}")
    lines.append("")

    not_sel = related_work_payload.get("not_selected_search_hits") or []
    if not_sel:
        lines.append("## Not-shortlisted search hits (one sentence from metadata / one abstract paragraph each)")
        lines.append("")
        for p in not_sel:
            t = p.get("title", "Unknown")
            lines.append(f"### {t}")
            lines.append(f"- {p.get('brief_abstract', p.get('abstract', ''))}")
            lines.append("")
    reports = related_work_payload.get("reports", [])
    if reports:
        lines.append("## Shortlisted related papers (target-focused + scratch summary)")
        lines.append("")
        for report in reports:
            title = report.get("title", "Unknown Title")
            year = report.get("year")
            year_str = f" ({year})" if year else ""
            lines.append(f"### {title}{year_str}")
            lines.append(f"- **Source:** {report.get('source', 'unknown').upper()}")
            lines.append(f"- **URL:** {report.get('url', 'none')}")
            if report.get("reference_pdf_path"):
                lines.append(f"- **Downloaded PDF:** `{report.get('reference_pdf_path', '')}`")
            lines.append("")
            tfn = report.get("target_focused_narrative", "").strip()
            if tfn:
                lines.append("**Target-focused narrative (1-2 paragraphs):**")
                lines.append(tfn)
                lines.append("")
            lines.append("**One-paragraph scratch:**")
            lines.append(report.get("summary", "No summary available."))
            lines.append("")
            if report.get("relevance_score"):
                lines.append(f"**Relevance Score:** {report['relevance_score']}/10")
            if report.get("should_read_full"):
                lines.append("**Note:** Full PDF may add rigor beyond the extract.")
            lines.append("")
            lines.append("---")
            lines.append("")

    (article_dir / "related_work_report.md").write_text("\n".join(lines), encoding="utf-8")


def _render_markdown(payload: dict, run_metadata: dict) -> str:
    understanding = payload.get("paper_understanding", {})
    q1 = payload.get("q1_derivations", [])
    q2 = payload.get("q2_experimental_and_processing_methods", [])
    q3 = payload.get("q3_explanation_uniqueness", {})
    q4 = payload.get("q4_data_and_figure_integrity", {})
    q5 = payload.get("q5_replications_and_related_systems", {})
    q6 = payload.get("q6_data_code_openness", {})
    q7 = payload.get("q7_problem_importance", {})
    q8 = payload.get("q8_technical_method_advance", {})
    q9 = payload.get("q9_area_change", {})
    q10 = payload.get("q10_community_impact", {})
    q11 = payload.get("q11_expansion_potential", {})
    scores = payload.get("scores", {})

    lines: list[str] = []
    model = run_metadata.get("model", "unknown")
    lines.append(f"# Paper Assessment — {_pipeline_header()} model={model}")
    lines.append("")

    # Assessment type info
    cat = run_metadata.get("paper_category")
    dist = run_metadata.get("theory_comp_exp_distribution")
    if cat:
        lines.append(f"- **Paper category:** {cat}")
    if dist:
        lines.append(f"- **Exp/Theo/Comp distribution:** {dist}")
    # Infer which assessment types were performed from category
    if cat and cat in CATEGORY_LABELS:
        lines.append(f"- **Assessment types:** {CATEGORY_LABELS[cat]}")
    lines.append("")

    lines.append("## Paper Understanding")
    lines.append("")
    lines.append(f"- **Methods:** {understanding.get('methods', '')}")
    lines.append(f"- **Results:** {understanding.get('results', '')}")
    lines.append(f"- **Conclusions:** {understanding.get('conclusions', '')}")
    lines.append("")
    lines.append("## Related Work Search Summary")
    lines.append("")
    lines.append(payload.get("related_work_summary", ""))
    lines.append("")
    lines.append("## (1) Derivations: Correctness and Assumptions")
    lines.append("")
    lines.extend(_render_list_of_dicts(q1))
    lines.append("")
    lines.append("## (2) Experimental and Data Processing Methods")
    lines.append("")
    lines.extend(_render_list_of_dicts(q2))
    lines.append("")
    lines.append("## (3) Explanation Uniqueness")
    lines.append("")
    lines.append(f"- **Unique explanation:** {q3.get('is_unique_explanation')}")
    lines.append(f"- **Analysis:** {q3.get('analysis', '')}")
    lines.extend(_render_simple_list("Alternative explanations", q3.get("alternative_explanations", [])))
    lines.append("")
    lines.append("## (4) Data/Figure Integrity and Artifact Risks")
    lines.append("")
    lines.append(f"- **Analysis:** {q4.get('analysis', '')}")
    lines.extend(_render_simple_list("Issues found", q4.get("issues_found", [])))
    lines.extend(_render_simple_list("Artifact risks", q4.get("processing_artifact_risks", [])))
    lines.append("")
    lines.append("## (5) Replications and Similar Systems")
    lines.append("")
    lines.append(f"- **Replication status:** {q5.get('replication_status', '')}")
    lines.append(f"- **Analysis:** {q5.get('analysis', '')}")
    lines.extend(_render_simple_list("Related works", q5.get("related_system_works", [])))
    lines.append("")
    lines.append("## (6) Data and Code Openness")
    lines.append("")
    lines.append(f"- **Data open-sourced:** {q6.get('data_open_sourced')}")
    lines.append(f"- **Code open-sourced:** {q6.get('code_open_sourced')}")
    if q6.get("reproducibility_what_is_shared"):
        lines.append(f"- **What is actually shared (reproducibility):** {q6.get('reproducibility_what_is_shared')}")
    lines.append(f"- **Details:** {q6.get('details', '')}")
    lines.extend(_render_simple_list("Evidence (files, structure, scale)", q6.get("evidence", [])))
    lines.append("")
    lines.append("## (7) Problem Importance / Relevance")
    lines.append("")
    lines.append(f"- **Importance level:** {q7.get('importance_level', q7.get('analysis', 'Not provided'))}")
    lines.append(f"- **Assessment:** {q7.get('problem_importance_assessment', q7.get('analysis', 'Not provided'))}")
    lines.extend(_render_simple_list("Evidence", q7.get("evidence", [])))
    lines.append("")
    lines.append("## (8) Technical/Method Advance")
    lines.append("")
    lines.append(f"- **Advance level:** {q8.get('advance_level', q8.get('analysis', 'Not provided'))}")
    lines.append(f"- **Assessment:** {q8.get('method_novelty_assessment', q8.get('analysis', 'Not provided'))}")
    lines.extend(_render_simple_list("Evidence", q8.get("evidence", [])))
    lines.append("")
    lines.append("## (9) How This Work Changes Its Area")
    lines.append("")
    lines.append(f"- **Change level:** {q9.get('change_level', q9.get('analysis', 'Not provided'))}")
    lines.append(f"- **Assessment:** {q9.get('area_change_assessment', q9.get('analysis', 'Not provided'))}")
    lines.extend(_render_simple_list("Evidence", q9.get("evidence", [])))
    lines.append("")
    lines.append("## (10) Impact on the Broader Scientific Community")
    lines.append("")
    lines.append(f"- **Impact type:** {q10.get('impact_type', q10.get('analysis', 'Not provided'))}")
    lines.append(f"- **Assessment:** {q10.get('community_impact_assessment', q10.get('analysis', 'Not provided'))}")
    if q10.get("textbook_or_commercial_potential"):
        lines.append(f"- **Textbook/commercial potential:** {q10.get('textbook_or_commercial_potential')}")
    lines.extend(_render_simple_list("Evidence", q10.get("evidence", [])))
    lines.append("")
    lines.append("## (11) Expansion Potential (Follow-up Research)")
    lines.append("")
    lines.append(f"- **Follow-up level:** {q11.get('follow_up_level', q11.get('analysis', 'Not provided'))}")
    lines.append(f"- **Assessment:** {q11.get('expansion_potential_assessment', q11.get('analysis', 'Not provided'))}")
    lines.extend(_render_simple_list("Evidence", q11.get("evidence", [])))
    lines.append("")

    # Scores
    reliability = scores.get("reliability", {}) if isinstance(scores.get("reliability"), dict) else scores
    novelty_impact = scores.get("novelty_impact", {}) if isinstance(scores.get("novelty_impact"), dict) else scores

    lines.append("## Scores (0-100)")
    lines.append("")
    lines.append("### Reliability")
    lines.append("")
    lines.append(f"- Derivation rigor: {reliability.get('derivation_rigor', 0)}")
    lines.append(f"- Experimental validity: {reliability.get('experimental_validity', 0)}")
    lines.append(f"- Evidence uniqueness: {reliability.get('evidence_uniqueness', 0)}")
    lines.append(f"- Data integrity: {reliability.get('data_integrity', 0)}")
    lines.append(f"- Replication support: {reliability.get('replication_support', 0)}")
    lines.append(f"- Openness: {reliability.get('openness', 0)}")
    lines.append(f"- **Reliability overall:** {reliability.get('overall', 0)}")

    # Show weighting information if available
    dist = run_metadata.get("theory_comp_exp_distribution")
    cat = run_metadata.get("paper_category")
    if dist:
        lines.append(f"  - *(weighted using distribution [exp={dist[0]:.2f}, theory={dist[1]:.2f}, comp={dist[2]:.2f}])*")
    if cat:
        lines.append(f"  - *(paper category: {cat})*")

    lines.append("")
    lines.append("### Novelty / Impact")
    lines.append("")
    lines.append(f"- Problem importance: {novelty_impact.get('problem_importance', 0)}")
    lines.append(f"- Technical/method advance: {novelty_impact.get('technical_method_advance', 0)}")
    lines.append(f"- Area change: {novelty_impact.get('area_change', 0)}")
    lines.append(f"- Community impact: {novelty_impact.get('community_impact', 0)}")
    lines.append(f"- Expansion potential: {novelty_impact.get('expansion_potential', 0)}")
    lines.append(f"- **Novelty/Impact overall:** {novelty_impact.get('overall', 0)}")
    lines.append("")
    lines.append(f"- **Overall paper score:** {scores.get('overall', 0)}")
    lines.append(f"- **Confidence:** {payload.get('confidence', 0)}")
    lines.append("")
    lines.append("## Final Reasoning Comment")
    lines.append("")
    lines.append(payload.get("final_reasoning_comment", ""))
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.extend(_render_dash_list(payload.get("limitations", []), default="No explicit limitations provided."))
    lines.append("")
    lines.append("## Citation Traces")
    lines.append("")
    traces = payload.get("citation_traces", [])
    if isinstance(traces, list) and traces:
        for idx, item in enumerate(traces, start=1):
            if isinstance(item, dict):
                lines.append(f"{idx}. `{item.get('source_id', '')}` - {item.get('claim', '')}")
                lines.append(f"   - Evidence: {item.get('supporting_excerpt', '')}")
    else:
        lines.append("- No citation traces were returned.")
    lines.append("")
    return "\n".join(lines)


def _render_list_of_dicts(items: list) -> list[str]:
    if not isinstance(items, list) or not items:
        return ["- No items were identified."]
    out: list[str] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        name = item.get("derivation_name") or item.get("method_name") or f"Item {idx}"
        out.append(f"{idx}. **{name}**")
        for key, value in item.items():
            if key in {"derivation_name", "method_name"}:
                continue
            if isinstance(value, list):
                if value:
                    out.append(f"   - {key}:")
                    for entry in value:
                        out.append(f"     - {entry}")
            else:
                out.append(f"   - {key}: {value}")
    return out or ["- No items were identified."]


def _render_simple_list(title: str, values: list) -> list[str]:
    if not isinstance(values, list) or not values:
        return [f"- **{title}:** none"]
    out = [f"- **{title}:**"]
    out.extend([f"  - {item}" for item in values])
    return out


def _render_dash_list(values: list, default: str) -> list[str]:
    if not isinstance(values, list) or not values:
        return [f"- {default}"]
    return [f"- {entry}" for entry in values]

