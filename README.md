# paper-assessment

version 0.4.0

This is AI-powered single-paper assessment workflow, with the detailed code written mostly by AI. The target is to quickly read and process papers, especially spotting the flawed or inconsistent data of paper, find the claims that are not fully supported or not very well founded, cross-ckeck other related papers, and assess the impact of the target papers.

I know that there exist some flaws in some papers that are relatively easy to check but hard to spot due to the large data quantity. Automatic AI checks can contribute a lot in these situations and save people's time (by not wasting time on these papers). Also, it can occasionally think from alternative perspectives or propose alternative explanations that the author of the paper does not know or does not want to know, which might be intriguing. These things might help people in reviewing and reading large quantities of papers in short time (of course the final judgement is still on you).

I think the current AI is still not as good as the best humans at intuitive physical understandings. But for many actual individuals inside the academy, the comparison to AI might lead to a very different result. It is certainly a question what people can still contribute to scientific research given the fast development of AI - some people says the taste and pointing the directions of research. But as somebody as small as a PhD student, postdoc, it is questionable whether your taste really matters, or the only thing you can do is just to follow those established professors. I think that's an important question to maintain people's creativity in acedemia in the AI era. I hope that AI-related tools can, at least, help people in academia to identify which people or articles might be not so trustworthy, and which people are actually capable of contrubuting to the research even with the current AI development level.

## Pipeline Overview

Given a paper DOI or local PDF path, the pipeline runs these stages:

| Stage | Description |
|-------|-------------|
| **1. Extraction** | Downloads the PDF from the publisher (browser-based, with doi2pdf fallback via Sci-Hub/OpenAlex/CrossRef). Download supplementary materials and reference source data files. |
| **2. Conversion** | Converts PDF to Markdown + images using Marker (configurable script path). |
| **3. Extraction of basic information** | Extract basic information from the paper, such as the title, type of study (theoretical/computational/experimental), and keywords for related-work search. |
| **4. Related-work search** | Searches for related papers via Crossref, arXiv, and Semantic Scholar. Followed by AI pre-filtering to shortlist the most relevant ones. Also searches for citing papers via Semantic Scholar citation graph (up to 400), AI-filtered to ~20. |
| **5. Related-pdf download** | Attempts to download PDFs for shortlisted related papers: publisher extraction → doi2pdf fallback → arXiv author+title search → direct arXiv download. |
| **6. Related full-text extraction** | Extracts plain text from downloaded related PDFs (pypdf / PyMuPDF, capped at 50K chars per paper). |
| **7. Related work assessment** | Sends the selected related papers (full text) in batches to an AI for assessment, summarizing the work and evoluting its relation to the target paper. |
| **8. Investigation of source data** | Convert the non-PDF supplementary materials (if exist, likely the source data files attached to the paper) to Markdown format. Perform an AI assessment to these files. |
| **9. Specific assessments** | Do three specific assessments. The type of these assessments are selected from below, and the number of times each assessment type is performed depends on the paper type obtained from Step 3. (1) Figure/Table assessment: Sends main-text, figures, tables, and selected source data files to an AI for per-panel analysis about method, explanation uniqueness, and internal consistency; (2) Derivation assessment: Sends main text and supplementary with formulas to an AI for per-step analysis about assumptions, mathematical derivations, and unterpretations; (3) Pipeline assessment: Sends main text and selected source code files to an AI for per-step analysis about pipeline advancement, open-source extent, and improvement evidences. The number of these types of assessments performed adds up to three, with some possibly not performed and some possibly performed more than once.|
| **10. AI assessment** | Sends the paper content + related-paper full-texts + figure/table assessment output to an LLM (OpenAI- or Gemini-compatible) for scoring across **two score categories**: **reliability** (derivations, methods, evidence uniqueness, integrity, replication, openness) and **novelty/impact** (problem importance, technical advance, area change, community impact, expansion potential). |

## Suggestions and Known Issues

1. The AI calls might consume over ~2M input tokens and over ~100k output tokens for each paper (tokens consumed in CoT not counted). Please be aware of the possible high cost asssociate with this.

2. It is recommanded to run this program under the campus network environment, because the download from journal websites (required for automatic supplementary extraction) sometimes need to pass the paywall. Although there are fallback plans like arxiv or sci-hub, the paper is sometimes not the final version or incomplete. However, due to the anti-bot mechanisms from some publishers (like APS and Science AAAS), some materials are unfortunately difficult to extarct from the official website (but you can still download manually and put the supplementary in the corresponding folder).

3. In downloading the resources from resources other than arXiv, Nature/Springer, and GitHub, the download can sometimes be blocked due to anti-bot detactions. Currently special care is implemented for Zenodo by using a library `zenodo-get`.

4. Some prompts might be tailored for physics area, and might not work for other research topics. The test was mostly performed on condensed matter experiments, and the test for other papers (including condensed matter theory) is far from complete. You might need to tweak some prompt wordings to get a better results for other topics.

5. APIs for molti-modal AIs are recommanded, because then the figures of the target paper can also be read.

6. Because all the responses will be generated through AI, some of them might contain wrong information, and AI might not be very good in understanding some physics concepts. But AI seems to be at least good at reading large amount of information in short time, and spot the basic flaws, errors, or inconsistencies during data processing. Please check yourself before you are sure that these comments are real.

## Setup

```bash
# Create environment
conda create -n paper-ext python=3.13
conda activate paper-ext

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# API credentials
# Create .secrets/assessment_api.json with:
# {
#   "api_key": "your-gemini-or-openai-key",
#   "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
#   "model": "gemini-2.0-flash"
# }
```

> **Note on marker-pdf**: `marker-pdf` is the heaviest dependency (pulls in `torch`, `transformers`, `surya-ocr`, etc.) and is only needed for PDF-to-markdown conversion. If you already have a converted `article.md`, run with `--skip-convert`. By default, the settings of `marker-pdf` are tuned to fit ~8 GB VRAM, and can be modified to further reduce the VRAM consumption.

## Usage

Use single-paper assessment

```bash
# Assess a paper by DOI
python assessment_cli.py 10.1038/s41586-026-10420-y

# Assess a local PDF
python assessment_cli.py paper/test/original.pdf

# With custom settings
python assessment_cli.py 10.1038/s41586-026-10420-y \
    --output-dir extraction/output_data \
    --headless \
    --real-browser-mode \
    --unpaywall-email your@email.com
```

Read a list of paper doi from a txt file (each line is a txt)
```bash
# Assess all paper doi in a txt file
python assessment_batch.py papers/doi.txt --output-dir papers
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `target` | — | DOI, URL, or local PDF path |
| `--headless` | true | Run browser headless |
| `--real-browser-mode` | — | Fewer automation tweaks (bypass Cloudflare) |
| `--mode` | both | Extraction mode: `structured`, `pdf`, or `both` |
| `--sci-hub-mirrors` | built-in | Custom Sci-Hub mirror URLs |
| `--no-doi2pdf` | — | Disable doi2pdf fallback |
| `--skip-convert` | — | Skip PDF-to-markdown conversion |
| `--skip-related-search` | — | Skip related-work search |
| `--related-download-count` | 20 | Max related PDFs to attempt downloading |
| `--no-related-fulltext` | — | Skip full-text extraction for related PDFs |
| `--provider` | auto | Force API provider: `openai` or `gemini` |

## Related-paper Download Sources

When browser-based publisher extraction fails, these fallbacks are tried in order for each related paper:

1. **doi2pdf** — OpenAlex OA URL → CrossRef PDF link → Sci-Hub mirrors
2. **arXiv author+title search** — Queries `export.arxiv.org/api/query` with author last names + title keywords, scored by author overlap + title overlap.
3. **Direct arXiv download** — If an arXiv ID was found by the resolver or pre-populated in search metadata.

## Output Files

All output files are written to the article directory (under `--output-dir`). Three sets are produced:

### 1. Related-work report (written early, before AI assessment)

| File | Content |
|------|---------|
| `related_work.json` | Raw search results and AI-generated shortlist metadata |
| `related_work_report.md` | Formatted report with search statistics, shortlisted papers with target-focused narratives, and not-shortlisted landscape summaries |

### 2. Specific assessments

| File | Content |
|------|---------|
| `figure_table_assessment_[index].json` | Raw AI response with per-panel analysis for figures and tables, as well as files flagged for final review |
| `figure_table_assessment_report_[index].md` | Formatted per-panel analysis for figures and tables |
| `derivation_assessment_[index].json` | Raw AI response with per-step analysis for derivations, as well as formulas flagged for final review  |
| `derivation_assessment_report_[index].md` | Formatted per-step analysis for derivations |
| `pipeline_assessment_[index].json` | Raw AI response with per-module analysis for pipelines, as well as code files flagged for final review  |
| `pipeline_assessment_report_[index].md` | Formatted per-module analysis for pipelines |

### 3. Final assessment report

| File | Content |
|------|---------|
| `assessment.json` | Structured JSON with all 11 assessment dimensions (q1–q11), scores split into **reliability** and **novelty/impact** categories, citation traces, and limitations |
| `assessment_run.json` | Run metadata: timestamp, input target, model, API base URL |
| `assessment.md` | Human-readable markdown report with per-section analysis and score tables |

### Trace directory (`ai_traces/`)

The subdirectory `ai_traces/` inside each article directory contains the raw prompt and raw model response for the final assessment step, useful for debugging.