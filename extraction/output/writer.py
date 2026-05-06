"""Persist extracted content to disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from extraction.core.models import ExtractionOutcome
from extraction.extractors.common import slug_from_url


def write_outcome(output_root: Path, outcome: ExtractionOutcome) -> Path:
    slug = slug_from_url(outcome.resolved_url)
    article_dir = output_root / slug
    figures_dir = article_dir / "figures"
    supplementary_dir = article_dir / "supplementary"
    article_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = article_dir / "metadata.json"
    metadata_payload = {
        **outcome.metadata,
        "source": outcome.source,
        "resolved_url": outcome.resolved_url,
        "doi": outcome.doi,
        "access_limited": outcome.access_limited,
        "access_warning": outcome.access_warning,
    }
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if outcome.article_markdown:
        (article_dir / "article.md").write_text(outcome.article_markdown, encoding="utf-8")

    for figure in outcome.figures:
        (figures_dir / figure.filename).write_bytes(figure.content)

    for supplementary in outcome.supplementary_files:
        (supplementary_dir / supplementary.filename).write_bytes(supplementary.content)

    if outcome.pdf is not None:
        (article_dir / "article.pdf").write_bytes(outcome.pdf.content)

    run_log = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "anti_bot_blocked": outcome.anti_bot_blocked,
        "access_limited": outcome.access_limited,
        "access_warning": outcome.access_warning,
        "attempts": [
            {"name": a.name, "success": a.success, "message": a.message}
            for a in outcome.attempts
        ],
        "errors": outcome.errors,
        "figure_count": len(outcome.figures),
        "supplementary_count": len(outcome.supplementary_files),
        "has_pdf": outcome.pdf is not None,
        "has_markdown": outcome.article_markdown is not None,
    }
    (article_dir / "run_log.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")

    return article_dir
