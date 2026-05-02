#!/usr/bin/env python3
"""
Batch runner for paper-assessment.

Usage:
    python assessment_batch.py dois.txt -- --headless --skip-convert

The first positional argument is a text file with one DOI per line.
Everything after ``--`` is forwarded to ``assessment_cli.py`` for each DOI.

Examples:
    python assessment_batch.py my_dois.txt
    python assessment_batch.py my_dois.txt -- --headless
    python assessment_batch.py my_dois.txt -- --skip-convert --output-dir papers
    python assessment_batch.py my_dois.txt -- --provider openai --model gpt-4o --headless
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1

    doi_file = Path(sys.argv[1])
    if not doi_file.exists():
        print(f"[error] file not found: {doi_file}", file=sys.stderr)
        return 1

    # Everything after -- is forwarded as CLI args
    forward_args = []
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        forward_args = sys.argv[idx + 1 :]

    # Read DOIs (skip blank lines and comments)
    dois = [
        line.strip()
        for line in doi_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not dois:
        print("[error] no DOIs found in the input file", file=sys.stderr)
        return 1

    print(f"[batch] loaded {len(dois)} DOIs from {doi_file}")
    print(f"[batch] forwarded CLI args: {forward_args}")

    cli_script = Path(__file__).resolve().parent / "assessment_cli.py"
    if not cli_script.exists():
        print(f"[error] assessment_cli.py not found at {cli_script}", file=sys.stderr)
        return 1

    successes = 0
    failures = 0
    total_start = time.perf_counter()

    for i, doi in enumerate(dois, start=1):
        paper_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"[batch] [{i}/{len(dois)}] processing: {doi}")
        print(f"{'='*60}")

        cmd = [sys.executable, str(cli_script), doi, *forward_args]

        try:
            result = subprocess.run(cmd, capture_output=False, timeout=7200)
            if result.returncode == 0:
                successes += 1
                print(f"[batch] [{i}/{len(dois)}] OK  ({time.perf_counter() - paper_start:.1f}s)")
            else:
                failures += 1
                print(
                    f"[batch] [{i}/{len(dois)}] FAILED (exit code {result.returncode}, "
                    f"{time.perf_counter() - paper_start:.1f}s)",
                    file=sys.stderr,
                )
        except subprocess.TimeoutExpired:
            failures += 1
            print(
                f"[batch] [{i}/{len(dois)}] TIMEOUT after 7200s ({time.perf_counter() - paper_start:.1f}s)",
                file=sys.stderr,
            )
        except Exception as exc:
            failures += 1
            print(
                f"[batch] [{i}/{len(dois)}] ERROR: {exc} ({time.perf_counter() - paper_start:.1f}s)",
                file=sys.stderr,
            )

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"[batch] done  total={total_elapsed:.1f}s  successes={successes}  failures={failures}")
    print(f"{'='*60}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
