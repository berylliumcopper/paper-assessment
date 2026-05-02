"""Centralized defaults for the extraction pipeline."""

from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_DELAY_SECONDS = 5.0
DEFAULT_JITTER_SECONDS = 3.0
DEFAULT_RETRIES = 2
DEFAULT_MODE = "both"
DEFAULT_OUTPUT_DIR = "extraction/output_data"

LOCAL_DIR = Path("extraction/.local")
PROFILE_DIR = LOCAL_DIR / "profile"
STATE_DIR = LOCAL_DIR / "state"
LOG_DIR = LOCAL_DIR / "logs"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 PaperExtractor/1.0"
)
