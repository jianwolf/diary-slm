"""Shared constants for diary-slm.

This module centralizes magic numbers, paths, and configuration values
used throughout the library. Import from here to ensure consistency.
"""

from datetime import datetime
from pathlib import Path
from typing import Final


# =============================================================================
# Bear App Configuration
# =============================================================================

# Bear stores its database in a sandboxed container
BEAR_CONTAINER_ID: Final[str] = "9K33E3U3T4.net.shinyfrog.bear"

# Relative path within the container to the database
BEAR_DB_RELATIVE_PATH: Final[str] = "Application Data/database.sqlite"

# Full default path to Bear's database on macOS
DEFAULT_BEAR_DB_PATH: Final[Path] = (
    Path.home() / "Library/Group Containers" / BEAR_CONTAINER_ID / BEAR_DB_RELATIVE_PATH
)

# Core Data epoch (Apple's reference date for timestamps)
# All Core Data timestamps are seconds since this date
CORE_DATA_EPOCH: Final[datetime] = datetime(2001, 1, 1)


# =============================================================================
# Token Estimation
# =============================================================================

# Approximate characters per token for estimation
# This is conservative; actual ratio varies by language and model:
# - English: ~4 chars/token
# - Chinese: ~1.5-2 chars/token
# - Mixed: ~3-4 chars/token
CHARS_PER_TOKEN: Final[int] = 4

# Common model context limits (in tokens)
CONTEXT_LIMIT_128K: Final[int] = 128_000
CONTEXT_LIMIT_32K: Final[int] = 32_000

# Safe context limit leaving room for system prompt and response
# We use 100k to leave ~28k buffer for prompts and generation
SAFE_CONTEXT_LIMIT: Final[int] = 100_000

# Default maximum tokens to generate in responses
DEFAULT_MAX_GENERATION_TOKENS: Final[int] = 4096


# =============================================================================
# Model Configuration
# =============================================================================

# Default model for analysis
DEFAULT_MODEL_ID: Final[str] = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# Pre-configured model presets with good performance
MODEL_PRESETS: Final[dict[str, str]] = {
    "qwen-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen-14b": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "llama-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "gemma-9b": "mlx-community/gemma-2-9b-it-4bit",
    "glm-4.7-flash": "mlx-community/GLM-4.7-Flash-4bit",  # 30B params, 128K context, ~17GB
}

# Default temperature for generation (0.7 balances creativity and coherence)
DEFAULT_TEMPERATURE: Final[float] = 0.7


# =============================================================================
# Period Configuration
# =============================================================================

# Valid period types for chunking diaries
VALID_PERIOD_TYPES: Final[tuple[str, ...]] = ("month", "quarter", "half_year", "year")

# Default period type (quarter provides good balance of context and specificity)
DEFAULT_PERIOD_TYPE: Final[str] = "quarter"

# Approximate tokens per period type (based on typical diary volume)
# These are rough estimates for a user writing ~1400 chars/day
ESTIMATED_TOKENS_PER_PERIOD: Final[dict[str, int]] = {
    "month": 10_000,
    "quarter": 32_000,
    "half_year": 64_000,
    "year": 125_000,
}


# =============================================================================
# Display Configuration
# =============================================================================

# Maximum tags to display in tag listing
MAX_TAGS_TO_DISPLAY: Final[int] = 30

# Maximum periods to show in period suggestions
MAX_PERIODS_TO_SUGGEST: Final[int] = 10
