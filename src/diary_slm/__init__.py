"""diary-slm: Analyze your diary entries using local LLMs.

This package provides tools for analyzing diary entries from Bear
using local language models via Apple's MLX framework.

Key Features:
    - Read-only access to Bear's SQLite database
    - Time-based chunking of diary entries
    - Full-context analysis (no RAG)
    - Streaming LLM responses
    - Pre-built analysis templates

Quick Start:
    >>> from diary_slm import BearReader, DiaryProcessor, DiaryAnalyzer, ModelManager
    >>>
    >>> # Read notes from Bear
    >>> reader = BearReader()
    >>> notes = reader.get_notes_by_tag("diary")
    >>>
    >>> # Chunk by quarter
    >>> processor = DiaryProcessor(notes)
    >>> chunk = processor.get_chunk_by_name("2024-Q1")
    >>>
    >>> # Analyze with LLM
    >>> model = ModelManager()
    >>> analyzer = DiaryAnalyzer(model)
    >>> for text in analyzer.analyze(chunk, "What patterns do you see?"):
    ...     print(text, end="")

CLI Usage:
    $ diary-slm list -t diary
    $ diary-slm analyze 2024-Q1 -q "What were my struggles?" -t diary
    $ diary-slm template 2024-Q1 mood -t diary

Safety:
    This package NEVER writes to Bear's database. All database access
    is strictly read-only (using SQLite's mode=ro flag).
"""

__version__ = "0.1.0"

# Core classes
from .bear_reader import BearReader, Note
from .processor import DiaryProcessor, DiaryChunk, PeriodType, estimate_tokens
from .model import ModelManager, list_available_models, resolve_model_name
from .analyzer import (
    DiaryAnalyzer,
    ANALYSIS_TEMPLATES,
    SYSTEM_PROMPT,
    list_analysis_templates,
)

# Exceptions
from .exceptions import (
    DiarySlmError,
    DatabaseError,
    DatabaseNotFoundError,
    DatabaseReadError,
    NoteError,
    NoNotesFoundError,
    PeriodError,
    PeriodNotFoundError,
    InvalidPeriodTypeError,
    ModelError,
    ModelNotLoadedError,
    ModelLoadError,
    AnalysisError,
    TemplateNotFoundError,
    ContextTooLargeError,
)

# Constants (for advanced users)
from .constants import (
    DEFAULT_BEAR_DB_PATH,
    CORE_DATA_EPOCH,
    CHARS_PER_TOKEN,
    SAFE_CONTEXT_LIMIT,
    DEFAULT_MODEL_ID,
    MODEL_PRESETS,
    VALID_PERIOD_TYPES,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "BearReader",
    "Note",
    "DiaryProcessor",
    "DiaryChunk",
    "PeriodType",
    "ModelManager",
    "DiaryAnalyzer",
    # Functions
    "estimate_tokens",
    "list_available_models",
    "resolve_model_name",
    "list_analysis_templates",
    # Constants
    "ANALYSIS_TEMPLATES",
    "SYSTEM_PROMPT",
    "DEFAULT_BEAR_DB_PATH",
    "CORE_DATA_EPOCH",
    "CHARS_PER_TOKEN",
    "SAFE_CONTEXT_LIMIT",
    "DEFAULT_MODEL_ID",
    "MODEL_PRESETS",
    "VALID_PERIOD_TYPES",
    # Exceptions
    "DiarySlmError",
    "DatabaseError",
    "DatabaseNotFoundError",
    "DatabaseReadError",
    "NoteError",
    "NoNotesFoundError",
    "PeriodError",
    "PeriodNotFoundError",
    "InvalidPeriodTypeError",
    "ModelError",
    "ModelNotLoadedError",
    "ModelLoadError",
    "AnalysisError",
    "TemplateNotFoundError",
    "ContextTooLargeError",
]
