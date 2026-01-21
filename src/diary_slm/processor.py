"""Process diary notes into time-based chunks for LLM analysis.

This module handles the chunking of diary entries into time periods
(month, quarter, half-year, year) suitable for LLM context windows.

The core philosophy is to avoid RAG by fitting entire time periods
into the model's context, giving it full visibility into patterns
and causal relationships.

Example:
    >>> from diary_slm.processor import DiaryProcessor
    >>> processor = DiaryProcessor(notes)
    >>> chunks = processor.get_chunks_by_period("quarter")
    >>> for chunk in chunks:
    ...     print(f"{chunk.period_name}: {chunk.estimated_tokens:,} tokens")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from .constants import (
    CHARS_PER_TOKEN,
    DEFAULT_PERIOD_TYPE,
    SAFE_CONTEXT_LIMIT,
    VALID_PERIOD_TYPES,
)
from .exceptions import InvalidPeriodTypeError, PeriodNotFoundError

if TYPE_CHECKING:
    from .bear_reader import Note

__all__ = ["DiaryChunk", "DiaryProcessor", "PeriodType", "estimate_tokens"]


# Type alias for period types
PeriodType = Literal["month", "quarter", "half_year", "year"]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class DiaryChunk:
    """An immutable chunk of diary entries for a specific time period.

    This represents a collection of notes grouped by time period,
    formatted and ready for LLM analysis.

    Attributes:
        period_name: Human-readable period identifier (e.g., "2024-Q1").
        period_type: The type of period (month, quarter, half_year, year).
        start_date: Earliest note date in this chunk.
        end_date: Latest note date in this chunk.
        note_count: Number of notes in this chunk.
        formatted_text: Pre-formatted text ready for LLM prompt.
        estimated_tokens: Estimated token count of formatted_text.

    Example:
        >>> chunk = chunks[0]
        >>> print(f"{chunk.period_name}: {chunk.note_count} notes")
        2024-Q1: 90 notes
    """

    period_name: str
    period_type: PeriodType
    start_date: datetime
    end_date: datetime
    note_count: int
    formatted_text: str
    estimated_tokens: int

    @property
    def total_chars(self) -> int:
        """Get total character count of formatted text."""
        return len(self.formatted_text)

    def fits_in_context(self, max_tokens: int = SAFE_CONTEXT_LIMIT) -> bool:
        """Check if this chunk fits within a context limit.

        Args:
            max_tokens: Maximum tokens allowed (default: 100k safe limit).

        Returns:
            True if chunk fits, False otherwise.
        """
        return self.estimated_tokens <= max_tokens


# =============================================================================
# Token Estimation
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses a conservative heuristic of ~4 characters per token.
    This works reasonably well for mixed English/Chinese content.

    Note:
        Actual token counts vary by model and content:
        - English prose: ~4-5 chars/token
        - Chinese text: ~1.5-2 chars/token
        - Code: ~3-4 chars/token
        - Mixed: ~3-4 chars/token

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated number of tokens.

    Example:
        >>> estimate_tokens("Hello, world!")
        3
    """
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


# =============================================================================
# Period Calculation Helpers
# =============================================================================


def _get_quarter(month: int) -> int:
    """Get quarter number (1-4) from month (1-12).

    Args:
        month: Month number (1-12).

    Returns:
        Quarter number (1-4).

    Example:
        >>> _get_quarter(1)  # January
        1
        >>> _get_quarter(4)  # April
        2
    """
    return (month - 1) // 3 + 1


def _get_half(month: int) -> int:
    """Get half-year number (1-2) from month (1-12).

    Args:
        month: Month number (1-12).

    Returns:
        Half number (1 for Jan-Jun, 2 for Jul-Dec).

    Example:
        >>> _get_half(3)   # March
        1
        >>> _get_half(9)   # September
        2
    """
    return 1 if month <= 6 else 2


def _get_period_key(dt: datetime, period_type: PeriodType) -> str:
    """Generate a period key for a datetime.

    Args:
        dt: Datetime to generate key for.
        period_type: Type of period to generate.

    Returns:
        Period key string (e.g., "2024-Q1", "2024-H1", "2024-01", "2024").

    Example:
        >>> from datetime import datetime
        >>> dt = datetime(2024, 3, 15)
        >>> _get_period_key(dt, "quarter")
        '2024-Q1'
    """
    year = dt.year

    if period_type == "month":
        return f"{year}-{dt.month:02d}"
    elif period_type == "quarter":
        return f"{year}-Q{_get_quarter(dt.month)}"
    elif period_type == "half_year":
        return f"{year}-H{_get_half(dt.month)}"
    else:  # year
        return str(year)


# =============================================================================
# Note Formatting
# =============================================================================


def _format_notes_for_prompt(notes: list[Note], period_name: str) -> str:
    """Format notes into a single text block for LLM prompt.

    Creates a chronologically ordered document with date headers
    that helps the LLM understand temporal relationships.

    Args:
        notes: List of notes to format.
        period_name: Name of the period for the header.

    Returns:
        Formatted text with date headers and note content.

    Example output:
        === Diary Entries: 2024-Q1 ===

        --- 2024-01-01 (Monday) ---
        ## My First Day
        Content of the note...

        --- 2024-01-02 (Tuesday) ---
        More content...
    """
    if not notes:
        return ""

    lines: list[str] = [f"=== Diary Entries: {period_name} ===\n"]

    # Sort notes chronologically
    sorted_notes = sorted(notes, key=lambda n: n.created_at)

    current_date: str | None = None
    for note in sorted_notes:
        note_date = note.created_at.strftime("%Y-%m-%d")

        # Add date header when date changes
        if note_date != current_date:
            current_date = note_date
            weekday = note.created_at.strftime("%A")
            lines.append(f"\n--- {note_date} ({weekday}) ---\n")

        # Process note content
        title = note.title.strip()
        content = note.content.strip()

        # Remove duplicate title from content if present
        title_header = f"# {title}"
        if content.startswith(title_header):
            content = content[len(title_header):].strip()

        # Add title if it's meaningful (not just a date)
        if title and not title.startswith("20"):
            lines.append(f"## {title}\n")

        lines.append(f"{content}\n")

    return "\n".join(lines)


# =============================================================================
# Main Processor Class
# =============================================================================


class DiaryProcessor:
    """Process diary notes into time-based chunks for LLM analysis.

    This class groups notes by time period and formats them for
    efficient LLM consumption. It supports various period types
    to balance context size and analysis granularity.

    Attributes:
        notes: List of notes being processed (sorted by date).

    Example:
        >>> processor = DiaryProcessor(notes)
        >>> for chunk in processor.get_chunks_by_period("quarter"):
        ...     if chunk.fits_in_context():
        ...         analyze(chunk)
    """

    def __init__(self, notes: list[Note]) -> None:
        """Initialize with a list of notes.

        Args:
            notes: List of Note objects to process.
                   Will be sorted by creation date internally.
        """
        # Store sorted copy to avoid modifying input
        self._notes: list[Note] = sorted(notes, key=lambda n: n.created_at)

    @property
    def notes(self) -> list[Note]:
        """Get the sorted list of notes (read-only access)."""
        return self._notes.copy()

    @property
    def note_count(self) -> int:
        """Get the total number of notes."""
        return len(self._notes)

    def _validate_period_type(self, period_type: str) -> PeriodType:
        """Validate and return a period type.

        Args:
            period_type: Period type to validate.

        Returns:
            Validated period type.

        Raises:
            InvalidPeriodTypeError: If period_type is not valid.
        """
        if period_type not in VALID_PERIOD_TYPES:
            raise InvalidPeriodTypeError(period_type)
        return period_type  # type: ignore

    def get_chunks_by_period(
        self,
        period_type: PeriodType | str = DEFAULT_PERIOD_TYPE,
    ) -> list[DiaryChunk]:
        """Split notes into chunks by time period.

        Groups notes by the specified period type and formats each
        group for LLM analysis.

        Args:
            period_type: How to split periods. One of:
                - "month": Monthly chunks (~10k tokens typical)
                - "quarter": Quarterly chunks (~32k tokens typical)
                - "half_year": Semi-annual chunks (~64k tokens typical)
                - "year": Annual chunks (~125k tokens typical)

        Returns:
            List of DiaryChunk objects, sorted chronologically.

        Raises:
            InvalidPeriodTypeError: If period_type is not valid.

        Example:
            >>> chunks = processor.get_chunks_by_period("quarter")
            >>> for chunk in chunks:
            ...     print(f"{chunk.period_name}: {chunk.estimated_tokens:,} tokens")
        """
        validated_type = self._validate_period_type(period_type)

        if not self._notes:
            return []

        # Group notes by period
        period_notes: dict[str, list[Note]] = {}
        for note in self._notes:
            key = _get_period_key(note.created_at, validated_type)
            if key not in period_notes:
                period_notes[key] = []
            period_notes[key].append(note)

        # Create chunks
        chunks: list[DiaryChunk] = []
        for period_name in sorted(period_notes.keys()):
            notes_in_period = period_notes[period_name]
            formatted = _format_notes_for_prompt(notes_in_period, period_name)
            tokens = estimate_tokens(formatted)

            # Calculate date range
            dates = [n.created_at for n in notes_in_period]

            chunk = DiaryChunk(
                period_name=period_name,
                period_type=validated_type,
                start_date=min(dates),
                end_date=max(dates),
                note_count=len(notes_in_period),
                formatted_text=formatted,
                estimated_tokens=tokens,
            )
            chunks.append(chunk)

        return chunks

    def get_chunk_by_name(
        self,
        period_name: str,
        period_type: PeriodType | str = DEFAULT_PERIOD_TYPE,
        *,
        raise_if_missing: bool = False,
    ) -> DiaryChunk | None:
        """Get a specific chunk by period name.

        Args:
            period_name: Period identifier (e.g., "2024-Q1", "2024-H1").
            period_type: Type of period to look for.
            raise_if_missing: If True, raise PeriodNotFoundError instead
                              of returning None.

        Returns:
            DiaryChunk if found, None otherwise (unless raise_if_missing=True).

        Raises:
            InvalidPeriodTypeError: If period_type is not valid.
            PeriodNotFoundError: If raise_if_missing=True and period not found.

        Example:
            >>> chunk = processor.get_chunk_by_name("2024-Q1")
            >>> if chunk:
            ...     print(chunk.formatted_text)
        """
        chunks = self.get_chunks_by_period(period_type)

        for chunk in chunks:
            if chunk.period_name == period_name:
                return chunk

        if raise_if_missing:
            available = [c.period_name for c in chunks]
            raise PeriodNotFoundError(period_name, available)

        return None

    def get_available_periods(
        self,
        period_type: PeriodType | str = DEFAULT_PERIOD_TYPE,
    ) -> list[tuple[str, int, int]]:
        """List available periods with statistics.

        Args:
            period_type: How to split periods.

        Returns:
            List of (period_name, note_count, estimated_tokens) tuples.

        Raises:
            InvalidPeriodTypeError: If period_type is not valid.

        Example:
            >>> for name, notes, tokens in processor.get_available_periods():
            ...     print(f"{name}: {notes} notes, {tokens:,} tokens")
        """
        chunks = self.get_chunks_by_period(period_type)
        return [
            (c.period_name, c.note_count, c.estimated_tokens)
            for c in chunks
        ]

    def get_total_stats(self) -> dict[str, int | tuple[datetime, datetime] | None]:
        """Get overall statistics for all notes.

        Returns:
            Dictionary with keys:
                - total_notes: Number of notes
                - total_chars: Total character count
                - estimated_tokens: Estimated total tokens
                - date_range: Tuple of (earliest, latest) dates, or None

        Example:
            >>> stats = processor.get_total_stats()
            >>> print(f"Total tokens: {stats['estimated_tokens']:,}")
        """
        if not self._notes:
            return {
                "total_notes": 0,
                "total_chars": 0,
                "estimated_tokens": 0,
                "date_range": None,
            }

        total_chars = sum(n.char_count for n in self._notes)
        combined_content = "".join(n.content for n in self._notes)
        dates = [n.created_at for n in self._notes]

        return {
            "total_notes": len(self._notes),
            "total_chars": total_chars,
            "estimated_tokens": estimate_tokens(combined_content),
            "date_range": (min(dates), max(dates)),
        }

    def suggest_period_type(
        self,
        max_tokens: int = SAFE_CONTEXT_LIMIT,
    ) -> PeriodType:
        """Suggest the best period type based on diary volume.

        Analyzes the token density and recommends a period type
        that will fit comfortably within the context limit.

        Args:
            max_tokens: Maximum tokens per chunk (default: 100k).

        Returns:
            Recommended period type.

        Example:
            >>> suggested = processor.suggest_period_type()
            >>> chunks = processor.get_chunks_by_period(suggested)
        """
        stats = self.get_total_stats()
        total_tokens = stats["estimated_tokens"]
        date_range = stats["date_range"]

        if not date_range or total_tokens == 0:
            return "quarter"

        # Calculate months of data
        start, end = date_range
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1

        if months == 0:
            return "month"

        tokens_per_month = total_tokens / months

        # Suggest based on what fits in max_tokens
        if tokens_per_month * 12 <= max_tokens:
            return "year"
        elif tokens_per_month * 6 <= max_tokens:
            return "half_year"
        elif tokens_per_month * 3 <= max_tokens:
            return "quarter"
        else:
            return "month"
