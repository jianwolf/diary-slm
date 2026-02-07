"""Read notes from Bear app's SQLite database.

This module provides read-only access to Bear's SQLite database.
It extracts notes, tags, and metadata without ever modifying the database.

SECURITY NOTE:
    This module ONLY performs read operations. The database connection
    is opened in read-only mode (mode=ro) to prevent any accidental writes.
    No INSERT, UPDATE, DELETE, or DROP statements are ever executed.

Example:
    >>> from diary_slm.bear_reader import BearReader
    >>> reader = BearReader()
    >>> notes = reader.get_notes_by_tag("diary")
    >>> for note in notes:
    ...     print(f"{note.created_at}: {note.title}")

Note:
    Bear must be installed and have created at least one note for the
    database to exist. The database location is auto-detected on macOS.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import CORE_DATA_EPOCH, DEFAULT_BEAR_DB_PATH
from .exceptions import DatabaseNotFoundError, DatabaseReadError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Note", "BearReader"]

_TAG_PATTERN = re.compile(r'(?<![#\w])#([a-zA-Z0-9\u4e00-\u9fff][\w\u4e00-\u9fff/]*)')


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class Note:
    """An immutable representation of a Bear note.

    This dataclass holds the essential data extracted from a Bear note.
    It is frozen (immutable) to prevent accidental modifications.

    Attributes:
        id: Unique identifier for the note (Bear's ZUNIQUEIDENTIFIER).
        title: Note title.
        content: Full note content (may include markdown and tags).
        created_at: When the note was created.
        modified_at: When the note was last modified.
        tags: List of tags extracted from the note content.

    Example:
        >>> note = Note(
        ...     id="abc123",
        ...     title="My Day",
        ...     content="Today was great! #diary",
        ...     created_at=datetime.now(),
        ...     modified_at=datetime.now(),
        ...     tags=["diary"],
        ... )
        >>> print(note.word_count)
        4
    """

    id: str
    title: str
    content: str
    created_at: datetime
    modified_at: datetime
    tags: tuple[str, ...] = field(default_factory=tuple)  # Immutable tuple instead of list

    @property
    def text(self) -> str:
        """Get full text combining title and content.

        Returns:
            Formatted text with title as markdown header.
        """
        return f"# {self.title}\n\n{self.content}"

    @property
    def word_count(self) -> int:
        """Estimate word count of the content.

        Returns:
            Approximate number of words (split by whitespace).
        """
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Get character count of the content.

        Returns:
            Number of characters in content.
        """
        return len(self.content)


# =============================================================================
# Helper Functions
# =============================================================================


def _core_data_to_datetime(timestamp: float | None) -> datetime:
    """Convert Apple Core Data timestamp to Python datetime.

    Core Data stores timestamps as seconds since January 1, 2001 (not Unix epoch).

    Args:
        timestamp: Core Data timestamp (seconds since 2001-01-01).
                   If None, returns current datetime.

    Returns:
        Python datetime object.

    Example:
        >>> _core_data_to_datetime(0)
        datetime.datetime(2001, 1, 1, 0, 0)
    """
    if timestamp is None:
        return datetime.now()
    return CORE_DATA_EPOCH + timedelta(seconds=timestamp)


def _datetime_to_core_data(dt: datetime) -> float:
    """Convert a Python datetime to Apple Core Data timestamp."""
    return (dt - CORE_DATA_EPOCH).total_seconds()


def _extract_tags_from_text(text: str) -> tuple[str, ...]:
    """Extract Bear-style tags from note text.

    Bear tags are formatted as #tag or #tag/subtag. This function
    extracts them while avoiding markdown headers (##).

    Args:
        text: Note content to extract tags from.

    Returns:
        Tuple of unique tag names (without the # prefix).

    Example:
        >>> _extract_tags_from_text("Hello #world #diary/2024")
        ('world', 'diary/2024')
    """
    # Preserve first-seen order while deduplicating.
    matches = _TAG_PATTERN.findall(text)
    return tuple(dict.fromkeys(matches))


def _tag_matches(note: Note, tag_lower: str) -> bool:
    """Check whether a note has a tag (or nested child tag), case-insensitively."""
    prefix = f"{tag_lower}/"
    for tag in note.tags:
        normalized = tag.lower()
        if normalized == tag_lower or normalized.startswith(prefix):
            return True
    return False


# =============================================================================
# Main Reader Class
# =============================================================================


class BearReader:
    """Read-only interface to Bear's SQLite database.

    This class provides safe, read-only access to Bear's note database.
    It never modifies the database in any way.

    SAFETY GUARANTEES:
        - Database is opened with `mode=ro` (read-only)
        - Only SELECT queries are executed
        - Connection is closed after each operation
        - No data is cached that could be modified

    Attributes:
        db_path: Path to the Bear SQLite database.

    Example:
        >>> reader = BearReader()
        >>> notes = reader.get_notes_by_tag("diary")
        >>> print(f"Found {len(notes)} diary entries")

    Raises:
        DatabaseNotFoundError: If the Bear database doesn't exist.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the reader with a database path.

        Args:
            db_path: Path to Bear's database.sqlite file.
                     If None, uses the default macOS location.

        Raises:
            DatabaseNotFoundError: If database file doesn't exist.
        """
        self.db_path: Path = Path(db_path) if db_path else DEFAULT_BEAR_DB_PATH

        if not self.db_path.exists():
            raise DatabaseNotFoundError(str(self.db_path))

    def _get_readonly_connection(self) -> sqlite3.Connection:
        """Get a read-only database connection.

        Uses SQLite URI mode with `mode=ro` to enforce read-only access.
        This prevents any accidental writes to the database.

        Returns:
            Read-only SQLite connection.

        Raises:
            DatabaseReadError: If connection fails.
        """
        try:
            # CRITICAL: mode=ro ensures read-only access
            uri = f"file:{self.db_path}?mode=ro"
            return sqlite3.connect(uri, uri=True)
        except sqlite3.Error as e:
            raise DatabaseReadError(f"Failed to connect to database: {e}") from e

    def _query_notes(
        self,
        *,
        include_archived: bool = False,
        include_trashed: bool = False,
        where_clauses: Sequence[str] = (),
        params: Sequence[object] = (),
    ) -> list[Note]:
        """Query notes with optional SQL-level filtering."""
        conditions = ["ZTEXT IS NOT NULL", "ZTEXT <> ''"]
        if not include_trashed:
            conditions.append("(ZTRASHED = 0 OR ZTRASHED IS NULL)")
        if not include_archived:
            conditions.append("(ZARCHIVED = 0 OR ZARCHIVED IS NULL)")
        conditions.extend(where_clauses)

        query = f"""
            SELECT
                ZUNIQUEIDENTIFIER,
                ZTITLE,
                ZTEXT,
                ZCREATIONDATE,
                ZMODIFICATIONDATE
            FROM ZSFNOTE
            WHERE {" AND ".join(conditions)}
            ORDER BY ZCREATIONDATE ASC
        """

        conn = self._get_readonly_connection()
        try:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            raise DatabaseReadError(f"Failed to query notes: {e}") from e
        finally:
            conn.close()

        notes: list[Note] = []
        for uid, title, text, created, modified in rows:
            note = Note(
                id=uid or "",
                title=title or "Untitled",
                content=text,
                created_at=_core_data_to_datetime(created),
                modified_at=_core_data_to_datetime(modified),
                tags=_extract_tags_from_text(text),
            )
            notes.append(note)

        return notes

    def get_all_notes(
        self,
        *,
        include_archived: bool = False,
        include_trashed: bool = False,
    ) -> list[Note]:
        """Get all notes from the database.

        Retrieves notes sorted by creation date (oldest first).

        Args:
            include_archived: If True, include archived notes.
            include_trashed: If True, include trashed notes.

        Returns:
            List of Note objects sorted by creation date ascending.

        Raises:
            DatabaseReadError: If database query fails.

        Example:
            >>> reader = BearReader()
            >>> all_notes = reader.get_all_notes()
            >>> active_notes = reader.get_all_notes(include_archived=False)
        """
        return self._query_notes(
            include_archived=include_archived,
            include_trashed=include_trashed,
        )

    def get_notes_by_tag(self, tag: str) -> list[Note]:
        """Get notes containing a specific tag.

        Matches tags case-insensitively and includes nested tags.
        For example, tag="diary" matches #diary, #Diary, and #diary/2024.

        Args:
            tag: Tag to filter by (without # prefix).

        Returns:
            List of Note objects with the tag, sorted by creation date.

        Example:
            >>> reader = BearReader()
            >>> diary_notes = reader.get_notes_by_tag("diary")
            >>> work_notes = reader.get_notes_by_tag("work")
        """
        tag_clean = tag.strip().lstrip("#")
        if not tag_clean:
            return []

        tag_lower = tag_clean.lower()
        # Coarse SQL pre-filter to avoid scanning all notes.
        notes = self._query_notes(
            where_clauses=("lower(ZTEXT) LIKE ?",),
            params=(f"%#{tag_lower}%",),
        )
        return [note for note in notes if _tag_matches(note, tag_lower)]

    def get_notes_by_date_range(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        *,
        tag: str | None = None,
    ) -> list[Note]:
        """Get notes within a date range.

        Filters notes by creation date. Both bounds are inclusive.

        Args:
            start_date: Start of range (inclusive). None means no lower bound.
            end_date: End of range (inclusive). None means no upper bound.
            tag: Optional tag to additionally filter by.

        Returns:
            List of Note objects in the date range, sorted by creation date.

        Example:
            >>> from datetime import datetime
            >>> reader = BearReader()
            >>> jan_notes = reader.get_notes_by_date_range(
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 1, 31),
            ...     tag="diary",
            ... )
        """
        clauses: list[str] = []
        params: list[object] = []

        if start_date:
            clauses.append("ZCREATIONDATE >= ?")
            params.append(_datetime_to_core_data(start_date))
        if end_date:
            clauses.append("ZCREATIONDATE <= ?")
            params.append(_datetime_to_core_data(end_date))

        tag_lower: str | None = None
        if tag:
            tag_clean = tag.strip().lstrip("#")
            if not tag_clean:
                return []
            tag_lower = tag_clean.lower()
            clauses.append("lower(ZTEXT) LIKE ?")
            params.append(f"%#{tag_lower}%")

        notes = self._query_notes(
            where_clauses=tuple(clauses),
            params=tuple(params),
        )

        if tag_lower is None:
            return notes
        return [note for note in notes if _tag_matches(note, tag_lower)]

    def get_available_tags(self) -> list[tuple[str, int]]:
        """Get all tags with their usage counts.

        Returns:
            List of (tag_name, count) tuples, sorted by count descending.

        Example:
            >>> reader = BearReader()
            >>> for tag, count in reader.get_available_tags()[:5]:
            ...     print(f"#{tag}: {count} notes")
        """
        all_notes = self.get_all_notes()
        tag_counts: dict[str, int] = {}

        for note in all_notes:
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return sorted(tag_counts.items(), key=lambda x: -x[1])

    def get_date_range(self, tag: str | None = None) -> tuple[datetime, datetime]:
        """Get the date range spanned by notes.

        Args:
            tag: Optional tag to filter notes by.

        Returns:
            Tuple of (earliest_date, latest_date).
            If no notes found, returns (now, now).

        Example:
            >>> reader = BearReader()
            >>> start, end = reader.get_date_range(tag="diary")
            >>> print(f"Diary spans {start} to {end}")
        """
        if tag:
            notes = self.get_notes_by_tag(tag)
            if not notes:
                now = datetime.now()
                return (now, now)
            return (notes[0].created_at, notes[-1].created_at)

        query = """
            SELECT MIN(ZCREATIONDATE), MAX(ZCREATIONDATE)
            FROM ZSFNOTE
            WHERE ZTEXT IS NOT NULL
                AND ZTEXT <> ''
                AND (ZTRASHED = 0 OR ZTRASHED IS NULL)
                AND (ZARCHIVED = 0 OR ZARCHIVED IS NULL)
        """
        conn = self._get_readonly_connection()
        try:
            min_ts, max_ts = conn.execute(query).fetchone()
        except sqlite3.Error as e:
            raise DatabaseReadError(f"Failed to query date range: {e}") from e
        finally:
            conn.close()

        if min_ts is None or max_ts is None:
            now = datetime.now()
            return (now, now)

        return (_core_data_to_datetime(min_ts), _core_data_to_datetime(max_ts))
