"""Custom exceptions for diary-slm.

This module defines a hierarchy of exceptions used throughout the library.
All exceptions inherit from DiarySlmError for easy catching.

Example:
    try:
        reader = BearReader()
        notes = reader.get_all_notes()
    except DiarySlmError as e:
        print(f"diary-slm error: {e}")
"""

from .constants import MAX_PERIODS_TO_SUGGEST


class DiarySlmError(Exception):
    """Base exception for all diary-slm errors.

    All custom exceptions in this library inherit from this class,
    making it easy to catch any diary-slm specific error.
    """

    pass


class DatabaseError(DiarySlmError):
    """Error related to database operations.

    Raised when there's an issue reading from or connecting to
    the Bear SQLite database.
    """

    pass


class DatabaseNotFoundError(DatabaseError):
    """Bear database file not found.

    Raised when the Bear SQLite database cannot be located at
    the expected path. This usually means Bear is not installed
    or has never been used.

    Attributes:
        path: The path where the database was expected.
    """

    def __init__(self, path: str, message: str | None = None):
        self.path = path
        if message is None:
            message = (
                f"Bear database not found at: {path}\n"
                "Possible solutions:\n"
                "  1. Make sure Bear app is installed\n"
                "  2. Open Bear and create at least one note\n"
                "  3. Use --db to specify a custom database path"
            )
        super().__init__(message)


class DatabaseReadError(DatabaseError):
    """Error reading from the database.

    Raised when a database query fails or returns unexpected results.
    """

    pass


class NoteError(DiarySlmError):
    """Error related to note operations."""

    pass


class NoNotesFoundError(NoteError):
    """No notes found matching the criteria.

    Raised when a query returns no notes, but notes were expected.

    Attributes:
        tag: The tag filter that was applied, if any.
        date_range: The date range filter that was applied, if any.
    """

    def __init__(
        self,
        tag: str | None = None,
        date_range: tuple | None = None,
        message: str | None = None,
    ):
        self.tag = tag
        self.date_range = date_range

        if message is None:
            parts = ["No notes found"]
            if tag:
                parts.append(f"with tag #{tag}")
            if date_range:
                start, end = date_range
                parts.append(f"between {start} and {end}")
            message = " ".join(parts) + "."

        super().__init__(message)


class PeriodError(DiarySlmError):
    """Error related to time period operations."""

    pass


class PeriodNotFoundError(PeriodError):
    """Specified period not found in the diary data.

    Attributes:
        period_name: The period that was requested.
        available_periods: List of available period names, if known.
    """

    def __init__(
        self,
        period_name: str,
        available_periods: list[str] | None = None,
        message: str | None = None,
    ):
        self.period_name = period_name
        self.available_periods = available_periods

        if message is None:
            message = f"Period not found: {period_name}"
            if available_periods:
                preview = available_periods[:MAX_PERIODS_TO_SUGGEST]
                message += f"\nAvailable periods: {', '.join(preview)}"
                if len(available_periods) > MAX_PERIODS_TO_SUGGEST:
                    remaining = len(available_periods) - MAX_PERIODS_TO_SUGGEST
                    message += f" (and {remaining} more)"

        super().__init__(message)


class InvalidPeriodTypeError(PeriodError):
    """Invalid period type specified.

    Attributes:
        period_type: The invalid period type.
        valid_types: List of valid period types.
    """

    VALID_TYPES = ("month", "quarter", "half_year", "year")

    def __init__(self, period_type: str, message: str | None = None):
        self.period_type = period_type

        if message is None:
            message = (
                f"Invalid period type: '{period_type}'. "
                f"Must be one of: {', '.join(self.VALID_TYPES)}"
            )

        super().__init__(message)


class ModelError(DiarySlmError):
    """Error related to LLM model operations."""

    pass


class ModelNotLoadedError(ModelError):
    """Model has not been loaded yet.

    Raised when attempting to use a model before it has been loaded.
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Model not loaded. Call load() first or access the model property."
        super().__init__(message)


class ModelLoadError(ModelError):
    """Failed to load the model.

    Attributes:
        model_name: The model that failed to load.
        reason: The underlying reason for the failure, if known.
    """

    def __init__(
        self,
        model_name: str,
        reason: str | None = None,
        message: str | None = None,
    ):
        self.model_name = model_name
        self.reason = reason

        if message is None:
            message = f"Failed to load model: {model_name}"
            if reason:
                message += f"\nReason: {reason}"

        super().__init__(message)


class AnalysisError(DiarySlmError):
    """Error during diary analysis."""

    pass


class TemplateNotFoundError(AnalysisError):
    """Analysis template not found.

    Attributes:
        template_name: The template that was requested.
        available_templates: List of available template names.
    """

    def __init__(
        self,
        template_name: str,
        available_templates: list[str] | None = None,
        message: str | None = None,
    ):
        self.template_name = template_name
        self.available_templates = available_templates

        if message is None:
            message = f"Unknown analysis template: '{template_name}'"
            if available_templates:
                message += f"\nAvailable templates: {', '.join(available_templates)}"

        super().__init__(message)


class ContextTooLargeError(AnalysisError):
    """Context exceeds model's maximum token limit.

    Attributes:
        estimated_tokens: Estimated token count of the context.
        max_tokens: Maximum tokens the model supports.
    """

    def __init__(
        self,
        estimated_tokens: int,
        max_tokens: int,
        message: str | None = None,
    ):
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens

        if message is None:
            message = (
                f"Context too large: ~{estimated_tokens:,} tokens "
                f"(max: {max_tokens:,}).\n"
                "Try using a shorter time period (e.g., month instead of quarter)."
            )

        super().__init__(message)
