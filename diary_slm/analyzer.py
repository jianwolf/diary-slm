"""Diary analysis using LLM with full context.

This module provides the high-level analysis interface for diary entries.
It combines the diary context with analysis prompts and sends them to
the LLM for processing.

Key Design Principle:
    NO RAG - We feed the complete diary period into the context,
    giving the LLM full visibility into patterns and causality.

Example:
    >>> from diary_slm.analyzer import DiaryAnalyzer
    >>> from diary_slm.model import ModelManager
    >>>
    >>> model = ModelManager()
    >>> analyzer = DiaryAnalyzer(model)
    >>> for text in analyzer.analyze(chunk, "What patterns do you see?"):
    ...     print(text, end="")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from .exceptions import TemplateNotFoundError

if TYPE_CHECKING:
    from .model import ModelManager
    from .processor import DiaryChunk

__all__ = [
    "DiaryAnalyzer",
    "ANALYSIS_TEMPLATES",
    "SYSTEM_PROMPT",
    "list_analysis_templates",
]


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT: str = """You are a thoughtful and insightful diary analyst. You have access to the user's diary entries and can see patterns, themes, and connections across time.

Your analysis should be:
- Empathetic and non-judgmental
- Focused on patterns and insights the writer might not see themselves
- Grounded in specific examples from the diary entries
- Constructive and forward-looking when appropriate

You can see causal relationships because you have the full context of the diary over time.
Respond in the same language as the diary entries."""


# =============================================================================
# Analysis Templates
# =============================================================================

# Pre-built analysis prompts for common use cases
# Each template is designed to extract specific insights
ANALYSIS_TEMPLATES: dict[str, str] = {
    "summary": """Based on these diary entries, provide a comprehensive summary including:
1. Main themes and topics that appeared
2. Key events and milestones
3. Emotional journey and mood patterns
4. Relationships and people mentioned
5. Goals, aspirations, and progress made

Be specific with dates and examples from the entries.""",

    "mood": """Analyze the emotional patterns in these diary entries:
1. Overall emotional tone during this period
2. Mood fluctuations and what triggered them
3. Recurring emotional themes
4. How emotions evolved over time
5. Any concerning patterns or positive trends

Reference specific entries to support your analysis.""",

    "themes": """Identify and analyze the recurring themes in these diary entries:
1. What topics come up repeatedly?
2. What concerns or interests persist over time?
3. Are there patterns in what the writer thinks about?
4. What implicit values or priorities emerge?
5. Any themes that appear, disappear, or transform?

Provide specific examples from the entries.""",

    "growth": """Analyze personal growth and development shown in these entries:
1. What challenges were faced and how were they handled?
2. What lessons were learned?
3. How has thinking or behavior evolved?
4. What goals were set and what progress was made?
5. Areas of improvement and remaining growth opportunities

Use specific examples to illustrate growth.""",

    "relationships": """Analyze the relationships and social dynamics in these entries:
1. Key people mentioned and their roles
2. How relationships evolved over this period
3. Patterns in social interactions
4. Sources of support and conflict
5. Any relationship insights the writer might benefit from

Reference specific mentions and interactions.""",

    "advice": """Based on everything in these diary entries, what advice would you give to the writer?

Consider:
1. Blind spots they might have
2. Patterns they might not see
3. Strengths they should leverage
4. Areas that need attention
5. Concrete suggestions for improvement

Be compassionate but honest. Ground advice in specific observations.""",

    "timeline": """Create a timeline of significant events from these diary entries:
1. List major events in chronological order
2. Note the emotional significance of each
3. Identify turning points or pivotal moments
4. Show how events connect and influence each other
5. Highlight any patterns in when events occur

Format as a clear timeline with dates and brief descriptions.""",

    "questions": """Based on these diary entries, generate thoughtful questions for the writer to reflect on:
1. Questions about patterns you noticed
2. Questions about unexplored topics
3. Questions about relationships
4. Questions about goals and values
5. Questions about emotional patterns

These should be questions that could lead to deeper self-understanding.""",
}


def list_analysis_templates() -> dict[str, str]:
    """Get available analysis templates with shortened descriptions.

    Returns:
        Dictionary mapping template names to truncated descriptions.

    Example:
        >>> for name, desc in list_analysis_templates().items():
        ...     print(f"{name}: {desc}")
    """
    return {
        name: (prompt[:100] + "...") if len(prompt) > 100 else prompt
        for name, prompt in ANALYSIS_TEMPLATES.items()
    }


# =============================================================================
# Analyzer Class
# =============================================================================


class DiaryAnalyzer:
    """Analyze diary entries using LLM with full context.

    This class orchestrates the analysis of diary chunks by:
    1. Combining diary text with analysis prompts
    2. Sending to the LLM via ModelManager
    3. Returning streaming or complete responses

    The key insight is that we feed the ENTIRE period's diary
    into the context, avoiding RAG and giving the LLM complete
    visibility into patterns and causality.

    Attributes:
        model: The ModelManager instance for LLM inference.

    Example:
        >>> analyzer = DiaryAnalyzer(model)
        >>> for text in analyzer.analyze(chunk, "What were my struggles?"):
        ...     print(text, end="")

    Note:
        For interactive sessions with multiple queries, use
        interactive_session() which provides a REPL-like interface.
    """

    def __init__(self, model: ModelManager) -> None:
        """Initialize with a model manager.

        Args:
            model: ModelManager instance for LLM inference.
        """
        self._model = model

    @property
    def model(self) -> ModelManager:
        """Get the model manager."""
        return self._model

    def _build_analysis_prompt(self, diary_text: str, query: str) -> str:
        """Build the complete prompt combining diary context and query.

        Args:
            diary_text: Formatted diary entries.
            query: The analysis question/request.

        Returns:
            Complete prompt string.
        """
        return f"""{diary_text}

---

Based on the diary entries above, please respond to the following:

{query}"""

    def analyze(
        self,
        chunk: DiaryChunk,
        query: str,
        *,
        stream: bool = True,
    ) -> str | Iterator[str]:
        """Analyze a diary chunk with a custom query.

        This is the main analysis method. It combines the diary text
        with your query and sends it to the LLM.

        Args:
            chunk: DiaryChunk containing formatted diary entries.
            query: Your analysis question or request.
            stream: If True, return an iterator for streaming output.
                    If False, return the complete response string.

        Returns:
            If stream=True: Iterator yielding text chunks.
            If stream=False: Complete response string.

        Example:
            >>> # Streaming (recommended for UX)
            >>> for text in analyzer.analyze(chunk, "What patterns?"):
            ...     print(text, end="", flush=True)
            >>>
            >>> # Non-streaming
            >>> response = analyzer.analyze(chunk, "What patterns?", stream=False)
            >>> print(response)
        """
        prompt = self._build_analysis_prompt(chunk.formatted_text, query)

        if stream:
            return self._model.stream_generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
        else:
            return self._model.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

    def analyze_with_template(
        self,
        chunk: DiaryChunk,
        template_name: str,
        *,
        stream: bool = True,
    ) -> str | Iterator[str]:
        """Analyze using a pre-built analysis template.

        Templates provide well-crafted prompts for common analysis types.
        See ANALYSIS_TEMPLATES for available options.

        Args:
            chunk: DiaryChunk containing formatted diary entries.
            template_name: Name of the template to use.
                           Options: summary, mood, themes, growth,
                           relationships, advice, timeline, questions.
            stream: If True, return an iterator for streaming output.

        Returns:
            If stream=True: Iterator yielding text chunks.
            If stream=False: Complete response string.

        Raises:
            TemplateNotFoundError: If template_name is not recognized.

        Example:
            >>> for text in analyzer.analyze_with_template(chunk, "mood"):
            ...     print(text, end="")
        """
        if template_name not in ANALYSIS_TEMPLATES:
            raise TemplateNotFoundError(
                template_name,
                available_templates=list(ANALYSIS_TEMPLATES.keys()),
            )

        query = ANALYSIS_TEMPLATES[template_name]
        return self.analyze(chunk, query, stream=stream)

    def compare_periods(
        self,
        chunk1: DiaryChunk,
        chunk2: DiaryChunk,
        query: str,
        *,
        stream: bool = True,
    ) -> str | Iterator[str]:
        """Compare two diary periods.

        This method combines two periods' entries and asks the LLM
        to analyze changes, progress, and evolution between them.

        Args:
            chunk1: First DiaryChunk (earlier period).
            chunk2: Second DiaryChunk (later period).
            query: Specific comparison question.
            stream: If True, return an iterator for streaming output.

        Returns:
            If stream=True: Iterator yielding text chunks.
            If stream=False: Complete response string.

        Warning:
            Combined token count may be large. Ensure it fits in context.

        Example:
            >>> for text in analyzer.compare_periods(q1, q2, "How did my mood change?"):
            ...     print(text, end="")
        """
        # Combine both periods with clear separation
        combined_text = f"""=== PERIOD 1: {chunk1.period_name} ===

{chunk1.formatted_text}

=== PERIOD 2: {chunk2.period_name} ===

{chunk2.formatted_text}"""

        # Wrap the user's query with comparison context
        full_query = f"""Compare these two periods of diary entries:

{query}

Focus on changes, progress, and evolution between the two periods."""

        prompt = self._build_analysis_prompt(combined_text, full_query)

        if stream:
            return self._model.stream_generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
        else:
            return self._model.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

    def interactive_session(self, chunk: DiaryChunk) -> None:
        """Start an interactive analysis session.

        Provides a REPL-like interface for asking multiple questions
        about the same diary period. The diary context is loaded once
        and reused for all queries.

        Commands:
            /templates - List available analysis templates
            /template <name> - Use a specific template
            /quit or /exit - End the session

        Args:
            chunk: DiaryChunk to analyze interactively.

        Example:
            >>> analyzer.interactive_session(chunk)
            Interactive Analysis: 2024-Q1
            Notes: 90, Tokens: ~31,250

            Type your questions. Commands: /templates, /quit

            You: What patterns do you see?
            Assistant: [analysis...]

            You: /quit
            Session ended.
        """
        from rich.console import Console

        console = Console()

        # Display session header
        console.print(f"\n[bold]Interactive Analysis: {chunk.period_name}[/bold]")
        console.print(f"Notes: {chunk.note_count}, Tokens: ~{chunk.estimated_tokens:,}")
        console.print("\nType your questions. Commands: /templates, /quit\n")

        # Build the context once
        context_prompt = self._build_analysis_prompt(chunk.formatted_text, "")

        # Interactive loop
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("/quit", "/exit", "/q"):
                break

            if user_input.lower() == "/templates":
                console.print("\n[bold]Available templates:[/bold]")
                for name, desc in list_analysis_templates().items():
                    console.print(f"  [cyan]{name}[/cyan]: {desc}")
                console.print("\nUse: /template <name>\n")
                continue

            if user_input.lower().startswith("/template "):
                template_name = user_input.split(maxsplit=1)[1]
                if template_name in ANALYSIS_TEMPLATES:
                    user_input = ANALYSIS_TEMPLATES[template_name]
                else:
                    console.print(f"[red]Unknown template: {template_name}[/red]")
                    console.print(f"Available: {', '.join(ANALYSIS_TEMPLATES.keys())}")
                    continue

            # Generate response
            console.print("\n[bold green]Assistant:[/bold green]")

            for text in self._model.stream_generate(
                prompt=f"{context_prompt}\n{user_input}",
                system_prompt=SYSTEM_PROMPT,
            ):
                console.print(text, end="")

            console.print("\n")

        console.print("\n[dim]Session ended.[/dim]")
