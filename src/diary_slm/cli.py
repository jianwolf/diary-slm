"""Command-line interface for diary-slm.

This module provides the CLI entry point for the diary-slm tool.
All commands are implemented using Click and output is formatted
using Rich.

Usage:
    diary-slm list              # List available periods
    diary-slm analyze PERIOD    # Analyze with custom query
    diary-slm template PERIOD   # Use analysis template
    diary-slm compare P1 P2     # Compare two periods
    diary-slm interactive P     # Interactive session
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.table import Table

from .analyzer import DiaryAnalyzer, list_analysis_templates
from .bear_reader import BearReader
from .constants import DEFAULT_MODEL_ID, MAX_TAGS_TO_DISPLAY, SAFE_CONTEXT_LIMIT
from .exceptions import DatabaseNotFoundError, PeriodNotFoundError, TemplateNotFoundError
from .model import ModelManager, list_available_models
from .processor import DiaryProcessor, PeriodType

if TYPE_CHECKING:
    pass

# Rich console for formatted output
console = Console()


# =============================================================================
# Helper Functions
# =============================================================================


def _create_processor(db_path: str | None, tag: str | None) -> DiaryProcessor:
    """Create a DiaryProcessor from Bear notes.

    Args:
        db_path: Optional custom database path.
        tag: Optional tag to filter notes by.

    Returns:
        Configured DiaryProcessor.

    Exits:
        With code 1 if database not found or no notes match.
    """
    try:
        reader = BearReader(db_path=Path(db_path) if db_path else None)
    except DatabaseNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if tag:
        notes = reader.get_notes_by_tag(tag)
        if not notes:
            console.print(f"[yellow]No notes found with tag #{tag}[/yellow]")
            console.print("Use 'diary-slm tags' to see available tags.")
            sys.exit(1)
    else:
        notes = reader.get_all_notes()
        if not notes:
            console.print("[yellow]No notes found in Bear database.[/yellow]")
            sys.exit(1)

    return DiaryProcessor(notes)


def _format_token_count(tokens: int) -> str:
    """Format token count with thousands separator."""
    return f"{tokens:,}"


# =============================================================================
# CLI Group
# =============================================================================


@click.group()
@click.version_option(package_name="diary-slm")
def main() -> None:
    """Analyze your Bear diary entries using local LLMs.

    diary-slm uses Apple's MLX framework to run language models locally
    on your Mac. It reads your diary entries from Bear and analyzes them
    with full context (no RAG).

    Quick start:

        diary-slm list -t diary          # See your diary periods

        diary-slm analyze 2024-Q1 \\
            -t diary \\
            -q "What patterns do you see?"
    """
    pass


# =============================================================================
# List Command
# =============================================================================


@main.command("list")
@click.option(
    "--db", "db_path",
    type=click.Path(exists=True),
    help="Path to Bear database (auto-detected if not specified).",
)
@click.option(
    "--tag", "-t",
    help="Filter notes by tag (e.g., 'diary').",
)
@click.option(
    "--period-type", "-p",
    type=click.Choice(["month", "quarter", "half_year", "year"]),
    default="quarter",
    show_default=True,
    help="How to split time periods.",
)
def list_cmd(
    db_path: str | None,
    tag: str | None,
    period_type: PeriodType,
) -> None:
    """List available diary periods and their statistics.

    Shows the number of notes and estimated token count for each period.
    Use this to see what periods are available for analysis.

    Examples:

        diary-slm list

        diary-slm list -t diary -p month
    """
    processor = _create_processor(db_path, tag)

    # Overall statistics
    stats = processor.get_total_stats()
    console.print("\n[bold]Diary Statistics[/bold]")
    console.print(f"Total notes: {stats['total_notes']:,}")
    console.print(f"Total characters: {stats['total_chars']:,}")
    console.print(f"Estimated tokens: {stats['estimated_tokens']:,}")

    if stats["date_range"]:
        start, end = stats["date_range"]
        console.print(
            f"Date range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        )

    suggested = processor.suggest_period_type()
    console.print(f"Suggested period type: [cyan]{suggested}[/cyan]")

    # Period breakdown
    periods = processor.get_available_periods(period_type)

    if not periods:
        console.print("[yellow]No periods found.[/yellow]")
        return

    table = Table(title=f"\nAvailable Periods ({period_type})")
    table.add_column("Period", style="cyan")
    table.add_column("Notes", justify="right")
    table.add_column("Est. Tokens", justify="right")
    table.add_column("Fits 128k?", justify="center")

    for name, count, tokens in periods:
        fits = "[green]Yes[/green]" if tokens < SAFE_CONTEXT_LIMIT else "[red]No[/red]"
        table.add_row(name, str(count), _format_token_count(tokens), fits)

    console.print(table)


# =============================================================================
# Analyze Command
# =============================================================================


@main.command()
@click.argument("period")
@click.option(
    "--query", "-q",
    required=True,
    help="Your analysis question.",
)
@click.option("--db", "db_path", help="Path to Bear database.")
@click.option("--tag", "-t", help="Filter notes by tag.")
@click.option(
    "--period-type", "-p",
    type=click.Choice(["month", "quarter", "half_year", "year"]),
    default="quarter",
)
@click.option(
    "--model", "-m",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Model to use (ID or alias).",
)
@click.option(
    "--max-tokens",
    default=4096,
    show_default=True,
    help="Max tokens to generate.",
)
def analyze(
    period: str,
    query: str,
    db_path: str | None,
    tag: str | None,
    period_type: PeriodType,
    model: str,
    max_tokens: int,
) -> None:
    """Analyze a specific diary period with a custom query.

    PERIOD is the time period to analyze (e.g., "2024-Q1", "2024-01").
    Use 'diary-slm list' to see available periods.

    Examples:

        diary-slm analyze 2024-Q1 -q "What were my main struggles?" -t diary

        diary-slm analyze 2024-01 -p month -q "Summarize this month"
    """
    processor = _create_processor(db_path, tag)
    chunk = processor.get_chunk_by_name(period, period_type)

    if not chunk:
        console.print(f"[red]Period not found: {period}[/red]")
        console.print("Use 'diary-slm list' to see available periods.")
        sys.exit(1)

    console.print(f"\n[bold]Analyzing {period}[/bold]")
    console.print(f"Notes: {chunk.note_count}, Tokens: ~{_format_token_count(chunk.estimated_tokens)}")
    console.print()

    model_mgr = ModelManager(model, max_tokens=max_tokens)
    analyzer = DiaryAnalyzer(model_mgr)

    console.print("[bold green]Analysis:[/bold green]\n")
    for text in analyzer.analyze(chunk, query, stream=True):
        console.print(text, end="")
    console.print("\n")


# =============================================================================
# Template Command
# =============================================================================


@main.command()
@click.argument("period")
@click.argument("template_name", metavar="TEMPLATE")
@click.option("--db", "db_path", help="Path to Bear database.")
@click.option("--tag", "-t", help="Filter notes by tag.")
@click.option(
    "--period-type", "-p",
    type=click.Choice(["month", "quarter", "half_year", "year"]),
    default="quarter",
)
@click.option("--model", "-m", default=DEFAULT_MODEL_ID, help="Model to use.")
@click.option("--max-tokens", default=4096, help="Max tokens to generate.")
def template(
    period: str,
    template_name: str,
    db_path: str | None,
    tag: str | None,
    period_type: PeriodType,
    model: str,
    max_tokens: int,
) -> None:
    """Run a pre-built analysis template on a diary period.

    TEMPLATE is one of: summary, mood, themes, growth, relationships,
    advice, timeline, questions.

    Use 'diary-slm templates' to see all available templates.

    Examples:

        diary-slm template 2024-Q1 mood -t diary

        diary-slm template 2024-Q1 summary -t diary
    """
    processor = _create_processor(db_path, tag)
    chunk = processor.get_chunk_by_name(period, period_type)

    if not chunk:
        console.print(f"[red]Period not found: {period}[/red]")
        sys.exit(1)

    templates = list_analysis_templates()
    if template_name not in templates:
        console.print(f"[red]Unknown template: {template_name}[/red]")
        console.print(f"Available: {', '.join(templates.keys())}")
        sys.exit(1)

    console.print(f"\n[bold]Running '{template_name}' analysis on {period}[/bold]")
    console.print(f"Notes: {chunk.note_count}, Tokens: ~{_format_token_count(chunk.estimated_tokens)}")
    console.print()

    model_mgr = ModelManager(model, max_tokens=max_tokens)
    analyzer = DiaryAnalyzer(model_mgr)

    console.print("[bold green]Analysis:[/bold green]\n")
    for text in analyzer.analyze_with_template(chunk, template_name, stream=True):
        console.print(text, end="")
    console.print("\n")


# =============================================================================
# Compare Command
# =============================================================================


@main.command()
@click.argument("period1")
@click.argument("period2")
@click.option(
    "--query", "-q",
    default="How did things change?",
    show_default=True,
    help="Comparison question.",
)
@click.option("--db", "db_path", help="Path to Bear database.")
@click.option("--tag", "-t", help="Filter notes by tag.")
@click.option(
    "--period-type", "-p",
    type=click.Choice(["month", "quarter", "half_year", "year"]),
    default="quarter",
)
@click.option("--model", "-m", default=DEFAULT_MODEL_ID, help="Model to use.")
@click.option("--max-tokens", default=4096, help="Max tokens to generate.")
def compare(
    period1: str,
    period2: str,
    query: str,
    db_path: str | None,
    tag: str | None,
    period_type: PeriodType,
    model: str,
    max_tokens: int,
) -> None:
    """Compare two diary periods to see changes over time.

    Compares PERIOD1 and PERIOD2 to identify changes, progress, and evolution.

    Examples:

        diary-slm compare 2024-Q1 2024-Q2 -t diary

        diary-slm compare 2024-Q1 2024-Q2 -q "How did my mood change?" -t diary
    """
    processor = _create_processor(db_path, tag)

    chunk1 = processor.get_chunk_by_name(period1, period_type)
    chunk2 = processor.get_chunk_by_name(period2, period_type)

    if not chunk1:
        console.print(f"[red]Period not found: {period1}[/red]")
        sys.exit(1)
    if not chunk2:
        console.print(f"[red]Period not found: {period2}[/red]")
        sys.exit(1)

    total_tokens = chunk1.estimated_tokens + chunk2.estimated_tokens
    console.print(f"\n[bold]Comparing {period1} vs {period2}[/bold]")
    console.print(f"Combined tokens: ~{_format_token_count(total_tokens)}")

    if total_tokens > SAFE_CONTEXT_LIMIT:
        console.print("[yellow]Warning: Combined context may exceed model limits[/yellow]")

    console.print()

    model_mgr = ModelManager(model, max_tokens=max_tokens)
    analyzer = DiaryAnalyzer(model_mgr)

    console.print("[bold green]Comparison:[/bold green]\n")
    for text in analyzer.compare_periods(chunk1, chunk2, query, stream=True):
        console.print(text, end="")
    console.print("\n")


# =============================================================================
# Interactive Command
# =============================================================================


@main.command()
@click.argument("period")
@click.option("--db", "db_path", help="Path to Bear database.")
@click.option("--tag", "-t", help="Filter notes by tag.")
@click.option(
    "--period-type", "-p",
    type=click.Choice(["month", "quarter", "half_year", "year"]),
    default="quarter",
)
@click.option("--model", "-m", default=DEFAULT_MODEL_ID, help="Model to use.")
@click.option("--max-tokens", default=4096, help="Max tokens to generate.")
def interactive(
    period: str,
    db_path: str | None,
    tag: str | None,
    period_type: PeriodType,
    model: str,
    max_tokens: int,
) -> None:
    """Start an interactive analysis session.

    Opens a REPL-like interface where you can ask multiple questions
    about the same diary period.

    Commands in interactive mode:
        /templates  - List available templates
        /template NAME  - Use a template
        /quit or /exit  - End session

    Example:

        diary-slm interactive 2024-Q1 -t diary
    """
    processor = _create_processor(db_path, tag)
    chunk = processor.get_chunk_by_name(period, period_type)

    if not chunk:
        console.print(f"[red]Period not found: {period}[/red]")
        sys.exit(1)

    model_mgr = ModelManager(model, max_tokens=max_tokens)
    analyzer = DiaryAnalyzer(model_mgr)
    analyzer.interactive_session(chunk)


# =============================================================================
# Info Commands
# =============================================================================


@main.command()
def models() -> None:
    """List available model presets.

    Shows alias names and their full HuggingFace model IDs.
    You can use either the alias or full ID with --model.
    """
    console.print("\n[bold]Available Model Presets[/bold]\n")

    table = Table()
    table.add_column("Alias", style="cyan")
    table.add_column("Model ID")

    for alias, model_id in list_available_models().items():
        table.add_row(alias, model_id)

    console.print(table)
    console.print("\nYou can also use any mlx-community model ID directly with --model")


@main.command()
def templates() -> None:
    """List available analysis templates.

    Templates are pre-built prompts for common analysis tasks.
    Use them with the 'template' command.
    """
    console.print("\n[bold]Available Analysis Templates[/bold]\n")

    for name, desc in list_analysis_templates().items():
        console.print(f"[cyan]{name}[/cyan]")
        console.print(f"  {desc}\n")


@main.command()
@click.option("--db", "db_path", help="Path to Bear database.")
def tags(db_path: str | None) -> None:
    """List all tags found in your Bear notes.

    Use this to find the right tag for your diary entries.
    """
    try:
        reader = BearReader(db_path=Path(db_path) if db_path else None)
    except DatabaseNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    tag_list = reader.get_available_tags()

    if not tag_list:
        console.print("[yellow]No tags found in Bear notes.[/yellow]")
        return

    table = Table(title="Tags in Bear Notes")
    table.add_column("Tag", style="cyan")
    table.add_column("Notes", justify="right")

    for tag_name, count in tag_list[:MAX_TAGS_TO_DISPLAY]:
        table.add_row(f"#{tag_name}", str(count))

    console.print(table)

    if len(tag_list) > MAX_TAGS_TO_DISPLAY:
        console.print(f"\n[dim]... and {len(tag_list) - MAX_TAGS_TO_DISPLAY} more tags[/dim]")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    main()
