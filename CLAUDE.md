# CLAUDE.md - Project Guide for AI Assistants

## Project Overview

**diary-slm** is a local-first diary analysis tool that uses Apple's MLX framework to run LLMs on Apple Silicon. It reads diary entries from the Bear note-taking app and analyzes them using large language models with full context (no RAG).

### Core Philosophy

- **No RAG**: Instead of retrieval-augmented generation, we directly inject diary content into the prompt. This gives the LLM a "god's eye view" to see causal relationships across time.
- **Local-first**: All processing happens locally using mlx-lm. No data leaves the machine.
- **Read-only**: We NEVER write to Bear's database. All database access is strictly read-only.

## Architecture

```
src/diary_slm/
├── bear_reader.py    # SQLite extraction (READ-ONLY)
├── processor.py      # Time-based chunking & token estimation
├── model.py          # mlx-lm wrapper for inference
├── analyzer.py       # Analysis prompts & workflows
└── cli.py            # Click-based CLI interface
```

### Data Flow

1. `BearReader` extracts notes from Bear's SQLite database (read-only)
2. `DiaryProcessor` groups notes by time period (month/quarter/half_year/year)
3. `ModelManager` loads an mlx-lm model and handles text generation
4. `DiaryAnalyzer` combines diary context with analysis prompts
5. `cli.py` provides the user interface

## Key Design Decisions

### Why No RAG?

A year of diaries (~500k characters) is roughly 125k tokens. Modern models support 128k+ context. By splitting into quarters (~32k tokens) or half-years (~64k tokens), we can fit the entire period in one prompt. This allows the LLM to see patterns and causality that RAG would miss.

### Token Estimation

We use a conservative estimate of 4 characters per token. This works well for mixed English/Chinese content. The actual token count depends on the model's tokenizer.

### Period Types

- `month`: ~10k tokens, fine-grained analysis
- `quarter`: ~32k tokens, recommended default
- `half_year`: ~64k tokens, broader patterns
- `year`: ~125k tokens, may exceed context limits

## Critical Safety Rules

### Database Safety (CRITICAL)

The Bear database is the user's personal data. We MUST:

1. **Always use read-only mode**: Open SQLite with `?mode=ro` URI parameter
2. **Never execute INSERT, UPDATE, DELETE, or DROP statements**
3. **Never modify the database file or its directory**
4. **Copy data to memory, never hold long-running connections**

### Privacy

- No data should be logged, cached permanently, or sent externally
- Prompt caches should use temp directories and be session-scoped
- No hardcoded paths that could expose usernames

## Common Development Tasks

### Adding a New Analysis Template

1. Add the template to `ANALYSIS_PROMPTS` dict in `analyzer.py`
2. The template should be a detailed prompt string
3. It will automatically be available via CLI and interactive mode

### Supporting a New Note Source

1. Create a new reader class following `BearReader` interface
2. Must implement: `get_all_notes()`, `get_notes_by_tag()`, `get_notes_by_date_range()`
3. Must return `Note` dataclass instances

### Adding a New Model Preset

1. Add to `DEFAULT_MODELS` dict in `model.py`
2. Use the mlx-community HuggingFace format

## Testing Locally

```bash
# Install in development mode
pip install -e ".[dev]"

# List available periods (tests Bear connection)
diary-slm list

# Run with a specific tag
diary-slm list -t diary

# Test analysis (requires mlx-lm model download)
diary-slm analyze 2024-Q1 -q "Summarize this quarter"
```

## Dependencies

- `mlx-lm`: Apple's MLX framework for LLM inference
- `click`: CLI framework
- `rich`: Terminal formatting

## Git Commits

**Do NOT add Claude as a coauthor** for commits in this repository. Commit messages should be attributed solely to the human developer.

## File Locations

- Bear database: `~/Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite`
- This is auto-detected but can be overridden with `--db` flag
