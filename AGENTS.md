# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the CLI entry point (`python main.py ...`).
- `diary_slm/` contains core package code:
- `bear_reader.py` (read-only Bear DB access), `processor.py` (period chunking/token estimates), `model.py` (MLX model wrapper), `analyzer.py` (prompt workflows), `cli.py` (Click commands), `constants.py`, and `exceptions.py`.
- `data/` stores local data artifacts and examples; do not commit private diary content.
- `README.md` is end-user documentation; `CLAUDE.md` captures architecture and safety context for contributors.

## Build, Test, and Development Commands
- `python -m pip install -e .`: install runtime dependencies in editable mode.
- `python -m pip install -e ".[dev]"`: install dev tooling (`pytest`, `ruff`).
- `python main.py --help`: list all CLI commands.
- `python main.py list -t diary`: quick smoke test for Bear connectivity and period detection.
- `ruff check diary_slm main.py`: run lint checks (line length and style).
- `pytest`: run tests (once tests are added under `tests/`).

## Coding Style & Naming Conventions
- Target Python `>=3.10`; use 4-space indentation and type hints for public functions.
- Keep lines within 100 chars (`[tool.ruff].line-length = 100`).
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Keep CLI behavior changes in `diary_slm/cli.py`; keep domain logic in module files, not command handlers.

## Testing Guidelines
- Use `pytest` as the test framework; place tests in `tests/` with names like `test_processor.py`.
- Prefer focused unit tests for chunking logic, prompt/template selection, and error paths.
- For CLI changes, add at least one smoke-style test case or documented manual check command.
- No fixed coverage gate is configured yet; cover all modified critical paths before merging.

## Commit & Pull Request Guidelines
- Follow existing history style: concise, imperative summaries (for example, `Fix GLM-4.7 preset mapping`).
- Keep commits scoped to one logical change.
- PRs should include:
- What changed and why.
- Commands run for validation (for example, `ruff check ...`, `pytest`, `python main.py list -t diary`).
- Linked issue/context and terminal output screenshots when CLI UX changes.

## Security & Configuration Tips
- Treat Bear data as sensitive: maintain read-only DB access and never add write queries.
- Do not hardcode personal paths; prefer `--db` overrides and environment-local configuration.
- Never commit raw diary exports or personal notes.
