# Optimization Areas and Improvements

This repository processes large diary datasets, so avoid full-data scans and repeated string work where possible.

## Implemented Improvements

### 1. SQL-first filtering in `BearReader`
- **Area**: `diary_slm/bear_reader.py`
- **Problem**: `get_notes_by_tag()` and `get_notes_by_date_range()` loaded all notes, then filtered in Python.
- **Change**:
- Added `_query_notes()` so date and coarse tag filters run in SQLite.
- Added `_datetime_to_core_data()` to push date filtering to `ZCREATIONDATE`.
- Kept correctness by validating tag matches from parsed tags after SQL pre-filter.
- Added direct `MIN/MAX` query in `get_date_range()` for non-tag lookups.
- **Impact**: Less memory usage and faster startup for large note sets.

### 2. Reduced repeated chunk computation in `DiaryProcessor`
- **Area**: `diary_slm/processor.py`
- **Problem**: `get_chunks_by_period()` rebuilt formatted chunks each call; `get_chunk_by_name()` often triggered duplicate work.
- **Change**:
- Added per-period chunk cache and period-name index cache.
- `get_chunk_by_name()` now resolves from cached index.
- Removed per-period date list allocations by using first/last note in already-sorted groups.
- **Impact**: Repeated operations (list + analyze + compare) are significantly cheaper.

### 3. Lower allocation overhead for total stats
- **Area**: `diary_slm/processor.py`
- **Problem**: `get_total_stats()` built one large concatenated string only to estimate tokens.
- **Change**:
- Replaced concatenation with `total_chars // CHARS_PER_TOKEN`.
- Cached computed total stats for reuse.
- **Impact**: Avoids large temporary strings and repeated recomputation.

### 4. Maintainability and dead code cleanup
- **Area**: `diary_slm/analyzer.py`, `diary_slm/cli.py`, `diary_slm/exceptions.py`
- **Problem**:
- Unused helper (`get_template_names`) and duplicated literals (`4096`, `128_000`) increased drift risk.
- CLI had repeated `data/` directory setup and broad exception handling in token counting.
- **Change**:
- Removed unused `get_template_names()` from analyzer.
- Replaced hardcoded CLI defaults/limits with constants and added typed helper signatures.
- Introduced `_get_data_dir()` to centralize output path creation.
- Narrowed tokenization error handling from `except Exception` to `ModelLoadError`.
- Wired `MAX_PERIODS_TO_SUGGEST` into `PeriodNotFoundError` preview logic.
- **Impact**: Less dead code, safer error handling, and fewer duplicated values to maintain.

## Remaining High-Value Opportunities
- Add targeted tests/benchmarks for large datasets (`10k+` notes) to quantify gains.
- Optimize keyword stats path in CLI to avoid repeated lowercase scans per keyword.
- Add optional persistent cache for expensive tokenization in `tokens` command.
