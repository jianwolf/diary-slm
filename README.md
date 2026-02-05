# diary-slm

Analyze your diary entries using local LLMs on Apple Silicon. No cloud, no RAG, just pure context.

## The Idea

Your diary (say, 500k characters/year) is roughly 125k tokens (at ~4 chars/token). Modern models like Qwen 2.5 and Llama 3.1 support 128k context windows. Instead of using RAG (which loses context), we split your diary by quarter or half-year and feed it directly into the prompt.

**The result**: The AI has a "god's eye view" of your life during that period. It can see patterns, causality, and connections that retrieval-based approaches would miss.

## Features

- **Local-first**: Runs entirely on your Mac using [mlx-lm](https://github.com/ml-explore/mlx-lm)
- **Read-only**: Never modifies your Bear database
- **Full context**: No RAG - the LLM sees your complete diary for the period
- **Pre-built analysis templates**: Mood, themes, growth, relationships, advice, and more
- **Interactive mode**: Multi-turn conversations about your diary
- **Period comparison**: Compare how you've changed across time periods

## Requirements

- macOS 15.0+ (Sequoia or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- [Bear](https://bear.app/) note-taking app with diary entries
- Python 3.10+

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/diary-slm.git
cd diary-slm
pip install -e .
```

This will install `mlx-lm`, `click`, and `rich` as dependencies.

## Quick Start

### 1. List your diary periods

```bash
# See all available periods and token counts
python main.py list

# Filter by a specific tag (e.g., your diary tag)
python main.py list -t diary
```

Output shows periods, note counts, and estimated tokens:

```
Diary Statistics
Total notes: 365
Total characters: 500,000
Estimated tokens: 125,000
Date range: 2024-01-01 to 2024-12-31
Suggested period type: quarter

Available Periods (quarter)
┏━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Period   ┃ Notes ┃ Est. Tokens ┃ Fits 128k?┃
┡━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 2024-Q1  │    90 │      31,250 │    Yes    │
│ 2024-Q2  │    91 │      32,000 │    Yes    │
│ 2024-Q3  │    92 │      30,500 │    Yes    │
│ 2024-Q4  │    92 │      31,250 │    Yes    │
└──────────┴───────┴─────────────┴───────────┘
```

### 2. Analyze a period

```bash
# Custom query
python main.py analyze 2024-Q1 -q "What were my main struggles this quarter?" -t diary

# Use a pre-built template
python main.py template 2024-Q1 mood -t diary
python main.py template 2024-Q1 themes -t diary
python main.py template 2024-Q1 advice -t diary
```

### 3. Compare periods

```bash
python main.py compare 2024-Q1 2024-Q2 -q "How did my priorities change?" -t diary
```

### 4. Interactive session

```bash
python main.py interactive 2024-Q1 -t diary
```

Then ask multiple questions:

```
Interactive Analysis: 2024-Q1
Notes: 90, Tokens: ~31,250

Type your questions. Commands: /templates, /quit

You: What patterns do you see in my mood?
Assistant: [analysis...]

You: /template growth
Assistant: [growth analysis...]

You: What should I focus on next quarter?
Assistant: [advice...]
```

## Analysis Templates

| Template | Description |
|----------|-------------|
| `summary` | Comprehensive overview: themes, events, emotions, relationships |
| `mood` | Emotional patterns, triggers, fluctuations over time |
| `themes` | Recurring topics, concerns, and how they evolve |
| `growth` | Personal development, lessons learned, progress |
| `relationships` | Social dynamics, key people, how relationships changed |
| `advice` | Personalized advice based on patterns in your diary |
| `timeline` | Chronological timeline of significant events |
| `questions` | Reflection questions to deepen self-understanding |

## Models

Default model: `mlx-community/Qwen2.5-7B-Instruct-4bit`

Available presets:

```bash
python main.py models
```

| Alias | Model |
|-------|-------|
| `qwen-7b` | mlx-community/Qwen2.5-7B-Instruct-4bit |
| `qwen-14b` | mlx-community/Qwen2.5-14B-Instruct-4bit |
| `llama-8b` | mlx-community/Meta-Llama-3.1-8B-Instruct-4bit |
| `gemma-9b` | mlx-community/gemma-2-9b-it-4bit |
| `glm-4.7-flash` | mlx-community/GLM-4.7-Flash-4bit (~17GB, 30B params, 128K context) |

Use any mlx-community model:

```bash
python main.py analyze 2024-Q1 -q "..." --model mlx-community/Qwen2.5-14B-Instruct-4bit
```

## CLI Reference

```bash
python main.py --help              # Show all commands
python main.py list --help         # List periods
python main.py analyze --help      # Analyze with custom query
python main.py template --help     # Use analysis template
python main.py compare --help      # Compare two periods
python main.py interactive --help  # Interactive session
python main.py models              # List model presets
python main.py templates           # List analysis templates
python main.py tags                # List tags in Bear
python main.py tokens              # Count tokens (estimated + exact)
```

### Common Options

| Option | Description |
|--------|-------------|
| `--db PATH` | Custom Bear database path (auto-detected by default) |
| `-t, --tag TAG` | Filter notes by tag (e.g., `-t diary`) |
| `-p, --period-type` | Period granularity: `month`, `quarter`, `half_year`, `year` |
| `-m, --model MODEL` | Model to use (HuggingFace ID or preset alias) |
| `--max-tokens N` | Maximum tokens to generate (default: 4096) |

## Privacy & Security

- **100% local**: All processing happens on your Mac. No data is sent anywhere.
- **Read-only**: The Bear database is opened in read-only mode. We never write to it.
- **No persistent storage**: Analysis results are only shown in the terminal.

## How It Works

1. **Read**: Extract notes from Bear's SQLite database (read-only)
2. **Chunk**: Group notes by time period (quarter recommended)
3. **Format**: Convert notes to a chronological text format with date headers
4. **Analyze**: Send the full context + your query to the local LLM
5. **Stream**: Display the response in real-time

## Token Budget

| Period Type | Typical Tokens | Fits 128k? |
|-------------|----------------|------------|
| Month | ~10k | Yes |
| Quarter | ~32k | Yes |
| Half Year | ~64k | Yes |
| Year | ~125k | Tight fit |

The tool automatically suggests the best period type based on your diary volume.

## Troubleshooting

### "Bear database not found"

Make sure Bear is installed and you've created at least one note. The database is at:
```
~/Library/Group Containers/9K33E3U3T4.net.shinyfrog.bear/Application Data/database.sqlite
```

### "No notes found with tag #diary"

Check your tag name. Use `python main.py tags` to see all available tags.

### Model download is slow

First run downloads the model (~4GB for 7B 4-bit). Subsequent runs use the cached model.

### Out of memory

Try a smaller model or shorter time period:
```bash
python main.py analyze 2024-01 -q "..." -p month --model mlx-community/Qwen2.5-3B-Instruct-4bit
```

## Contributing

Contributions welcome! Please read CLAUDE.md for architecture details.

## License

MIT
