# EchoChamber ⚖️

**A multi-LLM courtroom debate environment.** Give it a topic and a position, and three LLM agents — Prosecution, Defense, and a Moderator/Judge — argue it out over structured rounds until a verdict is reached. Each agent can run on a different provider, so you can pit Claude against Llama with DeepSeek on the bench.

> This started as my first LLM project and is shared as a learning artifact. The architecture notes below include an honest retrospective of what changed as I learned.

## What it does

```
Moderator opens the session
└── Round 1..N
    ├── Prosecution argues (may call web_search)
    ├── Defense rebuts (may call web_search)
    └── Moderator evaluates → submit_evaluation(continue?, winner?)
Moderator searches, then rules → submit_verdict(winner, reasoning)
```

- **Mixed providers per role** — Anthropic, OpenAI, Together.AI, Google Gemini, or free local models via LM Studio.
- **Native tool calling with graceful degradation** — agents on tool-capable providers get a real `web_search` tool and the moderator returns structured verdicts through `submit_evaluation`/`submit_verdict` tools; providers without tool support (or local models that reject tools) automatically fall back to a `[SEARCH: query]` / `FINAL VERDICT:` text protocol.
- **Evidence folders** — drop `.txt/.md/.json/.csv/.pdf` files into a case folder with `shared/`, `prosecution/`, `defense/`, and `moderator/` subfolders; each agent only sees what its role is entitled to.
- **Context strategies** — evidence is injected whole (`full`), condensed (`summarize`), or retrieved on demand via ChromaDB embeddings (`rag`), auto-selected by size.
- **Cost metering** — every provider call is recorded (tokens, latency, cost from a data-driven pricing table); runs print a per-role cost summary and can export JSONL usage events.
- **Batch runner** — run a matrix of provider/model combinations on a thread pool from a YAML config, with cost estimates up front and actual costs in the report.
- **Transcripts** — every session is saved as markdown and optionally JSON. See [examples/](examples/) for sample outputs, including the all-important *pineapple on pizza* proceedings.

## Quickstart

Uses [uv](https://docs.astral.sh/uv/) (`pip install -e ".[all]"` in a venv works too):

```bash
git clone https://github.com/<you>/echochamber && cd echochamber
uv sync --extra all
cp .env.example .env   # add keys for the providers you'll use

# Free, fully local (requires LM Studio running on localhost:1234)
uv run echochamber \
  --topic "Tabs vs Spaces" \
  --position "Tabs are superior to spaces"

# Cloud providers
uv run echochamber \
  --topic "Should AI systems be open source?" \
  --position "AI systems should be open source for safety and transparency" \
  --prosecution-provider anthropic \
  --defense-provider together \
  --moderator-provider anthropic
```

The default provider is `lmstudio`, so everything works with zero API keys if you have [LM Studio](https://lmstudio.ai) serving a local model.

### Debating over evidence

```bash
# Scaffold a case folder
uv run echochamber --init-case ./cases/my_case --topic "My Topic"

# Run against the bundled example case
uv run echochamber \
  --topic "Python vs Rust for High-Performance Data Processing" \
  --position "Rust should be chosen for the new system" \
  --case-folder cases/example_case
```

Each role gets its own private evidence plus everything in `shared/`. A `moderator/` folder can hold evaluation criteria the judge alone sees. Large evidence sets are automatically summarized or chunked into a vector store (`--context-strategy full|summarize|rag`).

> **Note:** `cases/` is gitignored except for the bundled example — real case material stays out of version control by default.

### Batch runs

```bash
uv run python -m echochamber.batch --config runs_example.yaml --parallel 3
```

Runs variations in-process on a thread pool, prints estimated cost per run before starting (with an approval gate above a threshold), and reports winner, rounds, and **actual** metered cost per run.

### Cost tracking

Every run ends with a usage summary (calls, tokens, dollars per role). Pass `--usage-log usage.jsonl` to append one normalized event per provider call:

```json
{"type": "usage", "at": "...", "host": "echochamber", "module": "debate.prosecution",
 "model": "gpt-4o", "input": 1204, "output": 512, "costUsd": 0.00813, "latencyMs": 2140, "note": "openai/gpt-4o"}
```

The event shape follows agent-stable's normalized usage-event schema (a companion cost/performance metering package), so the log can be ingested by its sinks/dashboard directly. Prices live in [echochamber/data/pricing.json](echochamber/data/pricing.json) — a data file, not code.

## Architecture

```
echochamber/
├── cli.py                  # Thin argparse layer over the runner
├── batch.py                # Thread-pool multi-run orchestration + cost gates
├── core/
│   ├── runner.py           # DebateSpec + run_debate(): the one entry point
│   ├── session.py          # CourtSession: round loop, structured verdicts + text fallback
│   ├── turns.py            # Agent turn loop: native tool transport or sentinel fallback
│   ├── agent.py            # Agent = name + role + provider + prompt (+ meter)
│   ├── usage.py            # UsageMeter → cost summaries + agent-stable JSONL events
│   ├── evidence.py         # Case folder loading (txt/md/json/csv/pdf)
│   ├── preprocessor.py     # full / summarize / RAG context strategies
│   ├── transcript.py       # Markdown + JSON transcript writing
│   └── costs.py            # Estimates, priced from data/pricing.json
├── data/pricing.json       # Per-model $/1M token table (edit freely)
├── providers/
│   ├── base.py             # Message / ToolDef / ToolCall / LLMResponse protocol
│   ├── retry.py            # Centralized exponential backoff
│   ├── openai_compat.py    # One implementation for OpenAI, Together, LM Studio
│   ├── anthropic.py        # Native tools via tool_use/tool_result blocks
│   └── gemini.py           # Text-only (sentinel fallback) on the legacy SDK
├── roles/prompts.py        # Prosecution / Defense / Moderator system prompts
└── tools/search.py         # DuckDuckGo search + web_search ToolDef + sentinel parsing
tests/                      # Fake-provider suite for the orchestration logic
```

Design choices worth noting:

- **Zero hard dependencies.** Every third-party library (provider SDKs, `pymupdf`, `chromadb`, `ddgs`, `yaml`, `dotenv`) is lazily imported, so the core installs clean and you only pull in what your chosen providers/features need.
- **Tools first, sentinels as fallback.** Control flow (searches, round evaluations, verdicts) uses native tool calling wherever the provider supports it, with the original text protocol kept as a per-call fallback — so a local model that rejects `tools` degrades mid-debate without crashing the session.
- **One OpenAI-compatible adapter.** OpenAI, Together.AI, and LM Studio share a single implementation; a new OpenAI-compatible provider is ~25 lines of client construction.
- **Threads, not asyncio.** Debate turns are inherently sequential; parallelism only exists *across* runs, and those are I/O-bound API calls — a thread pool gets the full benefit without forking the provider layer into sync/async variants.

### Retrospective

What this project taught me, now folded back in:

- ~~Sentinel parsing~~ → native tool use with structured verdicts (the original `[SEARCH:]` implementation never actually fed results back to advocates — the tool loop fixed a real bug, not just style).
- ~~Subprocess batch parallelism~~ → in-process thread pool sharing one metered runner.
- ~~No tests~~ → a fake-provider suite covering the session loop, tool transports, budgets, verdict parsing, metering, and config loading.
- ~~Hardcoded pricing in code~~ → `data/pricing.json`.
- ~~Per-callsite try/except~~ → centralized retry with backoff.

Still open: Gemini native tools (needs the `google-genai` SDK migration), streaming output, and a live pricing source instead of a static table.

## License

[MIT](LICENSE)
