# EchoChamber ⚖️

**A multi-LLM courtroom debate environment.** Give it a topic and a position, and three LLM agents — Prosecution, Defense, and a Moderator/Judge — argue it out over structured rounds until a verdict is reached. Each agent can run on a different provider, so you can pit Claude against Llama with DeepSeek on the bench.

> This was my first LLM project, shared as-is as a learning artifact. It works, and the architecture notes below describe both what it does and what I'd do differently today.

## What it does

```
Moderator opens the session
└── Round 1..N
    ├── Prosecution argues
    ├── Defense rebuts
    └── Moderator evaluates (CONTINUE? WINNER?)
Moderator issues a final ruling → verdict + transcript
```

- **Mixed providers per role** — Anthropic, OpenAI, Together.AI, Google Gemini, or free local models via LM Studio.
- **Evidence folders** — drop `.txt/.md/.json/.csv/.pdf` files into a case folder with `shared/`, `prosecution/`, `defense/`, and `moderator/` subfolders; each agent only sees what its role is entitled to.
- **Context strategies** — evidence is injected whole (`full`), condensed (`summarize`), or retrieved on demand via ChromaDB embeddings (`rag`), auto-selected by size.
- **Web search** — agents emit `[SEARCH: query]` tags that are executed via DuckDuckGo and fed back; the moderator is required to verify claims before ruling.
- **Batch runner** — run a matrix of provider/model combinations in parallel from a YAML config, with per-run cost estimation up front.
- **Transcripts** — every session is saved as markdown and optionally JSON. See [examples/](examples/) for sample outputs, including the all-important *pineapple on pizza* proceedings.

## Quickstart

```bash
git clone https://github.com/<you>/echochamber && cd echochamber
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env   # add keys for the providers you'll use

# Free, fully local (requires LM Studio running on localhost:1234)
python -m echochamber.cli \
  --topic "Tabs vs Spaces" \
  --position "Tabs are superior to spaces"

# Cloud providers
python -m echochamber.cli \
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
python -m echochamber.cli --init-case ./cases/my_case --topic "My Topic"

# Run against the bundled example case
python -m echochamber.cli \
  --topic "Python vs Rust for High-Performance Data Processing" \
  --position "Rust should be chosen for the new system" \
  --case-folder cases/example_case
```

Each role gets its own private evidence plus everything in `shared/`. A `moderator/` folder can hold evaluation criteria the judge alone sees. Large evidence sets are automatically summarized or chunked into a vector store (`--context-strategy full|summarize|rag`).

> **Note:** `cases/` is gitignored except for the bundled example — real case material stays out of version control by default.

### Batch runs

```bash
python -m echochamber.batch --config runs_example.yaml
```

Runs multiple provider/model matchups in parallel and prints estimated API costs per run before starting.

## Architecture

```
echochamber/
├── cli.py               # Argument parsing, agent wiring, session kickoff
├── batch.py             # Parallel multi-run orchestration + cost estimates
├── core/
│   ├── session.py       # CourtSession: round loop, verdict parsing, termination
│   ├── agent.py         # Agent = name + role + provider + system prompt
│   ├── evidence.py      # Case folder loading (txt/md/json/csv/pdf)
│   ├── preprocessor.py  # full / summarize / RAG context strategies
│   ├── transcript.py    # Markdown + JSON transcript writing
│   └── costs.py         # Per-model pricing tables and run cost estimates
├── providers/           # Thin adapters: Anthropic, OpenAI, Together, Gemini, LM Studio
├── roles/prompts.py     # Prosecution / Defense / Moderator system prompts
└── tools/search.py      # DuckDuckGo search + [SEARCH:] tag extraction
```

Design choices worth noting:

- **Zero hard dependencies.** Every third-party library (provider SDKs, `pymupdf`, `chromadb`, `ddgs`, `yaml`, `dotenv`) is lazily imported, so the core installs clean and you only pull in what your chosen providers/features need.
- **Providers are ~70-line adapters** over a common `Message` interface, which made adding each new provider trivial.
- **Structured-text protocol.** Control flow (verdicts, concessions, searches) is parsed from model output with sentinel phrases like `FINAL VERDICT: PROSECUTION WINS` and `[SEARCH: ...]`.

### What I'd do differently today

Honest retrospective, in rough priority order:

1. **Native tool use instead of sentinel parsing.** `[SEARCH: ...]` tags and `CONTINUE: YES/NO` regex parsing predate my understanding of structured outputs / tool calling. Modern provider APIs make the search tool and the verdict schema first-class, which would eliminate the brittleest code in the project.
2. **Async concurrency.** Agents take turns via blocking calls; `batch.py` parallelizes by spawning subprocesses. `asyncio` + provider async clients would be simpler and cheaper.
3. **Tests.** `pytest` is declared but there are no tests; the session loop and verdict parsing are pure functions begging for them.
4. **Centralized retry/backoff** instead of per-callsite try/except loops.
5. **Pricing tables drift.** `costs.py` hardcodes early-2025 prices; a live pricing source (or at least a `pricing.json` a human can update) would age better.

## License

[MIT](LICENSE)
