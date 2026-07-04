#!/usr/bin/env python3
"""
EchoChamber GUI - watch a debate live in the browser.

    uv run python -m echochamber.ui          # http://localhost:7860

Requires the ui extra:  uv sync --extra ui

Two tabs:
- Debate: run and watch debates (live agent/model status, token ticker,
  stop button, APA model recommendations).
- Setup: paste provider API keys, test them, and save to .env — no
  terminal needed.

Closing the tab while a debate runs triggers the browser's native leave
warning; whether the debate then aborts or continues in the background is a
user preference on the Debate tab (browsers do not allow custom dialogs on
tab close). Backgrounded debates appear in the "Background runs" panel and
still save their transcripts.

Set $ECHOCHAMBER_APA_DATA to an APA data directory (or merged export) to see
live procurement recommendations instead of the bundled sample.
"""

import os
import queue
import stat
import threading
from pathlib import Path

# The module stays importable without gradio (key helpers are GUI-independent
# and the test suite imports them); only building/launching the app needs it.
try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

_GRADIO_MISSING = "The GUI requires gradio. Install with: uv sync --extra ui"

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .core.runner import DebateSpec, run_debate
from .core.usage import UsageMeter
from .recommendations import load_apa_data, recommend_for_debate

PROVIDERS = ["lmstudio", "anthropic", "openai", "together", "gemini"]
DEFAULT_PROVIDER = os.environ.get("ECHOCHAMBER_DEFAULT_PROVIDER", "lmstudio")
ROLE_EMOJI = {"prosecution": "⚖️", "defense": "🛡️", "moderator": "👨‍⚖️", "system": "⚙️"}

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

ENV_FILE = Path.cwd() / ".env"

# The foreground run's stop event (single-user local app) and the run registry
# for the background panel.
_active_stop: dict = {"event": None}
_runs: list[dict] = []

# Warn (native browser dialog) when leaving while a debate is running.
_HEAD_JS = """
<script>
window.__ec_running = false;
window.addEventListener('beforeunload', (e) => {
  if (window.__ec_running) { e.preventDefault(); e.returnValue = ''; }
});
</script>
"""

# APA recommendations, loaded once at startup
try:
    _apa = load_apa_data()
    _recs = recommend_for_debate(_apa)
except Exception:
    _apa, _recs = None, {}


# ---------------------------------------------------------------- APA panel

def _recs_markdown() -> str:
    if not _recs or not any(_recs.values()):
        return "_No APA data available._"
    lines = [f"_Source: {_apa.source}_\n"]
    for role, items in _recs.items():
        lines.append(f"**{ROLE_EMOJI[role]} {role.capitalize()}**")
        for r in items:
            price = f"${r.price_in:g} / ${r.price_out:g} per 1M" if r.price_in is not None else "price unknown"
            lines.append(f"{r.rank}. `{r.model}` ({r.provider}) — {price}" +
                         (f" · bar: {r.benchmark}" if r.benchmark else ""))
            lines.append(f"   ↳ {r.justification}")
        lines.append("")
    return "\n".join(lines)


def _apply_recs(rank: int):
    """Provider/model values for all three roles from APA picks at the given rank."""
    out = []
    for role in ("prosecution", "defense", "moderator"):
        items = _recs.get(role) or []
        pick = next((r for r in items if r.rank == rank), items[-1] if items else None)
        if pick:
            out.extend([pick.provider, pick.model])
        else:
            out.extend([gr.skip(), gr.skip()])
    return out


# ------------------------------------------------------------- key handling

def _mask(value: str) -> str:
    if not value:
        return "not set"
    if len(value) <= 12:
        return "set (short key)"
    return f"set ({value[:7]}…{value[-4:]})"


def key_overview_md() -> str:
    lines = ["| Provider | Env var | Status |", "|---|---|---|"]
    for provider, env_key in PROVIDER_ENV_KEYS.items():
        lines.append(f"| {provider} | `{env_key}` | {_mask(os.environ.get(env_key, ''))} |")
    lines.append("| lmstudio | _(none needed)_ | local server on :1234 |")
    lines.append(
        "\nKeys are saved to `.env` in the folder EchoChamber was launched from "
        "(owner-read-only) and loaded on launch."
    )
    return "\n".join(lines)


def test_provider_key(provider: str, key: str) -> str:
    """Cheap auth check per provider (list-models style calls are free)."""
    key = (key or "").strip() or os.environ.get(PROVIDER_ENV_KEYS.get(provider, ""), "")
    try:
        if provider == "lmstudio":
            from .providers.lmstudio import get_lmstudio_model
            model = get_lmstudio_model()
            return f"✅ lmstudio: reachable, serving `{model}`" if model else \
                "❌ lmstudio: no server on localhost:1234 (start LM Studio and load a model)"
        if provider == "gemini":
            # Key OR Vertex env config both count as configured
            from .providers.gemini import build_client
            client, mode = build_client(key or None)
            next(iter(client.models.list()))
            return f"✅ gemini: works ({mode})"
        if not key:
            return f"❌ {provider}: no key entered or saved"
        if provider == "anthropic":
            import anthropic
            anthropic.Anthropic(api_key=key).models.list(limit=1)
        elif provider == "openai":
            from openai import OpenAI
            OpenAI(api_key=key).models.list()
        elif provider == "together":
            # Together's /models returns a bare array the OpenAI SDK can't
            # parse; a raw authenticated GET is the reliable check.
            import urllib.request
            req = urllib.request.Request(
                "https://api.together.xyz/v1/models",
                headers={
                    "Authorization": f"Bearer {key}",
                    # Together's edge rejects the default Python-urllib UA
                    "User-Agent": "echochamber-setup-check",
                },
            )
            urllib.request.urlopen(req, timeout=10)
        return f"✅ {provider}: key works"
    except ImportError as e:
        return f"❌ {provider}: SDK not installed ({e.name}) — run: uv sync --extra all"
    except Exception as e:
        return f"❌ {provider}: {str(e)[:200]}"


def save_keys(anthropic_key: str, openai_key: str, together_key: str, gemini_key: str):
    """Merge non-empty keys into .env (created owner-read-only) and the live env."""
    new_values = {
        "ANTHROPIC_API_KEY": anthropic_key.strip(),
        "OPENAI_API_KEY": openai_key.strip(),
        "TOGETHER_API_KEY": together_key.strip(),
        "GEMINI_API_KEY": gemini_key.strip(),
    }
    new_values = {k: v for k, v in new_values.items() if v}
    if not new_values:
        return key_overview_md(), "Nothing to save — all fields empty."

    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    for env_key, value in new_values.items():
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{env_key}="):
                lines[i] = f"{env_key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{env_key}={value}")
        os.environ[env_key] = value  # effective immediately, no restart

    ENV_FILE.write_text("\n".join(lines) + "\n")
    ENV_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    saved = ", ".join(new_values)
    return key_overview_md(), f"✅ Saved {saved} to {ENV_FILE} (permissions 600). Empty fields were left unchanged."


# ------------------------------------------------------------ debate running

def _fmt_tokens(meter: UsageMeter, budget: int | None) -> str:
    tin, tout = meter.total_tokens()
    total = tin + tout
    budget_str = f" / {budget:,} budget" if budget else ""
    pct = f" ({100 * total / budget:.0f}%)" if budget else ""
    return (f"**Tokens burned:** {total:,}{budget_str}{pct} · "
            f"{tin:,} in / {tout:,} out · **Cost:** ${meter.total_cost():.4f}")


def stop_debate():
    event = _active_stop["event"]
    if event:
        event.set()
        return "### ⏹️ Stopping after the current turn…"
    return "### 💤 No debate running."


def background_md() -> str:
    detached = [r for r in _runs if r["detached"]][-5:]
    if not detached:
        return "_No background runs._"
    lines = ["**Background runs** (tab was closed; debates kept going)\n"]
    for r in detached:
        tin, tout = r["meter"].total_tokens()
        lines.append(f"- *{r['topic']}* — {r['status']} · {tin + tout:,} tokens, "
                     f"${r['meter'].total_cost():.4f}" +
                     (f" · `{r['transcript']}`" if r["transcript"] else ""))
    return "\n".join(lines)


def inspect_evidence(path: str) -> str:
    """Preview what a case folder would load for each role."""
    path = (path or "").strip()
    if not path:
        return "_No case folder set — the debate runs on the topic alone._"
    try:
        from .core.evidence import EvidenceStore
        store = EvidenceStore.load(path)
        return f"```\n{store.summary()}\n```"
    except Exception as e:
        return f"❌ Could not load `{path}`: {e}"


def _text(value) -> str:
    """Normalize an optional Textbox value — untouched fields arrive as None."""
    return (value or "").strip()


def run_debate_ui(topic, position, pros_provider, pros_model, pros_instr,
                  def_provider, def_model, def_instr,
                  mod_provider, mod_model, mod_instr,
                  case_folder, context_strategy,
                  rounds, enable_search, token_budget, on_close):
    """Generator: streams (status, tokens, chat, verdict) updates while the debate runs."""
    chat: list[dict] = []

    if not topic or not position:
        yield "### ⚠️ Topic and position are required.", "", chat, ""
        return

    budget = int(token_budget) if token_budget else None
    meter = UsageMeter(hard_limit_tokens=budget)
    events: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    _active_stop["event"] = stop_event
    record = {"topic": topic, "meter": meter, "status": "running",
              "transcript": "", "detached": False}
    _runs.append(record)

    spec = DebateSpec(
        topic=topic,
        position=position,
        prosecution_provider=pros_provider,
        prosecution_model=_text(pros_model) or None,
        prosecution_instructions=_text(pros_instr),
        defense_provider=def_provider,
        defense_model=_text(def_model) or None,
        defense_instructions=_text(def_instr),
        moderator_provider=mod_provider,
        moderator_model=_text(mod_model) or None,
        moderator_instructions=_text(mod_instr),
        case_folder=_text(case_folder) or None,
        context_strategy=context_strategy or "auto",
        max_rounds=int(rounds),
        enable_search=bool(enable_search),
        max_total_tokens=budget,
        verbose=False,
    )

    def on_status(stage, agent):
        events.put(("status", (stage, agent.name, agent.role.value, agent.provider.name)))

    def on_turn(speaker, role, content):
        events.put(("turn", (speaker, role, content)))

    def worker():
        try:
            outcome = run_debate(
                spec, meter=meter,
                on_turn=on_turn, on_status=on_status,
                should_stop=stop_event.is_set,
            )
            record["status"] = (
                f"finished — {(outcome.result.winner or 'undecided').upper()} "
                f"({outcome.result.termination_reason.value})"
            )
            record["transcript"] = str(outcome.transcript_path or "")
            events.put(("done", outcome))
        except Exception as e:
            record["status"] = f"error: {str(e)[:120]}"
            events.put(("error", str(e)))

    threading.Thread(target=worker, daemon=True).start()

    status = "### 🏁 Starting debate…"
    try:
        yield status, _fmt_tokens(meter, budget), chat, ""

        while True:
            try:
                kind, payload = events.get(timeout=0.5)
            except queue.Empty:
                # No new event — refresh the live token/cost ticker
                yield status, _fmt_tokens(meter, budget), gr.skip(), gr.skip()
                continue

            if kind == "status":
                stage, name, role, provider_name = payload
                status = (f"### {ROLE_EMOJI.get(role, '💬')} Now running: **{name}** "
                          f"({stage})\n`{provider_name}`")
                yield status, _fmt_tokens(meter, budget), gr.skip(), gr.skip()

            elif kind == "turn":
                speaker, role, content = payload
                chat = chat + [{
                    "role": "assistant",
                    "content": f"**{ROLE_EMOJI.get(role, '💬')} {speaker} ({role})**\n\n{content}",
                }]
                yield status, _fmt_tokens(meter, budget), chat, gr.skip()

            elif kind == "error":
                yield f"### ❌ Error: {payload}", _fmt_tokens(meter, budget), chat, ""
                return

            elif kind == "done":
                outcome = payload
                result = outcome.result
                winner = (result.winner or "undecided").upper()
                verdict = (
                    f"## 🏛️ Verdict: **{winner}**\n"
                    f"- Termination: `{result.termination_reason.value}` "
                    f"after {result.rounds_completed} round(s)\n"
                    f"- Transcript: `{outcome.transcript_path}`\n\n"
                    f"```\n{outcome.usage.summary()}\n```"
                )
                yield "### ✅ Debate complete", _fmt_tokens(meter, budget), chat, verdict
                return
    except GeneratorExit:
        # Browser disconnected mid-run: honor the on-close preference.
        if record["status"] == "running":
            if str(on_close).startswith("Abort"):
                stop_event.set()
                record["status"] = "aborting (tab closed)"
            else:
                record["detached"] = True
        return
    finally:
        if _active_stop["event"] is stop_event:
            _active_stop["event"] = None


# ------------------------------------------------------------------- layout

def _debate_tab():
    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Textbox(label="Topic", value="Tabs vs Spaces")
            position = gr.Textbox(
                label="Prosecution position",
                value="Tabs are superior to spaces",
            )

            with gr.Accordion("📁 Case evidence (optional)", open=False):
                gr.Markdown(
                    "Link a local case folder with `shared/` (visible to all sides) "
                    "plus proprietary `prosecution/`, `defense/`, and `moderator/` "
                    "subfolders (`.txt/.md/.json/.csv/.pdf`). Scaffold one with "
                    "`echochamber --init-case ./cases/my_case`."
                )
                case_folder = gr.Textbox(
                    label="Case folder path", placeholder="cases/example_case",
                )
                with gr.Row():
                    context_strategy = gr.Dropdown(
                        ["auto", "full", "summarize", "rag"], value="auto",
                        label="Context strategy", scale=1,
                    )
                    inspect_btn = gr.Button("🔍 Inspect evidence", size="sm", scale=1)
                evidence_info = gr.Markdown("")
                inspect_btn.click(inspect_evidence, inputs=[case_folder], outputs=[evidence_info])

            role_inputs = {}
            for role in ("prosecution", "defense", "moderator"):
                with gr.Group():
                    gr.Markdown(f"**{ROLE_EMOJI[role]} {role.capitalize()}**")
                    with gr.Row():
                        provider = gr.Dropdown(
                            PROVIDERS, value=DEFAULT_PROVIDER, label="Provider", scale=1,
                        )
                        model = gr.Textbox(
                            label="Model (blank = provider default)", scale=2,
                        )
                    instructions = gr.Textbox(
                        label="Custom system prompt additions (optional)",
                        placeholder="Appended to this agent's system prompt",
                        lines=2,
                    )
                role_inputs[role] = (provider, model, instructions)

            with gr.Accordion("🤖 APA model recommendations", open=False):
                gr.Markdown(_recs_markdown())
                # Apply buttons set provider + model only (not instructions)
                rec_outputs = [w for triple in role_inputs.values() for w in triple[:2]]
                with gr.Row():
                    gr.Button("Apply APA winners", size="sm").click(
                        lambda: _apply_recs(1), outputs=rec_outputs
                    )
                    gr.Button("Apply APA fallbacks", size="sm").click(
                        lambda: _apply_recs(2), outputs=rec_outputs
                    )

            rounds = gr.Slider(1, 8, value=2, step=1, label="Max rounds")
            enable_search = gr.Checkbox(label="Enable web search", value=False)
            token_budget = gr.Number(
                label="Hard token budget (total across all agents)",
                value=200_000, precision=0,
            )
            on_close = gr.Radio(
                ["Abort the debate", "Continue in background"],
                value="Abort the debate",
                label="If I close this tab mid-debate",
            )

            with gr.Row():
                run_btn = gr.Button("▶️ Run debate", variant="primary")
                stop_btn = gr.Button("⏹️ Stop")

            background = gr.Markdown(background_md())
            gr.Timer(5).tick(background_md, outputs=[background])

        with gr.Column(scale=2):
            status = gr.Markdown("### 💤 Idle")
            tokens = gr.Markdown("")
            chatbot = gr.Chatbot(label="Proceedings", height=520)
            verdict = gr.Markdown("")

    inputs = [topic, position]
    for role in ("prosecution", "defense", "moderator"):
        inputs.extend(role_inputs[role])
    inputs.extend([case_folder, context_strategy, rounds, enable_search, token_budget, on_close])

    # The running-flag is toggled by dedicated js-only listeners so the main
    # event's input payload is never transformed client-side.
    run_btn.click(None, js="() => { window.__ec_running = true; }")
    run_event = run_btn.click(
        run_debate_ui,
        inputs=inputs,
        outputs=[status, tokens, chatbot, verdict],
    )
    run_event.then(None, js="() => { window.__ec_running = false; }")
    stop_btn.click(stop_debate, outputs=[status])


def _setup_tab():
    gr.Markdown(
        "### 🔑 Provider API keys\n"
        "Paste a key, **Test** it, then **Save**. Keys go into a local `.env` file "
        "readable only by your user account, and take effect immediately. "
        "Only fill the providers you plan to use — LM Studio (local) needs no key.\n\n"
        "Tip: create keys with a low monthly spend limit in each provider's console."
    )
    overview = gr.Markdown(key_overview_md())

    boxes = {}
    for provider in ("anthropic", "openai", "together", "gemini"):
        with gr.Row():
            box = gr.Textbox(
                label=f"{provider} — {PROVIDER_ENV_KEYS[provider]}",
                type="password", scale=3,
            )
            test_btn = gr.Button(f"Test", size="sm", scale=1)
        result = gr.Markdown("")
        test_btn.click(
            lambda key, p=provider: test_provider_key(p, key),
            inputs=[box], outputs=[result],
        )
        boxes[provider] = box

    with gr.Row():
        lms_btn = gr.Button("Test LM Studio (local, no key)", size="sm")
        lms_result = gr.Markdown("")
        lms_btn.click(lambda: test_provider_key("lmstudio", ""), outputs=[lms_result])

    save_btn = gr.Button("💾 Save keys", variant="primary")
    save_status = gr.Markdown("")
    save_btn.click(
        save_keys,
        inputs=[boxes["anthropic"], boxes["openai"], boxes["together"], boxes["gemini"]],
        outputs=[overview, save_status],
    )


def build_app() -> "gr.Blocks":
    if gr is None:  # pragma: no cover
        raise SystemExit(_GRADIO_MISSING)
    with gr.Blocks(title="EchoChamber", head=_HEAD_JS) as app:
        gr.Markdown("# ⚖️ EchoChamber — Multi-LLM Courtroom Debate")
        with gr.Tab("Debate"):
            _debate_tab()
        with gr.Tab("Setup"):
            _setup_tab()
    return app


def main():
    build_app().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
