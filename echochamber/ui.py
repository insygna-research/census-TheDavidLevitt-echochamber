#!/usr/bin/env python3
"""
EchoChamber GUI - watch a debate live in the browser.

    uv run python -m echochamber.ui          # http://localhost:7860

Requires the ui extra:  uv sync --extra ui
Set $ECHOCHAMBER_APA_DATA to an APA data directory (or merged export) to see
live procurement recommendations instead of the bundled sample.
"""

import queue
import threading

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    raise SystemExit("The GUI requires gradio. Install with: uv sync --extra ui")

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
ROLE_EMOJI = {"prosecution": "⚖️", "defense": "🛡️", "moderator": "👨‍⚖️", "system": "⚙️"}

_run_lock = threading.Lock()
_stop_event = threading.Event()

# APA recommendations, loaded once at startup
try:
    _apa = load_apa_data()
    _recs = recommend_for_debate(_apa)
except Exception:
    _apa, _recs = None, {}


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


def _fmt_tokens(meter: UsageMeter, budget: int | None) -> str:
    tin, tout = meter.total_tokens()
    total = tin + tout
    budget_str = f" / {budget:,} budget" if budget else ""
    pct = f" ({100 * total / budget:.0f}%)" if budget else ""
    return (f"**Tokens burned:** {total:,}{budget_str}{pct} · "
            f"{tin:,} in / {tout:,} out · **Cost:** ${meter.total_cost():.4f}")


def stop_debate():
    _stop_event.set()
    return "### ⏹️ Stopping after the current turn…"


def run_debate_ui(topic, position, pros_provider, pros_model, def_provider, def_model,
                  mod_provider, mod_model, rounds, enable_search, token_budget):
    """Generator: streams (status, tokens, chat, verdict) updates while the debate runs."""
    chat: list[dict] = []

    if not topic or not position:
        yield "### ⚠️ Topic and position are required.", "", chat, ""
        return
    if not _run_lock.acquire(blocking=False):
        yield "### ⚠️ A debate is already running.", "", chat, ""
        return

    try:
        _stop_event.clear()
        budget = int(token_budget) if token_budget else None
        meter = UsageMeter(hard_limit_tokens=budget)
        events: queue.Queue = queue.Queue()

        spec = DebateSpec(
            topic=topic,
            position=position,
            prosecution_provider=pros_provider,
            prosecution_model=pros_model.strip() or None,
            defense_provider=def_provider,
            defense_model=def_model.strip() or None,
            moderator_provider=mod_provider,
            moderator_model=mod_model.strip() or None,
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
                    should_stop=_stop_event.is_set,
                )
                events.put(("done", outcome))
            except Exception as e:
                events.put(("error", str(e)))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        status = "### 🏁 Starting debate…"
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
    finally:
        _run_lock.release()


def build_app() -> "gr.Blocks":
    with gr.Blocks(title="EchoChamber") as app:
        gr.Markdown("# ⚖️ EchoChamber — Multi-LLM Courtroom Debate")

        with gr.Row():
            with gr.Column(scale=1):
                topic = gr.Textbox(label="Topic", value="Tabs vs Spaces")
                position = gr.Textbox(
                    label="Prosecution position",
                    value="Tabs are superior to spaces",
                )

                role_inputs = {}
                for role, default_provider in (
                    ("prosecution", "lmstudio"),
                    ("defense", "lmstudio"),
                    ("moderator", "lmstudio"),
                ):
                    with gr.Group():
                        gr.Markdown(f"**{ROLE_EMOJI[role]} {role.capitalize()}**")
                        with gr.Row():
                            provider = gr.Dropdown(
                                PROVIDERS, value=default_provider,
                                label="Provider", scale=1,
                            )
                            model = gr.Textbox(
                                label="Model (blank = provider default)", scale=2,
                            )
                    role_inputs[role] = (provider, model)

                with gr.Accordion("🤖 APA model recommendations", open=False):
                    gr.Markdown(_recs_markdown())
                    rec_outputs = [w for pair in role_inputs.values() for w in pair]
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

                with gr.Row():
                    run_btn = gr.Button("▶️ Run debate", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop")

            with gr.Column(scale=2):
                status = gr.Markdown("### 💤 Idle")
                tokens = gr.Markdown("")
                chatbot = gr.Chatbot(label="Proceedings", height=520)
                verdict = gr.Markdown("")

        inputs = [topic, position]
        for role in ("prosecution", "defense", "moderator"):
            inputs.extend(role_inputs[role])
        inputs.extend([rounds, enable_search, token_budget])

        run_btn.click(run_debate_ui, inputs=inputs, outputs=[status, tokens, chatbot, verdict])
        stop_btn.click(stop_debate, outputs=[status])

    return app


def main():
    build_app().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
