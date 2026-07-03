#!/usr/bin/env python3
"""
EchoChamber CLI - Multi-LLM Courtroom Debate

Usage:
    python -m echochamber.cli --topic "Topic" --position "Position to argue for"
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads from .env in current directory or parent directories
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

from .core import (
    Agent, Role, CourtSession, SessionConfig,
    EvidenceStore, create_case_folder,
    ContextStrategy, PreprocessorConfig, DocumentPreprocessor,
    create_summarizer_from_provider,
)
from .core.preprocessor import auto_select_strategy
from .core.transcript import RunConfig
from .providers import create_provider
from .roles import get_role_prompt


# Default models per provider (for advocates: prosecution/defense)
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "together": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "lmstudio": "local-model",  # Will be auto-detected
    "gemini": "gemini-2.0-flash",
}

# Default models for moderator/judge role (can differ from advocates)
DEFAULT_MODERATOR_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "together": "deepseek-ai/DeepSeek-R1",
    "lmstudio": "local-model",
    "gemini": "gemini-2.0-flash",
}


def resolve_model(provider: str, model: str | None, role: str = "advocate") -> str:
    """Resolve model name based on provider and role if not explicitly specified."""
    if model:
        return model
    if role == "moderator":
        return DEFAULT_MODERATOR_MODELS.get(provider, "unknown")
    return DEFAULT_MODELS.get(provider, "unknown")


def parse_args():
    parser = argparse.ArgumentParser(
        description="EchoChamber: Multi-LLM Courtroom Debate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic debate using Anthropic for all agents
  python -m echochamber.cli \\
    --topic "Should AI systems be open source?" \\
    --position "AI systems should be open source for safety and transparency"

  # With evidence folder (auto-selects context strategy)
  python -m echochamber.cli \\
    --topic "Contract Dispute" \\
    --position "The defendant breached the contract" \\
    --case-folder ./cases/contract_dispute

  # Force a specific context strategy
  python -m echochamber.cli \\
    --topic "Patent Dispute" \\
    --position "The patent was infringed" \\
    --case-folder ./cases/patent_case \\
    --context-strategy rag

  # Enable web search for agents
  python -m echochamber.cli \\
    --topic "Current AI Regulations" \\
    --position "AI regulations are necessary" \\
    --enable-search

  # Local models via LM Studio
  python -m echochamber.cli \\
    --topic "Tabs vs Spaces" \\
    --position "Tabs are superior to spaces" \\
    --prosecution-provider lmstudio \\
    --defense-provider lmstudio \\
    --moderator-provider lmstudio

  # Create a new case folder
  python -m echochamber.cli --init-case ./cases/my_new_case --topic "My Topic"
        """,
    )

    # Case initialization (alternative mode)
    parser.add_argument(
        "--init-case",
        type=str,
        metavar="PATH",
        help="Create a new case folder with standard structure and exit",
    )

    # Required arguments (unless --init-case)
    parser.add_argument(
        "--topic",
        required=False,  # Made optional for --init-case
        help="The topic to debate",
    )
    parser.add_argument(
        "--position",
        required=False,  # Made optional for --init-case
        help="The position the prosecution will argue for",
    )

    # Evidence and case folder
    parser.add_argument(
        "--case-folder",
        type=str,
        metavar="PATH",
        help="Path to case folder containing evidence (shared/, prosecution/, defense/, moderator/)",
    )

    # Context strategy for large documents
    parser.add_argument(
        "--context-strategy",
        choices=["auto", "full", "summarize", "rag"],
        default="auto",
        help="Strategy for handling large documents: auto (default), full, summarize, or rag",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=100000,
        help="Target max tokens for evidence context (default: 100000)",
    )
    parser.add_argument(
        "--rag-chunks",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query in RAG mode (default: 10)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        metavar="PATH",
        help="Directory to cache summaries and embeddings",
    )

    # Web search (enabled by default)
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable web search for all agents (search is enabled by default)",
    )
    parser.add_argument(
        "--max-searches-per-turn",
        type=int,
        default=2,
        help="Maximum web searches per party turn (default: 2)",
    )
    parser.add_argument(
        "--moderator-searches-per-turn",
        type=int,
        default=5,
        help="Maximum web searches per moderator turn (default: 5)",
    )

    # Session config
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of debate rounds (default: 3)",
    )
    parser.add_argument(
        "--allow-concession",
        action="store_true",
        help="Allow parties to concede (default: disabled, only moderator decides)",
    )
    parser.add_argument(
        "--allow-conviction",
        action="store_true",
        help="Allow agents to be convinced by opposing arguments (default: strict adversarial)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output during debate",
    )

    # Provider configuration (defaults to LM Studio for local development)
    default_provider = "lmstudio"

    parser.add_argument(
        "--prosecution-provider",
        default=default_provider,
        choices=["anthropic", "openai", "together", "lmstudio", "gemini"],
        help=f"LLM provider for prosecution (default: {default_provider})",
    )
    parser.add_argument(
        "--prosecution-model",
        default=None,
        help="Model for prosecution (auto-detected for lmstudio, claude-sonnet-4-20250514 for anthropic)",
    )

    parser.add_argument(
        "--defense-provider",
        default=default_provider,
        choices=["anthropic", "openai", "together", "lmstudio", "gemini"],
        help=f"LLM provider for defense (default: {default_provider})",
    )
    parser.add_argument(
        "--defense-model",
        default=None,
        help="Model for defense (auto-detected for lmstudio, claude-sonnet-4-20250514 for anthropic)",
    )

    parser.add_argument(
        "--moderator-provider",
        default=default_provider,
        choices=["anthropic", "openai", "together", "lmstudio", "gemini"],
        help=f"LLM provider for moderator (default: {default_provider})",
    )
    parser.add_argument(
        "--moderator-model",
        default=None,
        help="Model for moderator (auto-detected for lmstudio, claude-sonnet-4-20250514 for anthropic)",
    )

    # Output
    parser.add_argument(
        "--save-transcript",
        type=str,
        help="Path to save transcript JSON",
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        default="./transcripts",
        help="Directory to save markdown transcripts (default: ./transcripts)",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Disable automatic transcript saving",
    )

    return parser.parse_args()


def create_agent_from_args(
    name: str,
    role: Role,
    provider: str,
    model: str,
    allow_conviction: bool = False,
    allow_concession: bool = True,
    enable_search: bool = False,
    extra_instructions: str = "",
) -> Agent:
    """Create an agent from CLI arguments."""
    llm_provider = create_provider(provider, model)

    # Get role-appropriate prompt with conviction/concession settings
    system_prompt = get_role_prompt(
        role,
        allow_conviction=allow_conviction,
        allow_concession=allow_concession,
    )

    # Add search instructions if enabled
    if enable_search:
        if role == Role.MODERATOR:
            search_instructions = """

WEB SEARCH (REQUIRED):
You MUST search the web to verify legal claims and find relevant precedents before making rulings.
Include [SEARCH: your query] in your response to perform a search.

IMPORTANT: You are REQUIRED to use at least one search per evaluation to verify the legal basis of arguments.
Example searches:
- [SEARCH: contract law breach of contract deposit refund precedent]
- [SEARCH: implied warranty consumer protection case law]
- [SEARCH: statute of limitations civil claims jurisdiction]

Search results will be provided to you. Base your rulings on established law, not just the arguments presented.
DO NOT skip searching - your rulings must be grounded in verified legal principles."""
        else:
            search_instructions = """

WEB SEARCH:
You can search the web for supporting evidence by including [SEARCH: your query] in your response.
For example: [SEARCH: statistics on remote work productivity 2024]
Search results will be provided to you, and you can reference them in your arguments.
Use searches strategically to find data, studies, or expert opinions that support your position.
Limit searches to the most impactful queries."""
        system_prompt += search_instructions

    # Add extra instructions from evidence files
    if extra_instructions:
        system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{extra_instructions}"

    return Agent(
        name=name,
        role=role,
        provider=llm_provider,
        system_prompt=system_prompt,
    )


def main():
    args = parse_args()

    # Handle --init-case mode
    if args.init_case:
        topic = args.topic or "Untitled Case"
        case_path = create_case_folder(args.init_case, topic)
        print(f"Created case folder: {case_path}")
        print(f"\nFolder structure:")
        print(f"  {case_path}/")
        print(f"  ├── shared/      # Evidence visible to all parties")
        print(f"  ├── prosecution/ # Private prosecution evidence")
        print(f"  ├── defense/     # Private defense evidence")
        print(f"  ├── moderator/   # Moderator-only guidance")
        print(f"  └── README.md    # Instructions")
        print(f"\nSupported files: .txt, .md, .json, .csv, .pdf")
        return

    # Validate required args for debate mode
    if not args.topic or not args.position:
        print("Error: --topic and --position are required for debates")
        print("Use --help for usage information")
        sys.exit(1)

    print("=" * 60)
    print("ECHOCHAMBER - Multi-LLM Courtroom Debate")
    print("=" * 60)
    print(f"\nTopic: {args.topic}")
    print(f"Position: {args.position}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Mode: {'Allow conviction' if args.allow_conviction else 'Strict adversarial'}")

    # Search is enabled by default unless --no-search is specified
    enable_search = not args.no_search
    if enable_search:
        print(f"Web search: Enabled (advocates: {args.max_searches_per_turn}/turn, moderator: {args.moderator_searches_per_turn}/turn)")
    else:
        print("Web search: Disabled")

    # Load evidence if case folder specified
    evidence = None
    raw_evidence = None
    search_tool = None

    if args.case_folder:
        try:
            raw_evidence = EvidenceStore.load(args.case_folder)
            print(f"\n{raw_evidence.summary()}")

            # Determine context strategy
            if args.context_strategy == "auto":
                strategy = auto_select_strategy(
                    total_chars=raw_evidence.total_chars(),
                    max_context_tokens=args.max_context_tokens,
                    num_documents=len(raw_evidence.shared) + len(raw_evidence.prosecution) +
                                  len(raw_evidence.defense) + len(raw_evidence.moderator),
                )
                print(f"\nAuto-selected context strategy: {strategy.value}")
            else:
                strategy = ContextStrategy(args.context_strategy)
                print(f"\nContext strategy: {strategy.value}")

            if strategy != ContextStrategy.FULL:
                print(f"Preprocessing evidence ({strategy.value})...")

                # Create preprocessor config
                preproc_config = PreprocessorConfig(
                    strategy=strategy,
                    max_tokens=args.max_context_tokens,
                    top_k=args.rag_chunks,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                )

                # For summarization, we need an LLM
                summarizer = None
                if strategy == ContextStrategy.SUMMARIZE:
                    # Use the moderator's provider for summarization
                    sum_provider = create_provider(
                        args.moderator_provider,
                        args.moderator_model,
                    )
                    summarizer = create_summarizer_from_provider(sum_provider)

                preprocessor = DocumentPreprocessor(preproc_config, summarizer=summarizer)
                evidence = preprocessor.process_evidence_store(raw_evidence)
                print(evidence.summary())
            else:
                evidence = raw_evidence

        except Exception as e:
            print(f"\nWarning: Could not load case folder: {e}")
            import traceback
            traceback.print_exc()
            print("Proceeding without evidence.\n")

    # Initialize search tool if enabled
    if enable_search:
        from .tools import WebSearchTool
        search_tool = WebSearchTool(max_results=5)

    print()

    # Create agents
    allow_concession = args.allow_concession  # Default is False (moderator decides)

    # Get instructions from evidence files (if any)
    prosecution_instructions = ""
    defense_instructions = ""
    moderator_instructions = ""
    if raw_evidence:
        prosecution_instructions = raw_evidence.get_instructions_for_role(Role.PROSECUTION)
        defense_instructions = raw_evidence.get_instructions_for_role(Role.DEFENSE)
        moderator_instructions = raw_evidence.get_instructions_for_role(Role.MODERATOR)

    # Resolve model names (auto-detect for lmstudio, role-specific defaults)
    prosecution_model = resolve_model(args.prosecution_provider, args.prosecution_model, "advocate")
    defense_model = resolve_model(args.defense_provider, args.defense_model, "advocate")
    moderator_model = resolve_model(args.moderator_provider, args.moderator_model, "moderator")

    try:
        print("Initializing agents...")
        prosecution = create_agent_from_args(
            name="Prosecution",
            role=Role.PROSECUTION,
            provider=args.prosecution_provider,
            model=prosecution_model,
            allow_conviction=args.allow_conviction,
            allow_concession=allow_concession,
            enable_search=enable_search,
            extra_instructions=prosecution_instructions,
        )
        print(f"  ✓ Prosecution: {prosecution}")

        defense = create_agent_from_args(
            name="Defense",
            role=Role.DEFENSE,
            provider=args.defense_provider,
            model=defense_model,
            allow_conviction=args.allow_conviction,
            allow_concession=allow_concession,
            enable_search=enable_search,
            extra_instructions=defense_instructions,
        )
        print(f"  ✓ Defense: {defense}")

        moderator_can_search = enable_search and args.moderator_searches_per_turn > 0
        moderator = create_agent_from_args(
            name="Moderator",
            role=Role.MODERATOR,
            provider=args.moderator_provider,
            model=moderator_model,
            allow_conviction=False,  # Moderator doesn't need conviction mode
            allow_concession=False,
            enable_search=moderator_can_search,
            extra_instructions=moderator_instructions,
        )
        if moderator_instructions:
            print(f"  ✓ Moderator: {moderator} (with custom instructions)")
        else:
            print(f"  ✓ Moderator: {moderator}")
        print()

    except Exception as e:
        print(f"\nError initializing agents: {e}")
        print("\nMake sure you have the required API keys set:")
        print("  - ANTHROPIC_API_KEY for Anthropic")
        print("  - OPENAI_API_KEY for OpenAI")
        print("  - TOGETHER_API_KEY for Together.AI")
        print("  - GEMINI_API_KEY for Google Gemini")
        print("  - LM Studio running on localhost:1234 for local models")
        sys.exit(1)

    # Configure session
    config = SessionConfig(
        max_rounds=args.max_rounds,
        allow_concession=allow_concession,
        allow_conviction=args.allow_conviction,
        verbose=not args.quiet,
    )

    # Capture run configuration for transcript (use actual detected model names)
    context_strategy_str = strategy.value if 'strategy' in locals() else "none"
    run_config = RunConfig(
        topic=args.topic,
        position=args.position,
        prosecution_provider=args.prosecution_provider,
        prosecution_model=prosecution.provider.model if hasattr(prosecution.provider, 'model') else prosecution_model,
        defense_provider=args.defense_provider,
        defense_model=defense.provider.model if hasattr(defense.provider, 'model') else defense_model,
        moderator_provider=args.moderator_provider,
        moderator_model=moderator.provider.model if hasattr(moderator.provider, 'model') else moderator_model,
        max_rounds=args.max_rounds,
        allow_concession=allow_concession,
        allow_conviction=args.allow_conviction,
        context_strategy=context_strategy_str,
        enable_search=enable_search,
        case_folder=args.case_folder or "",
    )

    # Create callback to handle search if enabled
    on_turn_callback = None
    if search_tool:
        from .tools.search import extract_search_queries

        def on_turn_with_search(speaker: str, role: str, content: str):
            """Process searches after each turn."""
            # Determine search limit based on role
            if role == "moderator":
                max_searches = args.moderator_searches_per_turn
            else:
                max_searches = args.max_searches_per_turn

            if max_searches <= 0:
                return  # Search disabled for this role

            queries = extract_search_queries(content)
            if queries:
                queries = queries[:max_searches]  # Limit searches
                print(f"\n  [{speaker} searching: {', '.join(queries)}]")
                for query in queries:
                    results = search_tool.search_formatted(query, max_results=3)
                    print(f"\n{results}")

        on_turn_callback = on_turn_with_search

    # Run session
    session = CourtSession(
        prosecution=prosecution,
        defense=defense,
        moderator=moderator,
        config=config,
        evidence=evidence,
        on_turn=on_turn_callback,
        search_tool=search_tool,
        max_moderator_searches=args.moderator_searches_per_turn,
    )

    try:
        result = session.run(
            topic=args.topic,
            prosecution_position=args.position,
        )
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during debate: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Attach run config to transcript
    result.transcript.run_config = run_config

    # Print summary
    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)
    print(f"Rounds completed: {result.rounds_completed}")
    print(f"Termination: {result.termination_reason.value}")
    if result.winner:
        print(f"Winner: {result.winner.upper()}")
    else:
        print("Winner: UNDECIDED")

    # Save markdown transcript (default behavior)
    if not args.no_transcript:
        transcript_path = result.transcript.save_markdown(args.transcript_dir)
        print(f"\nTranscript saved to: {transcript_path}")

    # Save JSON transcript if requested
    if args.save_transcript:
        path = Path(args.save_transcript)
        result.transcript.save(path)
        print(f"JSON transcript saved to: {path}")


if __name__ == "__main__":
    main()
