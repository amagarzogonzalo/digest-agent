# GitHub Digest Agent

A professional Hexagonal (Ports & Adapters) architecture Python agent that,
on a schedule, reads recent commits/PRs across configurable repositories,
selects a currently-free LLM (testing candidates until one works),
summarizes the activity, renders a PDF, and emails it.

## Overview

This project implements the **Hexagonal (Ports & Adapters)** design pattern at
the system level, solving the concrete problem of business rules being welded
to infrastructure choices. The core depends only on abstract interfaces
("ports"), making it testable and reusable across projects.

### Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    HEXAGONAL ARCHITECTURE                     │
│┌───────────────┐┌─────────────────────┐┌────────────────┐│
││  Port Interfaces││   Application Core   ││  Driven Adapters ││
││ (abstract, no   ││ - Orchestrator      ││ - GitHubAdapter  ││
││  vendor imports)││ - ModelSelectionPolicy││ - OpenRouterAdapter││
││                 ││ - DigestState       ││ - ReportLabAdapter││
│└─────────────────┘└─────────────────────┘└────────────────┘│
│           ▲                   ▲                   ▲           │
│           │                   │                   │           │
│           │                   │                   │           │
│           │                   │                   │           │
│           │                   │                   │           │
│           └───────┬─────────────┘───────────────┘           │
│                     │                                     │
│              (composition)                            │
│                                                                 │
│              SchedulerTrigger (GitHub Actions workflow)        │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

### Workflow Design

Two composable patterns from Anthropic's agent-design taxonomy:

- **Prompt Chaining** for the outer pipeline — a linear sequence with
  programmatic gates between steps
- **Evaluator-Optimizer** for model selection — generate a candidate,
  evaluate it against acceptance criteria, loop until pass or budget exhausted

### Pipeline

```
START
  -> fetch_github            [Orchestrator calls ActivitySourcePort]
  -> select_and_test_model   [LOOP: cycles until PASS, FAIL_EXHAUSTED, or FAIL_BUDGET]
  -> summarize                [Orchestrator calls LLMProviderPort]
  -> render_pdf               [Orchestrator calls DocumentRendererPort]
  -> send_notification        [Orchestrator calls NotifierPort]
END
```

### Key Features

- **Hexagonal Architecture** — core depends only on abstract ports, no vendor SDKs
- **Configurable LLM providers** — Groq, Mistral, and OpenRouter with automatic fallback
- **SMTP notifier** — works with Gmail or any SMTP host (Ethereal for testing)
- **Evaluator-Optimizer loop** — bounded by `min(len(candidate_pool), max_attempts)`
- **Three loop exit paths**: PASS / FAIL_EXHAUSTED / FAIL_BUDGET (all tested)
- **Observability** — logs model selection, candidates tried, rejection reasons
- **Idempotency** — re-running for same day/window deduplicates
- **Cost/rate-limit aware** — minimal test calls, counts against free-tier quota
- **Testability** — every node independently testable with mocked ports

### Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd digest-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e .[dev]

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials (provider key, GitHub token, SMTP)

# 4. Run locally
# Values can come from .env (loaded automatically) or environment variables.
python -m digest_agent.main

# 5. Or run via GitHub Actions (included)
```

### LLM Providers

Providers are tried in a fallback chain — **Groq → Mistral → OpenRouter** —
until a model passes the health check. Configure a key for each provider you
want available (all models used are free):

| Provider | Key var | Candidate models (free) |
|---|---|---|
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `groq/compound-mini` |
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest`, `open-mistral-7b` |
| OpenRouter | `OPENROUTER_API_KEY` | Live catalog filtered to price 0 |

The evaluator-optimizer loop tests candidates in chain order (Groq first,
then Mistral, then OpenRouter) and picks the first that passes. Providers
without a configured key are skipped. If none configure a key, the run
aborts with a clear error.

### Environment Variables

All values can be set in `.env` or as environment variables.

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | Yes | GitHub personal access token (repo scope) |
| `GITHUB_REPOS` | Yes | Comma-separated list of repos (owner/repo format) |
| `GROQ_API_KEY` | No* | Groq key (tried first) |
| `MISTRAL_API_KEY` | No* | Mistral key (tried second) |
| `OPENROUTER_API_KEY` | No* | OpenRouter key (tried last; alias `OPEN_ROUTER_API_KEY` accepted) |
| `SMTP_HOST` | No | SMTP host (default `smtp.gmail.com`) |
| `SMTP_PORT` | No | SMTP port (default `465`) |
| `SMTP_STARTTLS` | No | `true` to use STARTTLS (needed for port 587) |
| `SMTP_USER` / `SMTP_PASSWORD` | No | SMTP credentials (fall back to GMAIL_* ) |
| `GMAIL_ADDRESS` | No | Gmail address (used as SMTP user fallback) |
| `GMAIL_APP_PASSWORD` | No | Gmail app password |
| `RECIPIENT_EMAIL` | No | Recipient email (defaults to SMTP user) |
| `LOOKBACK_DAYS` | No | Days to look back (default: 1) |
| `DIGEST_OUTPUT_DIR` | No | Directory for generated PDFs (default: system temp dir) |

\* At least one of `GROQ_API_KEY` / `MISTRAL_API_KEY` / `OPENROUTER_API_KEY`
is required. They are tried in that order.

### Project Structure

```
digest-agent/
├── .github/
│   └── workflows/
│       └── daily-digest.yml    # GitHub Actions scheduler
├── src/
│   └── digest_agent/
│       ├── ports/              # Abstract port interfaces
│       │   ├── activity_source.py
│       │   ├── llm_provider.py
│       │   ├── document_renderer.py
│       │   └── notifier.py
│       ├── core/               # Application core (zero vendor imports)
│       │   ├── state.py
│       │   ├── policy.py
│       │   ├── orchestrator.py
│       │   └── fakes.py        # In-memory fakes for testing the core
│       ├── adapters/
│       │   └── driven/
│       │       ├── openai_compat.py       # Shared OpenAI-compatible base
│       │       ├── fallback_provider.py   # Groq -> Mistral -> OpenRouter chain
│       │       ├── github_adapter.py
│       │       ├── openrouter_adapter.py
│       │       ├── groq_adapter.py
│       │       ├── mistral_adapter.py
│       │       ├── reportlab_adapter.py
│       │       └── gmail_adapter.py       # SMTPAdapter (Gmail/Ethereal/SMTP)
│       └── main.py             # Entry point - wires adapters at composition boundary
├── tests/                      # Comprehensive test suite
│   ├── test_policy.py          # 10 unit tests for ModelSelectionPolicy
│   └── test_orchestrator.py    # 10 integration tests for Orchestrator
├── pyproject.toml              # UV package configuration
├── requirements.txt            # Pinned dependencies (generated by uv)
├── .env.example                # Environment template (copy to .env)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

### Definition of Done

- [x] Orchestrator and ModelSelectionPolicy contain no vendor imports
- [x] Graph is acyclic everywhere except the one designed cycle
- [x] Cycle has proven upper bound on iterations
- [x] All three loop exit paths (PASS / FAIL_EXHAUSTED / FAIL_BUDGET) are
      reachable in tests and each produces a distinct, observable outcome
- [x] No hardcoded model ID anywhere in the codebase
- [x] A failed model-selection loop never silently proceeds to summarize
- [x] Swapping any one adapter (e.g. Gmail → Resend) requires touching
      exactly one new file plus the composition boundary — nothing inside
      Orchestrator or ModelSelectionPolicy

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# All 20 tests pass:
# - 10 unit tests for ModelSelectionPolicy (PASS/FAIL_EXHAUSTED/FAIL_BUDGET paths)
# - 10 integration tests for Orchestrator (full pipeline + error handling)
```

### License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Design References

- **Design Specification**: `software-architecture.md` / `digest-agent-design-spec.md`
- **Architecture**: Hexagonal (Ports & Adapters) pattern, Dependency Inversion Principle
- **Patterns**: Prompt Chaining (outer pipeline), Evaluator-Optimizer (model selection)