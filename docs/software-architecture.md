# Software architecture: GitHub digest agent

## 1. Purpose of this document

The earlier design spec (`digest-agent-design-spec.md`) defines **control
flow** — the sequence of steps and the evaluator-optimizer loop for model
selection. This document defines **structure** — module boundaries and, most
importantly, *dependency direction*. These are complementary, not
duplicate: a system can have a correct workflow and a bad architecture (or
vice versa). Both are needed.

## 2. Architectural style: Hexagonal (Ports & Adapters)

**Problem this solves:** if application logic calls `requests.post()` to
OpenRouter directly, or `smtplib` directly, the business rules (*what a
digest is, how a model is selected, what "done" means*) become entangled
with infrastructure choices (*which HTTP client, which email vendor*).
That entanglement is what makes systems hard to test and hard to reuse.

**The rule:** the application core depends only on abstract interfaces
("ports"). Concrete implementations ("adapters") depend on the ports and
are supplied from outside the core at startup (dependency injection). The
core has zero imports of any vendor SDK or HTTP client.

```
[driven adapters] --implement--> [ports] <--depend on-- [application core]
```

This is the Dependency Inversion Principle (the "D" in SOLID) applied at
the system level, not just the class level.

## 3. Components

### 3.1 Application core (owns no I/O)

| Component | Responsibility |
|---|---|
| `Orchestrator` | The LangGraph `StateGraph`. Sequences nodes, owns the evaluator-optimizer loop's control flow (attempt counting, exit conditions). Calls ports, never adapters, by name. |
| `ModelSelectionPolicy` | Pure logic: given a list of candidate models and test results, decide pass/fail/next. No network calls — the port does the calling, the policy interprets the result. |
| `DigestState` | The shared state schema (repos, activity, selected model, summary, pdf path, outcome). |

### 3.2 Ports (interfaces the core depends on)

| Port | Contract (abstract) | Why it's a port, not a direct call |
|---|---|---|
| `ActivitySourcePort` | `fetch_activity(repos, since) -> RepoActivity[]` | Today: GitHub. Tomorrow: GitLab, Bitbucket, a local git clone — same contract. |
| `LLMProviderPort` | `list_free_models() -> ModelId[]`; `test(model) -> TestResult`; `complete(model, prompt) -> str` | Today: OpenRouter. Tomorrow: a direct Anthropic/OpenAI key, a local Ollama instance. |
| `DocumentRendererPort` | `render(summary: str) -> file path` | Today: PDF via reportlab. Tomorrow: HTML email body, Slack blocks, a Notion page. |
| `NotifierPort` | `send(recipient, subject, attachment)` | Today: Gmail SMTP. Tomorrow: Resend, Slack, a webhook. |

### 3.3 Driven adapters (implement the ports)

| Adapter | Implements | Vendor specifics live here, and only here |
|---|---|---|
| `GitHubAdapter` | `ActivitySourcePort` | GitHub REST API shape, pagination, auth headers |
| `OpenRouterAdapter` | `LLMProviderPort` | `/models` filtering logic, OpenRouter's request/response shape, fallback array syntax |
| `ReportLabAdapter` | `DocumentRendererPort` | reportlab's document/story API |
| `GmailAdapter` | `NotifierPort` | SMTP connection details, MIME construction |

### 3.4 Driving adapter (triggers the core)

| Adapter | Responsibility |
|---|---|
| `SchedulerTrigger` | GitHub Actions cron (or local cron) calling the entry point on a schedule. This is outside the core by the same logic — swapping to a cloud scheduler shouldn't touch orchestration code. |

## 4. Dependency rules (the part that's easy to violate by accident)

1. `Orchestrator` may import port interfaces and `DigestState`. It may
   **never** import `requests`, `smtplib`, `reportlab`, or any adapter
   class directly.
2. Adapters may import their vendor SDK freely — that's their entire job.
3. Adapters are wired into the orchestrator at the composition boundary
   (the entry point / `main`), not inside the graph definition. The graph
   definition should be constructible with **fake** adapters for testing,
   with no code changes, only different objects passed in.
4. If you find yourself writing `if provider == "openrouter"` inside the
   core, that's a leak — it belongs inside `OpenRouterAdapter` or in the
   composition step that chooses which adapter to inject.

## 5. Quality attributes this buys you

- **Testability**: `ModelSelectionPolicy` and `Orchestrator` are testable
  with in-memory fake ports — no network, no mocking framework fighting
  `requests`. The evaluator-optimizer loop's three exit paths (pass /
  exhausted / budget) become fast, deterministic unit tests.
- **Portability across "many projects"**: a second project that also wants
  a digest bot reuses the entire application core untouched. Only the
  adapter wiring changes (e.g. a project on GitLab swaps in
  `GitLabAdapter` implementing the same `ActivitySourcePort`).
- **Replaceability**: switching Gmail → Resend is a new class implementing
  `NotifierPort`, registered at the composition boundary. Zero diff inside
  `Orchestrator` or `ModelSelectionPolicy`.
- **Explicit vendor coupling**: instead of vendor specifics being scattered
  through the codebase, they are contained to exactly four files (one per
  adapter). Auditing "what depends on OpenRouter" becomes a `grep` for one
  filename, not a search through business logic.

## 6. How this maps back to the workflow patterns already chosen

- **Prompt Chaining** (the outer pipeline) and **Evaluator-Optimizer** (the
  model-selection loop) describe *control flow inside the application
  core*. They live entirely in `Orchestrator` and `ModelSelectionPolicy`.
- **Hexagonal architecture** describes *where the boundaries are* around
  that control flow. It doesn't replace the workflow patterns — it's the
  structural container they run inside.

## 7. What to show, concretely

For a portfolio or interview context, the artifact worth presenting is:
1. The layered diagram (trigger → core → ports → adapters).
2. This document's dependency rule in §4 — being able to articulate *why*
   the core never imports `requests` is the actual signal of architectural
   maturity, more than the diagram itself.
3. One concrete "swap" walkthrough as a talking point: "if I wanted this on
   GitLab instead of GitHub, here's exactly the one file I'd write and the
   zero files I wouldn't touch."
