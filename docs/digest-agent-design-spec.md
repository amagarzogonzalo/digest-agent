# Design spec: GitHub → OpenRouter → PDF → Gmail digest agent

**Audience:** a coding agent (e.g. opencode) implementing this from scratch.
**Pattern:** Prompt Chaining (outer pipeline) wrapping an Evaluator-Optimizer
loop (model selection). Implemented as a single LangGraph `StateGraph` with
one cycle.

---

## 1. Architecture

```
START
  -> fetch_github            [deterministic]
  -> select_and_test_model   [LOOP: cycles until PASS, FAIL_EXHAUSTED, or FAIL_BUDGET]
  -> summarize                [LLM call, uses the model selected above]
  -> render_pdf               [deterministic]
  -> send_gmail                [deterministic]
END
```

Only two nodes touch an LLM: `select_and_test_model` (cheap test calls) and
`summarize` (the real work). Everything else is deterministic and must not
retry on "logic" grounds — only on transient I/O failure (see §4).

---

## 2. Node specifications

### 2.1 `fetch_github`
- **Input:** repo list, lookback window.
- **Output:** structured activity per repo (commits, PRs).
- **Failure mode:** GitHub API error or empty token → hard fail, do not
  proceed. This is infrastructure, not something to loop on.

### 2.2 `select_and_test_model` — the loop node

This is the Evaluator-Optimizer core. Treat it as its own sub-state:

**Sub-state fields**
- `candidate_pool`: ranked list of currently-free OpenRouter models (fetched
  live from OpenRouter's public models endpoint, filtered to zero prompt/
  completion price, ranked by context length or another quality proxy —
  never hardcoded).
- `tried`: set of model IDs already attempted this run.
- `attempt_count` / `max_attempts`: hard cap (recommend 5) — this is the
  circuit breaker. Without it, an outage on OpenRouter's side turns into an
  infinite loop.
- `selected_model`: set only on PASS.

**Loop body, each iteration:**
1. Pop the next untried candidate from `candidate_pool`. If none remain →
   exit loop with `FAIL_EXHAUSTED`.
2. Issue one cheap, deterministic test call to the candidate (a short fixed
   prompt, not the real summarization prompt — keep this call minimal so
   testing itself doesn't burn meaningful free-tier quota).
3. **Acceptance criteria** (all must pass):
   - Returns HTTP 200 within a timeout budget (recommend 15s).
   - Response is non-empty and well-formed (valid completion object).
   - Not a rate-limit response (HTTP 429) — if rate-limited, this candidate
     is deprioritized but not the same failure class as "broken"; log it
     separately since it may work on a later run.
4. On PASS → set `selected_model`, exit loop, proceed to `summarize`.
5. On FAIL → add to `tried`, increment `attempt_count`, loop back to step 1.
6. If `attempt_count >= max_attempts` before exhausting the pool → exit with
   `FAIL_BUDGET` rather than continuing indefinitely.

**Termination guarantee:** the loop must be provably finite — bounded by
`min(len(candidate_pool), max_attempts)`. State this explicitly in the
implementation; an agent implementing "loop until resolved" without a cap
is the single most common bug in this pattern.

**On FAIL_EXHAUSTED or FAIL_BUDGET:** the graph must not silently continue
to `summarize` with no model. Route to a terminal failure state that still
produces an observable outcome (log + optionally a fallback notification),
not a crash mid-pipeline after GitHub data has already been fetched.

### 2.3 `summarize`
- **Input:** activity data + `selected_model` from the loop.
- **Output:** structured digest text.
- Uses the *real* summarization prompt — separate from the lightweight
  test prompt used during selection.
- If this call itself fails (selected model degrades between test and use),
  that is a distinct, rarer failure — handle with a small bounded retry
  (e.g. 2 attempts) rather than re-entering the full selection loop.

### 2.4 `render_pdf`
- Deterministic transform, no LLM, no external network call. Should never
  be a source of loop/retry logic.

### 2.5 `send_gmail`
- Deterministic I/O. Retry on transient SMTP failure only (e.g. 2 attempts
  with backoff), never loop indefinitely.

---

## 3. Non-functional requirements

- **Observability:** log which model was selected, how many candidates were
  tried, and why each rejected candidate failed. This loop is exactly the
  kind of thing that fails silently if unlogged.
- **Idempotency:** re-running the graph for the same day/window should not
  produce duplicate emails if triggered twice (e.g. manual + scheduled run
  on the same day) — dedupe on date or run ID if this matters to you.
- **Secrets:** GitHub token, OpenRouter key, Gmail credentials via
  environment/secret store only — never hardcoded, never logged.
- **Cost/rate-limit awareness:** the test call in §2.2 step 2 counts against
  the free tier's request budget too — keep it minimal, don't over-test.
- **Testability:** each node must be independently unit-testable, including
  the loop node with a mocked candidate pool (test the PASS path, the
  FAIL_EXHAUSTED path, and the FAIL_BUDGET path as three separate cases).

## 4. What NOT to loop on

To keep the Evaluator-Optimizer pattern from creeping into the rest of the
graph: only `select_and_test_model` loops on *logical* failure (model
doesn't work). Every other node's retries, if any, are bounded and only for
*transient* infrastructure failure (timeout, 5xx) — never for "the LLM gave
a bad answer," which has no place outside the evaluator loop as designed.

## 5. Definition of done

- [ ] Graph is acyclic everywhere except the one designed cycle.
- [ ] The cycle has a proven upper bound on iterations.
- [ ] All three loop exit paths (PASS / FAIL_EXHAUSTED / FAIL_BUDGET) are
      reachable in tests and each produces a distinct, observable outcome.
- [ ] No hardcoded model ID anywhere in the codebase.
- [ ] A failed model-selection loop never silently proceeds to `summarize`.
