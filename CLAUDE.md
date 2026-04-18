# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo purpose

Benchmark suite for ZK-ML audit agents. Three components are intended to work together end-to-end: `dataset_generator/` curates the ground-truth audit-finding dataset, `dataset_loader/` materializes (paper PDF, codebase) pairs for an agent run, and `grader/` scores the agent's JSON output against the ground truth. **Only `grader/` is implemented today** — the other two are placeholder folders with READMEs describing the planned interface.

## Common commands

```bash
# Run all tests
python -m pytest grader/tests/ -v

# Run a single test file / class / test
python -m pytest grader/tests/test_scorers.py -v
python -m pytest grader/tests/test_scorers.py::TestScoreSeverity -v
python -m pytest grader/tests/test_scorers.py::TestScoreSeverity::test_exact_critical -v

# Run the grader end-to-end
python -m grader \
    --ground-truth zkMLDataset.xlsx \
    --agent-output agent_results.json \
    --output grade_report.json \
    --output-md grade_report.md
```

`pyproject.toml` exposes the grader as a `zkml-grader` console script after `pip install -e .`.

## Test fixtures

`grader/tests/conftest.py` builds fictional ground-truth (xlsx) and agent-output (JSON) fixtures at session start using synthetic project names (`alpha`, `beta`). **No real dataset content is ever in tests** — deliberately, to avoid leaking ground-truth to anything that reads the test suite. All tests are self-contained; none skip. LLM tests use `MockLLMProvider` from `grader.llm` (no API is ever contacted).

## Example agent output

`examples/agent_output.example.json` is the canonical reference file for agents producing output for the grader, with a companion `examples/README.md` documenting the full schema. The findings use fictional project names (`zkFoo`, `zkBar`) — **do not** add real dataset content to these examples (avoids leaking ground-truth content to agents).

## Architecture

### Grader pipeline

```
xlsx GT     ─┐
             ├─► loader ─► dict[entry_id, list[Finding]] ─┐
agent JSON  ─┘                                            ├─► matcher (per-project)
                                                          │     ├─► matched pairs
                                                          │     ├─► missed_gt
                                                          │     └─► extra_agent
                                                          ▼
                                                       scorers
                                                          ▼
                                                       report
                                                          ▼
                                              JSON + markdown output
```

Module responsibilities:

- **`grader/loader.py`** — parses the xlsx (ground truth) and the agent JSON. Both must include all 7 fields (severity, category, security-concern, relevant-code, paper-reference, issue-name, issue-explanation). `parse_code_refs` handles the `file:line[-line], file:line` format including unicode en-dashes. Entry IDs are normalized to lowercase as the grouping key. Validation is strict: invalid closed-list values raise `ValueError`.

- **`grader/similarity.py`** — `SimilarityBackend` ABC with two concrete backends: `JaccardSimilarity` (word-overlap baseline, zero deps) and `LLMJudgeSimilarity` (LLM-as-judge via `LLMProvider`). The judge exposes a primary `judge_bulk(agent_text, candidates)` method that scores one agent finding against all GT candidates in a single LLM call, plus a `score(a, b)` compat path (degenerate 1-candidate bulk call) so paper-reference quote scoring still works. `JUDGE_SCHEMA` is strict and used verbatim by both OpenAI `response_format=json_schema` and Anthropic tool-use. Results are cached in-memory by SHA-256 of agent + sorted candidates; the richer `{match_score, same_root_cause, reasoning}` per-pair judgment is available via `last_result_for(agent_text, gt_id)` for future report enrichment (not yet wired into the report).

- **`grader/llm.py`** — `LLMProvider` ABC with `OpenAIProvider`, `AnthropicProvider`, `MockLLMProvider`. Provider SDK imports are lazy (inside `__init__`) so the package works without `openai` or `anthropic` installed. `build_config_from_env()` reads `LLM_PROVIDER` (default `openai`), `OPENAI_API_KEY` / `OPENAI_MODEL` (default `gpt-4o`), `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` (default `claude-opus-4-5`). `.env` loading via `python-dotenv` is a soft dependency (silent no-op if not installed).

- **`grader/matcher.py`** — operates **per-project** (never matches findings across entry-ids). Builds an `M×N` similarity matrix on `name + " " + explanation`, then does greedy 1:1 assignment above a threshold (default 0.3). Issue-ids are GT-only labels — agents produce findings in arbitrary order with no ID correspondence. Unmatched agent findings are exposed via `extra_by_severity` because hallucinated Critical findings are qualitatively different from extra Info suggestions. **The matcher detects `backend.judge_bulk` via `getattr` and prefers it** (one call per agent finding) over the M×N per-pair `score()` loop; do not assume either branch is always taken.

- **`grader/scorers.py`** — one function per graded field, each returning `FieldScore(score, detail)`:
  - **Severity** uses an asymmetric 3×3 matrix where under-reporting (Warning when GT is Critical) scores 0.0 but over-reporting gets partial credit (0.25 for one level, 0.1 for two). This is deliberate — missing severity is the worst failure.
  - **Category** and **security-concern** support partial credit via a small proximity table for semantically adjacent values (e.g., "Under-constrained Circuit" ↔ "Witness/Commitment Mismatch" → 0.4).
  - **Code location** does basename matching (so `verifier.rs:38` matches `src/util/verifier.rs:36-42`) and grades by line-range proximity (overlap → 1.0, within 30 lines → 0.7, etc.). Returns 1.0 with a "skip" detail when GT has no code refs.
  - **Paper reference** averages two sub-scores: section-ID extraction via regex (matches `Section X.Y.Z`, `Protocol N`, `Theorem N`, `Eq. N`, `Example N`) and quote similarity through the pluggable backend. Same skip semantics as code location when GT is `-` or empty.

- **`grader/report.py`** — `_compute_pair_score` does weighted averaging of the field scores with **automatic redistribution of skipped fields' weights** to the remaining active ones (this is critical — don't change without updating tests). Default weights: code 0.30, paper 0.25, severity/category/security-concern 0.15 each. Aggregates into per-project metrics (precision, recall, F1, severity-weighted recall, quality, composite) and an overall `benchmark_score = 0.4 * f1 + 0.6 * quality`.

- **`grader/cli.py` / `__main__.py`** — argparse orchestration. `_parse_weights` accepts `key=value` overrides. Projects with no GT are skipped with a printed notice. Weights and threshold are written into the report's `meta` block for reproducibility. `--backend llm-judge` activates the LLM backend; keys are read from `.env` at the repo root (see `.env.example`). `_LLM_PROVIDER_OVERRIDE` is a module-level test seam — tests monkeypatch it with a `MockLLMProvider` to avoid any real API calls.

### Closed-list constants

Defined in `grader/__init__.py`. Updating these (adding a category, renaming a severity) requires touching multiple places: the constants, the proximity tables in `scorers.py`, the severity matrix and weight in `report.py`, and the corresponding tests. There are 3 severities, 7 categories + Other, and 7 security-concerns + Other.

### Backend selection tradeoffs

Jaccard is intentionally rough — some obvious matches (different vocabulary for the same root cause) fall below 0.3. The upgrade path is `--backend llm-judge`, which is now built. Reach for the LLM backend when matching quality matters; keep Jaccard for CI / fast local iteration. The `same_root_cause` and `reasoning` fields the judge returns are collected (see `LLMJudgeSimilarity.last_result_for`) but are not yet wired into the report output — that's the next iteration.
