# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo purpose

Benchmark suite for ZK-ML audit agents. Three components work together end-to-end: `dataset_loader/` downloads papers, codebases, and artifacts from the [`Netzerep/zkml-audit-benchmark`](https://huggingface.co/datasets/Netzerep/zkml-audit-benchmark) HF dataset and materializes (paper PDF, codebase) pairs for agent runs; `dataset_generator/` produces benchmark cases by applying bug artifacts to the clean codebases and emitting ground-truth findings JSON; and `grader/` scores the agent's JSON output against the ground truth.

## Common commands

```bash
# Run all tests
python -m pytest grader/tests/ -v

# Run a single test file / class / test
python -m pytest grader/tests/test_scorers.py -v
python -m pytest grader/tests/test_scorers.py::TestScoreSeverity -v
python -m pytest grader/tests/test_scorers.py::TestScoreSeverity::test_exact_critical -v

# Run the grader end-to-end (requires .env with an API key — see .env.example)
python -m grader \
    --ground-truth findings.json \
    --agent-output agent_results.json \
    --output grade_report.json \
    --output-md grade_report.md

# Materialize a run-set from HF for agent auditing
python -m dataset_loader materialize --output ./run_set
python -m dataset_loader materialize --output ./run_set --pairs zkllm,zkml

# List available pairs / artifacts
python -m dataset_loader list-pairs
python -m dataset_loader list-artifacts --pair zkllm

# Generate benchmark test cases (downloads sources from HF)
python -m dataset_generator test --output ./dataset/ --num-cases 2 --artifacts-per-case 3
```

`pyproject.toml` exposes three console scripts after `pip install -e .`: `zkml-grader`, `zkml-dataset-gen`, and `zkml-loader`. The grader makes real LLM calls (OpenAI by default) to match agent findings against ground truth — tests use `MockLLMProvider` to avoid any API traffic. The loader and generator use `huggingface_hub` to download dataset files (cached locally after first download).

## Test fixtures

`grader/tests/conftest.py` builds fictional ground-truth (JSON) and agent-output (JSON) fixtures at session start using synthetic project names (`alpha`, `beta`). **No real dataset content is ever in tests** — deliberately, to avoid leaking ground-truth to anything that reads the test suite. All tests are self-contained; none skip. LLM tests use `MockLLMProvider` from `grader.llm` (no API is ever contacted).

## Example agent output

`examples/agent_output.example.json` is the canonical reference file for agents producing output for the grader, with a companion `examples/README.md` documenting the full schema. The findings use fictional project names (`zkFoo`, `zkBar`) — **do not** add real dataset content to these examples (avoids leaking ground-truth content to agents).

## Architecture

### Grader pipeline

```
GT JSON     ─┐
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

- **`grader/loader.py`** — parses the ground-truth JSON and the agent JSON. Both are flat JSON arrays; each finding must include all 7 fields (severity, category, security-concern, relevant-code, paper-reference, issue-name, issue-explanation). Ground-truth findings also carry an `issue-id` field (auto-generated if absent). `parse_code_refs` handles the `file:line[-line], file:line` format including unicode en-dashes. Entry IDs are normalized to lowercase as the grouping key. Validation is strict: invalid closed-list values raise `ValueError`.

- **`grader/similarity.py`** — `SimilarityBackend` ABC plus one production backend: `LLMJudgeSimilarity`. Exposes `judge_bulk(agent_text, candidates)` — the primary matching API, one LLM call returns a ranked list of per-candidate judgments `{gt_id, match_score, same_root_cause, reasoning}` — and `score(a, b)` for single-pair text similarity (used by paper-reference quote scoring). `JUDGE_SCHEMA` is strict and used verbatim by both OpenAI `response_format=json_schema` and Anthropic tool-use. Results are cached in-memory by SHA-256 of agent + sorted candidates. The richer per-pair judgment (beyond `match_score`) is reachable via `last_result_for(agent_text, gt_id)` for future report enrichment — it is collected today but not yet written into the report.

- **`grader/llm.py`** — `LLMProvider` ABC with `OpenAIProvider`, `AnthropicProvider`, `MockLLMProvider`. Provider SDK imports are lazy (inside `__init__`) for a cleaner import-error message when a specific provider is missing. `build_config_from_env()` reads `LLM_PROVIDER` (default `openai`), `OPENAI_API_KEY` / `OPENAI_MODEL` (default `gpt-4o`), `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` (default `claude-opus-4-5`). `.env` loading via `python-dotenv` is required at runtime; both SDKs are required dependencies in `pyproject.toml`.

- **`grader/matcher.py`** — operates **per-project** (never matches findings across entry-ids). Builds the judge text from each finding via `_build_matching_text` (`issue-name + issue-explanation + paper-reference` — code refs and closed-list fields are intentionally excluded), issues one `judge_bulk` call per agent finding, and applies an **AND gate**: a pair is considered a match only when `match_score >= threshold` AND `same_root_cause == True`. Greedy 1:1 assignment on the surviving triples resolves conflicts in favor of higher-scored pairs. Candidate IDs passed to the judge are real GT `issue_id`s so the judge's reasoning remains traceable. Unmatched agent findings are exposed via `extra_by_severity` because hallucinated Critical findings are qualitatively different from extra Info suggestions.

- **`grader/scorers.py`** — one function per graded field, each returning `FieldScore(score, detail)`:
  - **Severity** uses an asymmetric 3×3 matrix where under-reporting (Warning when GT is Critical) scores 0.0 but over-reporting gets partial credit (0.25 for one level, 0.1 for two). This is deliberate — missing severity is the worst failure.
  - **Category** and **security-concern** support partial credit via a proximity table for semantically adjacent values. `_CATEGORY_PROXIMITY` in `scorers.py` covers three tiers: **0.4** for near-synonyms ("Under-constrained Circuit" ↔ "Witness/Commitment Mismatch", "Under-constrained Circuit" ↔ "Specification Mismatch"); **0.3** for close cousins ("Engineering/Prototype Gap" ↔ "Specification Mismatch", "Protocol/Transcript Logic" ↔ "Witness/Commitment Mismatch", "Protocol/Transcript Logic" ↔ "Engineering/Prototype Gap"); **0.2** for weaker overlaps ("Under-constrained Circuit" ↔ "Engineering/Prototype Gap", "Numerical/Quantization Bug" ↔ "Engineering/Prototype Gap"). Security-concern uses the same structure with its own pairs. Unlisted mismatches score **0.0**; "Other" on either side gives **0.1** for security-concern (category has no "Other" shortcut). When updating either table, update the tests that assert specific scores.
  - **Code location** does basename matching (so `verifier.rs:38` matches `src/util/verifier.rs:36-42`) and grades by line-range proximity (overlap → 1.0, within 30 lines → 0.7, etc.). Returns 1.0 with a "skip" detail when GT has no code refs.
  - **Paper reference** averages two sub-scores: section-ID extraction via regex (matches `Section X.Y.Z`, `Protocol N`, `Theorem N`, `Eq. N`, `Example N`) and quote similarity through the LLM judge (or, in tests, a word-overlap stub). Same skip semantics as code location when GT is `-` or empty.

- **`grader/report.py`** — `_compute_pair_score` does weighted averaging of the field scores with **automatic redistribution of skipped fields' weights** to the remaining active ones (this is critical — don't change without updating tests). Default weights: code 0.30, paper 0.25, severity/category/security-concern 0.15 each. Aggregates into per-project metrics (precision, recall, F1, severity-weighted recall, quality, composite) and an overall `benchmark_score = 0.4 * f1 + 0.6 * quality`.

- **`grader/cli.py` / `__main__.py`** — argparse orchestration. `_parse_weights` accepts `key=value` overrides. Projects with no GT are skipped with a printed notice. Weights and threshold are written into the report's `meta` block for reproducibility. Backend construction (`_build_backend`) reads keys from `.env` (see `.env.example`) and raises `ValueError` with a pointer to `.env.example` if configuration is missing. `_LLM_PROVIDER_OVERRIDE` is a module-level test seam — tests monkeypatch it with a `MockLLMProvider` to avoid any real API calls.

### Closed-list constants

Defined in `grader/__init__.py`. Updating these (adding a category, renaming a severity) requires touching multiple places: the constants, the proximity tables in `scorers.py`, the severity matrix and weight in `report.py`, and the corresponding tests. There are 3 severities, 7 categories + Other, and 7 security-concerns + Other.

### Matching decisions

- **What the judge sees per finding**: `issue-name`, `issue-explanation`, and `paper-reference`. Code refs are excluded because the judge doesn't have the codebase loaded and a file/line alone is not self-sufficient evidence. The three closed-list fields (severity, category, security-concern) are excluded because they are graded independently by `scorers.py`; feeding them to the matcher would produce a circular signal.
- **Why the AND gate**: the judge returns both a numeric `match_score` and a boolean `same_root_cause`. We require both to agree before declaring a match. This trades some recall for precision — a pair with high score but `same_root_cause=False` (model picked up keyword overlap but recognized the components differ) is correctly rejected.
- **Why bulk, not pairwise**: the matcher issues one `judge_bulk` call per agent finding, passing all GT findings in that project as candidates. The LLM ranks comparatively in one call, which is both cheaper and usually more accurate than M×N isolated pair comparisons.
- **Future report enrichment**: the judge's per-pair `same_root_cause` and `reasoning` are accumulated inside `LLMJudgeSimilarity` (see `last_result_for()`) but are not yet written into the JSON/markdown report. Wiring them in is the next iteration.
