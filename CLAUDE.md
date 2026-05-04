# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo purpose

Benchmark suite for ZK-ML audit agents. Three components work together end-to-end: `dataset_loader/` downloads papers, codebases, and artifacts from the [`Anonymous648/zkml-audit-benchmark`](https://huggingface.co/datasets/Anonymous648/zkml-audit-benchmark) HF dataset and materializes (paper PDF, codebase) pairs for agent runs; `dataset_generator/` produces benchmark cases by applying bug artifacts to the clean codebases and emitting ground-truth findings JSON; and `grader/` scores the agent's JSON output against the ground truth.

## Common commands

```bash
# Run all tests
python -m pytest grader/tests/ -v

# Run a single test file / class / test
python -m pytest grader/tests/test_scorers.py -v
python -m pytest grader/tests/test_scorers.py::TestScoreCodeLocation -v
python -m pytest grader/tests/test_scorers.py::TestScoreCodeLocation::test_overlap -v

# Run the grader end-to-end (requires .env with an API key — see .env.example)
python -m grader \
    --ground-truth findings.json \
    --agent-output agent_results.json \
    --output grade_report.json \
    --output-md grade_report.md

# Run the grader with baseline exclusion (removes known pre-existing issues)
python -m grader \
    --ground-truth findings.json \
    --agent-output agent_results.json \
    --baseline baseline_findings.json \
    --output grade_report.json

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

- **`grader/loader.py`** — parses the ground-truth JSON and the agent JSON. Both are flat JSON arrays; each finding must include 5 required fields (entry-id, issue-name, issue-explanation, relevant-code, paper-reference). Ground-truth findings also carry an `issue-id` field (auto-generated if absent). `parse_code_refs` handles the `file:line[-line], file:line` format including unicode en-dashes. Entry IDs are normalized to lowercase as the grouping key.

- **`grader/similarity.py`** — `SimilarityBackend` ABC plus one production backend: `LLMJudgeSimilarity`. Exposes `judge_bulk(agent_text, candidates)` — the primary matching API, one LLM call returns a ranked list of per-candidate judgments `{gt_id, match_score, reasoning}` — and `score(a, b)` for single-pair text similarity (used by paper-reference quote scoring). `match_score` is an ordinal 1..5 scale. `JUDGE_SCHEMA` is strict and used verbatim by both OpenAI `response_format=json_schema` and Anthropic tool-use. Results are cached in-memory by SHA-256 of agent + sorted candidates.

- **`grader/llm.py`** — `LLMProvider` ABC with `OpenAIProvider`, `AnthropicProvider`, `MockLLMProvider`. Provider SDK imports are lazy (inside `__init__`) for a cleaner import-error message when a specific provider is missing. `build_config_from_env()` reads `LLM_PROVIDER` (default `openai`), `OPENAI_API_KEY` / `OPENAI_MODEL` (default `gpt-4o`), `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` (default `claude-opus-4-5`). `.env` loading via `python-dotenv` is required at runtime; both SDKs are required dependencies in `pyproject.toml`.

- **`grader/matcher.py`** — operates **per-project** (never matches findings across entry-ids). Builds the judge text from each finding via `_build_matching_text` (`issue-name + issue-explanation + paper-reference` — code refs are intentionally excluded), issues one `judge_bulk` call per agent finding, and accepts a match when `match_score >= threshold` (default 4, ordinal 1..5). Greedy assignment resolves conflicts in favor of higher-scored pairs; GTs can absorb multiple agents (N:1, flagged with `dup_rank`). Candidate IDs passed to the judge are real GT `issue_id`s so the judge's reasoning remains traceable.

- **`grader/scorers.py`** — two field scorers, each returning `FieldScore(score, detail)`:
  - **Code location** does basename matching (so `verifier.rs:38` matches `src/util/verifier.rs:36-42`) and grades by line-range proximity (overlap → 1.0, within 2 lines → 1.0, within 30 → 0.7, within 100 → 0.4, else 0.2). Returns 1.0 with a "skip" detail when GT has no code refs.
  - **Paper reference** averages two sub-scores: section-ID extraction via regex (matches `Section X.Y.Z`, `Protocol N`, `Theorem N`, `Eq. N`, `Example N`, `Figure N`, `Appendix`, `Definition`, `Step`, `Line`) and quote similarity through the LLM judge (or, in tests, a word-overlap stub). Same skip semantics as code location when GT is `-` or empty.

- **`grader/report.py`** — uses a **quality-gate model**: each matched pair gets a quality score via `QUALITY_WEIGHTS` (50% match_score, 30% code_location, 20% paper_reference) and a pass/fail gate at `QUALITY_THRESHOLD` (default 0.55). `grade_pair()` scores code location and paper reference, computes quality, and sets `passed = quality >= threshold`. `grade_project()` computes recall (unique GTs with ≥1 passing match), precision (passed agent matches / total agent), F1, and avg_quality. `build_report()` aggregates all projects into overall metrics. Includes serialization helpers (`_grade_to_dict`, `_dict_to_project_grade`) for checkpoint save/load. Markdown output uses quality badges (🟢/🟡/🔴).

- **`grader/cli.py` / `__main__.py`** — argparse orchestration. `--quality-threshold` overrides the pass/fail gate. `--checkpoint` enables incremental JSONL saves (one project per line, appended after grading each project; resumes from existing checkpoint on restart). Projects with no GT are skipped with a printed notice. Threshold and quality-threshold are written into the report's `meta` block for reproducibility. Backend construction (`_build_backend`) reads keys from `.env` (see `.env.example`) and raises `ValueError` with a pointer to `.env.example` if configuration is missing. `_LLM_PROVIDER_OVERRIDE` is a module-level test seam — tests monkeypatch it with a `MockLLMProvider` to avoid any real API calls.

### Matching decisions

- **What the judge sees per finding**: `issue-name`, `issue-explanation`, and `paper-reference`. Code refs are excluded because the judge doesn't have the codebase loaded and a file/line alone is not self-sufficient evidence.
- **Match threshold**: the judge returns an ordinal `match_score` (1..5). A match is accepted when `match_score >= threshold` (default 4).
- **Why bulk, not pairwise**: the matcher issues one `judge_bulk` call per agent finding, passing all GT findings in that project as candidates. The LLM ranks comparatively in one call, which is both cheaper and usually more accurate than M×N isolated pair comparisons.
