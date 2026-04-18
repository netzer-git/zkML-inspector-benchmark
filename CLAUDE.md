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

`grader/tests/test_loader.py` and `grader/tests/test_cli.py` look for `zkMLDataset.xlsx` at the repo root via `Path(__file__).parent.parent.parent / "zkMLDataset.xlsx"`. The xlsx is **not** committed — those tests skip cleanly when it's absent. When the dataset is present, expect 153 tests to pass; without it, 137 pass and 16 skip. `grader/tests/test_agent_output.json` is a hand-crafted agent output used by the CLI integration tests.

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

- **`grader/similarity.py`** — `SimilarityBackend` ABC plus a `JaccardSimilarity` baseline. The matcher and the paper-reference scorer both depend on this interface, so swapping in TF-IDF, embeddings, or LLM-as-judge backends requires no changes elsewhere. Optional dependency groups in `pyproject.toml` (`similarity-tfidf`, `similarity-embedding`, `similarity-llm`) are placeholders for those future backends.

- **`grader/matcher.py`** — operates **per-project** (never matches findings across entry-ids). Builds an `M×N` similarity matrix on `name + " " + explanation`, then does greedy 1:1 assignment above a threshold (default 0.3). Issue-ids are GT-only labels — agents produce findings in arbitrary order with no ID correspondence. Unmatched agent findings are exposed via `extra_by_severity` because hallucinated Critical findings are qualitatively different from extra Info suggestions.

- **`grader/scorers.py`** — one function per graded field, each returning `FieldScore(score, detail)`:
  - **Severity** uses an asymmetric 3×3 matrix where under-reporting (Warning when GT is Critical) scores 0.0 but over-reporting gets partial credit (0.25 for one level, 0.1 for two). This is deliberate — missing severity is the worst failure.
  - **Category** and **security-concern** support partial credit via a small proximity table for semantically adjacent values (e.g., "Under-constrained Circuit" ↔ "Witness/Commitment Mismatch" → 0.4).
  - **Code location** does basename matching (so `verifier.rs:38` matches `src/util/verifier.rs:36-42`) and grades by line-range proximity (overlap → 1.0, within 30 lines → 0.7, etc.). Returns 1.0 with a "skip" detail when GT has no code refs.
  - **Paper reference** averages two sub-scores: section-ID extraction via regex (matches `Section X.Y.Z`, `Protocol N`, `Theorem N`, `Eq. N`, `Example N`) and quote similarity through the pluggable backend. Same skip semantics as code location when GT is `-` or empty.

- **`grader/report.py`** — `_compute_pair_score` does weighted averaging of the field scores with **automatic redistribution of skipped fields' weights** to the remaining active ones (this is critical — don't change without updating tests). Default weights: code 0.30, paper 0.25, severity/category/security-concern 0.15 each. Aggregates into per-project metrics (precision, recall, F1, severity-weighted recall, quality, composite) and an overall `benchmark_score = 0.4 * f1 + 0.6 * quality`.

- **`grader/cli.py` / `__main__.py`** — argparse orchestration. `_parse_weights` accepts `key=value` overrides. Projects with no GT are skipped with a printed notice. Weights and threshold are written into the report's `meta` block for reproducibility.

### Closed-list constants

Defined in `grader/__init__.py`. Updating these (adding a category, renaming a severity) requires touching multiple places: the constants, the proximity tables in `scorers.py`, the severity matrix and weight in `report.py`, and the corresponding tests. There are 3 severities, 7 categories + Other, and 7 security-concerns + Other.

### Known limitation: matching threshold

The Jaccard baseline is intentionally rough. Some obvious matches (e.g., "Lasso not implemented" vs "Lasso Lookup Not Implemented") fall below 0.3 because the texts use different vocabulary. The pluggable `SimilarityBackend` interface is the upgrade path — don't try to fix matching by tweaking the Jaccard scorer or threshold; add a real backend instead.
