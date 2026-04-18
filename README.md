# zkML-inspector-benchmark

A benchmark suite for evaluating ZK-ML audit agents on (paper, codebase) pairs.

The benchmark has three components that together let users assemble an evaluation set, run their agent, and grade the output:

```
zkML-inspector-benchmark/
  dataset_generator/   # Build the ground-truth audit findings dataset
  dataset_loader/      # Materialize a (paper, codebase) set for an agent run
  grader/              # Score agent output against the ground truth
```

## Workflow

1. **Generate / curate the dataset.** Use `dataset_generator/` to produce the ground-truth audit-finding dataset (xlsx).
2. **Load a run set.** Use `dataset_loader/` to fetch the paper PDFs and codebases referenced by the dataset and lay them out for the agent.
3. **Run your agent.** Your agent reads each (paper, codebase) pair and produces a finding JSON. The expected format mirrors the dataset columns — see [`examples/agent_output.example.json`](examples/agent_output.example.json) for a complete reference file and [`examples/README.md`](examples/README.md) for the full schema including all closed-list values:
   ```json
   [
     {
       "entry-id": "<project-key>",
       "issue-name": "...",
       "issue-explanation": "...",
       "severity": "Critical|Warning|Info",
       "category": "...",
       "security-concern": "...",
       "relevant-code": "file.rs:10-20, other.cu:3",
       "paper-reference": "Section 6.1.3: ..."
     }
   ]
   ```
4. **Grade.** Run `grader/` against the ground-truth xlsx and the agent JSON to get per-project and overall scores.

## Components

### `grader/`

Compares an agent's audit findings against a ground-truth dataset and produces JSON + markdown reports.

```bash
python -m grader \
    --ground-truth zkMLDataset.xlsx \
    --agent-output agent_results.json \
    --output grade_report.json \
    --output-md grade_report.md
```

Scores 5 fields per matched finding (severity, category, security-concern, relevant-code, paper-reference) plus precision/recall/F1 across the matched set. Extra unmatched findings are reported broken down by severity. Finding matching uses a pluggable similarity backend (Jaccard baseline; TF-IDF / embeddings / LLM-as-judge can plug in via the `SimilarityBackend` interface).

See `grader/` for full module docs and tests under `grader/tests/`.

#### LLM-as-judge backend

The matcher can use an LLM to decide whether an agent finding describes the same root cause as a ground-truth finding, instead of word-overlap similarity. One LLM call is made per agent finding per project (not per pair), so cost scales linearly with agent findings.

```bash
# Install provider extras (pick one, or use `.[llm]` for both)
pip install -e .[similarity-openai]     # or .[similarity-anthropic]

# Configure keys
cp .env.example .env
# edit .env: set OPENAI_API_KEY (or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic)

# Run
python -m grader \
    --ground-truth zkMLDataset.xlsx \
    --agent-output agent_results.json \
    --backend llm-judge \
    --output grade_report.json
```

Defaults: `OPENAI_MODEL=gpt-4o`, `ANTHROPIC_MODEL=claude-opus-4-5`. All env vars are documented in `.env.example`.

### `dataset_loader/` *(planned)*

Given a dataset xlsx, materialize the corresponding (paper PDF, codebase) pairs into a runnable directory layout for the agent. Handles fetching from configured sources (local paths, URLs, git repos).

### `dataset_generator/` *(planned)*

Tools for assembling new ground-truth audit-finding datasets: schema validation, manual curation helpers, and pipelines for proposing candidate findings from existing audit reports.

## Dataset schema

Each finding in the ground-truth xlsx has these columns:

| Column | Type | Notes |
|--------|------|-------|
| entry-id | str | Project identifier (e.g. `zkLLM`) |
| issue-id | str | `<entry-id>-NN` |
| issue-name | str | 3-7 words |
| issue-explanation | str | One paragraph |
| severity | enum | `Critical` / `Warning` / `Info` |
| category | enum | 7 options + `Other` (see `grader/__init__.py`) |
| security-concern | enum | 6 options + `Other` |
| relevant-code | str | `file:line[-line], file:line` (can be empty) |
| paper-reference | str | Section / Protocol / Theorem cite + quote (can be empty) |

## Development

```bash
# Run grader tests
python -m pytest grader/tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
