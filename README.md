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

1. **Generate / curate the dataset.** Use `dataset_generator/` to produce benchmark cases and the ground-truth findings JSON from bug artifacts.
2. **Load a run set.** Use `dataset_loader/` to fetch the paper PDFs and codebases referenced by the dataset and lay them out for the agent.
3. **Run your agent.** Your agent reads each (paper, codebase) pair and produces a finding JSON. The expected format mirrors the dataset columns — see [`examples/agent_output.example.json`](examples/agent_output.example.json) for a complete reference file and [`examples/README.md`](examples/README.md) for the full schema:
   ```json
   [
     {
       "entry-id": "<project-key>",
       "issue-name": "...",
       "issue-explanation": "...",
       "relevant-code": "file.rs:10-20, other.cu:3",
       "paper-reference": "Section 6.1.3: ..."
     }
   ]
   ```
   (Agents produce 5 fields. `issue-id` is a ground-truth-only label used by the dataset for cross-referencing.)
4. **Grade.** Run `grader/` against the ground-truth JSON and the agent JSON to get per-project and overall scores.

## Components

### `grader/`

Compares an agent's audit findings against a ground-truth dataset and produces JSON + markdown reports. Matching uses an LLM judge (OpenAI or Anthropic) that decides whether an agent finding describes the same root cause as a ground-truth finding. One LLM call is made per agent finding per project.

```bash
# Install
pip install -e .

# Configure keys
cp .env.example .env
# edit .env: set OPENAI_API_KEY (or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic)

# Run
python -m grader \
    --ground-truth findings.json \
    --agent-output agent_results.json \
    --output grade_report.json \
    --output-md grade_report.md \
    --checkpoint checkpoint.jsonl
```

Uses a quality-gate model: each matched pair is scored via `quality = 0.5*(match_score/5) + 0.3*code_location + 0.2*paper_reference` with a pass/fail threshold (default 0.55). Reports recall, precision, F1, and average quality per project and overall. Supports `--checkpoint` for incremental saves (JSONL, one project per line) with automatic resume on restart.

Defaults: `OPENAI_MODEL=gpt-4o`, `ANTHROPIC_MODEL=claude-opus-4-5`. All env vars are documented in `.env.example`. See `grader/` for module docs and tests under `grader/tests/`.

### `dataset_loader/`

Given a dataset, materializes the corresponding (paper PDF, codebase) pairs into a runnable directory layout for the agent. Fetches from the HuggingFace dataset repo.

### `dataset_generator/`

Produces benchmark cases by applying bug artifacts to clean codebases and emitting ground-truth findings JSON. Supports multiple selection strategies (random, all, isolated, fixed-subset) and an iterative runner for composition experiments.

## Dataset schema

Each finding in the ground-truth JSON has these fields:

| Column | Type | Notes |
|--------|------|-------|
| entry-id | str | Project identifier (e.g. `zkLLM`) |
| issue-id | str | `<entry-id>-NN` — GT-only; agents do not produce this. |
| issue-name | str | 3-7 words |
| issue-explanation | str | One paragraph |
| relevant-code | str | `file:line[-line], file:line` (can be empty) |
| paper-reference | str | Section / Protocol / Theorem cite + quote (can be empty) |

## Development

```bash
# Run grader tests
python -m pytest grader/tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
