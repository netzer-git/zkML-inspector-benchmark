# dataset_generator

Generates benchmark cases by applying strict-v2 bug artifacts to fixed codebases and emitting ground-truth findings JSON.

Sources (papers, codebases, artifacts) are loaded from the
[`Anonymous648/zkml-audit-benchmark`](https://huggingface.co/datasets/Anonymous648/zkml-audit-benchmark)
Hugging Face dataset via `dataset_loader`.

## Usage

```bash
python -m dataset_generator test \
  --output ./dataset/ \
  --num-cases 2 \
  --artifacts-per-case 3 \
  --strategy random \
  --seed 42

# Override the HF repo (for local forks / testing):
python -m dataset_generator test \
  --repo-id myuser/zkml-audit-benchmark \
  --output ./dataset/
```

## Output

```
output/
  dataset_manifest.json   # case metadata (strategy, seed, case list)
  findings.json           # aggregated ground-truth (flat array, grader-compatible)
  cases/<entry-id>/
    paper.pdf
    codebase/             # modified codebase with injected bugs
    case.json             # per-case metadata (artifact_ids, source_codebase)
```
