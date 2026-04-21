# dataset_generator

Generates benchmark cases by applying strict-v2 bug artifacts to fixed codebases and emitting ground-truth findings JSON.

## Usage

```bash
python -m dataset_generator test \
  --sources ./sources/ \
  --output ./dataset/ \
  --num-cases 2 \
  --artifacts-per-case 3 \
  --strategy random \
  --seed 42
```

## Sources layout (placeholder — subject to change)

```
sources/
  sources.json            # [{entry-id, paper, codebase_zip, codebase_name}]
  papers/*.pdf
  codebases/*.zip
  artifacts/<codebase_name>/*.json   # strict v2 artifacts
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
