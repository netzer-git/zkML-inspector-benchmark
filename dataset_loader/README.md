# dataset_loader

Downloads and materializes (paper, codebase) pairs from the
[`Anonymous648/zkml-audit-benchmark`](https://huggingface.co/datasets/Anonymous648/zkml-audit-benchmark)
Hugging Face dataset for agent auditing runs.

## Installation

```bash
pip install -e .          # installs huggingface_hub and other deps
```

## CLI

```bash
# List available pairs
python -m dataset_loader list-pairs

# List artifacts (optionally filtered by pair)
python -m dataset_loader list-artifacts --pair zkllm

# Materialize a run-set directory
python -m dataset_loader materialize --output ./run_set
python -m dataset_loader materialize --output ./run_set --pairs zkllm,zkml
```

The `materialize` command downloads papers and codebases, extracts them to a
flat layout, and emits a `batch_manifest.json` compatible with
an external audit agent's batch-analyze workflow:

```
run_set/
  zkllm/
    paper.pdf
    codebase/
  zkml/
    paper.pdf
    codebase/
  batch_manifest.json
```

## Python API

```python
from dataset_loader import BenchmarkDataset
from dataset_loader.materialize import materialize

ds = BenchmarkDataset()        # defaults to Anonymous648/zkml-audit-benchmark
ds.pair_ids()                  # ['zkgpt', 'zkllm', 'zkml', 'zktorch']
ds.artifact_ids(pair_id="zkllm")  # ['zkLLM-001', ..., 'zkLLM-014']

# Download individual files (cached by huggingface_hub)
paper = ds.paper_path("zkllm")        # -> Path to local cached PDF
art = ds.load_artifact_json("zkLLM-001")  # -> dict

# Materialize full run-set
from pathlib import Path
materialize(ds, Path("./run_set"), pair_ids=["zkllm"])
```
