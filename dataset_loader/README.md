# dataset_loader (planned)

Materializes a runnable (paper, codebase) layout from a ground-truth dataset, so an agent can be pointed at the prepared directory and produce findings.

## Planned interface

```bash
python -m dataset_loader \
    --dataset zkMLDataset.xlsx \
    --manifest sources.json \
    --output ./run_set/
```

Where `sources.json` maps each `entry-id` to its paper PDF location and codebase source (local path, URL, or git ref). The output directory has the layout expected by an agent run:

```
run_set/
  zkLLM/
    paper.pdf
    codebase/
  zkGPT/
    ...
```

## Status

Not yet implemented.
