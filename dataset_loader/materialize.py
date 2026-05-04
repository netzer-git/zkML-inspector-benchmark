"""Materialize a run-set directory from the HF benchmark dataset.

The output layout is directly consumable by an audit agent's
batch-analyze workflow.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from dataset_loader import BenchmarkDataset


def materialize(
    dataset: BenchmarkDataset,
    output_dir: Path,
    pair_ids: list[str] | None = None,
    emit_batch_manifest: bool = True,
) -> Path:
    """Download papers + codebases and lay them out for an agent run.

    Creates::

        output_dir/
          {pair_id}/
            paper.pdf
            codebase/
              ...
          batch_manifest.json   (if *emit_batch_manifest*)

    Parameters
    ----------
    dataset:
        :class:`BenchmarkDataset` instance (already configured with repo ID).
    output_dir:
        Root directory for the materialized run-set.
    pair_ids:
        Subset of pairs to materialize.  ``None`` means all pairs.
    emit_batch_manifest:
        If ``True`` (default), write a ``batch_manifest.json`` compatible
        with an audit agent's batch-analyze workflow.

    Returns
    -------
    Path
        *output_dir* (same value passed in), for convenience.
    """
    if pair_ids is None:
        pair_ids = dataset.pair_ids()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses: list[dict[str, str]] = []

    for pid in pair_ids:
        pair_dir = output_dir / pid
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Paper
        paper_src = dataset.paper_path(pid)
        paper_dst = pair_dir / "paper.pdf"
        shutil.copy2(paper_src, paper_dst)

        # Codebase
        codebase_dst = pair_dir / "codebase"
        if codebase_dst.exists():
            shutil.rmtree(codebase_dst)

        # extract_codebase writes to dest/{pair_id}/, we want dest/codebase/
        # So we extract to a temp sibling then rename.
        tmp_extract = pair_dir / "_codebase_extract"
        dataset.extract_codebase(pid, tmp_extract)

        # extract_codebase creates tmp_extract/{pair_id}/...; we want
        # the inner content under codebase_dst directly.
        inner = tmp_extract / pid
        if inner.is_dir():
            inner.rename(codebase_dst)
            shutil.rmtree(tmp_extract, ignore_errors=True)
        else:
            # Fallback: the zip extracted directly into tmp_extract
            tmp_extract.rename(codebase_dst)

        analyses.append({
            "entry-id": pid,
            "paper": f"./{pid}/paper.pdf",
            "codebase": f"./{pid}/codebase/",
        })

    if emit_batch_manifest:
        manifest = {"analyses": analyses}
        manifest_path = output_dir / "batch_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    return output_dir
