"""Revert artifact edits from a modified codebase using clean file snapshots.

Given a modified codebase directory, a set of artifact IDs to revert, and
access to the clean codebase, this module restores any files touched by those
artifacts to their clean state, then re-applies the remaining artifacts.

The approach:
1. Load the clean codebase from disk (or from Sources).
2. For each artifact to revert, collect the set of files it edits.
3. Replace those files in the modified codebase with their clean versions.
4. Re-apply the remaining (non-reverted) artifacts' edits to handle
   any artifacts that share files with the reverted ones.

For the composition experiment's use case (condition D), this provides
oracle-quality reversion that is independent of agent patch quality.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dataset_generator.artifacts import Artifact
from dataset_generator.assembler import build_case, BuiltCase


def revert_artifacts(
    *,
    clean_codebase_dir: Path,
    paper_path: Path | None,
    all_artifacts: list[Artifact],
    revert_ids: set[str],
    entry_id: str,
    codebase_name: str,
    output_dir: Path,
) -> BuiltCase:
    """Build a new case with a subset of artifacts removed.

    Instead of trying to surgically undo individual edits (fragile when
    multiple artifacts share files and interact via line-offset tracking),
    this rebuilds from the clean codebase applying only the remaining
    artifacts.

    Parameters
    ----------
    clean_codebase_dir
        Path to the unmodified codebase directory.
    paper_path
        Path to the paper PDF (copied into the case directory).
    all_artifacts
        The full list of artifacts that were applied in the original case.
    revert_ids
        Set of artifact IDs to remove.
    entry_id
        Entry ID for the output case.
    codebase_name
        Codebase name (e.g. "zkgpt").
    output_dir
        Parent directory for the case output (case_dir = output_dir/entry_id).

    Returns
    -------
    BuiltCase
        The newly built case with only the surviving artifacts applied.

    Raises
    ------
    ValueError
        If any revert_id is not in all_artifacts.
    """
    known_ids = {a.artifact_id for a in all_artifacts}
    unknown = revert_ids - known_ids
    if unknown:
        raise ValueError(f"Cannot revert unknown artifact IDs: {unknown}")

    surviving = [a for a in all_artifacts if a.artifact_id not in revert_ids]

    # Clean up any prior output for this entry
    case_dir = output_dir / entry_id
    if case_dir.exists():
        shutil.rmtree(case_dir)

    if not surviving:
        # All reverted → return a clean codebase case with no artifacts
        case_codebase = case_dir / "codebase"
        shutil.copytree(clean_codebase_dir, case_codebase)
        if paper_path and paper_path.exists():
            shutil.copy2(paper_path, case_dir / "paper.pdf")
        return BuiltCase(
            entry_id=entry_id,
            source_codebase=codebase_name,
            artifact_ids=[],
            case_dir=case_dir,
            paper_path=case_dir / "paper.pdf",
        )

    # Rebuild from clean with only the surviving artifacts
    return build_case(
        entry_id=entry_id,
        codebase_dir=clean_codebase_dir,
        codebase_name=codebase_name,
        paper_path=paper_path,
        artifacts=surviving,
        output_dir=output_dir,
    )
