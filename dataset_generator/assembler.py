"""Case assembler — builds individual benchmark cases from sources + artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset_generator.artifacts import Artifact
from dataset_generator.conflict import topological_sort
from dataset_generator.edits import Codebase, EditError, apply_artifact_edits
from dataset_generator.probes import ProbeFailure, evaluate_probes


class CaseBuildError(Exception):
    """Raised when building a case fails."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.details = details or []


@dataclass
class BuiltCase:
    """Result of successfully building a case."""

    entry_id: str
    source_codebase: str
    artifact_ids: list[str]
    case_dir: Path
    paper_path: Path


def _load_codebase_from_disk(codebase_dir: Path) -> Codebase:
    """Load all text files from a directory into an in-memory Codebase."""
    codebase: Codebase = {}
    for file_path in sorted(codebase_dir.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(codebase_dir).as_posix()
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                codebase[rel] = content.split("\n")
            except (UnicodeDecodeError, PermissionError):
                # Skip binary or inaccessible files
                pass
    return codebase


def _write_codebase_to_disk(codebase: Codebase, dest: Path) -> None:
    """Write an in-memory Codebase to disk."""
    for rel_path, lines in codebase.items():
        out_path = dest / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")


def build_case(
    entry_id: str,
    codebase_dir: Path,
    codebase_name: str,
    paper_path: Path | None,
    artifacts: list[Artifact],
    output_dir: Path,
) -> BuiltCase:
    """Build a single benchmark case.

    1. Load codebase into memory
    2. Topologically sort artifacts by requires
    3. Apply edits atomically for each artifact
    4. Run all presence probes
    5. Write modified codebase + paper + case.json

    Raises CaseBuildError on any failure (with cleanup).
    """
    case_dir = output_dir / entry_id
    case_codebase_dir = case_dir / "codebase"
    case_paper_path = case_dir / "paper.pdf"

    try:
        # Load codebase
        codebase = _load_codebase_from_disk(codebase_dir)

        # Sort artifacts
        sorted_artifacts = topological_sort(artifacts)

        # Apply edits
        for artifact in sorted_artifacts:
            try:
                apply_artifact_edits(
                    codebase, artifact.edits, artifact.artifact_id
                )
            except EditError as e:
                raise CaseBuildError(
                    f"Edit failed for {artifact.artifact_id}: {e}",
                    [{"artifact_id": artifact.artifact_id, "error": str(e)}],
                ) from e

        # Run probes
        all_failures: list[ProbeFailure] = []
        for artifact in sorted_artifacts:
            failures = evaluate_probes(
                codebase, artifact.presence_probes, artifact.artifact_id
            )
            all_failures.extend(failures)

        if all_failures:
            details = [
                {
                    "artifact_id": f.artifact_id,
                    "probe_index": f.probe_index,
                    "kind": f.kind,
                    "file": f.file,
                    "detail": f.detail,
                }
                for f in all_failures
            ]
            raise CaseBuildError(
                f"{len(all_failures)} probe(s) failed for case '{entry_id}'",
                details,
            )

        # Write output
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_codebase_to_disk(codebase, case_codebase_dir)

        # Copy paper
        if paper_path is not None and paper_path.exists():
            shutil.copy2(paper_path, case_paper_path)

        # Write case metadata
        case_meta = {
            "entry_id": entry_id,
            "source_codebase": codebase_name,
            "artifact_ids": [a.artifact_id for a in sorted_artifacts],
            "num_artifacts": len(sorted_artifacts),
        }
        (case_dir / "case.json").write_text(
            json.dumps(case_meta, indent=2), encoding="utf-8"
        )

        return BuiltCase(
            entry_id=entry_id,
            source_codebase=codebase_name,
            artifact_ids=[a.artifact_id for a in sorted_artifacts],
            case_dir=case_dir,
            paper_path=case_paper_path,
        )

    except CaseBuildError:
        # Clean up on failure
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
        raise
    except Exception as e:
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
        raise CaseBuildError(f"Unexpected error building case '{entry_id}': {e}") from e
