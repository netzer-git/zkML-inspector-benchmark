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


# ---------------------------------------------------------------------------
# Line-offset tracking: when multiple artifacts edit the same file, earlier
# edits may change the line count, shifting all subsequent line numbers.
# We track each edit's effect in *clean-codebase* coordinates and adjust
# later artifacts' anchors accordingly.
# ---------------------------------------------------------------------------

@dataclass
class _AppliedEdit:
    """Record of one edit's line-count effect in clean-codebase coordinates."""

    file: str
    original_end: int  # 1-based inclusive end of the replaced range
    delta: int  # (new_lines - original_lines)


def _compute_line_adjustment(
    line_1based: int,
    applied_edits: list[_AppliedEdit],
    file: str,
) -> int:
    """Cumulative adjustment for a clean-codebase line number."""
    return sum(
        ae.delta
        for ae in applied_edits
        if ae.file == file and ae.original_end < line_1based
    )


def _adjust_edit(
    edit: dict[str, Any],
    applied_edits: list[_AppliedEdit],
) -> dict[str, Any]:
    """Return a shallow copy of *edit* with ``line_range`` anchors adjusted."""
    anchor = edit.get("anchor", {})
    if anchor.get("kind", "line_range") != "line_range":
        return edit

    start = anchor.get("start")
    end = anchor.get("end")
    if start is None or end is None:
        return edit

    file = edit["file"]
    adj_s = _compute_line_adjustment(start, applied_edits, file)
    adj_e = _compute_line_adjustment(end, applied_edits, file)
    if adj_s == 0 and adj_e == 0:
        return edit

    return {
        **edit,
        "anchor": {**anchor, "start": start + adj_s, "end": end + adj_e},
    }


def _record_edit_effect(
    edit: dict[str, Any],
    applied_edits: list[_AppliedEdit],
) -> None:
    """Append the line-count delta of *edit* (using its clean-codebase coords)."""
    anchor = edit.get("anchor", {})
    if anchor.get("kind", "line_range") != "line_range":
        return

    op = edit.get("op", "")
    if op in ("create_file", "replace_regex"):
        return

    file = edit["file"]
    start = anchor.get("start", 1)
    end = anchor.get("end", start)
    new_content = edit.get("new_content", "")
    new_lines = len(new_content.split("\n")) if new_content else 0

    if op == "replace_block":
        delta = new_lines - (end - start + 1)
        ref_end = end
    elif op == "delete_block":
        delta = -(end - start + 1)
        ref_end = end
    elif op == "insert_after":
        delta = new_lines
        ref_end = end
    elif op == "insert_before":
        delta = new_lines
        ref_end = start - 1
    else:
        return

    if delta != 0:
        applied_edits.append(
            _AppliedEdit(file=file, original_end=ref_end, delta=delta)
        )


def _adjust_probe(
    probe: dict[str, Any],
    applied_edits: list[_AppliedEdit],
) -> dict[str, Any]:
    """Return a copy of a ``line_equals`` probe with an adjusted line number."""
    if probe.get("kind") != "line_equals":
        return probe
    line = probe.get("line")
    if line is None:
        return probe
    file = probe.get("file", "")
    adj = _compute_line_adjustment(line, applied_edits, file)
    if adj == 0:
        return probe
    return {**probe, "line": line + adj}


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

        # Apply edits with line-offset tracking
        applied_edits: list[_AppliedEdit] = []

        for artifact in sorted_artifacts:
            try:
                adjusted = [_adjust_edit(e, applied_edits) for e in artifact.edits]
                apply_artifact_edits(
                    codebase, adjusted, artifact.artifact_id
                )
                # Record effects using original clean-codebase coordinates
                for edit in artifact.edits:
                    _record_edit_effect(edit, applied_edits)
            except EditError as e:
                raise CaseBuildError(
                    f"Edit failed for {artifact.artifact_id}: {e}",
                    [{"artifact_id": artifact.artifact_id, "error": str(e)}],
                ) from e

        # Run probes (adjust line_equals probes for line shifts)
        all_failures: list[ProbeFailure] = []
        for artifact in sorted_artifacts:
            adjusted_probes = [
                _adjust_probe(p, applied_edits)
                for p in artifact.presence_probes
            ]
            failures = evaluate_probes(
                codebase, adjusted_probes, artifact.artifact_id
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
