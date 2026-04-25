"""Presence-probe evaluator for verifying bug injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataset_generator.edits import Codebase


@dataclass
class ProbeFailure:
    """Structured record of a failed presence probe."""

    artifact_id: str
    probe_index: int
    kind: str
    file: str
    detail: str


def evaluate_probe(
    codebase: Codebase, probe: dict[str, Any], artifact_id: str, probe_index: int
) -> ProbeFailure | None:
    """Evaluate a single probe. Returns None on success, a ProbeFailure otherwise."""
    kind = probe["kind"]
    file_path = probe["file"]

    if file_path not in codebase:
        return ProbeFailure(
            artifact_id=artifact_id,
            probe_index=probe_index,
            kind=kind,
            file=file_path,
            detail=f"File '{file_path}' not found in codebase",
        )

    lines = codebase[file_path]
    full_text = "\n".join(lines)

    if kind == "contains":
        text = probe["text"]
        if text not in full_text:
            return ProbeFailure(
                artifact_id=artifact_id,
                probe_index=probe_index,
                kind=kind,
                file=file_path,
                detail=f"Expected substring not found: '{text[:80]}'",
            )

    elif kind == "not_contains":
        text = probe["text"]
        if text in full_text:
            return ProbeFailure(
                artifact_id=artifact_id,
                probe_index=probe_index,
                kind=kind,
                file=file_path,
                detail=f"Forbidden substring still present: '{text[:80]}'",
            )

    elif kind == "line_equals":
        line_num = probe["line"]  # 1-based
        text = probe["text"]
        idx = line_num - 1
        if idx < 0 or idx >= len(lines):
            return ProbeFailure(
                artifact_id=artifact_id,
                probe_index=probe_index,
                kind=kind,
                file=file_path,
                detail=f"Line {line_num} out of range (file has {len(lines)} lines)",
            )
        actual = lines[idx].rstrip("\n")
        if actual != text:
            return ProbeFailure(
                artifact_id=artifact_id,
                probe_index=probe_index,
                kind=kind,
                file=file_path,
                detail=f"Line {line_num}: expected '{text[:60]}', got '{actual[:60]}'",
            )

    else:
        return ProbeFailure(
            artifact_id=artifact_id,
            probe_index=probe_index,
            kind=kind,
            file=file_path,
            detail=f"Unknown probe kind: {kind}",
        )

    return None


def evaluate_probes(
    codebase: Codebase,
    probes: list[dict[str, Any]],
    artifact_id: str,
) -> list[ProbeFailure]:
    """Evaluate all probes for an artifact. Returns list of failures (empty = all pass)."""
    failures = []
    for i, probe in enumerate(probes):
        result = evaluate_probe(codebase, probe, artifact_id, i)
        if result is not None:
            failures.append(result)
    return failures
