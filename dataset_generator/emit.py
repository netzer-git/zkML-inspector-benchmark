"""Emitters for dataset manifest and aggregated ground-truth findings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_generator.assembler import BuiltCase
from dataset_generator.artifacts import Artifact


def _artifact_to_finding(
    artifact: Artifact, entry_id: str
) -> dict[str, str]:
    """Convert an artifact's finding block to a grader-compatible finding dict."""
    f = artifact.finding
    labels = f["labels"]
    return {
        "entry-id": entry_id,
        "issue-id": artifact.artifact_id,
        "issue-name": f["name"],
        "issue-explanation": f["explanation"],
        "severity": labels["severity"],
        "category": labels["category"],
        "security-concern": labels["security_concern"],
        "relevant-code": labels["relevant_code"],
        "paper-reference": labels["paper_reference"],
    }


def write_findings_json(
    output_path: Path,
    cases: list[BuiltCase],
    artifacts_by_id: dict[str, Artifact],
) -> Path:
    """Write aggregated findings.json (flat array, grader-compatible).

    Returns the path to the written file.
    """
    findings: list[dict[str, str]] = []
    for case in cases:
        for aid in case.artifact_ids:
            artifact = artifacts_by_id[aid]
            findings.append(_artifact_to_finding(artifact, case.entry_id))

    output_path.write_text(
        json.dumps(findings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def write_dataset_manifest(
    output_path: Path,
    cases: list[BuiltCase],
    strategy: str,
    seed: int | None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write dataset_manifest.json with case metadata.

    Returns the path to the written file.
    """
    manifest = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "seed": seed,
        "num_cases": len(cases),
        "cases": [
            {
                "entry_id": case.entry_id,
                "source_codebase": case.source_codebase,
                "artifact_ids": case.artifact_ids,
                "num_artifacts": len(case.artifact_ids),
                "case_dir": str(case.case_dir),
                "paper_path": str(case.paper_path),
            }
            for case in cases
        ],
    }
    if extra_meta:
        manifest["meta"] = extra_meta

    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
