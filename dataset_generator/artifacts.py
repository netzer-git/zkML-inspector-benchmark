"""Load and validate strict-v2 artifact JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "artifact.v2.schema.json"
_SCHEMA: dict | None = None


def _get_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


@dataclass
class Artifact:
    """A validated strict-v2 bug artifact."""

    artifact_id: str
    codebase: str
    source: str
    finding: dict[str, Any]
    edits: list[dict[str, Any]]
    conflict_keys: dict[str, Any]
    presence_probes: list[dict[str, Any]]
    raw: dict[str, Any] = field(repr=False)

    @property
    def requires(self) -> list[str]:
        return self.conflict_keys.get("requires", [])

    @property
    def incompatible(self) -> list[str]:
        return self.conflict_keys.get("incompatible", [])

    @property
    def regions(self) -> list[dict[str, Any]]:
        return self.conflict_keys.get("regions", [])

    @property
    def semantic_tags(self) -> list[str]:
        return self.conflict_keys.get("semantic_tags", [])

    @property
    def files(self) -> list[str]:
        return self.conflict_keys.get("files", [])


def validate_artifact(data: dict[str, Any]) -> None:
    """Validate a dict against the strict-v2 schema. Raises on failure."""
    jsonschema.validate(instance=data, schema=_get_schema())


def load_artifact(path: Path) -> Artifact:
    """Load and validate a single artifact JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact(data)
    return Artifact(
        artifact_id=data["artifact_id"],
        codebase=data["codebase"],
        source=data["source"],
        finding=data["finding"],
        edits=data["edits"],
        conflict_keys=data["conflict_keys"],
        presence_probes=data["presence_probes"],
        raw=data,
    )


def load_artifacts_from_dir(directory: Path) -> list[Artifact]:
    """Load all *.json artifacts from a directory, validating each."""
    artifacts = []
    for p in sorted(directory.glob("*.json")):
        artifacts.append(load_artifact(p))
    return artifacts
