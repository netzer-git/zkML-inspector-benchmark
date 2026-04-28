"""Load ground truth (JSON) and agent output (JSON), parse and validate."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple




class CodeRef(NamedTuple):
    """A reference to a code location: file, optional start/end lines."""
    filename: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class GroundTruthFinding:
    entry_id: str
    issue_id: str
    issue_name: str
    issue_explanation: str
    relevant_code: list[CodeRef] = field(default_factory=list)
    paper_reference: str = ""


@dataclass
class AgentFinding:
    entry_id: str
    issue_name: str
    issue_explanation: str
    relevant_code: list[CodeRef] = field(default_factory=list)
    paper_reference: str = ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CODE_REF_PATTERN = re.compile(
    r"([A-Za-z0-9_.\/\-]+\.[A-Za-z0-9]+)"  # filename with extension
    r"(?::(\d+)(?:\s*[-–]\s*(\d+))?)?"       # optional :start[-end]
)


def parse_code_refs(raw: str | None) -> list[CodeRef]:
    """Parse a code-reference string like 'file.rs:10-15, other.cu:3' into CodeRefs."""
    if not raw or raw.strip().lower() in ("none", "-", ""):
        return []
    refs: list[CodeRef] = []
    # Split on comma, semicolon, or standalone whitespace between refs
    for segment in re.split(r"[,;]\s*", raw.strip()):
        segment = segment.strip()
        if not segment:
            continue
        m = _CODE_REF_PATTERN.search(segment)
        if m:
            filename = m.group(1)
            start = int(m.group(2)) if m.group(2) else None
            end = int(m.group(3)) if m.group(3) else start
            refs.append(CodeRef(filename, start, end))
    return refs


def _normalize_entry_id(raw: str) -> str:
    """Normalize project name to lowercase for consistent keying."""
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# Ground truth loader (JSON)
# ---------------------------------------------------------------------------

_REQUIRED_GT_FIELDS = {
    "entry-id", "issue-name", "issue-explanation",
    "relevant-code", "paper-reference",
}


def load_ground_truth(json_path: str | Path) -> dict[str, list[GroundTruthFinding]]:
    """Load ground truth from a flat JSON array.

    Returns findings grouped by normalized entry_id. Each object must have
    the 7 required fields plus an optional ``issue-id``. If ``issue-id`` is
    absent it is synthesized as ``{entry-id}-{index}``.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Ground truth must be a JSON array of finding objects")

    result: dict[str, list[GroundTruthFinding]] = defaultdict(list)
    _entry_counters: dict[str, int] = defaultdict(int)

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"GT finding #{i}: expected object, got {type(obj).__name__}")

        missing = _REQUIRED_GT_FIELDS - set(obj.keys())
        if missing:
            raise ValueError(f"GT finding #{i}: missing required fields: {missing}")

        entry_id_raw = str(obj["entry-id"]).strip()
        issue_id = str(obj.get("issue-id", "")).strip()
        if not issue_id:
            _entry_counters[entry_id_raw] += 1
            issue_id = f"{entry_id_raw}-{_entry_counters[entry_id_raw]:02d}"

        finding = GroundTruthFinding(
            entry_id=entry_id_raw,
            issue_id=issue_id,
            issue_name=str(obj["issue-name"]).strip(),
            issue_explanation=str(obj["issue-explanation"]).strip(),
            relevant_code=parse_code_refs(str(obj.get("relevant-code", "") or "")),
            paper_reference=str(obj.get("paper-reference", "") or "").strip(),
        )
        result[_normalize_entry_id(entry_id_raw)].append(finding)

    return dict(result)


# ---------------------------------------------------------------------------
# Agent output loader (JSON)
# ---------------------------------------------------------------------------

_REQUIRED_AGENT_FIELDS = {
    "entry-id", "issue-name", "issue-explanation",
    "relevant-code", "paper-reference",
}


def load_agent_output(json_path: str | Path) -> dict[str, list[AgentFinding]]:
    """Load agent output from a flat JSON array. Returns findings grouped by normalized entry_id."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Agent output must be a JSON array of finding objects")

    result: dict[str, list[AgentFinding]] = defaultdict(list)

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"Agent finding #{i}: expected object, got {type(obj).__name__}")

        missing = _REQUIRED_AGENT_FIELDS - set(obj.keys())
        if missing:
            raise ValueError(f"Agent finding #{i}: missing required fields: {missing}")

        entry_id_raw = str(obj["entry-id"]).strip()

        finding = AgentFinding(
            entry_id=entry_id_raw,
            issue_name=str(obj["issue-name"]).strip(),
            issue_explanation=str(obj["issue-explanation"]).strip(),
            relevant_code=parse_code_refs(str(obj.get("relevant-code", "") or "")),
            paper_reference=str(obj.get("paper-reference", "") or "").strip(),
        )
        result[_normalize_entry_id(entry_id_raw)].append(finding)

    return dict(result)
