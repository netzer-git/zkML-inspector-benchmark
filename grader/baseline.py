"""Baseline finding exclusion: load known pre-existing issues and filter
matching agent findings before GT matching.

Baseline findings represent genuine gaps inherent to research-prototype
codebases (not injected bugs). Agent findings that match a baseline entry
are silently excluded from scoring — they don't count toward recall or
precision.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from grader.loader import AgentFinding, parse_code_refs, CodeRef
from grader.matcher import _build_matching_text
from grader.similarity import JudgeCandidate, LLMJudgeSimilarity


@dataclass
class BaselineFinding:
    """A known pre-existing issue in a clean codebase."""
    pair: str                  # pair prefix, e.g. "zkllm"
    baseline_id: str           # synthetic id: "{pair}-baseline-{index:02d}"
    issue_name: str
    issue_explanation: str
    relevant_code: list[CodeRef] = field(default_factory=list)
    paper_reference: str = ""


def load_baseline(path: str | Path) -> dict[str, list[BaselineFinding]]:
    """Load baseline findings from a flat JSON array.

    Returns a dict keyed by pair prefix (lowercase), mapping to the list
    of baseline findings for that pair.

    The JSON uses the same schema as agent output:
    ``entry-id``, ``issue-name``, ``issue-explanation``,
    ``relevant-code``, ``paper-reference``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("baseline JSON must be a flat array")

    result: dict[str, list[BaselineFinding]] = defaultdict(list)
    for i, item in enumerate(raw):
        pair = item["entry-id"].strip().lower()
        bf = BaselineFinding(
            pair=pair,
            baseline_id=f"{pair}-baseline-{i:02d}",
            issue_name=item["issue-name"],
            issue_explanation=item["issue-explanation"],
            relevant_code=parse_code_refs(item.get("relevant-code", "")),
            paper_reference=item.get("paper-reference", ""),
        )
        result[pair].append(bf)
    return dict(result)


def pair_prefix(entry_id: str) -> str:
    """Extract the pair prefix from an experiment entry-id.

    Examples:
        ``zkllm-A-zkLLM-003-rep02`` → ``zkllm``
        ``zktorch-B-max-rep01``     → ``zktorch``
        ``zkgpt``                   → ``zkgpt``
    """
    return entry_id.strip().lower().split("-")[0]


def filter_baseline(
    agent_findings: list[AgentFinding],
    baseline_findings: list[BaselineFinding],
    backend: LLMJudgeSimilarity,
    threshold: int = 4,
    verbose: bool = False,
) -> tuple[list[AgentFinding], list[AgentFinding]]:
    """Partition agent findings into (remaining, excluded).

    Each agent finding is compared against all baseline findings via one
    ``judge_bulk`` call. If the top baseline match scores >= *threshold*,
    the agent finding is excluded.

    Returns:
        A 2-tuple ``(remaining, excluded)`` where *remaining* should be
        passed to ``match_findings`` and *excluded* are baseline matches
        that should not affect scoring.
    """
    if not agent_findings or not baseline_findings:
        return list(agent_findings), []

    candidates = [
        JudgeCandidate(
            gt_id=bf.baseline_id,
            text=_build_matching_text(
                bf.issue_name, bf.issue_explanation, bf.paper_reference,
            ),
        )
        for bf in baseline_findings
    ]

    remaining: list[AgentFinding] = []
    excluded: list[AgentFinding] = []

    for i, af in enumerate(agent_findings):
        agent_text = _build_matching_text(
            af.issue_name, af.issue_explanation, af.paper_reference,
        )
        if verbose:
            name_preview = (af.issue_name or "")[:50]
            print(f"    [baseline {i + 1}/{len(agent_findings)}] {name_preview!r}", end="", flush=True)

        results = backend.judge_bulk(agent_text, candidates)
        best = max(results, key=lambda r: r.match_score, default=None)

        if best is not None and best.match_score >= threshold:
            excluded.append(af)
            if verbose:
                print(f" -> EXCLUDED (matched {best.gt_id}, score={best.match_score})")
        else:
            remaining.append(af)
            if verbose:
                top_score = best.match_score if best else 0
                print(f" -> kept (top={top_score})")

    return remaining, excluded
