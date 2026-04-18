"""Finding matching: discover which agent findings correspond to which GT findings.

Uses a similarity matrix and greedy assignment. Operates per-project.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from grader.loader import AgentFinding, GroundTruthFinding
from grader.similarity import SimilarityBackend


@dataclass
class MatchedPair:
    agent: AgentFinding
    gt: GroundTruthFinding
    similarity: float


@dataclass
class MatchResult:
    matched: list[MatchedPair] = field(default_factory=list)
    missed_gt: list[GroundTruthFinding] = field(default_factory=list)
    extra_agent: list[AgentFinding] = field(default_factory=list)

    @property
    def extra_by_severity(self) -> dict[str, list[AgentFinding]]:
        """Group extra findings by severity for differentiated reporting."""
        groups: dict[str, list[AgentFinding]] = defaultdict(list)
        for af in self.extra_agent:
            groups[af.severity].append(af)
        return dict(groups)


def _build_similarity_text(name: str, explanation: str) -> str:
    """Concatenate name and explanation for similarity comparison."""
    return f"{name} {explanation}"


def match_findings(
    agent_findings: list[AgentFinding],
    gt_findings: list[GroundTruthFinding],
    backend: SimilarityBackend,
    threshold: float = 0.3,
) -> MatchResult:
    """Match agent findings to GT findings using greedy assignment on similarity.

    Args:
        agent_findings: Unordered list of agent findings for one project.
        gt_findings: GT findings for the same project.
        backend: Similarity scoring backend.
        threshold: Minimum similarity to consider a match.

    Returns:
        MatchResult with matched pairs, missed GT, and extra agent findings.
    """
    if not agent_findings and not gt_findings:
        return MatchResult()

    if not agent_findings:
        return MatchResult(missed_gt=list(gt_findings))

    if not gt_findings:
        return MatchResult(extra_agent=list(agent_findings))

    m = len(agent_findings)
    n = len(gt_findings)

    # Build M x N similarity matrix
    agent_texts = [
        _build_similarity_text(af.issue_name, af.issue_explanation)
        for af in agent_findings
    ]
    gt_texts = [
        _build_similarity_text(gf.issue_name, gf.issue_explanation)
        for gf in gt_findings
    ]

    triples: list[tuple[float, int, int]] = []
    judge_bulk = getattr(backend, "judge_bulk", None)
    if callable(judge_bulk):
        # Bulk LLM-judge path: one call per agent finding, ranked output
        # against all GT candidates for this project.
        from grader.similarity import JudgeCandidate

        candidates = [
            JudgeCandidate(gt_id=str(j), text=gt_texts[j]) for j in range(n)
        ]
        for i in range(m):
            results = judge_bulk(agent_texts[i], candidates)
            by_id = {r.gt_id: r for r in results}
            for j in range(n):
                r = by_id.get(str(j))
                if r is not None and r.match_score >= threshold:
                    triples.append((r.match_score, i, j))
    else:
        # Per-pair path for symmetric backends (Jaccard, future TF-IDF, etc.)
        for i in range(m):
            for j in range(n):
                sim = backend.score(agent_texts[i], gt_texts[j])
                if sim >= threshold:
                    triples.append((sim, i, j))

    # Greedy assignment: sort descending, assign greedily
    triples.sort(key=lambda t: t[0], reverse=True)

    assigned_agent: set[int] = set()
    assigned_gt: set[int] = set()
    matched: list[MatchedPair] = []

    for sim, i, j in triples:
        if i in assigned_agent or j in assigned_gt:
            continue
        matched.append(MatchedPair(
            agent=agent_findings[i],
            gt=gt_findings[j],
            similarity=sim,
        ))
        assigned_agent.add(i)
        assigned_gt.add(j)

    missed_gt = [gt_findings[j] for j in range(n) if j not in assigned_gt]
    extra_agent = [agent_findings[i] for i in range(m) if i not in assigned_agent]

    return MatchResult(matched=matched, missed_gt=missed_gt, extra_agent=extra_agent)
