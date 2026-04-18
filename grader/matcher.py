"""Finding matching: discover which agent findings correspond to which GT findings.

Uses the LLM judge (one call per agent finding, comparing to all GT findings
in the same project) and greedy 1:1 assignment. Matching requires both the
judge's numeric confidence (match_score >= threshold) AND its semantic verdict
(same_root_cause == True).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from grader.loader import AgentFinding, GroundTruthFinding
from grader.similarity import JudgeCandidate, LLMJudgeSimilarity


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


def _build_matching_text(
    name: str, explanation: str, paper_reference: str
) -> str:
    """Format a finding for the LLM judge: name + explanation + paper reference.

    Whitespace-normalized. Code refs and the three closed-list fields
    (severity/category/security-concern) are intentionally excluded — code
    needs the codebase in hand to evaluate, and the closed-list fields are
    graded independently downstream.
    """
    name = " ".join(name.split())
    explanation = " ".join(explanation.split())
    paper = " ".join(paper_reference.split()) if paper_reference else ""
    if not paper or paper == "-":
        paper_line = "Paper reference: (none)"
    else:
        paper_line = f"Paper reference: {paper}"
    return f"{name}\n{explanation}\n{paper_line}"


def match_findings(
    agent_findings: list[AgentFinding],
    gt_findings: list[GroundTruthFinding],
    backend: LLMJudgeSimilarity,
    threshold: float = 0.3,
) -> MatchResult:
    """Match agent findings to GT findings using the LLM judge.

    Issues one bulk LLM call per agent finding, ranking against all GT
    findings in the project. Matching requires BOTH numeric confidence
    (match_score >= threshold) AND the judge's semantic verdict
    (same_root_cause == True). After per-pair scoring, greedy 1:1 assignment
    resolves conflicts in favor of higher-scored pairs.

    Args:
        agent_findings: Agent findings for one project, in arbitrary order.
        gt_findings: GT findings for the same project.
        backend: An LLMJudgeSimilarity instance (must expose judge_bulk).
        threshold: Minimum match_score to consider a match (applied in AND
            with same_root_cause).

    Returns:
        MatchResult with matched pairs, missed GT findings, and extra agent
        findings.

    Raises:
        AttributeError: If the backend does not expose judge_bulk().
    """
    if not agent_findings and not gt_findings:
        return MatchResult()
    if not agent_findings:
        return MatchResult(missed_gt=list(gt_findings))
    if not gt_findings:
        return MatchResult(extra_agent=list(agent_findings))

    if not hasattr(backend, "judge_bulk"):
        raise AttributeError(
            "match_findings requires a backend that exposes judge_bulk() — "
            f"got {type(backend).__name__}"
        )

    m = len(agent_findings)
    n = len(gt_findings)

    agent_texts = [
        _build_matching_text(
            af.issue_name, af.issue_explanation, af.paper_reference
        )
        for af in agent_findings
    ]
    candidates = [
        JudgeCandidate(
            gt_id=gf.issue_id,
            text=_build_matching_text(
                gf.issue_name, gf.issue_explanation, gf.paper_reference
            ),
        )
        for gf in gt_findings
    ]
    id_to_j = {gf.issue_id: j for j, gf in enumerate(gt_findings)}

    triples: list[tuple[float, int, int]] = []
    for i in range(m):
        results = backend.judge_bulk(agent_texts[i], candidates)
        for r in results:
            j = id_to_j.get(r.gt_id)
            if j is None:
                continue  # defensive: judge returned an id we didn't send
            # AND gate: both numeric confidence and semantic verdict required
            if r.match_score >= threshold and r.same_root_cause:
                triples.append((r.match_score, i, j))

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
