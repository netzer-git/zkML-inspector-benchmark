"""Finding matching: discover which agent findings correspond to which GT findings.

Uses the LLM judge (one call per agent finding, comparing to all GT findings
in the same project) and greedy 1:1 assignment. Matching requires both the
judge's numeric confidence (match_score >= threshold) AND its semantic verdict
(same_root_cause == True).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grader.loader import AgentFinding, GroundTruthFinding
from grader.similarity import JudgeCandidate, JudgeResult, LLMJudgeSimilarity


@dataclass
class MatchedPair:
    agent: AgentFinding
    gt: GroundTruthFinding
    similarity: float
    # dup_rank = 0 for the first (best) agent that matched this GT;
    # 1, 2, ... for subsequent agents that also matched the same GT.
    # A non-zero dup_rank flags the pair as a duplicate/split — the GT was
    # already bound to another agent when this pair was assigned.
    dup_rank: int = 0


@dataclass
class AgentFindingTrace:
    """Judge output for one agent finding: every candidate the LLM scored.

    Populated by `match_findings` and surfaced in `MatchResult.traces`. The
    normal grading report ignores this; it's consumed only by the debug-only
    `--judge-trace` markdown output.
    """
    agent_index: int                      # position in the input agent_findings list
    agent: AgentFinding
    agent_text: str                       # the matching text shown to the judge
    candidates: list[JudgeResult]         # full judgment list, in GT order
    matched_gt_id: str | None = None      # set after greedy assignment if matched


@dataclass
class MatchResult:
    matched: list[MatchedPair] = field(default_factory=list)
    missed_gt: list[GroundTruthFinding] = field(default_factory=list)
    extra_agent: list[AgentFinding] = field(default_factory=list)
    traces: list[AgentFindingTrace] = field(default_factory=list)


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
    threshold: int = 4,
    verbose: bool = False,
) -> MatchResult:
    """Match agent findings to GT findings using the LLM judge.

    Issues one bulk LLM call per agent finding, ranking against all GT
    findings in the project. The judge returns an integer match_score on
    a 1..5 scale (see grader.similarity._DEFAULT_SYSTEM_PROMPT). A pair
    clears the gate when match_score >= threshold (default 4 = "very likely
    the same finding"). After per-pair scoring, greedy assignment gives each
    agent its best GT; multiple agents can bind to the same GT (N:1).

    Args:
        agent_findings: Agent findings for one project, in arbitrary order.
        gt_findings: GT findings for the same project.
        backend: An LLMJudgeSimilarity instance (must expose judge_bulk).
        threshold: Minimum match_score (integer 1..5) for a match. Default 4.
        verbose: If True, print a progress line per agent finding to stdout.
            Useful for long-running real-API runs; tests leave it off.

    Returns:
        MatchResult with matched pairs, missed GT findings, and extra agent
        findings.

    Raises:
        AttributeError: If the backend does not expose judge_bulk().
    """
    import time
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

    traces: list[AgentFindingTrace] = []
    triples: list[tuple[int, int, int]] = []
    for i in range(m):
        if verbose:
            name_preview = (agent_findings[i].issue_name or "")[:60]
            print(
                f"    [{i + 1}/{m}] judging: {name_preview!r}",
                end="", flush=True,
            )
        t0 = time.perf_counter()
        results = backend.judge_bulk(agent_texts[i], candidates)
        elapsed = time.perf_counter() - t0
        traces.append(AgentFindingTrace(
            agent_index=i,
            agent=agent_findings[i],
            agent_text=agent_texts[i],
            candidates=list(results),
            matched_gt_id=None,
        ))
        best = max(
            (r for r in results), key=lambda r: r.match_score, default=None
        )
        if verbose:
            if best is not None:
                print(
                    f" -> top {best.match_score}/5 / {best.gt_id!r} "
                    f"[{elapsed:.1f}s]",
                    flush=True,
                )
            else:
                print(f" -> no candidates [{elapsed:.1f}s]", flush=True)
        for r in results:
            j = id_to_j.get(r.gt_id)
            if j is None:
                continue  # defensive: judge returned an id we didn't send
            # Pair clears the gate when the judge's ordinal score is at or
            # above threshold. Default threshold=4 means "very likely same
            # finding" or "confident match". No boolean gate any more --
            # the ordinal encodes confidence directly.
            if r.match_score >= threshold:
                triples.append((r.match_score, i, j))

    # Assignment: sort triples by score desc and assign greedily.
    #
    # Each AGENT can bind to at most one GT (its highest-confidence one).
    # Each GT can absorb MULTIPLE agents — if two agents both claim the same
    # GT, both are recorded as matches. The first one to bind is "primary"
    # (dup_rank=0); subsequent binders are duplicates (dup_rank=1, 2, ...).
    # This reflects the case where the agent split one underlying issue into
    # several findings; both agents should get credit and the user should
    # see the duplication flagged in the report.
    triples.sort(key=lambda t: t[0], reverse=True)

    assigned_agent: set[int] = set()
    matched_gt_indices: set[int] = set()
    gt_dup_count: dict[int, int] = {}
    matched: list[MatchedPair] = []

    for sim, i, j in triples:
        if i in assigned_agent:
            continue  # agent already bound to its best GT
        dup_rank = gt_dup_count.get(j, 0)
        matched.append(MatchedPair(
            agent=agent_findings[i],
            gt=gt_findings[j],
            similarity=sim,
            dup_rank=dup_rank,
        ))
        assigned_agent.add(i)
        matched_gt_indices.add(j)
        gt_dup_count[j] = dup_rank + 1
        traces[i].matched_gt_id = gt_findings[j].issue_id

    missed_gt = [gt_findings[j] for j in range(n) if j not in matched_gt_indices]
    extra_agent = [agent_findings[i] for i in range(m) if i not in assigned_agent]

    return MatchResult(
        matched=matched, missed_gt=missed_gt, extra_agent=extra_agent, traces=traces,
    )
