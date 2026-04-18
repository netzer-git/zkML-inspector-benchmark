"""Debug-only markdown renderer for the LLM judge's per-pair decisions.

Produced by the CLI when the user passes `--judge-trace <path>`. Surfaces
every candidate the judge scored for each agent finding AND the per-field
pair-score breakdown for every matched pair, so the user can audit both the
judge's matching quality and the scorer's grading quality. Not part of the
default run.
"""

from __future__ import annotations

from typing import Any

from grader.matcher import AgentFindingTrace, MatchResult
from grader.similarity import JudgeResult


_VERDICT_MATCHED = "**MATCHED**"
_VERDICT_ELIGIBLE_LOST = "eligible (lost)"
_VERDICT_REJECTED_SCORE = "rejected (score)"
_VERDICT_REJECTED_VERDICT = "rejected (verdict)"
_VERDICT_REJECTED = "rejected"


def _classify(
    result: JudgeResult,
    matched_gt_id: str | None,
    threshold: float,
) -> str:
    """Assign a human-readable verdict to one candidate row."""
    if matched_gt_id == result.gt_id:
        return _VERDICT_MATCHED
    score_ok = result.match_score >= threshold
    verdict_ok = result.same_root_cause
    if score_ok and verdict_ok:
        # Cleared the AND gate but greedy assignment picked another pair.
        return _VERDICT_ELIGIBLE_LOST
    if verdict_ok and not score_ok:
        return _VERDICT_REJECTED_SCORE
    if score_ok and not verdict_ok:
        return _VERDICT_REJECTED_VERDICT
    return _VERDICT_REJECTED


def _escape_table(text: str) -> str:
    """Make a text safe for a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _find_gt_name(gt_id: str, gt_name_by_id: dict[str, str]) -> str:
    return gt_name_by_id.get(gt_id, "(unknown)")


def _render_pair_score_breakdown(
    gt_id: str,
    project_grade: Any,  # ProjectGrade | None — kept loose to avoid import cycle
) -> list[str]:
    """Render the per-field pair_score breakdown for a MATCHED pair.

    Looks up the PairGrade(s) for this gt_id in the project grade. If the GT
    has multiple matching agents (N:1), shows each pair individually so the
    user can see how the GT's combined score was assembled.
    """
    if project_grade is None:
        return []
    # There may be multiple PairGrades for the same gt_id (N:1 duplicates).
    pairs = [m for m in project_grade.matches if m.gt_id == gt_id]
    if not pairs:
        return []

    lines: list[str] = ["**Pair-score breakdown:**", ""]
    if len(pairs) > 1:
        lines.append(
            f"*{len(pairs)} agent findings matched this GT — each pair is "
            f"shown below; the GT's combined score is the average.*"
        )
        lines.append("")

    for idx, pair in enumerate(pairs):
        if len(pairs) > 1:
            dup_label = " (primary)" if pair.dup_rank == 0 else f" (dup #{pair.dup_rank})"
            lines.append(
                f"*Pair #{idx + 1}{dup_label}: agent \"{pair.agent_name}\" "
                f"-> pair_score {pair.pair_score:.2f}*"
            )
            lines.append("")
        lines.append("| Field | Score | Weight | Detail |")
        lines.append("|-------|-------|--------|--------|")
        # Iterate in the canonical field order.
        for field_name in (
            "severity", "category", "security_concern",
            "code_location", "paper_reference",
        ):
            fs = pair.scores.get(field_name)
            if fs is None:
                continue
            # Weights are not carried on PairGrade; show blank column for now.
            lines.append(
                f"| {field_name} | {fs.score:.2f} | - | "
                f"{_escape_table(fs.detail)} |"
            )
        lines.append("")
        lines.append(f"**pair_score = {pair.pair_score:.3f}**")
        lines.append("")
    return lines


def _render_one_trace(
    trace: AgentFindingTrace,
    threshold: float,
    gt_name_by_id: dict[str, str],
    project_grade: Any,  # ProjectGrade | None
) -> list[str]:
    """Render one agent-finding section. Returns list of markdown lines."""
    lines: list[str] = []
    af = trace.agent
    header = f"### Agent finding [{trace.agent_index}]: \"{af.issue_name}\" ({af.severity})"
    lines.append(header)
    lines.append("")

    if trace.matched_gt_id:
        lines.append(f"**Outcome:** MATCHED `{trace.matched_gt_id}`")
    else:
        lines.append("**Outcome:** no match")
    lines.append("")

    lines.append("**Judge input:**")
    lines.append("")
    lines.append("```")
    lines.append(trace.agent_text)
    lines.append("```")
    lines.append("")

    # Table of candidates with reasoning column, sorted by score desc.
    sorted_candidates = sorted(
        trace.candidates, key=lambda r: r.match_score, reverse=True
    )

    lines.append("**Candidates considered (sorted by score, desc):**")
    lines.append("")
    lines.append(
        "| GT ID | GT Name | Score | Same root cause | Verdict | Reasoning |"
    )
    lines.append(
        "|-------|---------|-------|-----------------|---------|-----------|"
    )
    for r in sorted_candidates:
        verdict = _classify(r, trace.matched_gt_id, threshold)
        gt_name = _escape_table(_find_gt_name(r.gt_id, gt_name_by_id))
        srh = "yes" if r.same_root_cause else "no"
        reasoning = _escape_table(r.reasoning or "(none)")
        lines.append(
            f"| `{r.gt_id}` | {gt_name} | {r.match_score:.2f} | {srh} | "
            f"{verdict} | {reasoning} |"
        )
    lines.append("")

    # Pair-score breakdown — only for MATCHED pairs; pulls from project_grade.
    if trace.matched_gt_id:
        lines.extend(
            _render_pair_score_breakdown(trace.matched_gt_id, project_grade)
        )

    lines.append("---")
    lines.append("")
    return lines


def _project_header(
    project: str, match_result: MatchResult
) -> list[str]:
    m = len(match_result.matched)
    unique_gts_matched = len({mp.gt.issue_id for mp in match_result.matched})
    n_gt_unique = unique_gts_matched + len(match_result.missed_gt)
    n_agent = m + len(match_result.extra_agent)
    dup_count = sum(1 for mp in match_result.matched if mp.dup_rank > 0)
    lines = [
        f"## Project: `{project}`",
        "",
        f"- Agent findings: {n_agent}",
        f"- GT findings: {n_gt_unique}",
        f"- Unique GTs matched: {unique_gts_matched}",
        f"- Matched pairs (incl. N:1 duplicates): {m}",
    ]
    if dup_count > 0:
        lines.append(f"- **N:1 duplicates flagged: {dup_count}**")
    lines.extend([
        f"- Missed GT: {len(match_result.missed_gt)}",
        f"- Extra agent findings: {len(match_result.extra_agent)}",
        "",
    ])
    return lines


def write_judge_trace(
    path: str,
    match_results: dict[str, MatchResult],
    meta: dict[str, Any],
    project_grades: dict[str, Any] | None = None,
) -> None:
    """Write the debug judge-trace markdown file.

    Args:
        path: Output file path.
        match_results: Per-project MatchResult (each carrying .traces).
        meta: Dict with at least: grader_version, timestamp, threshold,
            backend, entry_ids.
        project_grades: Optional per-project ProjectGrade. When provided, each
            MATCHED pair in the trace gains a per-field pair-score breakdown
            section so the user can audit the scorer's decisions.
    """
    project_grades = project_grades or {}

    lines: list[str] = []
    lines.append("# Judge Trace")
    lines.append("")
    lines.append(f"**Grader version:** {meta.get('grader_version', '?')}  ")
    lines.append(f"**Date:** {meta.get('timestamp', '?')}  ")
    lines.append(f"**Backend:** {meta.get('backend', '?')}  ")
    lines.append(f"**Threshold:** {meta.get('threshold', '?')}  ")
    entry_ids = meta.get("entry_ids")
    if entry_ids:
        lines.append(f"**Entry IDs filter:** {', '.join(entry_ids)}")
    else:
        lines.append("**Entry IDs filter:** (none -- all projects)")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not match_results:
        lines.append("_No projects graded in this run._")
        lines.append("")
    else:
        for project, mr in sorted(match_results.items()):
            # Build a gt_id -> gt_name lookup from the MatchResult.
            gt_name_by_id: dict[str, str] = {}
            for mp in mr.matched:
                gt_name_by_id[mp.gt.issue_id] = mp.gt.issue_name
            for gf in mr.missed_gt:
                gt_name_by_id[gf.issue_id] = gf.issue_name

            lines.extend(_project_header(project, mr))
            if not mr.traces:
                lines.append("_(no traces -- empty agent or GT input)_")
                lines.append("")
                lines.append("---")
                lines.append("")
                continue
            pg = project_grades.get(project)
            for trace in mr.traces:
                lines.extend(
                    _render_one_trace(
                        trace,
                        threshold=float(meta.get("threshold", 0.3)),
                        gt_name_by_id=gt_name_by_id,
                        project_grade=pg,
                    )
                )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
