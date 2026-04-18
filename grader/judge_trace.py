"""Debug-only markdown renderer for the LLM judge's per-pair decisions.

Produced by the CLI when the user passes `--judge-trace <path>`. Surfaces
every candidate the judge scored for each agent finding so the user can
audit the judge's matching quality. Not part of the default run.

The per-field pair-score breakdown lives in the normal grading report, not
here — the trace focuses strictly on the LLM judge's view.
"""

from __future__ import annotations

from typing import Any

from grader.matcher import AgentFindingTrace, MatchResult
from grader.similarity import JudgeResult


_VERDICT_MATCHED = "**MATCHED**"
_VERDICT_ELIGIBLE_LOST = "eligible (lost to greedy)"
_VERDICT_REJECTED = "rejected"


def _classify(
    result: JudgeResult,
    matched_gt_id: str | None,
    threshold: int,
) -> str:
    """Assign a human-readable verdict to one candidate row."""
    if matched_gt_id == result.gt_id:
        return _VERDICT_MATCHED
    if result.match_score >= threshold:
        # Cleared the score gate but greedy assignment picked another pair
        # (typically because another agent scored higher for this GT, or
        # because this agent scored even higher on a different GT).
        return _VERDICT_ELIGIBLE_LOST
    return _VERDICT_REJECTED


def _escape_table(text: str) -> str:
    """Make a text safe for a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _find_gt_name(gt_id: str, gt_name_by_id: dict[str, str]) -> str:
    return gt_name_by_id.get(gt_id, "(unknown)")


def _render_one_trace(
    trace: AgentFindingTrace,
    threshold: float,
    gt_name_by_id: dict[str, str],
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
        "| GT ID | GT Name | Score (1-5) | Verdict | Reasoning |"
    )
    lines.append(
        "|-------|---------|-------------|---------|-----------|"
    )
    for r in sorted_candidates:
        verdict = _classify(r, trace.matched_gt_id, threshold)
        gt_name = _escape_table(_find_gt_name(r.gt_id, gt_name_by_id))
        reasoning = _escape_table(r.reasoning or "(none)")
        lines.append(
            f"| `{r.gt_id}` | {gt_name} | {r.match_score} | "
            f"{verdict} | {reasoning} |"
        )
    lines.append("")

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
) -> None:
    """Write the debug judge-trace markdown file.

    Args:
        path: Output file path.
        match_results: Per-project MatchResult (each carrying .traces).
        meta: Dict with at least: grader_version, timestamp, threshold,
            backend, entry_ids.
    """
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
            for trace in mr.traces:
                lines.extend(
                    _render_one_trace(
                        trace,
                        threshold=int(meta.get("threshold", 4)),
                        gt_name_by_id=gt_name_by_id,
                    )
                )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
