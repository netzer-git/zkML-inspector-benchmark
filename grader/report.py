"""Aggregate scoring and output formatting (JSON + markdown).

Recall-focused grading: quality is a pass/fail gate combining the LLM judge
match_score with code-location and paper-reference evidence scores. A match
counts toward recall only if quality >= QUALITY_THRESHOLD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import grader
from grader.loader import AgentFinding, GroundTruthFinding
from grader.matcher import MatchResult, MatchedPair
from grader.scorers import (
    FieldScore,
    score_code_location,
    score_paper_reference,
)
from grader.similarity import SimilarityBackend

# Quality gate: combines LLM judge confidence with evidence scores.
# quality = W_MATCH*(match_score/5) + W_CODE*code_location + W_PAPER*paper_reference
QUALITY_WEIGHTS = {"match_score": 0.50, "code_location": 0.30, "paper_reference": 0.20}
QUALITY_THRESHOLD = 0.55


def _quality_badge(quality: float, passed: bool) -> str:
    """Return an emoji badge for quality display in markdown."""
    if not passed:
        return "🔴"
    if quality >= 0.75:
        return "🟢"
    return "🟡"


@dataclass
class PairGrade:
    """Grading result for a single matched (agent, GT) pair."""
    gt_id: str
    gt_name: str
    agent_name: str
    match_similarity: float
    code_location_score: FieldScore
    paper_reference_score: FieldScore
    quality: float
    passed: bool
    dup_rank: int = 0


@dataclass
class ProjectGrade:
    """Grading result for all findings in one project."""
    project: str
    recall: float
    precision: float
    f1: float
    avg_quality: float
    matches: list[PairGrade]
    missed_gt: list[dict[str, str]]
    extra_agent: list[dict[str, str]]
    extra_count: int = 0


@dataclass
class GradeReport:
    """Full grading report across all projects."""
    meta: dict[str, Any]
    projects: dict[str, ProjectGrade]
    overall: dict[str, float]


def _compute_quality(
    match_score: float,
    code_location: FieldScore,
    paper_reference: FieldScore,
) -> float:
    """Compute quality score combining LLM match confidence with evidence."""
    w = QUALITY_WEIGHTS
    code_val = code_location.score if "skip" not in code_location.detail else 0.5
    paper_val = paper_reference.score if "skip" not in paper_reference.detail else 0.5
    return (
        w["match_score"] * (match_score / 5.0)
        + w["code_location"] * code_val
        + w["paper_reference"] * paper_val
    )


def grade_pair(
    pair: MatchedPair,
    similarity_backend: SimilarityBackend,
    quality_threshold: float = QUALITY_THRESHOLD,
) -> PairGrade:
    """Grade a single matched pair. Quality is a pass/fail gate."""
    code_loc = score_code_location(
        pair.agent.relevant_code, pair.gt.relevant_code
    )
    paper_ref = score_paper_reference(
        pair.agent.paper_reference, pair.gt.paper_reference, similarity_backend
    )

    quality = _compute_quality(pair.similarity, code_loc, paper_ref)
    passed = quality >= quality_threshold

    return PairGrade(
        gt_id=pair.gt.issue_id,
        gt_name=pair.gt.issue_name,
        agent_name=pair.agent.issue_name,
        match_similarity=pair.similarity,
        code_location_score=code_loc,
        paper_reference_score=paper_ref,
        quality=quality,
        passed=passed,
        dup_rank=pair.dup_rank,
    )


def grade_project(
    project: str,
    match_result: MatchResult,
    similarity_backend: SimilarityBackend,
    gt_findings: list[GroundTruthFinding],
    agent_findings: list[AgentFinding],
    quality_threshold: float = QUALITY_THRESHOLD,
) -> ProjectGrade:
    """Grade all findings for one project.

    Recall = unique GTs with at least one passing match / total GTs.
    Precision = passed agent matches / total agent findings.
    Quality = mean quality score of passed matches (informational).
    """
    pair_grades = [
        grade_pair(mp, similarity_backend, quality_threshold)
        for mp in match_result.matched
    ]

    total_gt = len(gt_findings)
    total_agent = len(agent_findings)

    # Recall: unique GT issue_ids with at least one passing match.
    passed_gt_ids = {pg.gt_id for pg in pair_grades if pg.passed}
    recall = len(passed_gt_ids) / total_gt if total_gt > 0 else 0.0

    # Precision: passed agent matches / total agent findings.
    passed_count = sum(1 for pg in pair_grades if pg.passed)
    precision = passed_count / total_agent if total_agent > 0 else 0.0

    # F1
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Avg quality of passed matches (informational).
    passed_qualities = [pg.quality for pg in pair_grades if pg.passed]
    avg_quality = (
        sum(passed_qualities) / len(passed_qualities)
        if passed_qualities
        else 0.0
    )

    # Format missed and extra for output
    missed_gt_list = [
        {"id": gf.issue_id, "name": gf.issue_name}
        for gf in match_result.missed_gt
    ]
    extra_agent_list = [
        {"name": af.issue_name}
        for af in match_result.extra_agent
    ]

    return ProjectGrade(
        project=project,
        recall=recall,
        precision=precision,
        f1=f1,
        avg_quality=avg_quality,
        matches=pair_grades,
        missed_gt=missed_gt_list,
        extra_agent=extra_agent_list,
        extra_count=len(match_result.extra_agent),
    )


def build_report(
    project_grades: dict[str, ProjectGrade],
    threshold: float,
    quality_threshold: float,
    backend_name: str,
    skipped_projects: list[str] | None = None,
    failed_projects: list[dict[str, str]] | None = None,
) -> GradeReport:
    """Build the full grading report across all projects.

    Recall is the primary metric. Quality is a pass/fail gate.
    """
    meta = {
        "grader_version": grader.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "match_threshold": threshold,
        "quality_threshold": quality_threshold,
        "quality_weights": QUALITY_WEIGHTS,
        "similarity_backend": backend_name,
        "skipped_projects": list(skipped_projects or []),
        "failed_projects": list(failed_projects or []),
    }

    total_gt = 0
    total_agent = 0
    total_passed = 0
    total_passed_agents = 0
    total_extra = 0
    quality_sum = 0.0
    quality_count = 0

    for pg in project_grades.values():
        n_gt = len({m.gt_id for m in pg.matches if m.passed}) + len(pg.missed_gt)
        # Also count GTs that matched but didn't pass
        unpassed_gts = {m.gt_id for m in pg.matches if not m.passed} - {m.gt_id for m in pg.matches if m.passed}
        n_gt += len(unpassed_gts)
        n_agent = len(pg.matches) + len(pg.extra_agent)
        passed_gt_count = len({m.gt_id for m in pg.matches if m.passed})
        passed_agent_count = sum(1 for m in pg.matches if m.passed)

        total_gt += n_gt
        total_agent += n_agent
        total_passed += passed_gt_count
        total_passed_agents += passed_agent_count
        total_extra += pg.extra_count

        passed_quals = [m.quality for m in pg.matches if m.passed]
        quality_sum += sum(passed_quals)
        quality_count += len(passed_quals)

    overall_recall = total_passed / total_gt if total_gt > 0 else 0.0
    overall_precision = (
        total_passed_agents / total_agent if total_agent > 0 else 0.0
    )
    overall_f1 = (
        2 * overall_precision * overall_recall
        / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )
    overall_avg_quality = quality_sum / quality_count if quality_count > 0 else 0.0

    overall = {
        "recall": round(overall_recall, 4),
        "precision": round(overall_precision, 4),
        "f1": round(overall_f1, 4),
        "avg_quality": round(overall_avg_quality, 4),
        "total_gt": total_gt,
        "total_agent": total_agent,
        "total_passed": total_passed,
        "total_passed_agents": total_passed_agents,
        "total_extra": total_extra,
    }

    return GradeReport(meta=meta, projects=project_grades, overall=overall)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _grade_to_dict(report: GradeReport) -> dict:
    """Convert GradeReport to a JSON-serializable dict."""
    projects = {}
    for name, pg in report.projects.items():
        projects[name] = _project_grade_to_dict(pg)

    return {
        "meta": report.meta,
        "projects": projects,
        "overall": report.overall,
    }


def _project_grade_to_dict(pg: ProjectGrade) -> dict:
    """Convert a single ProjectGrade to a JSON-serializable dict."""
    return {
        "project": pg.project,
        "recall": round(pg.recall, 4),
        "precision": round(pg.precision, 4),
        "f1": round(pg.f1, 4),
        "avg_quality": round(pg.avg_quality, 4),
        "matches": [
            {
                "gt_id": m.gt_id,
                "gt_name": m.gt_name,
                "agent_name": m.agent_name,
                "match_similarity": round(m.match_similarity, 4),
                "code_location": {"score": round(m.code_location_score.score, 4), "detail": m.code_location_score.detail},
                "paper_reference": {"score": round(m.paper_reference_score.score, 4), "detail": m.paper_reference_score.detail},
                "quality": round(m.quality, 4),
                "passed": m.passed,
                "dup_rank": m.dup_rank,
            }
            for m in pg.matches
        ],
        "missed_gt": pg.missed_gt,
        "extra_agent": pg.extra_agent,
        "extra_count": pg.extra_count,
    }


def _dict_to_project_grade(d: dict) -> ProjectGrade:
    """Reconstruct a ProjectGrade from a serialized dict."""
    matches = []
    for m in d["matches"]:
        code_loc = m.get("code_location", {})
        paper_ref = m.get("paper_reference", {})
        matches.append(PairGrade(
            gt_id=m["gt_id"],
            gt_name=m["gt_name"],
            agent_name=m["agent_name"],
            match_similarity=m["match_similarity"],
            code_location_score=FieldScore(
                score=code_loc.get("score", 0.0),
                detail=code_loc.get("detail", ""),
            ),
            paper_reference_score=FieldScore(
                score=paper_ref.get("score", 0.0),
                detail=paper_ref.get("detail", ""),
            ),
            quality=m.get("quality", 0.0),
            passed=m.get("passed", False),
            dup_rank=m.get("dup_rank", 0),
        ))
    return ProjectGrade(
        project=d["project"],
        recall=d["recall"],
        precision=d["precision"],
        f1=d["f1"],
        avg_quality=d.get("avg_quality", 0.0),
        matches=matches,
        missed_gt=d["missed_gt"],
        extra_agent=d["extra_agent"],
        extra_count=d.get("extra_count", len(d.get("extra_agent", []))),
    )


def write_json_report(report: GradeReport, path: str) -> None:
    """Write grading report as JSON."""
    data = _grade_to_dict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown_report(report: GradeReport, path: str) -> None:
    """Write grading report as human-readable markdown."""
    lines: list[str] = []
    o = report.overall

    lines.append("# zkML Benchmark Grading Report\n")
    lines.append(f"**Grader version:** {report.meta['grader_version']}  ")
    lines.append(f"**Date:** {report.meta['timestamp']}  ")
    lines.append(f"**Similarity backend:** {report.meta['similarity_backend']}  ")
    lines.append(f"**Match threshold:** {report.meta['match_threshold']}  ")
    lines.append(f"**Quality threshold:** {report.meta['quality_threshold']}\n")

    lines.append("## Overall Results\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Recall** | **{o['recall']:.4f}** |")
    lines.append(f"| Precision | {o['precision']:.4f} |")
    lines.append(f"| F1 | {o['f1']:.4f} |")
    lines.append(f"| Avg Quality | {o['avg_quality']:.4f} |")
    lines.append(f"| Total GT | {o['total_gt']} |")
    lines.append(f"| Total Agent | {o['total_agent']} |")
    lines.append(f"| Total Passed | {o['total_passed']} |")
    extra_count = o.get("total_extra", 0)
    if extra_count:
        lines.append(f"| Extra Findings | {extra_count} |")
    lines.append("")

    skipped = report.meta.get("skipped_projects") or []
    failed = report.meta.get("failed_projects") or []
    if skipped or failed:
        lines.append("## Skipped / failed projects\n")
        if skipped:
            lines.append("**Skipped (no ground truth available):** "
                         + ", ".join(sorted(skipped)))
            lines.append("")
        if failed:
            lines.append("**Failed during grading:**")
            lines.append("")
            lines.append("| Project | Error type | Error |")
            lines.append("|---------|------------|-------|")
            for f in failed:
                err_text = str(f.get("error", "")).replace("|", "\\|")
                lines.append(
                    f"| {f.get('project', '')} | {f.get('error_type', '')} | {err_text} |"
                )
            lines.append("")

    for name, pg in sorted(report.projects.items()):
        lines.append(f"## {name}\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| **Recall** | **{pg.recall:.4f}** |")
        lines.append(f"| Precision | {pg.precision:.4f} |")
        lines.append(f"| F1 | {pg.f1:.4f} |")
        lines.append(f"| Avg Quality | {pg.avg_quality:.4f} |")
        if pg.extra_count:
            lines.append(f"| Extra Findings | {pg.extra_count} |")
        lines.append("")

        if pg.matches:
            lines.append("### Matches\n")
            lines.append(
                "| GT ID | GT Name | Agent Name | Similarity | Quality | Passed | Code | Paper |"
            )
            lines.append(
                "|-------|---------|------------|------------|---------|--------|------|-------|"
            )
            for m in pg.matches:
                badge = _quality_badge(m.quality, m.passed)
                dup_marker = "" if m.dup_rank == 0 else f" (dup #{m.dup_rank})"
                passed_str = "✅" if m.passed else "❌"
                lines.append(
                    f"| {m.gt_id}{dup_marker} | {m.gt_name} | {m.agent_name} "
                    f"| {m.match_similarity:.2f} | {badge} {m.quality:.2f} | {passed_str} "
                    f"| {m.code_location_score.score:.2f} "
                    f"| {m.paper_reference_score.score:.2f} |"
                )
            lines.append("")

        if pg.missed_gt:
            lines.append("### Missed GT Findings\n")
            for mg in pg.missed_gt:
                lines.append(f"- {mg['id']}: {mg['name']}")
            lines.append("")

        if pg.extra_agent:
            lines.append("### Extra Agent Findings\n")
            for ea in pg.extra_agent:
                lines.append(f"- {ea['name']}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
