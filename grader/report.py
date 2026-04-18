"""Aggregate scoring and output formatting (JSON + markdown)."""

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
    score_category,
    score_code_location,
    score_paper_reference,
    score_security_concern,
    score_severity,
)
from grader.similarity import SimilarityBackend

# Default weights for the 5 core graded fields
DEFAULT_WEIGHTS = {
    "severity": 0.15,
    "category": 0.15,
    "security_concern": 0.15,
    "code_location": 0.30,
    "paper_reference": 0.25,
}

_SEVERITY_WEIGHT = {"Critical": 3, "Warning": 2, "Info": 1}


@dataclass
class PairGrade:
    """Grading result for a single matched (agent, GT) pair."""
    gt_id: str
    gt_name: str
    agent_name: str
    match_similarity: float
    scores: dict[str, FieldScore]
    pair_score: float


@dataclass
class ProjectGrade:
    """Grading result for all findings in one project."""
    project: str
    recall: float
    precision: float
    f1: float
    severity_weighted_recall: float
    quality: float
    composite: float
    matches: list[PairGrade]
    missed_gt: list[dict[str, str]]
    extra_agent: list[dict[str, str]]
    extra_by_severity: dict[str, int]


@dataclass
class GradeReport:
    """Full grading report across all projects."""
    meta: dict[str, Any]
    projects: dict[str, ProjectGrade]
    overall: dict[str, float]


def _compute_pair_score(
    scores: dict[str, FieldScore], weights: dict[str, float]
) -> float:
    """Compute weighted pair score, redistributing weight for skipped fields."""
    active_weights: dict[str, float] = {}
    for field_name, w in weights.items():
        fs = scores.get(field_name)
        if fs and "skip" not in fs.detail:
            active_weights[field_name] = w

    if not active_weights:
        return 0.0

    total_w = sum(active_weights.values())
    normalized = {k: v / total_w for k, v in active_weights.items()}
    return sum(normalized[k] * scores[k].score for k in normalized)


def grade_pair(
    pair: MatchedPair,
    similarity_backend: SimilarityBackend,
    weights: dict[str, float] | None = None,
) -> PairGrade:
    """Grade a single matched pair across all fields."""
    w = weights or DEFAULT_WEIGHTS

    scores: dict[str, FieldScore] = {
        "severity": score_severity(pair.agent.severity, pair.gt.severity),
        "category": score_category(pair.agent.category, pair.gt.category),
        "security_concern": score_security_concern(
            pair.agent.security_concern, pair.gt.security_concern
        ),
        "code_location": score_code_location(
            pair.agent.relevant_code, pair.gt.relevant_code
        ),
        "paper_reference": score_paper_reference(
            pair.agent.paper_reference, pair.gt.paper_reference, similarity_backend
        ),
    }

    pair_score = _compute_pair_score(scores, w)

    return PairGrade(
        gt_id=pair.gt.issue_id,
        gt_name=pair.gt.issue_name,
        agent_name=pair.agent.issue_name,
        match_similarity=pair.similarity,
        scores=scores,
        pair_score=pair_score,
    )


def grade_project(
    project: str,
    match_result: MatchResult,
    similarity_backend: SimilarityBackend,
    gt_findings: list[GroundTruthFinding],
    agent_findings: list[AgentFinding],
    weights: dict[str, float] | None = None,
) -> ProjectGrade:
    """Grade all findings for one project."""
    pair_grades = [
        grade_pair(mp, similarity_backend, weights) for mp in match_result.matched
    ]

    total_gt = len(gt_findings)
    total_agent = len(agent_findings)
    matched_gt = len(match_result.matched)
    matched_agent = len(match_result.matched)

    recall = matched_gt / total_gt if total_gt > 0 else 0.0
    precision = matched_agent / total_agent if total_agent > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    quality = (
        sum(pg.pair_score for pg in pair_grades) / len(pair_grades)
        if pair_grades
        else 0.0
    )

    # Severity-weighted recall
    swr_num = 0.0
    swr_den = 0.0
    matched_gt_severities = {mp.gt.issue_id for mp in match_result.matched}
    for gf in gt_findings:
        w = _SEVERITY_WEIGHT.get(gf.severity, 1)
        swr_den += w
        if gf.issue_id in matched_gt_severities:
            swr_num += w
    severity_weighted_recall = swr_num / swr_den if swr_den > 0 else 0.0

    composite = 0.4 * f1 + 0.6 * quality

    # Format missed and extra for output
    missed_gt_list = [
        {"id": gf.issue_id, "name": gf.issue_name, "severity": gf.severity}
        for gf in match_result.missed_gt
    ]
    extra_agent_list = [
        {"name": af.issue_name, "severity": af.severity}
        for af in match_result.extra_agent
    ]
    extra_by_sev = {
        sev: len(findings)
        for sev, findings in match_result.extra_by_severity.items()
    }

    return ProjectGrade(
        project=project,
        recall=recall,
        precision=precision,
        f1=f1,
        severity_weighted_recall=severity_weighted_recall,
        quality=quality,
        composite=composite,
        matches=pair_grades,
        missed_gt=missed_gt_list,
        extra_agent=extra_agent_list,
        extra_by_severity=extra_by_sev,
    )


def build_report(
    project_grades: dict[str, ProjectGrade],
    threshold: float,
    weights: dict[str, float],
    backend_name: str,
) -> GradeReport:
    """Build the full grading report across all projects."""
    meta = {
        "grader_version": grader.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "match_threshold": threshold,
        "weights": weights,
        "similarity_backend": backend_name,
    }

    # Overall metrics
    total_gt = 0
    total_agent = 0
    total_matched = 0
    weighted_quality_sum = 0.0
    weighted_quality_den = 0
    total_extra_by_sev: dict[str, int] = {}

    for pg in project_grades.values():
        n_gt = len(pg.matches) + len(pg.missed_gt)
        n_agent = len(pg.matches) + len(pg.extra_agent)
        total_gt += n_gt
        total_agent += n_agent
        total_matched += len(pg.matches)
        weighted_quality_sum += pg.quality * n_gt
        weighted_quality_den += n_gt
        for sev, count in pg.extra_by_severity.items():
            total_extra_by_sev[sev] = total_extra_by_sev.get(sev, 0) + count

    overall_recall = total_matched / total_gt if total_gt > 0 else 0.0
    overall_precision = total_matched / total_agent if total_agent > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )
    overall_quality = (
        weighted_quality_sum / weighted_quality_den if weighted_quality_den > 0 else 0.0
    )
    benchmark_score = 0.4 * overall_f1 + 0.6 * overall_quality

    overall = {
        "recall": round(overall_recall, 4),
        "precision": round(overall_precision, 4),
        "f1": round(overall_f1, 4),
        "quality": round(overall_quality, 4),
        "benchmark_score": round(benchmark_score, 4),
        "total_gt": total_gt,
        "total_agent": total_agent,
        "total_matched": total_matched,
        "extra_by_severity": total_extra_by_sev,
    }

    return GradeReport(meta=meta, projects=project_grades, overall=overall)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _grade_to_dict(report: GradeReport) -> dict:
    """Convert GradeReport to a JSON-serializable dict."""
    projects = {}
    for name, pg in report.projects.items():
        projects[name] = {
            "recall": round(pg.recall, 4),
            "precision": round(pg.precision, 4),
            "f1": round(pg.f1, 4),
            "severity_weighted_recall": round(pg.severity_weighted_recall, 4),
            "quality": round(pg.quality, 4),
            "composite": round(pg.composite, 4),
            "matches": [
                {
                    "gt_id": m.gt_id,
                    "gt_name": m.gt_name,
                    "agent_name": m.agent_name,
                    "match_similarity": round(m.match_similarity, 4),
                    "scores": {
                        k: {"score": round(v.score, 4), "detail": v.detail}
                        for k, v in m.scores.items()
                    },
                    "pair_score": round(m.pair_score, 4),
                }
                for m in pg.matches
            ],
            "missed_gt": pg.missed_gt,
            "extra_agent": pg.extra_agent,
            "extra_by_severity": pg.extra_by_severity,
        }

    return {
        "meta": report.meta,
        "projects": projects,
        "overall": report.overall,
    }


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
    lines.append(f"**Match threshold:** {report.meta['match_threshold']}\n")

    lines.append("## Overall Results\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Benchmark Score | **{o['benchmark_score']:.4f}** |")
    lines.append(f"| Recall | {o['recall']:.4f} |")
    lines.append(f"| Precision | {o['precision']:.4f} |")
    lines.append(f"| F1 | {o['f1']:.4f} |")
    lines.append(f"| Quality | {o['quality']:.4f} |")
    lines.append(f"| Total GT | {o['total_gt']} |")
    lines.append(f"| Total Agent | {o['total_agent']} |")
    lines.append(f"| Total Matched | {o['total_matched']} |")
    extra_sev = o.get("extra_by_severity", {})
    if extra_sev:
        extra_str = ", ".join(f"{s}: {c}" for s, c in sorted(extra_sev.items()))
        lines.append(f"| Extra by Severity | {extra_str} |")
    lines.append("")

    for name, pg in sorted(report.projects.items()):
        lines.append(f"## {name}\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Recall | {pg.recall:.4f} |")
        lines.append(f"| Precision | {pg.precision:.4f} |")
        lines.append(f"| F1 | {pg.f1:.4f} |")
        lines.append(f"| Sev-Weighted Recall | {pg.severity_weighted_recall:.4f} |")
        lines.append(f"| Quality | {pg.quality:.4f} |")
        lines.append(f"| Composite | **{pg.composite:.4f}** |")
        if pg.extra_by_severity:
            extra_str = ", ".join(
                f"{s}: {c}" for s, c in sorted(pg.extra_by_severity.items())
            )
            lines.append(f"| Extra by Severity | {extra_str} |")
        lines.append("")

        if pg.matches:
            lines.append("### Matches\n")
            lines.append(
                "| GT ID | GT Name | Agent Name | Similarity | Pair Score | Severity | Category | Security | Code | Paper |"
            )
            lines.append(
                "|-------|---------|------------|------------|------------|----------|----------|----------|------|-------|"
            )
            for m in pg.matches:
                s = m.scores
                lines.append(
                    f"| {m.gt_id} | {m.gt_name} | {m.agent_name} "
                    f"| {m.match_similarity:.2f} | {m.pair_score:.2f} "
                    f"| {s['severity'].score:.2f} | {s['category'].score:.2f} "
                    f"| {s['security_concern'].score:.2f} "
                    f"| {s['code_location'].score:.2f} "
                    f"| {s['paper_reference'].score:.2f} |"
                )
            lines.append("")

        if pg.missed_gt:
            lines.append("### Missed GT Findings\n")
            for mg in pg.missed_gt:
                lines.append(f"- **[{mg['severity']}]** {mg['id']}: {mg['name']}")
            lines.append("")

        if pg.extra_agent:
            lines.append("### Extra Agent Findings\n")
            for ea in pg.extra_agent:
                lines.append(f"- **[{ea['severity']}]** {ea['name']}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
