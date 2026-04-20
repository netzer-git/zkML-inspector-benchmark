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

# Severity weights for severity-weighted recall (used to prioritize
# matching Critical > Warning > Info misses).
_SEVERITY_WEIGHT = {"Critical": 3, "Warning": 2, "Info": 1}

# Per-severity weight used in PRECISION. Rationale: the current dataset
# focuses on Critical findings; Warning is under-represented, so an extra
# Warning outside the GT should not be penalized as heavily as an extra
# Critical would be. Info-severity agent findings are typically defensive
# observations ("X is correctly implemented", "no impact but noted") rather
# than vulnerability claims — they are excluded from precision entirely
# (weight 0) so that producing helpful Info notes never hurts an agent's
# score. Info findings still appear in `extras_by_severity` for transparency.
_PRECISION_SEVERITY_WEIGHT = {"Critical": 1.0, "Warning": 0.5, "Info": 0.0}


def _pair_score_badge(pair_score: float) -> str:
    """Return a green/yellow/red emoji badge for a pair_score.

    Used in markdown reports to highlight match quality at a glance. Zero
    external dependencies; renders in any Markdown viewer.
    """
    if pair_score >= 0.7:
        return "🟢"
    if pair_score >= 0.4:
        return "🟡"
    return "🔴"


@dataclass
class PairGrade:
    """Grading result for a single matched (agent, GT) pair."""
    gt_id: str
    gt_name: str
    agent_name: str
    match_similarity: float
    scores: dict[str, FieldScore]
    pair_score: float
    # 0 = primary / only match for this GT; >=1 = duplicate (same GT matched
    # by another agent earlier in greedy order). Duplicates contribute to
    # quality via the per-GT average but are flagged in the report.
    dup_rank: int = 0


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
    # Numerator / denominator used to compute severity_weighted_recall, kept
    # so the overall roll-up can aggregate them across projects without
    # recomputing from gt_findings (which build_report doesn't see).
    swr_num: float = 0.0
    swr_den: float = 0.0


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
        dup_rank=pair.dup_rank,
    )


def grade_project(
    project: str,
    match_result: MatchResult,
    similarity_backend: SimilarityBackend,
    gt_findings: list[GroundTruthFinding],
    agent_findings: list[AgentFinding],
    weights: dict[str, float] | None = None,
) -> ProjectGrade:
    """Grade all findings for one project.

    Scoring handles N-to-1 matches (multiple agent findings bound to the same
    GT, which happens when the agent split one underlying issue into several
    findings). Behavior:

    - recall: unique GTs matched / total GTs (reported, not used in composite).
    - severity_weighted_recall: used in the F1 term of the composite. Weights
      Critical=3, Warning=2, Info=1 so missing a Critical is worse than an Info.
    - quality: severity-weighted mean across GT groups; each GT group
      contributes its pair_score weighted by GT severity. Duplicates are
      averaged within the group first.
    - precision: severity-weighted at 1.0/0.5/0.1. Extras subtract in
      proportion to their severity; all matched agents count toward matched.
    - composite: 0.4 * F1(precision, severity_weighted_recall) + 0.6 * quality.
    """
    pair_grades = [
        grade_pair(mp, similarity_backend, weights) for mp in match_result.matched
    ]

    total_gt = len(gt_findings)
    total_agent = len(agent_findings)
    gt_by_id = {gf.issue_id: gf for gf in gt_findings}

    # Recall: unique GT issue_ids matched.
    unique_matched_gts = {mp.gt.issue_id for mp in match_result.matched}
    recall = len(unique_matched_gts) / total_gt if total_gt > 0 else 0.0

    # Severity-weighted precision. All matched agents contribute; extras
    # subtract in proportion to their severity weight.
    if total_agent > 0:
        matched_agent_ids = {id(mp.agent) for mp in match_result.matched}
        weighted_total = sum(
            _PRECISION_SEVERITY_WEIGHT.get(af.severity, 0.0) for af in agent_findings
        )
        weighted_matched = sum(
            _PRECISION_SEVERITY_WEIGHT.get(af.severity, 0.0)
            for af in agent_findings
            if id(af) in matched_agent_ids
        )
        precision = weighted_matched / weighted_total if weighted_total > 0 else 0.0
    else:
        precision = 0.0

    # Severity-weighted recall: weights GT hits/misses by severity so
    # missing a Critical costs more than missing an Info.
    swr_num = 0.0
    swr_den = 0.0
    for gf in gt_findings:
        w = _SEVERITY_WEIGHT.get(gf.severity, 1)
        swr_den += w
        if gf.issue_id in unique_matched_gts:
            swr_num += w
    severity_weighted_recall = swr_num / swr_den if swr_den > 0 else 0.0

    # F1 uses severity-weighted recall so the composite reflects audit quality:
    # a Critical miss hurts more than an Info miss.
    f1 = (
        2 * precision * severity_weighted_recall
        / (precision + severity_weighted_recall)
        if (precision + severity_weighted_recall) > 0
        else 0.0
    )

    # Quality: severity-weighted mean of per-GT-group pair_scores. A GT
    # matched by N agents contributes the average of those pair_scores,
    # weighted by the GT's severity weight.
    if pair_grades:
        gt_group_scores: dict[str, list[float]] = {}
        for pg in pair_grades:
            gt_group_scores.setdefault(pg.gt_id, []).append(pg.pair_score)
        num = 0.0
        den = 0.0
        for gt_id, pair_scores in gt_group_scores.items():
            gt = gt_by_id.get(gt_id)
            w = _SEVERITY_WEIGHT.get(gt.severity, 1) if gt else 1
            num += w * (sum(pair_scores) / len(pair_scores))
            den += w
        quality = num / den if den > 0 else 0.0
    else:
        quality = 0.0

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
        swr_num=swr_num,
        swr_den=swr_den,
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
    skipped_projects: list[str] | None = None,
    failed_projects: list[dict[str, str]] | None = None,
) -> GradeReport:
    """Build the full grading report across all projects.

    Args:
        project_grades: Successfully graded projects.
        threshold: Match threshold used.
        weights: Field weights used.
        backend_name: Label for the similarity backend.
        skipped_projects: Agent project keys that had no GT counterpart and
            were therefore skipped. Recorded in meta for transparency.
        failed_projects: Projects where matching/grading raised an exception.
            Each entry is ``{"project", "error_type", "error"}``. Recorded in
            meta so the user can see which projects didn't get a score.
    """
    meta = {
        "grader_version": grader.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "match_threshold": threshold,
        "weights": weights,
        "similarity_backend": backend_name,
        "skipped_projects": list(skipped_projects or []),
        "failed_projects": list(failed_projects or []),
    }

    # Overall metrics.
    #
    # Recall and quality operate on UNIQUE GTs (N:1 duplicates collapse to
    # one GT). Precision is severity-weighted against all agent findings.
    # total_matched = unique GTs with at least one matching agent.
    total_gt = 0
    total_agent = 0
    total_matched = 0                     # unique GTs matched, across projects
    total_matched_agents = 0              # agent edges (including duplicates)
    total_weighted_agent = 0.0            # severity-weighted agent total
    total_weighted_matched_agent = 0.0    # severity-weighted matched agents
    total_swr_num = 0.0                   # severity-weighted recall numerator
    total_swr_den = 0.0                   # severity-weighted recall denominator
    weighted_quality_sum = 0.0            # per-GT quality sum (weight: severity-weighted GT mass)
    weighted_quality_den = 0.0
    total_extra_by_sev: dict[str, int] = {}

    for pg in project_grades.values():
        n_gt_unique = len({m.gt_id for m in pg.matches}) + len(pg.missed_gt)
        n_agent_edges = len(pg.matches) + len(pg.extra_agent)
        unique_matched_gt_count = len({m.gt_id for m in pg.matches})

        total_gt += n_gt_unique
        total_agent += n_agent_edges
        total_matched += unique_matched_gt_count
        total_matched_agents += len(pg.matches)

        # Accumulate precision weights. Extras are in pg.extra_agent; matched
        # agents are in pg.matches. Severity for matched agents is looked up
        # from the ProjectGrade — but we stored only the pair's agent_name
        # in PairGrade, not severity. To avoid losing fidelity we recompute
        # weighted totals from pg.precision * weighted_total. An equivalent
        # approximation: assume matched agents are all-Critical (weight 1.0)
        # because recall/precision at project level already baked the real
        # weights in. Here we aggregate via the already-computed pg.precision.
        # (See tests — this aggregation matches per-project precision exactly
        # when there's only one project.)
        # Cleaner path: pg.precision is (weighted_matched / weighted_total);
        # use it to recover both numerator and denominator when combined with
        # extras' severities.
        extras_weight = sum(
            _PRECISION_SEVERITY_WEIGHT.get(sev, 0.0) * count
            for sev, count in pg.extra_by_severity.items()
        )
        # At the project level: precision = weighted_matched / (weighted_matched + extras_weight)
        # => weighted_matched = precision * (weighted_matched + extras_weight)
        # => weighted_matched * (1 - precision) = precision * extras_weight
        # => weighted_matched = precision * extras_weight / (1 - precision)
        if pg.precision >= 1.0:
            # No extras, or all matched → weighted_matched is all the matched
            # severity weights. We don't carry these numerically, so treat
            # matched agents as contributing 1.0 each (upper bound).
            weighted_matched = float(len(pg.matches))
            weighted_total = weighted_matched
        elif pg.precision > 0.0:
            weighted_matched = pg.precision * extras_weight / (1.0 - pg.precision)
            weighted_total = weighted_matched + extras_weight
        else:
            weighted_matched = 0.0
            weighted_total = extras_weight
        total_weighted_matched_agent += weighted_matched
        total_weighted_agent += weighted_total

        total_swr_num += pg.swr_num
        total_swr_den += pg.swr_den

        # Weight each project's quality contribution by its severity-weighted
        # GT mass so projects with more Criticals dominate the roll-up quality
        # proportionally to audit importance.
        weighted_quality_sum += pg.quality * pg.swr_den
        weighted_quality_den += pg.swr_den
        for sev, count in pg.extra_by_severity.items():
            total_extra_by_sev[sev] = total_extra_by_sev.get(sev, 0) + count

    overall_recall = total_matched / total_gt if total_gt > 0 else 0.0
    overall_severity_weighted_recall = (
        total_swr_num / total_swr_den if total_swr_den > 0 else 0.0
    )
    overall_precision = (
        total_weighted_matched_agent / total_weighted_agent
        if total_weighted_agent > 0 else 0.0
    )
    # F1 in the composite uses severity-weighted recall so the headline score
    # reflects audit quality (missing a Critical counts more than missing an Info).
    overall_f1 = (
        2 * overall_precision * overall_severity_weighted_recall
        / (overall_precision + overall_severity_weighted_recall)
        if (overall_precision + overall_severity_weighted_recall) > 0
        else 0.0
    )
    overall_quality = (
        weighted_quality_sum / weighted_quality_den if weighted_quality_den > 0 else 0.0
    )
    benchmark_score = 0.4 * overall_f1 + 0.6 * overall_quality

    overall = {
        "recall": round(overall_recall, 4),
        "severity_weighted_recall": round(overall_severity_weighted_recall, 4),
        "precision": round(overall_precision, 4),
        "f1": round(overall_f1, 4),
        "quality": round(overall_quality, 4),
        "benchmark_score": round(benchmark_score, 4),
        "total_gt": total_gt,
        "total_agent": total_agent,
        "total_matched": total_matched,          # unique GTs matched
        "total_matched_agents": total_matched_agents,  # includes N:1 duplicates
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
                    "dup_rank": m.dup_rank,
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
    lines.append(f"| Recall (unweighted) | {o['recall']:.4f} |")
    lines.append(f"| Sev-Weighted Recall | {o.get('severity_weighted_recall', 0.0):.4f} |")
    lines.append(f"| Precision | {o['precision']:.4f} |")
    lines.append(f"| F1 (sev-weighted) | {o['f1']:.4f} |")
    lines.append(f"| Quality (sev-weighted) | {o['quality']:.4f} |")
    lines.append(f"| Total GT | {o['total_gt']} |")
    lines.append(f"| Total Agent | {o['total_agent']} |")
    lines.append(f"| Total Matched | {o['total_matched']} |")
    extra_sev = o.get("extra_by_severity", {})
    if extra_sev:
        extra_str = ", ".join(f"{s}: {c}" for s, c in sorted(extra_sev.items()))
        lines.append(f"| Extra by Severity | {extra_str} |")
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
                badge = _pair_score_badge(m.pair_score)
                dup_marker = "" if m.dup_rank == 0 else f" (dup #{m.dup_rank})"
                lines.append(
                    f"| {m.gt_id}{dup_marker} | {m.gt_name} | {m.agent_name} "
                    f"| {m.match_similarity:.2f} | {badge} {m.pair_score:.2f} "
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
