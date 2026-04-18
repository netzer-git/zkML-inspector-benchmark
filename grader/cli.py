"""CLI entry point for the zkML benchmark grader."""

from __future__ import annotations

import argparse
import sys

from grader.loader import load_agent_output, load_ground_truth
from grader.matcher import match_findings
from grader.report import (
    DEFAULT_WEIGHTS,
    build_report,
    grade_project,
    write_json_report,
    write_markdown_report,
)
from grader.similarity import JaccardSimilarity, SimilarityBackend


def _parse_weights(raw: str | None) -> dict[str, float]:
    """Parse comma-separated key=value weight overrides."""
    if not raw:
        return dict(DEFAULT_WEIGHTS)
    weights = dict(DEFAULT_WEIGHTS)
    for pair in raw.split(","):
        k, v = pair.strip().split("=")
        k = k.strip()
        if k not in weights:
            raise ValueError(f"Unknown weight field: {k}. Valid: {list(weights.keys())}")
        weights[k] = float(v.strip())
    return weights


def _get_backend(name: str) -> SimilarityBackend:
    backends = {
        "jaccard": JaccardSimilarity,
    }
    if name not in backends:
        raise ValueError(f"Unknown similarity backend: {name}. Available: {list(backends.keys())}")
    return backends[name]()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="grader",
        description="zkML Agents Benchmark Grader",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth xlsx file",
    )
    parser.add_argument(
        "--agent-output", required=True,
        help="Path to agent output JSON file",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Minimum similarity for a match (default: 0.3)",
    )
    parser.add_argument(
        "--backend", default="jaccard",
        help="Similarity backend: jaccard (default)",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Comma-separated weight overrides, e.g. severity=0.2,code_location=0.25",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for JSON report output",
    )
    parser.add_argument(
        "--output-md", default=None,
        help="Path for markdown report output",
    )
    args = parser.parse_args(argv)

    weights = _parse_weights(args.weights)
    backend = _get_backend(args.backend)

    # Load data
    print(f"Loading ground truth from {args.ground_truth}...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  {sum(len(v) for v in gt.values())} findings across {len(gt)} projects")

    print(f"Loading agent output from {args.agent_output}...")
    agent = load_agent_output(args.agent_output)
    print(f"  {sum(len(v) for v in agent.values())} findings across {len(agent)} projects")

    # Grade each project
    all_projects = set(gt.keys()) | set(agent.keys())
    project_grades = {}

    for project in sorted(all_projects):
        gt_findings = gt.get(project, [])
        agent_findings = agent.get(project, [])

        if not gt_findings:
            print(f"  {project}: no ground truth available, skipping")
            continue

        match_result = match_findings(agent_findings, gt_findings, backend, args.threshold)
        pg = grade_project(
            project, match_result, backend, gt_findings, agent_findings, weights
        )
        project_grades[project] = pg

        n_matched = len(pg.matches)
        n_missed = len(pg.missed_gt)
        n_extra = len(pg.extra_agent)
        extra_sev = pg.extra_by_severity
        extra_detail = ""
        if extra_sev:
            extra_detail = " (" + ", ".join(f"{s}:{c}" for s, c in sorted(extra_sev.items())) + ")"
        print(
            f"  {project}: {n_matched} matched, {n_missed} missed, "
            f"{n_extra} extra{extra_detail}, composite={pg.composite:.3f}"
        )

    # Build report
    report = build_report(project_grades, args.threshold, weights, args.backend)

    o = report.overall
    print(f"\nOverall: benchmark_score={o['benchmark_score']:.4f}, "
          f"recall={o['recall']:.4f}, precision={o['precision']:.4f}, "
          f"f1={o['f1']:.4f}, quality={o['quality']:.4f}")

    if args.output:
        write_json_report(report, args.output)
        print(f"JSON report written to {args.output}")

    if args.output_md:
        write_markdown_report(report, args.output_md)
        print(f"Markdown report written to {args.output_md}")

    if not args.output and not args.output_md:
        print("\nTip: use --output and/or --output-md to save the report.")


if __name__ == "__main__":
    main()
