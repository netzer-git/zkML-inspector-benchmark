"""CLI entry point for the zkML benchmark grader."""

from __future__ import annotations

import argparse
import sys


def _force_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 so non-ASCII progress output
    (agent finding names, judge reasoning previews) doesn't crash on
    Windows' default cp1252 console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

from grader.loader import load_agent_output, load_ground_truth
from grader.matcher import match_findings
from grader.report import (
    DEFAULT_WEIGHTS,
    build_report,
    grade_project,
    write_json_report,
    write_markdown_report,
)
from grader.similarity import LLMJudgeSimilarity


# Test seam: tests may monkeypatch this attribute with a MockLLMProvider to
# exercise the pipeline without hitting a real API.
_LLM_PROVIDER_OVERRIDE = None


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


def _build_backend() -> LLMJudgeSimilarity:
    """Construct the LLM judge backend from env, or from the test override."""
    if _LLM_PROVIDER_OVERRIDE is not None:
        return LLMJudgeSimilarity(_LLM_PROVIDER_OVERRIDE)

    from grader.llm import (
        build_config_from_env,
        build_provider,
        load_dotenv_if_available,
    )
    load_dotenv_if_available()
    try:
        cfg = build_config_from_env()
    except Exception as e:
        raise ValueError(
            f"The grader requires LLM configuration. See .env.example. {e}"
        ) from e
    return LLMJudgeSimilarity(build_provider(cfg))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="grader",
        description=(
            "zkML Agents Benchmark Grader. Uses an LLM judge (OpenAI or "
            "Anthropic) to match agent findings to ground-truth findings. "
            "Configure via .env (see .env.example)."
        ),
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth JSON file (flat array of findings)",
    )
    parser.add_argument(
        "--agent-output", required=True,
        help="Path to agent output JSON file",
    )
    parser.add_argument(
        "--threshold", type=int, default=4,
        help="Minimum match_score (integer 1..5) for a match. Default 4. "
             "The judge returns an ordinal score on a 1..5 scale; scores "
             ">= threshold are treated as matches.",
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
    parser.add_argument(
        "--entry-id", action="append", default=None, metavar="ID",
        help="Restrict grading to one or more entry-ids "
             "(repeatable, case-insensitive). Default: all projects.",
    )
    parser.add_argument(
        "--judge-trace", default=None, metavar="PATH",
        help="Debug: write a markdown file with per-pair judge reasoning. "
             "Shows every GT candidate the LLM scored for each agent finding.",
    )
    args = parser.parse_args(argv)

    _force_utf8_stdout()

    weights = _parse_weights(args.weights)
    backend = _build_backend()

    # Load data
    print(f"Loading ground truth from {args.ground_truth}...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  {sum(len(v) for v in gt.values())} findings across {len(gt)} projects")

    print(f"Loading agent output from {args.agent_output}...")
    agent = load_agent_output(args.agent_output)
    print(f"  {sum(len(v) for v in agent.values())} findings across {len(agent)} projects")

    # Optional entry-id filter — restricts both GT and agent dicts to the
    # selected projects (case-insensitive).
    if args.entry_id:
        wanted = {eid.strip().lower() for eid in args.entry_id}
        gt = {k: v for k, v in gt.items() if k in wanted}
        agent = {k: v for k, v in agent.items() if k in wanted}
        print(f"Filtering to entry-ids: {sorted(wanted)}")

    # Grade each project. Projects with no GT counterpart are skipped.
    # Projects that raise during matching/grading are isolated — logged,
    # recorded in report meta, and the rest of the run continues.
    all_projects = set(gt.keys()) | set(agent.keys())
    project_grades: dict = {}
    match_results: dict = {}  # retained for --judge-trace rendering
    skipped_projects: list[str] = []
    failed_projects: list[dict[str, str]] = []

    for project in sorted(all_projects):
        gt_findings = gt.get(project, [])
        agent_findings = agent.get(project, [])

        if not gt_findings:
            skipped_projects.append(project)
            print(f"  {project}: no ground truth available, skipping")
            continue

        print(
            f"\n==> Project {project}: "
            f"{len(agent_findings)} agent findings vs {len(gt_findings)} GT -- "
            f"matching (1 LLM call per agent finding)..."
        )
        try:
            match_result = match_findings(
                agent_findings, gt_findings, backend, args.threshold,
                verbose=True,
            )
            print(
                f"    matching done. Grading {len(match_result.matched)} "
                f"matched pair(s) (paper-ref LLM calls)..."
            )
            pg = grade_project(
                project, match_result, backend, gt_findings, agent_findings, weights
            )
        except Exception as e:
            failed_projects.append({
                "project": project,
                "error_type": type(e).__name__,
                "error": str(e)[:300],
            })
            print(f"  {project}: FAILED ({type(e).__name__}: {e}); skipping")
            continue

        project_grades[project] = pg
        match_results[project] = match_result

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

    # Build report (backend is fixed now — "llm-judge")
    report = build_report(
        project_grades, args.threshold, weights, "llm-judge",
        skipped_projects=skipped_projects, failed_projects=failed_projects,
    )

    o = report.overall
    print(f"\nOverall: benchmark_score={o['benchmark_score']:.4f}, "
          f"recall={o['recall']:.4f}, precision={o['precision']:.4f}, "
          f"f1={o['f1']:.4f}, quality={o['quality']:.4f}")
    if failed_projects:
        print(f"NOTE: {len(failed_projects)} project(s) failed and are excluded "
              f"from the scores above: "
              f"{', '.join(f['project'] for f in failed_projects)}")

    if args.output:
        write_json_report(report, args.output)
        print(f"JSON report written to {args.output}")

    if args.output_md:
        write_markdown_report(report, args.output_md)
        print(f"Markdown report written to {args.output_md}")

    if args.judge_trace:
        from grader.judge_trace import write_judge_trace
        write_judge_trace(
            path=args.judge_trace,
            match_results=match_results,
            meta={
                "grader_version": report.meta["grader_version"],
                "timestamp": report.meta["timestamp"],
                "threshold": args.threshold,
                "backend": "llm-judge",
                "entry_ids": args.entry_id,
            },
        )
        print(f"Judge trace written to {args.judge_trace}")

    if not args.output and not args.output_md and not args.judge_trace:
        print("\nTip: use --output, --output-md, or --judge-trace to save output.")


if __name__ == "__main__":
    main()
