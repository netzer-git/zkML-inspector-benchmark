"""CLI entry point for the dataset generator."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from dataset_loader import DEFAULT_REPO_ID
from dataset_generator.artifacts import Artifact
from dataset_generator.assembler import BuiltCase, CaseBuildError, build_case
from dataset_generator.emit import write_dataset_manifest, write_findings_json
from dataset_generator.sources import load_sources
from dataset_generator.strategies import STRATEGIES


def _test_command(args: argparse.Namespace) -> None:
    """Execute the 'test' subcommand: generate a dataset from sources."""
    repo_id = args.repo_id
    revision = args.revision
    output_dir = Path(args.output)
    seed = args.seed
    num_cases = args.num_cases
    artifacts_per_case = args.artifacts_per_case
    strategy_name = args.strategy

    # Load sources from HF
    print(f"Loading sources from {repo_id}...")
    sources = load_sources(repo_id=repo_id, revision=revision)
    entries = sources.iter_entries()
    print(f"  Found {len(entries)} source entries")

    # Build strategy
    strategy_cls = STRATEGIES.get(strategy_name)
    if strategy_cls is None:
        print(f"Unknown strategy: {strategy_name}", file=sys.stderr)
        sys.exit(1)
    # RandomStrategy requires k; AllStrategy takes no args
    if strategy_name == "random":
        strategy = strategy_cls(artifacts_per_case)
    else:
        strategy = strategy_cls()

    rng = random.Random(seed)

    # Prepare output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir()

    # Build cases
    built_cases: list[BuiltCase] = []
    all_artifacts: dict[str, Artifact] = {}
    errors: list[dict] = []

    # Create a flat pool of (entry, artifacts) assignments
    assignments: list[tuple] = []
    for entry in entries:
        pool = sources.get_artifact_pool(entry.entry_id)
        if not pool:
            print(f"  Warning: no artifacts for {entry.entry_id}, skipping")
            continue
        for a in pool:
            all_artifacts[a.artifact_id] = a

        for case_idx in range(num_cases):
            assignments.append((entry, pool, case_idx))

    print(f"  Building {len(assignments)} cases...")

    for entry, pool, case_idx in assignments:
        entry_id = f"{entry.entry_id}-{case_idx}" if case_idx > 0 else entry.entry_id

        try:
            selected = strategy.assign(pool, rng)
        except (ValueError, RuntimeError) as e:
            errors.append({
                "entry_id": entry_id,
                "phase": "assignment",
                "error": str(e),
            })
            print(f"  SKIP {entry_id}: {e}", file=sys.stderr)
            continue

        # Record all selected artifacts
        for a in selected:
            all_artifacts[a.artifact_id] = a

        # Extract codebase
        try:
            codebase_dir = sources.extract_codebase(
                entry.entry_id, cases_dir / entry_id
            )
        except Exception as e:
            errors.append({
                "entry_id": entry_id,
                "phase": "extraction",
                "error": str(e),
            })
            print(f"  SKIP {entry_id}: extraction failed: {e}", file=sys.stderr)
            continue

        # Get paper path (may not exist for test runs)
        try:
            paper_path = sources.get_paper_path(entry.entry_id)
        except FileNotFoundError:
            paper_path = None

        # Build case
        try:
            case = build_case(
                entry_id=entry_id,
                codebase_dir=codebase_dir,
                codebase_name=entry.codebase_name,
                paper_path=paper_path,
                artifacts=selected,
                output_dir=cases_dir,
            )
            built_cases.append(case)
            print(f"  OK {entry_id}: {len(selected)} artifacts applied")
        except CaseBuildError as e:
            errors.append({
                "entry_id": entry_id,
                "phase": "build",
                "error": str(e),
                "details": e.details,
            })
            print(f"  FAIL {entry_id}: {e}", file=sys.stderr)

        # Clean up extracted codebase (build_case wrote a modified copy)
        case_output_dir = cases_dir / entry_id
        if codebase_dir.exists() and codebase_dir != case_output_dir / "codebase":
            shutil.rmtree(codebase_dir, ignore_errors=True)

    if not built_cases:
        print("\nNo cases built successfully.", file=sys.stderr)
        if errors:
            json.dump(errors, sys.stderr, indent=2)
        sys.exit(1)

    # Emit outputs
    manifest_path = write_dataset_manifest(
        output_dir / "dataset_manifest.json",
        built_cases,
        strategy=strategy_name,
        seed=seed,
    )
    print(f"\nManifest: {manifest_path}")

    findings_path = write_findings_json(
        output_dir / "findings.json",
        built_cases,
        all_artifacts,
    )
    print(f"Findings: {findings_path}")

    if errors:
        err_path = output_dir / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Errors: {err_path}")

    print(f"\nDone: {len(built_cases)} cases, "
          f"{sum(len(c.artifact_ids) for c in built_cases)} total artifacts")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dataset-generator",
        description="Generate benchmark datasets from sources + artifacts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test subcommand
    test_parser = subparsers.add_parser(
        "test",
        help="Generate a test dataset from a sources directory",
    )
    test_parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"HF dataset repo (default: {DEFAULT_REPO_ID})",
    )
    test_parser.add_argument(
        "--revision", default=None,
        help="Git revision / branch / tag (default: main)",
    )
    test_parser.add_argument(
        "--output", required=True,
        help="Path for the output dataset directory",
    )
    test_parser.add_argument(
        "--num-cases", type=int, default=1,
        help="Number of cases to generate per source entry (default: 1)",
    )
    test_parser.add_argument(
        "--artifacts-per-case", type=int, default=1,
        help="Number of artifacts to select per case (default: 1)",
    )
    test_parser.add_argument(
        "--strategy", default="random", choices=list(STRATEGIES.keys()),
        help="Artifact selection strategy (default: random)",
    )
    test_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args(argv)

    if args.command == "test":
        _test_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
