"""CLI entry point for the dataset loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_loader import DEFAULT_REPO_ID, BenchmarkDataset
from dataset_loader.materialize import materialize


def _list_pairs(args: argparse.Namespace) -> None:
    ds = BenchmarkDataset(repo_id=args.repo_id, revision=args.revision)
    pairs = ds.pairs()
    print(f"{'pair_id':<12} {'artifacts':>9}  paper")
    print("-" * 50)
    for p in pairs:
        print(f"{p.pair_id:<12} {len(p.artifact_paths):>9}  {p.paper_path}")


def _list_artifacts(args: argparse.Namespace) -> None:
    ds = BenchmarkDataset(repo_id=args.repo_id, revision=args.revision)
    pair_filter = args.pair or None
    ids = ds.artifact_ids(pair_id=pair_filter)
    for aid in ids:
        print(aid)
    print(f"\n{len(ids)} artifact(s)")


def _materialize(args: argparse.Namespace) -> None:
    ds = BenchmarkDataset(repo_id=args.repo_id, revision=args.revision)
    pair_ids = args.pairs.split(",") if args.pairs else None
    output = Path(args.output)

    print(f"Materializing run-set to {output} ...")
    materialize(
        ds,
        output,
        pair_ids=pair_ids,
        emit_batch_manifest=not args.no_manifest,
    )
    if pair_ids is None:
        pair_ids = ds.pair_ids()
    print(f"Done: {len(pair_ids)} pair(s) materialized.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dataset-loader",
        description="Load and materialize the zkml-audit-benchmark HF dataset",
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"HF dataset repo (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision", default=None,
        help="Git revision / branch / tag (default: main)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # list-pairs
    lp = sub.add_parser("list-pairs", help="List available (paper, codebase) pairs")
    lp.set_defaults(func=_list_pairs)

    # list-artifacts
    la = sub.add_parser("list-artifacts", help="List artifact IDs")
    la.add_argument("--pair", default=None, help="Filter by pair_id")
    la.set_defaults(func=_list_artifacts)

    # materialize
    mat = sub.add_parser(
        "materialize",
        help="Download and lay out a run-set for agent auditing",
    )
    mat.add_argument("--output", required=True, help="Output directory")
    mat.add_argument(
        "--pairs", default=None,
        help="Comma-separated pair_ids to include (default: all)",
    )
    mat.add_argument(
        "--no-manifest", action="store_true",
        help="Skip emitting batch_manifest.json",
    )
    mat.set_defaults(func=_materialize)

    args = parser.parse_args(argv)
    args.func(args)
