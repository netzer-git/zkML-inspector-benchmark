"""Iterative runner for condition D of the composition experiment.

Between iterations this script:
1. Reads the agent's output JSON from the previous iteration
2. Runs the grader matcher to identify which GT artifacts were detected
3. Oracle-reverts those artifacts (rebuilds codebase from clean + remaining artifacts)
4. Emits the next iteration's batch manifest + findings.json

The agent invocation itself is external (run /analyze-batch manually or via
a wrapper). This script only handles the *between-iteration* bookkeeping.

Usage:
    python -m dataset_generator.iterative_runner prepare \\
        --dataset-manifest path/to/dataset_manifest.json \\
        --clean-codebases path/to/clean_codebases/ \\
        --output path/to/composition_exp/ \\
        --max-iters 3 \\
        --reps 15

    # After each iteration, run the agent, then:
    python -m dataset_generator.iterative_runner advance \\
        --state path/to/composition_exp/iterative_state.json \\
        --agent-output path/to/agent_output.json \\
        --threshold 4
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dataset_generator.artifacts import Artifact, load_artifact
from dataset_generator.revert import revert_artifacts
from dataset_generator.emit import write_findings_json


@dataclass
class PairState:
    """Tracks per-(pair, rep) state across iterations."""
    pair_id: str
    rep: int
    remaining_artifact_ids: list[str]
    detected_artifact_ids: list[str] = field(default_factory=list)
    done: bool = False


@dataclass
class IterativeState:
    """Full state of the iterative experiment."""
    current_iter: int
    max_iters: int
    reps: int
    pairs: list[str]
    pair_states: list[PairState]
    clean_codebases_dir: str
    artifacts_dir: str
    output_dir: str
    threshold: int = 4

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "IterativeState":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["pair_states"] = [PairState(**ps) for ps in data["pair_states"]]
        return cls(**data)


def _entry_id_for(pair_id: str, rep: int, iteration: int) -> str:
    """Generate a unique entry-id for a (pair, rep, iter) combination."""
    return f"{pair_id}-D-rep{rep:02d}-iter{iteration}"


def prepare(
    *,
    dataset_manifest_path: Path,
    clean_codebases_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    max_iters: int = 3,
    reps: int = 15,
) -> IterativeState:
    """Set up the initial iterative state from a dataset manifest.

    Reads the dataset manifest to discover pairs and their artifact pools.
    Creates the iter-1 batch manifest (identical to condition B: all artifacts).

    Parameters
    ----------
    dataset_manifest_path
        Path to a dataset_manifest.json (from dataset_generator).
    clean_codebases_dir
        Directory containing clean codebases: <pair_id>/codebase/
    artifacts_dir
        Directory containing artifact JSON files: <pair_prefix>/<artifact_id>.json
    output_dir
        Where to write iterative experiment state and batch manifests.
    max_iters
        Maximum number of iterations (default 3).
    reps
        Number of repetitions per pair (default 15).

    Returns
    -------
    IterativeState
    """
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover unique pairs and their artifact pools
    pairs_artifacts: dict[str, list[str]] = {}
    for case in manifest["cases"]:
        source = case["source_codebase"]
        if source not in pairs_artifacts:
            pairs_artifacts[source] = []
        # Collect all artifact IDs for this codebase
        for aid in case["artifact_ids"]:
            if aid not in pairs_artifacts[source]:
                pairs_artifacts[source].append(aid)

    pair_states = []
    for pair_id, artifact_ids in sorted(pairs_artifacts.items()):
        for rep in range(1, reps + 1):
            pair_states.append(PairState(
                pair_id=pair_id,
                rep=rep,
                remaining_artifact_ids=list(artifact_ids),
            ))

    state = IterativeState(
        current_iter=1,
        max_iters=max_iters,
        reps=reps,
        pairs=sorted(pairs_artifacts.keys()),
        pair_states=pair_states,
        clean_codebases_dir=str(clean_codebases_dir),
        artifacts_dir=str(artifacts_dir),
        output_dir=str(output_dir),
    )
    state.save(output_dir / "iterative_state.json")
    return state


def build_iteration_manifest(
    state: IterativeState,
) -> tuple[Path, Path]:
    """Build the batch_manifest.json and findings.json for the current iteration.

    Only includes pairs that are not yet done. Each active (pair, rep)
    gets a case entry built from the clean codebase + remaining artifacts.

    Returns (batch_manifest_path, findings_path).
    """
    output_dir = Path(state.output_dir)
    iter_dir = output_dir / f"iter{state.current_iter}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = iter_dir / "cases"
    cases_dir.mkdir(exist_ok=True)

    clean_dir = Path(state.clean_codebases_dir)
    artifacts_dir = Path(state.artifacts_dir)

    manifest_entries = []
    all_findings: list[dict[str, str]] = []
    artifacts_cache: dict[str, Artifact] = {}

    for ps in state.pair_states:
        if ps.done:
            continue

        entry_id = _entry_id_for(ps.pair_id, ps.rep, state.current_iter)
        pair_clean = clean_dir / ps.pair_id / "codebase"
        paper_path = clean_dir / ps.pair_id / "paper.pdf"

        # Load remaining artifacts
        remaining = []
        for aid in ps.remaining_artifact_ids:
            if aid not in artifacts_cache:
                # Find artifact JSON file
                prefix = aid.split("-")[0].lower()
                art_path = artifacts_dir / prefix / f"{aid}.json"
                if not art_path.exists():
                    # Try with original casing in directory
                    for d in artifacts_dir.iterdir():
                        candidate = d / f"{aid}.json"
                        if candidate.exists():
                            art_path = candidate
                            break
                artifacts_cache[aid] = load_artifact(art_path)
            remaining.append(artifacts_cache[aid])

        if not remaining:
            ps.done = True
            continue

        # Build case from clean codebase + remaining artifacts
        from dataset_generator.assembler import build_case, CaseBuildError
        try:
            case = build_case(
                entry_id=entry_id,
                codebase_dir=pair_clean,
                codebase_name=ps.pair_id,
                paper_path=paper_path if paper_path.exists() else None,
                artifacts=remaining,
                output_dir=cases_dir,
            )
        except CaseBuildError as e:
            print(f"  WARN: build failed for {entry_id}: {e}")
            ps.done = True
            continue

        case_dir = cases_dir / entry_id
        manifest_entries.append({
            "entry-id": entry_id,
            "paper": str(case_dir / "paper.pdf"),
            "codebase": str(case_dir / "codebase"),
        })

        # Add findings for this entry
        from dataset_generator.emit import _artifact_to_finding
        for art in remaining:
            all_findings.append(_artifact_to_finding(art, entry_id))

    # Write batch manifest
    batch_manifest_path = iter_dir / "batch_manifest.json"
    batch_manifest_path.write_text(
        json.dumps(manifest_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write findings.json
    findings_path = iter_dir / "findings.json"
    findings_path.write_text(
        json.dumps(all_findings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return batch_manifest_path, findings_path


def advance(
    *,
    state_path: Path,
    agent_output_path: Path,
    threshold: int = 4,
) -> IterativeState:
    """Process agent output and advance to the next iteration.

    1. Loads the agent's output JSON for the current iteration
    2. For each (pair, rep), runs grader matcher to find detected GT artifacts
    3. Removes detected artifacts from remaining_artifact_ids
    4. Advances current_iter
    5. Saves updated state

    Parameters
    ----------
    state_path
        Path to iterative_state.json.
    agent_output_path
        Path to the agent's output JSON (flat array of findings).
    threshold
        Grader match threshold (default 4).

    Returns
    -------
    Updated IterativeState.
    """
    state = IterativeState.load(state_path)

    if state.current_iter > state.max_iters:
        print("Already at max iterations.")
        return state

    # Load agent output grouped by entry-id
    from grader.loader import load_agent_output, load_ground_truth

    agent_by_entry = load_agent_output(agent_output_path)

    # Load the current iteration's findings.json as ground truth
    iter_dir = Path(state.output_dir) / f"iter{state.current_iter}"
    gt_by_entry = load_ground_truth(iter_dir / "findings.json")

    # For each active (pair, rep), match and identify detected artifacts
    for ps in state.pair_states:
        if ps.done:
            continue

        entry_id = _entry_id_for(ps.pair_id, ps.rep, state.current_iter)
        entry_key = entry_id.lower()

        agent_findings = agent_by_entry.get(entry_key, [])
        gt_findings = gt_by_entry.get(entry_key, [])

        if not agent_findings:
            # Agent found nothing → this rep is done
            ps.done = True
            continue

        if not gt_findings:
            ps.done = True
            continue

        # Use a simple text-overlap matcher (no LLM calls) for the
        # iterative runner. The full LLM judge is used in the final
        # grading step; here we just need to identify which GT artifacts
        # the agent plausibly found.
        detected_ids = _simple_match(agent_findings, gt_findings, threshold)

        if not detected_ids:
            # Agent reported findings but none matched GT → done
            ps.done = True
            continue

        # Remove detected artifacts
        ps.detected_artifact_ids.extend(detected_ids)
        ps.remaining_artifact_ids = [
            aid for aid in ps.remaining_artifact_ids
            if aid not in detected_ids
        ]

        if not ps.remaining_artifact_ids:
            ps.done = True

    # Advance iteration
    state.current_iter += 1
    state.save(state_path)

    return state


def _simple_match(
    agent_findings: list,
    gt_findings: list,
    threshold: int,
) -> set[str]:
    """Lightweight keyword-overlap matching for the iterative runner.

    Returns the set of GT issue_ids that were "detected" by the agent.
    This avoids LLM calls during the iteration loop; the real LLM-based
    grading happens in the final analysis phase.

    For the composition experiment, we use the LLM judge for the final
    scoring. This simple matcher is just for deciding what to revert.
    """
    detected = set()
    for gt in gt_findings:
        gt_words = set(gt.issue_name.lower().split()) | set(
            gt.issue_explanation.lower().split()[:30]
        )
        for af in agent_findings:
            af_words = set(af.issue_name.lower().split()) | set(
                af.issue_explanation.lower().split()[:30]
            )
            overlap = len(gt_words & af_words)
            total = len(gt_words | af_words)
            if total > 0 and overlap / total > 0.3:
                detected.add(gt.issue_id)
                break
    return detected
