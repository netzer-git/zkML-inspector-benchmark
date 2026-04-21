"""Static conflict detection for artifact sets."""

from __future__ import annotations

from dataclasses import dataclass

from dataset_generator.artifacts import Artifact


@dataclass
class ConflictEdge:
    """A conflict between two artifacts, with a reason."""

    a: str
    b: str
    reason: str


def _regions_overlap(r1: dict, r2: dict) -> bool:
    """Check if two region dicts overlap (same file, overlapping line ranges)."""
    if r1["file"] != r2["file"]:
        return False
    return r1["start"] <= r2["end"] and r2["start"] <= r1["end"]


def detect_conflicts(artifacts: list[Artifact]) -> list[ConflictEdge]:
    """Detect static conflicts among a set of artifacts.

    Two artifacts conflict if:
    - Their regions overlap in the same file.
    - They share a semantic_tag.
    - Either lists the other in incompatible.
    """
    conflicts: list[ConflictEdge] = []
    by_id = {a.artifact_id: a for a in artifacts}

    for i, a in enumerate(artifacts):
        for b in artifacts[i + 1 :]:
            # Region overlap
            for ra in a.regions:
                for rb in b.regions:
                    if _regions_overlap(ra, rb):
                        conflicts.append(ConflictEdge(
                            a.artifact_id, b.artifact_id,
                            f"Overlapping regions in {ra['file']}: "
                            f"{ra['start']}-{ra['end']} vs {rb['start']}-{rb['end']}",
                        ))

            # Shared semantic tags
            shared = set(a.semantic_tags) & set(b.semantic_tags)
            if shared:
                conflicts.append(ConflictEdge(
                    a.artifact_id, b.artifact_id,
                    f"Shared semantic tags: {shared}",
                ))

            # Explicit incompatibility
            if b.artifact_id in a.incompatible or a.artifact_id in b.incompatible:
                conflicts.append(ConflictEdge(
                    a.artifact_id, b.artifact_id,
                    "Explicitly incompatible",
                ))

    return conflicts


def detect_requires_cycle(artifacts: list[Artifact]) -> list[str] | None:
    """Check for cycles in the requires graph. Returns the cycle path or None."""
    by_id = {a.artifact_id: a for a in artifacts}
    visited: set[str] = set()
    in_stack: set[str] = set()
    path: list[str] = []

    def dfs(aid: str) -> list[str] | None:
        if aid in in_stack:
            cycle_start = path.index(aid)
            return path[cycle_start:] + [aid]
        if aid in visited:
            return None
        visited.add(aid)
        in_stack.add(aid)
        path.append(aid)
        if aid in by_id:
            for dep in by_id[aid].requires:
                result = dfs(dep)
                if result is not None:
                    return result
        path.pop()
        in_stack.discard(aid)
        return None

    for a in artifacts:
        result = dfs(a.artifact_id)
        if result is not None:
            return result

    return None


def topological_sort(artifacts: list[Artifact]) -> list[Artifact]:
    """Sort artifacts by requires (dependencies first). Raises on cycle."""
    cycle = detect_requires_cycle(artifacts)
    if cycle is not None:
        raise ValueError(f"Dependency cycle detected: {' → '.join(cycle)}")

    by_id = {a.artifact_id: a for a in artifacts}
    visited: set[str] = set()
    order: list[str] = []

    def visit(aid: str) -> None:
        if aid in visited:
            return
        visited.add(aid)
        if aid in by_id:
            for dep in by_id[aid].requires:
                visit(dep)
        order.append(aid)

    for a in artifacts:
        visit(a.artifact_id)

    return [by_id[aid] for aid in order if aid in by_id]


def is_compatible_set(artifacts: list[Artifact]) -> bool:
    """Check if a set of artifacts has no conflicts and no dependency cycles."""
    if detect_conflicts(artifacts):
        return False
    if detect_requires_cycle(artifacts):
        return False
    return True
