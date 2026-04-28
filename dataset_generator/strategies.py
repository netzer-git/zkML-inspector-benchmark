"""Assignment strategies for selecting artifact subsets per case."""

from __future__ import annotations

import random
from typing import Protocol

from dataset_generator.artifacts import Artifact
from dataset_generator.conflict import detect_conflicts, detect_requires_cycle


class AssignmentStrategy(Protocol):
    """Protocol for artifact assignment strategies."""

    def assign(
        self,
        artifact_pool: list[Artifact],
        rng: random.Random,
    ) -> list[Artifact]:
        """Select a subset of artifacts from the pool."""
        ...


class RandomStrategy:
    """Select k artifacts uniformly at random, rejecting conflict sets."""

    def __init__(self, k: int, max_attempts: int = 100) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.max_attempts = max_attempts

    def assign(
        self,
        artifact_pool: list[Artifact],
        rng: random.Random,
    ) -> list[Artifact]:
        if len(artifact_pool) < self.k:
            raise ValueError(
                f"Pool has {len(artifact_pool)} artifacts but k={self.k} requested"
            )

        for _ in range(self.max_attempts):
            sample = rng.sample(artifact_pool, self.k)
            # Check for conflicts
            if not detect_conflicts(sample) and not detect_requires_cycle(sample):
                # Collect all required artifacts that aren't already in sample
                sample_ids = {a.artifact_id for a in sample}
                pool_by_id = {a.artifact_id: a for a in artifact_pool}
                needed = set()
                for a in sample:
                    for req in a.requires:
                        if req not in sample_ids:
                            needed.add(req)

                # Add required dependencies
                extended = list(sample)
                valid = True
                for req_id in needed:
                    if req_id not in pool_by_id:
                        valid = False
                        break
                    extended.append(pool_by_id[req_id])

                if valid and not detect_conflicts(extended) and not detect_requires_cycle(extended):
                    return extended

        raise RuntimeError(
            f"Failed to find a conflict-free set of {self.k} artifacts "
            f"after {self.max_attempts} attempts"
        )


class AllStrategy:
    """Select all compatible artifacts from the pool (for testing)."""

    def assign(
        self,
        artifact_pool: list[Artifact],
        rng: random.Random,
    ) -> list[Artifact]:
        if not detect_conflicts(artifact_pool) and not detect_requires_cycle(artifact_pool):
            return list(artifact_pool)
        raise ValueError("Pool contains conflicting artifacts; cannot use 'all' strategy")


class IsolatedStrategy:
    """Yield one case per artifact in the pool (deterministic, ignores rng).

    Used by the composition experiment to create N single-artifact variants
    from a pool of N artifacts. The ``assign`` method is called N times by
    the outer loop (once per ``num_cases``); call index ``i`` returns the
    ``i``-th artifact (sorted by artifact_id for reproducibility).

    The caller must set ``num_cases = len(pool)`` so the outer loop calls
    ``assign`` exactly N times.
    """

    def __init__(self) -> None:
        self._sorted_pool: list[Artifact] | None = None
        self._call_index: int = 0

    def assign(
        self,
        artifact_pool: list[Artifact],
        rng: random.Random,
    ) -> list[Artifact]:
        if self._sorted_pool is None:
            self._sorted_pool = sorted(artifact_pool, key=lambda a: a.artifact_id)
        if self._call_index >= len(self._sorted_pool):
            raise ValueError(
                f"IsolatedStrategy exhausted: called {self._call_index + 1} times "
                f"but pool has only {len(self._sorted_pool)} artifacts"
            )
        artifact = self._sorted_pool[self._call_index]
        self._call_index += 1

        # Include transitive dependencies
        pool_by_id = {a.artifact_id: a for a in artifact_pool}
        result = [artifact]
        for req_id in artifact.requires:
            if req_id in pool_by_id:
                result.append(pool_by_id[req_id])
        return result


class FixedSubsetStrategy:
    """Select a fixed list of artifact IDs (deterministic, ignores rng).

    Used by the composition experiment for reproducible intermediate-k
    variants (condition C). The caller provides the exact artifact IDs to
    include; the strategy resolves transitive dependencies and validates
    against conflicts.
    """

    def __init__(self, artifact_ids: list[str]) -> None:
        if not artifact_ids:
            raise ValueError("artifact_ids must be non-empty")
        self._artifact_ids = artifact_ids

    def assign(
        self,
        artifact_pool: list[Artifact],
        rng: random.Random,
    ) -> list[Artifact]:
        pool_by_id = {a.artifact_id: a for a in artifact_pool}
        missing = [aid for aid in self._artifact_ids if aid not in pool_by_id]
        if missing:
            raise ValueError(f"Artifacts not in pool: {missing}")

        selected = [pool_by_id[aid] for aid in self._artifact_ids]

        # Resolve transitive dependencies
        selected_ids = set(self._artifact_ids)
        for a in list(selected):
            for req_id in a.requires:
                if req_id not in selected_ids:
                    if req_id not in pool_by_id:
                        raise ValueError(
                            f"Artifact {a.artifact_id} requires {req_id} "
                            f"which is not in the pool"
                        )
                    selected.append(pool_by_id[req_id])
                    selected_ids.add(req_id)

        if detect_conflicts(selected):
            raise ValueError(
                f"Fixed subset {self._artifact_ids} has conflicts"
            )
        if detect_requires_cycle(selected):
            raise ValueError(
                f"Fixed subset {self._artifact_ids} has dependency cycles"
            )
        return selected


# Registry for CLI dispatch
STRATEGIES: dict[str, type] = {
    "random": RandomStrategy,
    "all": AllStrategy,
    "isolated": IsolatedStrategy,
    "fixed": FixedSubsetStrategy,
}
