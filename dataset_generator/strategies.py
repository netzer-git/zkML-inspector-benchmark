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


# Registry for CLI dispatch
STRATEGIES: dict[str, type] = {
    "random": RandomStrategy,
    "all": AllStrategy,
}
