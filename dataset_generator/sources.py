"""Sources adapter — loads papers, codebases, and artifacts from HF.

Wraps :class:`dataset_loader.BenchmarkDataset` to provide the public API
consumed by the rest of the dataset_generator package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataset_loader import DEFAULT_REPO_ID, BenchmarkDataset
from dataset_generator.artifacts import Artifact, load_artifact


@dataclass
class SourceEntry:
    """A single entry mapping a paper to a codebase and its artifact pool."""

    entry_id: str          # = pair_id
    codebase_name: str     # e.g. "zkml-fixed" (not used for HF lookups)


class Sources:
    """Adapter over the HF benchmark dataset."""

    def __init__(self, dataset: BenchmarkDataset) -> None:
        self._ds = dataset
        pairs = dataset.pairs()
        self._entries = [
            SourceEntry(entry_id=p.pair_id, codebase_name=p.pair_id)
            for p in pairs
        ]
        self._by_id = {e.entry_id: e for e in self._entries}

    def iter_entries(self) -> list[SourceEntry]:
        return list(self._entries)

    def get_entry(self, entry_id: str) -> SourceEntry:
        if entry_id not in self._by_id:
            raise KeyError(f"Unknown entry-id: {entry_id}")
        return self._by_id[entry_id]

    def get_paper_path(self, entry_id: str) -> Path:
        self.get_entry(entry_id)  # validate
        return self._ds.paper_path(entry_id)

    def extract_codebase(self, entry_id: str, dest: Path) -> Path:
        """Extract the codebase zip for an entry into *dest*.

        Returns the path to the extracted codebase root.
        """
        self.get_entry(entry_id)  # validate
        return self._ds.extract_codebase(entry_id, dest)

    def get_artifact_pool(self, entry_id: str) -> list[Artifact]:
        """Download and load all strict-v2 artifacts for *entry_id*."""
        artifact_ids = self._ds.artifact_ids(pair_id=entry_id)
        artifacts: list[Artifact] = []
        for aid in artifact_ids:
            path = self._ds.artifact_json_path(aid)
            artifacts.append(load_artifact(path))
        return artifacts


def load_sources(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
) -> Sources:
    """Create a :class:`Sources` backed by the HF benchmark dataset."""
    return Sources(BenchmarkDataset(repo_id=repo_id, revision=revision))
