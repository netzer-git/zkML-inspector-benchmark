"""Sources adapter — loads papers, codebases, and artifacts from a sources directory.

This is the swap-point for future HF-repo ingestion. Everything downstream
depends only on the public API here.

Expected layout (placeholder — subject to change):
    sources_dir/
      sources.json        # [{entry-id, paper, codebase_zip, codebase_name}]
      papers/*.pdf
      codebases/*.zip     # each extracts to a root named <codebase_name>/
      artifacts/<codebase_name>/*.json   # strict v2 artifacts
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from dataset_generator.artifacts import Artifact, load_artifacts_from_dir


@dataclass
class SourceEntry:
    """A single entry mapping a paper to a codebase and its artifact pool."""

    entry_id: str
    paper: str  # relative path within sources_dir
    codebase_zip: str  # relative path within sources_dir
    codebase_name: str  # e.g. "zkml-fixed"


class Sources:
    """Adapter over a sources directory."""

    def __init__(self, sources_dir: Path) -> None:
        self._dir = sources_dir
        manifest_path = sources_dir / "sources.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"sources.json not found in {sources_dir}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("sources.json must be a JSON array")
        self._entries = [
            SourceEntry(
                entry_id=e["entry-id"],
                paper=e["paper"],
                codebase_zip=e["codebase_zip"],
                codebase_name=e["codebase_name"],
            )
            for e in raw
        ]
        self._by_id = {e.entry_id: e for e in self._entries}

    def iter_entries(self) -> list[SourceEntry]:
        return list(self._entries)

    def get_entry(self, entry_id: str) -> SourceEntry:
        if entry_id not in self._by_id:
            raise KeyError(f"Unknown entry-id: {entry_id}")
        return self._by_id[entry_id]

    def get_paper_path(self, entry_id: str) -> Path:
        entry = self.get_entry(entry_id)
        p = self._dir / entry.paper
        if not p.exists():
            raise FileNotFoundError(f"Paper not found: {p}")
        return p

    def extract_codebase(self, entry_id: str, dest: Path) -> Path:
        """Extract the codebase zip for an entry to dest/<codebase_name>/.

        Returns the path to the extracted codebase root.
        """
        entry = self.get_entry(entry_id)
        zip_path = self._dir / entry.codebase_zip
        if not zip_path.exists():
            raise FileNotFoundError(f"Codebase zip not found: {zip_path}")

        codebase_dest = dest / entry.codebase_name
        if codebase_dest.exists():
            shutil.rmtree(codebase_dest)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Security: reject entries that would escape dest
            for info in zf.infolist():
                member_path = (dest / info.filename).resolve()
                if not str(member_path).startswith(str(dest.resolve())):
                    raise ValueError(
                        f"Unsafe path in zip (escapes dest): {info.filename}"
                    )
            zf.extractall(dest)

        if not codebase_dest.exists():
            # The zip might extract to a differently-named root; find it
            extracted = [d for d in dest.iterdir() if d.is_dir()]
            if len(extracted) == 1 and extracted[0] != codebase_dest:
                extracted[0].rename(codebase_dest)
            elif not codebase_dest.exists():
                raise FileNotFoundError(
                    f"Expected extracted directory '{entry.codebase_name}' "
                    f"not found in {dest}"
                )

        return codebase_dest

    def get_artifact_pool(self, codebase_name: str) -> list[Artifact]:
        """Load all strict-v2 artifacts for a codebase."""
        artifacts_dir = self._dir / "artifacts" / codebase_name
        if not artifacts_dir.exists():
            return []
        return load_artifacts_from_dir(artifacts_dir)


def load_sources(sources_dir: str | Path) -> Sources:
    return Sources(Path(sources_dir))
