"""HF-backed dataset loader for zkml-audit-benchmark.

Downloads papers, codebases, and artifacts from the Hugging Face dataset
``Netzerep/zkml-audit-benchmark`` and provides a typed API for the
dataset_generator and grader pipelines.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_tree

__version__ = "0.1.0"

DEFAULT_REPO_ID = "Netzerep/zkml-audit-benchmark"

# Artifact-ID prefix → pair_id mapping (authoritative)
_PREFIX_TO_PAIR: dict[str, str] = {
    "zkLLM": "zkllm",
    "zkML": "zkml",
    "zkTorch": "zktorch",
    "zkGPT": "zkgpt",
}

PAIR_IDS = sorted(_PREFIX_TO_PAIR.values())


def _pair_id_for_artifact(artifact_id: str) -> str:
    """Derive pair_id from an artifact_id like ``zkLLM-001``."""
    prefix = artifact_id.rsplit("-", 1)[0]
    if prefix not in _PREFIX_TO_PAIR:
        raise ValueError(
            f"Unknown artifact prefix '{prefix}' in '{artifact_id}'. "
            f"Known: {sorted(_PREFIX_TO_PAIR)}"
        )
    return _PREFIX_TO_PAIR[prefix]


@dataclass(frozen=True)
class PairInfo:
    """Lightweight metadata for one (paper, codebase) pair."""

    pair_id: str
    paper_path: str          # relative path inside the repo, e.g. "papers/zkllm.pdf"
    codebase_path: str       # e.g. "codebases/zkllm.zip"
    artifact_paths: list[str]  # e.g. ["artifacts/zkllm/zkLLM-001.json", ...]


class BenchmarkDataset:
    """Facade over the HF dataset repo.

    All heavy downloads are deferred to the individual accessor methods
    and cached by ``huggingface_hub``.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str | None = None,
    ) -> None:
        self._repo_id = repo_id
        self._revision = revision
        self._manifest: dict[str, Any] | None = None
        self._pairs_cache: list[PairInfo] | None = None

    # -- manifest ----------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            path = hf_hub_download(
                repo_id=self._repo_id,
                filename="MANIFEST.json",
                repo_type="dataset",
                revision=self._revision,
            )
            self._manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        return self._manifest

    # -- pairs -------------------------------------------------------------

    def pair_ids(self) -> list[str]:
        """Return sorted list of available pair_ids."""
        return [p.pair_id for p in self.pairs()]

    def pairs(self) -> list[PairInfo]:
        """Return metadata for every (paper, codebase) pair."""
        if self._pairs_cache is not None:
            return list(self._pairs_cache)

        manifest = self._load_manifest()
        files_by_prefix: dict[str, dict[str, Any]] = {}

        for entry in manifest["files"]:
            p = entry["path"]
            for pid in PAIR_IDS:
                if p == f"papers/{pid}.pdf":
                    files_by_prefix.setdefault(pid, {})["paper"] = p
                elif p == f"codebases/{pid}.zip":
                    files_by_prefix.setdefault(pid, {})["codebase"] = p
                elif p.startswith(f"artifacts/{pid}/"):
                    files_by_prefix.setdefault(pid, {}).setdefault(
                        "artifacts", []
                    ).append(p)

        pairs: list[PairInfo] = []
        for pid in sorted(files_by_prefix):
            info = files_by_prefix[pid]
            pairs.append(
                PairInfo(
                    pair_id=pid,
                    paper_path=info.get("paper", f"papers/{pid}.pdf"),
                    codebase_path=info.get("codebase", f"codebases/{pid}.zip"),
                    artifact_paths=sorted(info.get("artifacts", [])),
                )
            )
        self._pairs_cache = pairs
        return list(pairs)

    def get_pair(self, pair_id: str) -> PairInfo:
        """Return :class:`PairInfo` for a single pair. Raises ``KeyError``."""
        for p in self.pairs():
            if p.pair_id == pair_id:
                return p
        raise KeyError(f"Unknown pair_id: {pair_id!r}")

    # -- file downloads ----------------------------------------------------

    def _download(self, repo_path: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=self._repo_id,
                filename=repo_path,
                repo_type="dataset",
                revision=self._revision,
            )
        )

    def paper_path(self, pair_id: str) -> Path:
        """Download (if needed) and return local path to the paper PDF."""
        info = self.get_pair(pair_id)
        return self._download(info.paper_path)

    def codebase_zip_path(self, pair_id: str) -> Path:
        """Download (if needed) and return local path to the codebase zip."""
        info = self.get_pair(pair_id)
        return self._download(info.codebase_path)

    def extract_codebase(self, pair_id: str, dest: Path) -> Path:
        """Extract the codebase zip for *pair_id* into *dest*.

        Returns the path to the extracted root directory
        (``dest/{pair_id}/``).  Raises ``ValueError`` on unsafe zip entries.
        """
        zip_path = self.codebase_zip_path(pair_id)
        codebase_dest = dest / pair_id

        if codebase_dest.exists():
            shutil.rmtree(codebase_dest)
        codebase_dest.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                resolved = (codebase_dest / info.filename).resolve()
                if not str(resolved).startswith(str(codebase_dest.resolve())):
                    raise ValueError(
                        f"Unsafe path in zip (escapes dest): {info.filename}"
                    )
            zf.extractall(codebase_dest)

        # Flatten single top-level directory so that codebase files live
        # directly under codebase_dest (e.g. src/main.rs, not
        # zkTransformer-main-fixed/src/main.rs).
        children = [c for c in codebase_dest.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            inner = children[0]
            # Move all contents of the inner directory up one level.
            # Use shutil.move to handle cross-device and symlink edge cases.
            for item in list(inner.iterdir()):
                shutil.move(str(item), str(codebase_dest / item.name))
            inner.rmdir()

        return codebase_dest

    # -- artifacts ---------------------------------------------------------

    def artifact_json_path(self, artifact_id: str) -> Path:
        """Download and return the local path to a single artifact JSON."""
        pair_id = _pair_id_for_artifact(artifact_id)
        repo_path = f"artifacts/{pair_id}/{artifact_id}.json"
        return self._download(repo_path)

    def load_artifact_json(self, artifact_id: str) -> dict[str, Any]:
        """Download, parse, and return an artifact as a dict."""
        path = self.artifact_json_path(artifact_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_ids(self, pair_id: str | None = None) -> list[str]:
        """Return sorted artifact IDs, optionally filtered by *pair_id*."""
        ids: list[str] = []
        for p in self.pairs():
            if pair_id is not None and p.pair_id != pair_id:
                continue
            for ap in p.artifact_paths:
                # "artifacts/zkllm/zkLLM-001.json" → "zkLLM-001"
                ids.append(Path(ap).stem)
        return sorted(ids)
