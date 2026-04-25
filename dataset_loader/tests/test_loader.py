"""Tests for dataset_loader — BenchmarkDataset + materialize."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dataset_loader import (
    DEFAULT_REPO_ID,
    PAIR_IDS,
    BenchmarkDataset,
    PairInfo,
    _pair_id_for_artifact,
)
from dataset_loader.materialize import materialize


# ---------------------------------------------------------------------------
# Fixtures — tiny on-disk files that mock HF hub downloads
# ---------------------------------------------------------------------------

SAMPLE_MANIFEST = {
    "generated_by": "test",
    "total_files": 6,
    "files": [
        {"path": "papers/zkllm.pdf", "sha256": "a" * 64, "size": 100},
        {"path": "papers/zkml.pdf", "sha256": "b" * 64, "size": 100},
        {"path": "codebases/zkllm.zip", "sha256": "c" * 64, "size": 200},
        {"path": "codebases/zkml.zip", "sha256": "d" * 64, "size": 200},
        {"path": "artifacts/zkllm/zkLLM-001.json", "sha256": "e" * 64, "size": 50},
        {"path": "artifacts/zkml/zkML-001.json", "sha256": "f" * 64, "size": 50},
    ],
}


def _write_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "MANIFEST.json"
    p.write_text(json.dumps(SAMPLE_MANIFEST), encoding="utf-8")
    return p


def _write_fake_pdf(tmp_path: Path, name: str = "paper.pdf") -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _write_fake_zip(tmp_path: Path, name: str = "codebase.zip") -> Path:
    """Create a zip that contains a single file ``src/main.rs``."""
    zip_path = tmp_path / name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("src/main.rs", "fn main() {}\n")
    return zip_path


def _write_unsafe_zip(tmp_path: Path) -> Path:
    """Create a zip with a path-traversal entry."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../etc/passwd", "root::0:0:::\n")
    return zip_path


@pytest.fixture()
def hub_root(tmp_path: Path) -> Path:
    """Create a mini file tree that mirrors HF cache layout."""
    root = tmp_path / "hub"
    root.mkdir()

    _write_manifest(root)
    _write_fake_pdf(root, "papers_zkllm.pdf")
    _write_fake_pdf(root, "papers_zkml.pdf")
    _write_fake_zip(root, "codebases_zkllm.zip")
    _write_fake_zip(root, "codebases_zkml.zip")

    return root


def _fake_hf_hub_download(
    hub_root: Path,
    *,
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> str:
    """Replacement for ``huggingface_hub.hf_hub_download``.

    Maps repo filenames to flat files under *hub_root* (slashes → underscores).
    """
    flat_name = filename.replace("/", "_")
    path = hub_root / flat_name
    if not path.exists():
        raise FileNotFoundError(f"Mock: {flat_name} not in {hub_root}")
    return str(path)


# ---------------------------------------------------------------------------
# BenchmarkDataset — unit tests
# ---------------------------------------------------------------------------

class TestPrefixMapping:
    def test_known_prefixes(self):
        assert _pair_id_for_artifact("zkLLM-001") == "zkllm"
        assert _pair_id_for_artifact("zkML-014") == "zkml"
        assert _pair_id_for_artifact("zkTorch-003") == "zktorch"
        assert _pair_id_for_artifact("zkGPT-010") == "zkgpt"

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown artifact prefix"):
            _pair_id_for_artifact("zkFoo-001")


class TestBenchmarkDataset:
    def _make_ds(self, hub_root: Path) -> BenchmarkDataset:
        ds = BenchmarkDataset(repo_id="test/repo")
        # Patch the download helper to use local files
        ds._download = lambda repo_path: Path(
            _fake_hf_hub_download(hub_root, repo_id="test/repo", filename=repo_path)
        )
        # Pre-load manifest from local file
        ds._manifest = SAMPLE_MANIFEST
        return ds

    def test_pair_ids(self, hub_root):
        ds = self._make_ds(hub_root)
        ids = ds.pair_ids()
        assert "zkllm" in ids
        assert "zkml" in ids

    def test_pairs_returns_pair_info(self, hub_root):
        ds = self._make_ds(hub_root)
        pairs = ds.pairs()
        assert all(isinstance(p, PairInfo) for p in pairs)
        zkllm = next(p for p in pairs if p.pair_id == "zkllm")
        assert zkllm.paper_path == "papers/zkllm.pdf"
        assert zkllm.codebase_path == "codebases/zkllm.zip"
        assert "artifacts/zkllm/zkLLM-001.json" in zkllm.artifact_paths

    def test_get_pair_raises_for_unknown(self, hub_root):
        ds = self._make_ds(hub_root)
        with pytest.raises(KeyError, match="nonexistent"):
            ds.get_pair("nonexistent")

    def test_paper_path(self, hub_root):
        ds = self._make_ds(hub_root)
        p = ds.paper_path("zkllm")
        assert p.exists()
        assert p.name == "papers_zkllm.pdf"

    def test_codebase_zip_path(self, hub_root):
        ds = self._make_ds(hub_root)
        p = ds.codebase_zip_path("zkllm")
        assert p.exists()
        assert p.suffix == ".zip"

    def test_extract_codebase(self, hub_root, tmp_path):
        ds = self._make_ds(hub_root)
        dest = tmp_path / "extract"
        dest.mkdir()
        extracted = ds.extract_codebase("zkllm", dest)
        assert extracted.is_dir()
        # Should contain the file from the zip
        assert any(extracted.rglob("main.rs"))

    def test_extract_codebase_rejects_traversal(self, hub_root, tmp_path):
        # Replace the zkllm zip with an unsafe one
        _write_unsafe_zip(hub_root)
        ds = self._make_ds(hub_root)
        ds._download = lambda repo_path: hub_root / "evil.zip"

        dest = tmp_path / "extract_evil"
        dest.mkdir()
        with pytest.raises(ValueError, match="Unsafe path"):
            ds.extract_codebase("zkllm", dest)

    def test_artifact_ids(self, hub_root):
        ds = self._make_ds(hub_root)
        ids = ds.artifact_ids(pair_id="zkllm")
        assert ids == ["zkLLM-001"]

    def test_artifact_ids_all(self, hub_root):
        ds = self._make_ds(hub_root)
        ids = ds.artifact_ids()
        assert "zkLLM-001" in ids
        assert "zkML-001" in ids


# ---------------------------------------------------------------------------
# Materialize — integration tests
# ---------------------------------------------------------------------------

class TestMaterialize:
    def _make_ds(self, hub_root: Path) -> BenchmarkDataset:
        ds = BenchmarkDataset(repo_id="test/repo")
        ds._download = lambda repo_path: Path(
            _fake_hf_hub_download(hub_root, repo_id="test/repo", filename=repo_path)
        )
        ds._manifest = SAMPLE_MANIFEST
        return ds

    def test_creates_expected_layout(self, hub_root, tmp_path):
        ds = self._make_ds(hub_root)
        out = tmp_path / "run_set"
        materialize(ds, out, pair_ids=["zkllm"])

        assert (out / "zkllm" / "paper.pdf").exists()
        assert (out / "zkllm" / "codebase").is_dir()
        assert (out / "batch_manifest.json").exists()

    def test_batch_manifest_format(self, hub_root, tmp_path):
        ds = self._make_ds(hub_root)
        out = tmp_path / "run_set"
        materialize(ds, out, pair_ids=["zkllm", "zkml"])

        manifest = json.loads(
            (out / "batch_manifest.json").read_text(encoding="utf-8")
        )
        assert "analyses" in manifest
        entries = manifest["analyses"]
        assert len(entries) == 2
        for e in entries:
            assert "entry-id" in e
            assert "paper" in e
            assert "codebase" in e

    def test_no_manifest_flag(self, hub_root, tmp_path):
        ds = self._make_ds(hub_root)
        out = tmp_path / "run_set"
        materialize(ds, out, pair_ids=["zkllm"], emit_batch_manifest=False)

        assert not (out / "batch_manifest.json").exists()
        assert (out / "zkllm" / "paper.pdf").exists()

    def test_all_pairs_when_none(self, hub_root, tmp_path):
        ds = self._make_ds(hub_root)
        out = tmp_path / "run_set"
        materialize(ds, out, pair_ids=None)

        manifest = json.loads(
            (out / "batch_manifest.json").read_text(encoding="utf-8")
        )
        ids = {e["entry-id"] for e in manifest["analyses"]}
        assert "zkllm" in ids
        assert "zkml" in ids


# ---------------------------------------------------------------------------
# CLI — smoke test
# ---------------------------------------------------------------------------

class TestCli:
    def test_list_pairs_runs(self, hub_root, capsys):
        with patch(
            "dataset_loader.hf_hub_download",
            side_effect=lambda **kw: _fake_hf_hub_download(hub_root, **kw),
        ):
            from dataset_loader.cli import main as cli_main
            cli_main(["--repo-id", "test/repo", "list-pairs"])
        captured = capsys.readouterr()
        assert "zkllm" in captured.out
