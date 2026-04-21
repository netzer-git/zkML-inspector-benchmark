"""End-to-end smoke test for the dataset generator.

Builds a minimal sources tree in tmp_path, runs the generator, and validates
that the output is correct and the grader can consume the findings.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from dataset_generator.cli import main as gen_main
from grader.loader import load_ground_truth


def _build_sources(tmp_path: Path) -> Path:
    """Build a minimal sources directory for testing."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()

    # Paper (just a text file pretending to be a PDF)
    papers_dir = sources_dir / "papers"
    papers_dir.mkdir()
    paper = papers_dir / "test_paper.pdf"
    paper.write_text("Fake paper content for testing", encoding="utf-8")

    # Codebase zip
    codebases_dir = sources_dir / "codebases"
    codebases_dir.mkdir()
    zip_path = codebases_dir / "test-codebase.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test-codebase/src/main.rs", "fn main() {\n    println!(\"hello\");\n}\n")
        zf.writestr("test-codebase/src/lib.rs", "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n")
        zf.writestr("test-codebase/Cargo.toml", "[package]\nname = \"test\"\nversion = \"0.1.0\"\n")

    # Artifacts — two non-conflicting artifacts targeting different files
    artifacts_dir = sources_dir / "artifacts" / "test-codebase"
    artifacts_dir.mkdir(parents=True)

    import hashlib
    main_line1 = "fn main() {\n"
    main_sha = hashlib.sha256(main_line1.encode("utf-8")).hexdigest()

    lib_line1 = "pub fn add(a: i32, b: i32) -> i32 {\n"
    lib_sha = hashlib.sha256(lib_line1.encode("utf-8")).hexdigest()

    artifact1 = {
        "artifact_id": "zkML-001",
        "codebase": "zkml-fixed",
        "source": "synthetic",
        "finding": {
            "name": "Hardcoded main function",
            "explanation": "The main function is hardcoded and should be replaced with a configurable entry point.",
            "labels": {
                "severity": "Critical",
                "category": "Engineering/Prototype Gap",
                "security_concern": "Proof Forgery (Soundness)",
                "relevant_code": "src/main.rs:1",
                "paper_reference": "Section 1: Entry point must be configurable",
            },
        },
        "edits": [
            {
                "file": "src/main.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1, "expect_sha256": main_sha},
                "new_content": "fn main_buggy() {",
            }
        ],
        "conflict_keys": {
            "files": ["src/main.rs"],
            "regions": [{"file": "src/main.rs", "start": 1, "end": 3}],
            "semantic_tags": ["main_entry"],
        },
        "presence_probes": [
            {"kind": "contains", "file": "src/main.rs", "text": "main_buggy"},
            {"kind": "not_contains", "file": "src/main.rs", "text": "fn main() {"},
        ],
    }

    artifact2 = {
        "artifact_id": "zkML-002",
        "codebase": "zkml-fixed",
        "source": "synthetic",
        "finding": {
            "name": "Unconstrained addition operation",
            "explanation": "The add function does not check for overflow, which could lead to incorrect results in the circuit.",
            "labels": {
                "severity": "Warning",
                "category": "Under-constrained Circuit",
                "security_concern": "Semantic Subversion (Integrity)",
                "relevant_code": "src/lib.rs:1-2",
                "paper_reference": "Section 3.2: All arithmetic must be range-checked",
            },
        },
        "edits": [
            {
                "file": "src/lib.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1, "expect_sha256": lib_sha},
                "new_content": "pub fn add(a: i32, b: i32) -> i32 { // UNCHECKED",
            }
        ],
        "conflict_keys": {
            "files": ["src/lib.rs"],
            "regions": [{"file": "src/lib.rs", "start": 1, "end": 3}],
            "semantic_tags": ["add_function"],
        },
        "presence_probes": [
            {"kind": "contains", "file": "src/lib.rs", "text": "UNCHECKED"},
        ],
    }

    (artifacts_dir / "zkML-001.json").write_text(
        json.dumps(artifact1, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "zkML-002.json").write_text(
        json.dumps(artifact2, indent=2), encoding="utf-8"
    )

    # sources.json
    sources_json = [
        {
            "entry-id": "test-project",
            "paper": "papers/test_paper.pdf",
            "codebase_zip": "codebases/test-codebase.zip",
            "codebase_name": "test-codebase",
        }
    ]
    (sources_dir / "sources.json").write_text(
        json.dumps(sources_json, indent=2), encoding="utf-8"
    )

    return sources_dir


class TestEndToEnd:
    def test_generate_single_case(self, tmp_path):
        sources_dir = _build_sources(tmp_path)
        output_dir = tmp_path / "output"

        gen_main([
            "test",
            "--sources", str(sources_dir),
            "--output", str(output_dir),
            "--num-cases", "1",
            "--artifacts-per-case", "2",
            "--strategy", "random",
            "--seed", "42",
        ])

        # Verify outputs exist
        assert (output_dir / "dataset_manifest.json").exists()
        assert (output_dir / "findings.json").exists()

        # Verify manifest
        manifest = json.loads(
            (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["strategy"] == "random"
        assert manifest["seed"] == 42
        assert manifest["num_cases"] == 1
        assert len(manifest["cases"]) == 1

        # Verify findings
        findings = json.loads(
            (output_dir / "findings.json").read_text(encoding="utf-8")
        )
        assert len(findings) == 2  # 2 artifacts applied
        assert all(f["entry-id"] == "test-project" for f in findings)
        assert {f["issue-id"] for f in findings} == {"zkML-001", "zkML-002"}

        # Verify each finding has all required fields
        required_fields = {
            "entry-id", "issue-id", "issue-name", "issue-explanation",
            "severity", "category", "security-concern",
            "relevant-code", "paper-reference",
        }
        for f in findings:
            assert required_fields <= set(f.keys()), f"Missing fields in {f}"

    def test_findings_loadable_by_grader(self, tmp_path):
        sources_dir = _build_sources(tmp_path)
        output_dir = tmp_path / "output"

        gen_main([
            "test",
            "--sources", str(sources_dir),
            "--output", str(output_dir),
            "--num-cases", "1",
            "--artifacts-per-case", "2",
            "--strategy", "random",
            "--seed", "42",
        ])

        # The grader's load_ground_truth should accept the generated findings
        gt = load_ground_truth(output_dir / "findings.json")
        assert "test-project" in gt
        assert len(gt["test-project"]) == 2
        severities = {f.severity for f in gt["test-project"]}
        assert "Critical" in severities

    def test_case_directory_structure(self, tmp_path):
        sources_dir = _build_sources(tmp_path)
        output_dir = tmp_path / "output"

        gen_main([
            "test",
            "--sources", str(sources_dir),
            "--output", str(output_dir),
            "--num-cases", "1",
            "--artifacts-per-case", "1",
            "--strategy", "random",
            "--seed", "7",
        ])

        cases_dir = output_dir / "cases"
        assert cases_dir.exists()
        case_dirs = list(cases_dir.iterdir())
        assert len(case_dirs) >= 1

        # Check case.json exists in the case directory
        case_dir = case_dirs[0]
        case_json = case_dir / "case.json"
        assert case_json.exists()
        case_meta = json.loads(case_json.read_text(encoding="utf-8"))
        assert "entry_id" in case_meta
        assert "artifact_ids" in case_meta
        assert "source_codebase" in case_meta
