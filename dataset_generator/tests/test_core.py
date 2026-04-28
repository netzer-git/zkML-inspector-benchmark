"""Unit tests for the dataset generator core modules."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from dataset_generator.artifacts import Artifact, validate_artifact
from dataset_generator.conflict import (
    ConflictEdge,
    detect_conflicts,
    detect_requires_cycle,
    is_compatible_set,
    topological_sort,
)
from dataset_generator.edits import Codebase, EditError, apply_artifact_edits, apply_edit
from dataset_generator.probes import ProbeFailure, evaluate_probes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_artifact(
    artifact_id: str = "zkML-001",
    codebase: str = "zkml-fixed",
    edits: list | None = None,
    regions: list | None = None,
    semantic_tags: list | None = None,
    requires: list | None = None,
    incompatible: list | None = None,
    probes: list | None = None,
) -> Artifact:
    """Build a minimal Artifact for testing."""
    raw = {
        "artifact_id": artifact_id,
        "codebase": codebase,
        "source": "synthetic",
        "finding": {
            "name": f"Test finding {artifact_id}",
            "explanation": "A test explanation for this artifact finding.",
            "labels": {
                "relevant_code": "src/test.rs:10",
                "paper_reference": "-",
            },
        },
        "edits": edits or [{"file": "src/test.rs", "op": "replace_block",
                            "anchor": {"kind": "line_range", "start": 1, "end": 1},
                            "new_content": "// replaced"}],
        "conflict_keys": {
            "files": ["src/test.rs"],
            "regions": regions or [{"file": "src/test.rs", "start": 1, "end": 5}],
            "semantic_tags": semantic_tags or [],
            "requires": requires or [],
            "incompatible": incompatible or [],
        },
        "presence_probes": probes or [{"kind": "contains", "file": "src/test.rs", "text": "replaced"}],
    }
    return Artifact(
        artifact_id=raw["artifact_id"],
        codebase=raw["codebase"],
        source=raw["source"],
        finding=raw["finding"],
        edits=raw["edits"],
        conflict_keys=raw["conflict_keys"],
        presence_probes=raw["presence_probes"],
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------

class TestEdits:
    def test_replace_block(self):
        codebase: Codebase = {"src/test.rs": ["line1", "line2", "line3"]}
        sha = _sha256("line2\n")
        edit = {
            "file": "src/test.rs",
            "op": "replace_block",
            "anchor": {"kind": "line_range", "start": 2, "end": 2, "expect_sha256": sha},
            "new_content": "replaced_line",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["line1", "replaced_line", "line3"]

    def test_replace_block_sha_mismatch(self):
        codebase: Codebase = {"src/test.rs": ["line1", "line2"]}
        edit = {
            "file": "src/test.rs",
            "op": "replace_block",
            "anchor": {"kind": "line_range", "start": 1, "end": 1, "expect_sha256": "0" * 64},
            "new_content": "x",
        }
        with pytest.raises(EditError, match="SHA-256 mismatch"):
            apply_edit(codebase, edit)

    def test_delete_block(self):
        codebase: Codebase = {"src/test.rs": ["a", "b", "c", "d"]}
        edit = {
            "file": "src/test.rs",
            "op": "delete_block",
            "anchor": {"kind": "line_range", "start": 2, "end": 3},
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["a", "d"]

    def test_insert_after_line_range(self):
        codebase: Codebase = {"src/test.rs": ["a", "b", "c"]}
        edit = {
            "file": "src/test.rs",
            "op": "insert_after",
            "anchor": {"kind": "line_range", "start": 2, "end": 2},
            "new_content": "inserted",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["a", "b", "inserted", "c"]

    def test_insert_before_unique_string(self):
        codebase: Codebase = {"src/test.rs": ["hello", "world", "foo"]}
        edit = {
            "file": "src/test.rs",
            "op": "insert_before",
            "anchor": {"kind": "unique_string", "text": "world"},
            "new_content": "before",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["hello", "before", "world", "foo"]

    def test_create_file(self):
        codebase: Codebase = {}
        edit = {
            "file": "new_file.rs",
            "op": "create_file",
            "anchor": {"kind": "line_range"},
            "new_content": "// new file\nfn main() {}",
        }
        apply_edit(codebase, edit)
        assert "new_file.rs" in codebase
        assert codebase["new_file.rs"] == ["// new file", "fn main() {}"]

    def test_create_file_already_exists(self):
        codebase: Codebase = {"existing.rs": ["x"]}
        edit = {
            "file": "existing.rs",
            "op": "create_file",
            "anchor": {"kind": "line_range"},
            "new_content": "y",
        }
        with pytest.raises(EditError, match="already exists"):
            apply_edit(codebase, edit)

    def test_replace_regex(self):
        codebase: Codebase = {"src/test.rs": ["let x = 42;", "let y = 42;"]}
        edit = {
            "file": "src/test.rs",
            "op": "replace_regex",
            "anchor": {"kind": "line_range"},
            "regex_pattern": r"42",
            "new_content": "99",
            "expected_match_count": 2,
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["let x = 99;", "let y = 99;"]

    def test_replace_regex_count_mismatch(self):
        codebase: Codebase = {"src/test.rs": ["let x = 42;"]}
        edit = {
            "file": "src/test.rs",
            "op": "replace_regex",
            "anchor": {"kind": "line_range"},
            "regex_pattern": r"42",
            "new_content": "99",
            "expected_match_count": 3,
        }
        with pytest.raises(EditError, match="expected 3 matches"):
            apply_edit(codebase, edit)

    def test_path_traversal_rejected(self):
        codebase: Codebase = {}
        edit = {
            "file": "../etc/passwd",
            "op": "create_file",
            "anchor": {"kind": "line_range"},
            "new_content": "bad",
        }
        with pytest.raises(EditError, match="Path traversal"):
            apply_edit(codebase, edit)

    def test_atomic_rollback(self):
        codebase: Codebase = {"src/test.rs": ["line1", "line2"]}
        edits = [
            {
                "file": "src/test.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1},
                "new_content": "modified",
            },
            {
                "file": "src/test.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1, "expect_sha256": "0" * 64},
                "new_content": "will_fail",
            },
        ]
        with pytest.raises(EditError):
            apply_artifact_edits(codebase, edits, "zkML-001")
        # Should be rolled back
        assert codebase["src/test.rs"] == ["line1", "line2"]

    def test_missing_file_raises(self):
        codebase: Codebase = {}
        edit = {
            "file": "nonexistent.rs",
            "op": "replace_block",
            "anchor": {"kind": "line_range", "start": 1, "end": 1},
            "new_content": "x",
        }
        with pytest.raises(EditError, match="not found"):
            apply_edit(codebase, edit)

    def test_insert_after_unique_string(self):
        codebase: Codebase = {"src/test.rs": ["hello", "world", "foo"]}
        edit = {
            "file": "src/test.rs",
            "op": "insert_after",
            "anchor": {"kind": "unique_string", "text": "world"},
            "new_content": "inserted",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["hello", "world", "inserted", "foo"]

    def test_insert_after_regex(self):
        codebase: Codebase = {"src/test.rs": ["fn main() {", "    let x = 1;", "}"]}
        edit = {
            "file": "src/test.rs",
            "op": "insert_after",
            "anchor": {"kind": "regex", "pattern": r"fn main"},
            "new_content": "    // inserted",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["fn main() {", "    // inserted", "    let x = 1;", "}"]

    def test_insert_before_line_range(self):
        codebase: Codebase = {"src/test.rs": ["alpha", "beta", "gamma"]}
        edit = {
            "file": "src/test.rs",
            "op": "insert_before",
            "anchor": {"kind": "line_range", "start": 2, "end": 2},
            "new_content": "before_beta",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["alpha", "before_beta", "beta", "gamma"]

    def test_insert_before_regex(self):
        codebase: Codebase = {"src/test.rs": ["aaa", "bbb", "ccc"]}
        edit = {
            "file": "src/test.rs",
            "op": "insert_before",
            "anchor": {"kind": "regex", "pattern": r"^bbb$"},
            "new_content": "before_bbb",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["aaa", "before_bbb", "bbb", "ccc"]

    def test_replace_regex_without_expected_count(self):
        codebase: Codebase = {"src/test.rs": ["let x = 42;", "let y = 42;"]}
        edit = {
            "file": "src/test.rs",
            "op": "replace_regex",
            "anchor": {"kind": "line_range"},
            "regex_pattern": r"42",
            "new_content": "99",
        }
        apply_edit(codebase, edit)
        assert codebase["src/test.rs"] == ["let x = 99;", "let y = 99;"]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class TestProbes:
    def test_contains_pass(self):
        codebase: Codebase = {"src/test.rs": ["hello world"]}
        probes = [{"kind": "contains", "file": "src/test.rs", "text": "hello"}]
        failures = evaluate_probes(codebase, probes, "zkML-001")
        assert failures == []

    def test_contains_fail(self):
        codebase: Codebase = {"src/test.rs": ["hello world"]}
        probes = [{"kind": "contains", "file": "src/test.rs", "text": "goodbye"}]
        failures = evaluate_probes(codebase, probes, "zkML-001")
        assert len(failures) == 1
        assert failures[0].kind == "contains"

    def test_not_contains_pass(self):
        codebase: Codebase = {"src/test.rs": ["hello world"]}
        probes = [{"kind": "not_contains", "file": "src/test.rs", "text": "goodbye"}]
        failures = evaluate_probes(codebase, probes, "zkML-001")
        assert failures == []

    def test_not_contains_fail(self):
        codebase: Codebase = {"src/test.rs": ["hello world"]}
        probes = [{"kind": "not_contains", "file": "src/test.rs", "text": "hello"}]
        failures = evaluate_probes(codebase, probes, "zkML-001")
        assert len(failures) == 1

    def test_line_equals_pass(self):
        codebase: Codebase = {"src/test.rs": ["alpha", "beta", "gamma"]}
        probes = [{"kind": "line_equals", "file": "src/test.rs", "line": 2, "text": "beta"}]
        failures = evaluate_probes(codebase, probes, "zkML-001")
        assert failures == []

    def test_line_equals_fail(self):
        codebase: Codebase = {"src/test.rs": ["alpha", "beta", "gamma"]}
        probes = [{"kind": "line_equals", "file": "src/test.rs", "line": 2, "text": "delta"}]
        failures = evaluate_probes(codebase, probes, "zkML-001")
        assert len(failures) == 1

    def test_missing_file(self):
        codebase: Codebase = {}
        probes = [{"kind": "contains", "file": "missing.rs", "text": "x"}]
        failures = evaluate_probes(codebase, probes, "zkML-001")
        assert len(failures) == 1
        assert "not found" in failures[0].detail


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

class TestConflict:
    def test_no_conflict(self):
        a = _make_artifact("zkML-001", regions=[{"file": "a.rs", "start": 1, "end": 10}])
        b = _make_artifact("zkML-002", regions=[{"file": "a.rs", "start": 20, "end": 30}])
        assert detect_conflicts([a, b]) == []

    def test_region_overlap(self):
        a = _make_artifact("zkML-001", regions=[{"file": "a.rs", "start": 1, "end": 15}])
        b = _make_artifact("zkML-002", regions=[{"file": "a.rs", "start": 10, "end": 25}])
        conflicts = detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert "Overlapping" in conflicts[0].reason

    def test_shared_semantic_tag(self):
        a = _make_artifact("zkML-001", semantic_tags=["freivalds"],
                          regions=[{"file": "a.rs", "start": 1, "end": 5}])
        b = _make_artifact("zkML-002", semantic_tags=["freivalds"],
                          regions=[{"file": "b.rs", "start": 1, "end": 5}])
        conflicts = detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert "semantic" in conflicts[0].reason.lower()

    def test_explicit_incompatible(self):
        a = _make_artifact("zkML-001", incompatible=["zkML-002"],
                          regions=[{"file": "a.rs", "start": 1, "end": 5}])
        b = _make_artifact("zkML-002",
                          regions=[{"file": "b.rs", "start": 1, "end": 5}])
        conflicts = detect_conflicts([a, b])
        assert len(conflicts) == 1

    def test_no_cycle(self):
        a = _make_artifact("zkML-001", requires=["zkML-002"],
                          regions=[{"file": "a.rs", "start": 10, "end": 20}])
        b = _make_artifact("zkML-002",
                          regions=[{"file": "a.rs", "start": 1, "end": 5}])
        assert detect_requires_cycle([a, b]) is None

    def test_cycle_detected(self):
        a = _make_artifact("zkML-001", requires=["zkML-002"],
                          regions=[{"file": "a.rs", "start": 10, "end": 15}])
        b = _make_artifact("zkML-002", requires=["zkML-001"],
                          regions=[{"file": "a.rs", "start": 20, "end": 25}])
        cycle = detect_requires_cycle([a, b])
        assert cycle is not None

    def test_topological_sort_respects_requires(self):
        a = _make_artifact("zkML-001", requires=["zkML-002"],
                          regions=[{"file": "a.rs", "start": 10, "end": 15}])
        b = _make_artifact("zkML-002",
                          regions=[{"file": "a.rs", "start": 1, "end": 5}])
        order = topological_sort([a, b])
        ids = [x.artifact_id for x in order]
        assert ids.index("zkML-002") < ids.index("zkML-001")

    def test_is_compatible_set_true(self):
        a = _make_artifact("zkML-001", regions=[{"file": "a.rs", "start": 1, "end": 5}])
        b = _make_artifact("zkML-002", regions=[{"file": "b.rs", "start": 1, "end": 5}])
        assert is_compatible_set([a, b])

    def test_is_compatible_set_false(self):
        a = _make_artifact("zkML-001", regions=[{"file": "a.rs", "start": 1, "end": 10}])
        b = _make_artifact("zkML-002", regions=[{"file": "a.rs", "start": 5, "end": 15}])
        assert not is_compatible_set([a, b])


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_random_strategy_selects_k(self):
        from dataset_generator.strategies import RandomStrategy
        pool = [
            _make_artifact(f"zkML-{i:03d}", regions=[{"file": f"f{i}.rs", "start": 1, "end": 5}])
            for i in range(1, 6)
        ]
        strategy = RandomStrategy(k=2)
        selected = strategy.assign(pool, random.Random(42))
        assert len(selected) == 2

    def test_random_strategy_insufficient_pool(self):
        from dataset_generator.strategies import RandomStrategy
        pool = [_make_artifact("zkML-001")]
        strategy = RandomStrategy(k=3)
        with pytest.raises(ValueError, match="Pool has 1"):
            strategy.assign(pool, random.Random(42))

    def test_all_strategy(self):
        from dataset_generator.strategies import AllStrategy
        pool = [
            _make_artifact(f"zkML-{i:03d}", regions=[{"file": f"f{i}.rs", "start": 1, "end": 5}])
            for i in range(1, 4)
        ]
        strategy = AllStrategy()
        selected = strategy.assign(pool, random.Random(42))
        assert len(selected) == 3

    def test_all_strategy_with_conflicts_raises(self):
        from dataset_generator.strategies import AllStrategy
        pool = [
            _make_artifact("zkML-001", regions=[{"file": "a.rs", "start": 1, "end": 10}]),
            _make_artifact("zkML-002", regions=[{"file": "a.rs", "start": 5, "end": 15}]),
        ]
        strategy = AllStrategy()
        with pytest.raises(ValueError, match="conflicting"):
            strategy.assign(pool, random.Random(42))

    def test_isolated_strategy_emits_n_single_artifact_cases(self):
        from dataset_generator.strategies import IsolatedStrategy
        pool = [
            _make_artifact(f"zkML-{i:03d}", regions=[{"file": f"f{i}.rs", "start": 1, "end": 5}])
            for i in range(1, 6)
        ]
        strategy = IsolatedStrategy()
        results = [strategy.assign(pool, random.Random(0)) for _ in range(5)]
        # Each call returns exactly one artifact
        assert all(len(r) == 1 for r in results)
        # Artifact IDs are sorted and non-repeating
        ids = [r[0].artifact_id for r in results]
        assert ids == sorted(ids)
        assert len(set(ids)) == 5

    def test_isolated_strategy_exhaustion_raises(self):
        from dataset_generator.strategies import IsolatedStrategy
        pool = [_make_artifact("zkML-001")]
        strategy = IsolatedStrategy()
        strategy.assign(pool, random.Random(0))  # first call OK
        with pytest.raises(ValueError, match="exhausted"):
            strategy.assign(pool, random.Random(0))

    def test_isolated_strategy_includes_dependencies(self):
        from dataset_generator.strategies import IsolatedStrategy
        a = _make_artifact("zkML-001", requires=["zkML-002"],
                          regions=[{"file": "a.rs", "start": 10, "end": 15}])
        b = _make_artifact("zkML-002",
                          regions=[{"file": "a.rs", "start": 1, "end": 5}])
        pool = [a, b]
        strategy = IsolatedStrategy()
        # First call: zkML-001 (sorted first) requires zkML-002
        result = strategy.assign(pool, random.Random(0))
        ids = {r.artifact_id for r in result}
        assert "zkML-001" in ids
        assert "zkML-002" in ids

    def test_fixed_subset_strategy(self):
        from dataset_generator.strategies import FixedSubsetStrategy
        pool = [
            _make_artifact(f"zkML-{i:03d}", regions=[{"file": f"f{i}.rs", "start": 1, "end": 5}])
            for i in range(1, 6)
        ]
        strategy = FixedSubsetStrategy(["zkML-002", "zkML-004"])
        selected = strategy.assign(pool, random.Random(0))
        ids = {a.artifact_id for a in selected}
        assert ids == {"zkML-002", "zkML-004"}

    def test_fixed_subset_strategy_missing_artifact_raises(self):
        from dataset_generator.strategies import FixedSubsetStrategy
        pool = [_make_artifact("zkML-001")]
        strategy = FixedSubsetStrategy(["zkML-001", "zkML-999"])
        with pytest.raises(ValueError, match="not in pool"):
            strategy.assign(pool, random.Random(0))

    def test_fixed_subset_strategy_conflict_raises(self):
        from dataset_generator.strategies import FixedSubsetStrategy
        pool = [
            _make_artifact("zkML-001", regions=[{"file": "a.rs", "start": 1, "end": 10}]),
            _make_artifact("zkML-002", regions=[{"file": "a.rs", "start": 5, "end": 15}]),
        ]
        strategy = FixedSubsetStrategy(["zkML-001", "zkML-002"])
        with pytest.raises(ValueError, match="conflicts"):
            strategy.assign(pool, random.Random(0))

    def test_fixed_subset_strategy_resolves_deps(self):
        from dataset_generator.strategies import FixedSubsetStrategy
        a = _make_artifact("zkML-001", requires=["zkML-002"],
                          regions=[{"file": "a.rs", "start": 10, "end": 15}])
        b = _make_artifact("zkML-002",
                          regions=[{"file": "a.rs", "start": 1, "end": 5}])
        pool = [a, b]
        strategy = FixedSubsetStrategy(["zkML-001"])
        selected = strategy.assign(pool, random.Random(0))
        ids = {s.artifact_id for s in selected}
        assert ids == {"zkML-001", "zkML-002"}

    def test_fixed_subset_strategy_empty_raises(self):
        from dataset_generator.strategies import FixedSubsetStrategy
        with pytest.raises(ValueError, match="non-empty"):
            FixedSubsetStrategy([])


# ---------------------------------------------------------------------------
# Artifact validation (schema)
# ---------------------------------------------------------------------------

class TestArtifactValidation:
    def test_valid_v2_artifact(self, tmp_path):
        data = {
            "artifact_id": "zkML-001",
            "codebase": "zkml-fixed",
            "source": "synthetic",
            "finding": {
                "name": "Test finding name here",
                "explanation": "A detailed explanation of the test finding.",
                "labels": {
                    "relevant_code": "",
                    "paper_reference": "-",
                },
            },
            "edits": [
                {
                    "file": "src/test.rs",
                    "op": "replace_block",
                    "anchor": {"kind": "line_range", "start": 1, "end": 1},
                    "new_content": "// replaced",
                }
            ],
            "conflict_keys": {
                "files": ["src/test.rs"],
                "regions": [{"file": "src/test.rs", "start": 1, "end": 5}],
                "semantic_tags": [],
            },
            "presence_probes": [
                {"kind": "contains", "file": "src/test.rs", "text": "replaced"}
            ],
        }
        validate_artifact(data)  # Should not raise

    def test_invalid_artifact_id_pattern(self):
        data = {
            "artifact_id": "BAD-001",  # invalid prefix
            "codebase": "zkml-fixed",
            "source": "synthetic",
            "finding": {
                "name": "Test finding name here",
                "explanation": "A detailed explanation of the test finding.",
                "labels": {
                    "relevant_code": "",
                    "paper_reference": "-",
                },
            },
            "edits": [
                {
                    "file": "src/test.rs",
                    "op": "replace_block",
                    "anchor": {"kind": "line_range", "start": 1, "end": 1},
                    "new_content": "x",
                }
            ],
            "conflict_keys": {
                "files": [],
                "regions": [],
                "semantic_tags": [],
            },
            "presence_probes": [
                {"kind": "contains", "file": "src/test.rs", "text": "x"}
            ],
        }
        with pytest.raises(Exception):  # jsonschema.ValidationError
            validate_artifact(data)

    def test_missing_required_field(self):
        data = {
            "artifact_id": "zkML-001",
            "codebase": "zkml-fixed",
            # missing source, finding, edits, etc.
        }
        with pytest.raises(Exception):
            validate_artifact(data)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class TestSources:
    def test_unknown_entry_id_raises(self, tmp_path):
        from dataset_generator.sources import Sources
        from dataset_loader import BenchmarkDataset

        # Empty manifest → no pairs
        ds = BenchmarkDataset.__new__(BenchmarkDataset)
        ds._repo_id = "test/repo"
        ds._revision = None
        ds._manifest = {"total_files": 0, "files": []}
        ds._pairs_cache = None

        sources = Sources(ds)
        with pytest.raises(KeyError, match="nonexistent"):
            sources.get_entry("nonexistent")

    def test_empty_artifact_pool(self, tmp_path):
        from dataset_generator.sources import Sources
        from dataset_loader import BenchmarkDataset

        ds = BenchmarkDataset.__new__(BenchmarkDataset)
        ds._repo_id = "test/repo"
        ds._revision = None
        ds._manifest = {"total_files": 0, "files": []}
        ds._pairs_cache = None

        sources = Sources(ds)
        assert sources.get_artifact_pool("nonexistent") == []


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class TestAssembler:
    def test_edit_failure_raises_case_build_error(self, tmp_path):
        from dataset_generator.assembler import CaseBuildError, build_case

        codebase_dir = tmp_path / "cb"
        codebase_dir.mkdir()
        (codebase_dir / "src").mkdir()
        (codebase_dir / "src" / "test.rs").write_text("line1\nline2\n", encoding="utf-8")

        # Artifact with an edit that will fail (SHA mismatch)
        artifact = _make_artifact(
            "zkML-001",
            edits=[{
                "file": "src/test.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1,
                           "expect_sha256": "0" * 64},
                "new_content": "bad",
            }],
            probes=[{"kind": "contains", "file": "src/test.rs", "text": "bad"}],
        )

        with pytest.raises(CaseBuildError, match="Edit failed"):
            build_case(
                entry_id="test",
                codebase_dir=codebase_dir,
                codebase_name="cb",
                paper_path=None,
                artifacts=[artifact],
                output_dir=tmp_path / "out",
            )
        # Case dir should be cleaned up
        assert not (tmp_path / "out" / "test").exists()

    def test_probe_failure_raises_case_build_error(self, tmp_path):
        from dataset_generator.assembler import CaseBuildError, build_case

        codebase_dir = tmp_path / "cb"
        codebase_dir.mkdir()
        (codebase_dir / "src").mkdir()
        (codebase_dir / "src" / "test.rs").write_text("line1\nline2\n", encoding="utf-8")

        # Artifact whose edit succeeds but probe fails
        artifact = _make_artifact(
            "zkML-001",
            edits=[{
                "file": "src/test.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1},
                "new_content": "modified",
            }],
            probes=[{"kind": "contains", "file": "src/test.rs", "text": "DOES_NOT_EXIST"}],
        )

        with pytest.raises(CaseBuildError, match="probe"):
            build_case(
                entry_id="test",
                codebase_dir=codebase_dir,
                codebase_name="cb",
                paper_path=None,
                artifacts=[artifact],
                output_dir=tmp_path / "out",
            )

    def test_line_offset_across_artifacts(self, tmp_path):
        """Two artifacts edit the same file at non-overlapping ranges.

        The first artifact changes the line count, which should NOT break
        the second artifact's anchor thanks to offset tracking.
        """
        from dataset_generator.assembler import build_case

        codebase_dir = tmp_path / "cb"
        codebase_dir.mkdir()
        (codebase_dir / "src").mkdir()

        # 10-line file
        lines = [f"line{i}" for i in range(1, 11)]
        (codebase_dir / "src" / "f.rs").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        sha_l2_l3 = _sha256("line2\nline3\n")  # L2-L3
        sha_l8 = _sha256("line8\n")  # L8

        # Artifact A: replace L2-L3 (2 lines) with 1 line → delta=-1
        a = _make_artifact(
            "zkML-001",
            edits=[{
                "file": "src/f.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 2, "end": 3,
                           "expect_sha256": sha_l2_l3},
                "new_content": "REPLACED_A",
            }],
            regions=[{"file": "src/f.rs", "start": 1, "end": 4}],
            probes=[{"kind": "contains", "file": "src/f.rs", "text": "REPLACED_A"}],
        )

        # Artifact B: replace L8 (1 line) with 1 line → delta=0
        # Without offset tracking, L8 would be wrong after A removes a line.
        b = _make_artifact(
            "zkML-002",
            edits=[{
                "file": "src/f.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 8, "end": 8,
                           "expect_sha256": sha_l8},
                "new_content": "REPLACED_B",
            }],
            regions=[{"file": "src/f.rs", "start": 7, "end": 9}],
            probes=[{"kind": "contains", "file": "src/f.rs", "text": "REPLACED_B"}],
        )

        case = build_case(
            entry_id="offset-test",
            codebase_dir=codebase_dir,
            codebase_name="cb",
            paper_path=None,
            artifacts=[a, b],
            output_dir=tmp_path / "out",
        )

        # Both edits should have been applied
        result = (case.case_dir / "codebase" / "src" / "f.rs").read_text("utf-8")
        assert "REPLACED_A" in result
        assert "REPLACED_B" in result
        # Original line8 content should be gone
        assert "line8" not in result


# ---------------------------------------------------------------------------
# Conflict — probe contamination
# ---------------------------------------------------------------------------

class TestProbeContamination:
    def test_not_contains_vs_new_content(self):
        """Two artifacts where B's new_content introduces text that
        violates A's not_contains probe should be detected as a conflict."""
        a = _make_artifact(
            "zkML-001",
            edits=[{
                "file": "src/test.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1},
                "new_content": "safe_content",
            }],
            regions=[{"file": "src/test.rs", "start": 1, "end": 3}],
            probes=[
                {"kind": "contains", "file": "src/test.rs", "text": "safe_content"},
                {"kind": "not_contains", "file": "src/test.rs", "text": "FORBIDDEN"},
            ],
        )

        b = _make_artifact(
            "zkML-002",
            edits=[{
                "file": "src/test.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 5, "end": 5},
                "new_content": "this has FORBIDDEN text",
            }],
            regions=[{"file": "src/test.rs", "start": 5, "end": 7}],
            probes=[{"kind": "contains", "file": "src/test.rs", "text": "FORBIDDEN"}],
        )

        conflicts = detect_conflicts([a, b])
        reasons = [c.reason for c in conflicts]
        assert any("not_contains" in r for r in reasons)

    def test_no_false_positive_different_file(self):
        """not_contains on file X should not conflict with new_content on file Y."""
        a = _make_artifact(
            "zkML-001",
            edits=[{
                "file": "src/a.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1},
                "new_content": "ok",
            }],
            regions=[{"file": "src/a.rs", "start": 1, "end": 3}],
            probes=[
                {"kind": "not_contains", "file": "src/a.rs", "text": "FORBIDDEN"},
            ],
        )

        b = _make_artifact(
            "zkML-002",
            edits=[{
                "file": "src/b.rs",
                "op": "replace_block",
                "anchor": {"kind": "line_range", "start": 1, "end": 1},
                "new_content": "FORBIDDEN content here",
            }],
            regions=[{"file": "src/b.rs", "start": 1, "end": 3}],
            probes=[{"kind": "contains", "file": "src/b.rs", "text": "FORBIDDEN"}],
        )

        conflicts = detect_conflicts([a, b])
        probe_conflicts = [c for c in conflicts if "not_contains" in c.reason]
        assert len(probe_conflicts) == 0
