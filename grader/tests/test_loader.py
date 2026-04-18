"""Tests for grader.loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from grader.loader import (
    AgentFinding,
    CodeRef,
    GroundTruthFinding,
    load_agent_output,
    load_ground_truth,
    parse_code_refs,
)


# ---------------------------------------------------------------------------
# parse_code_refs
# ---------------------------------------------------------------------------

class TestParseCodeRefs:
    def test_single_ref_with_range(self):
        refs = parse_code_refs("rmsnorm.cu:21-31")
        assert refs == [CodeRef("rmsnorm.cu", 21, 31)]

    def test_single_ref_single_line(self):
        refs = parse_code_refs("verifier.cpp:658")
        assert refs == [CodeRef("verifier.cpp", 658, 658)]

    def test_multiple_comma_separated(self):
        refs = parse_code_refs("rmsnorm.cu:21-31, llama-rmsnorm.py:34-38")
        assert len(refs) == 2
        assert refs[0] == CodeRef("rmsnorm.cu", 21, 31)
        assert refs[1] == CodeRef("llama-rmsnorm.py", 34, 38)

    def test_file_without_lines(self):
        refs = parse_code_refs("config.yaml")
        assert refs == [CodeRef("config.yaml", None, None)]

    def test_mixed_with_and_without_lines(self):
        refs = parse_code_refs("config.yaml, config.rs:31")
        assert len(refs) == 2
        assert refs[0] == CodeRef("config.yaml", None, None)
        assert refs[1] == CodeRef("config.rs", 31, 31)

    def test_path_with_directories(self):
        refs = parse_code_refs("src/util/verifier.rs:36-42")
        assert refs == [CodeRef("src/util/verifier.rs", 36, 42)]

    def test_none_returns_empty(self):
        assert parse_code_refs(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_code_refs("") == []

    def test_dash_returns_empty(self):
        assert parse_code_refs("-") == []

    def test_none_string_returns_empty(self):
        assert parse_code_refs("None") == []

    def test_unicode_dash_in_range(self):
        # The xlsx uses en-dash (–) in some ranges
        refs = parse_code_refs("tlookup.cu:267–370")
        assert refs == [CodeRef("tlookup.cu", 267, 370)]

    def test_semicolon_separator(self):
        refs = parse_code_refs("file1.rs:10; file2.rs:20")
        assert len(refs) == 2


# ---------------------------------------------------------------------------
# load_ground_truth (real xlsx)
# ---------------------------------------------------------------------------

XLSX_PATH = Path(__file__).parent.parent.parent / "zkMLDataset.xlsx"


@pytest.mark.skipif(not XLSX_PATH.exists(), reason="xlsx not found")
class TestLoadGroundTruth:
    def test_loads_all_projects(self):
        gt = load_ground_truth(XLSX_PATH)
        assert "zkllm" in gt
        assert "zkgpt" in gt
        assert "zkml" in gt
        assert "zktorch" in gt

    def test_total_finding_count(self):
        gt = load_ground_truth(XLSX_PATH)
        total = sum(len(v) for v in gt.values())
        assert total == 30

    def test_finding_has_all_fields(self):
        gt = load_ground_truth(XLSX_PATH)
        f = gt["zkllm"][0]
        assert isinstance(f, GroundTruthFinding)
        assert f.entry_id
        assert f.issue_id
        assert f.issue_name
        assert f.issue_explanation
        assert f.severity in {"Critical", "Warning", "Info"}
        assert f.category
        assert f.security_concern

    def test_code_refs_parsed(self):
        gt = load_ground_truth(XLSX_PATH)
        # zkLLM-01 has two code refs
        zkllm_01 = next(f for f in gt["zkllm"] if f.issue_id == "zkLLM-01")
        assert len(zkllm_01.relevant_code) == 2
        assert zkllm_01.relevant_code[0].filename == "rmsnorm.cu"

    def test_empty_code_refs(self):
        gt = load_ground_truth(XLSX_PATH)
        # zkLLM-02 has no code refs ("None")
        zkllm_02 = next(f for f in gt["zkllm"] if f.issue_id == "zkLLM-02")
        assert zkllm_02.relevant_code == []

    def test_paper_reference_dash(self):
        gt = load_ground_truth(XLSX_PATH)
        # zkLLM-02 has paper_reference = "-"
        zkllm_02 = next(f for f in gt["zkllm"] if f.issue_id == "zkLLM-02")
        assert zkllm_02.paper_reference == "-"

    def test_severity_values(self):
        gt = load_ground_truth(XLSX_PATH)
        all_severities = {f.severity for findings in gt.values() for f in findings}
        assert all_severities <= {"Critical", "Warning", "Info"}

    def test_category_values(self):
        gt = load_ground_truth(XLSX_PATH)
        from grader import CATEGORIES
        for findings in gt.values():
            for f in findings:
                assert f.category in CATEGORIES, f"{f.issue_id}: {f.category}"

    def test_security_concern_values(self):
        gt = load_ground_truth(XLSX_PATH)
        from grader import SECURITY_CONCERNS
        for findings in gt.values():
            for f in findings:
                assert f.security_concern in SECURITY_CONCERNS, f"{f.issue_id}: {f.security_concern}"


# ---------------------------------------------------------------------------
# load_agent_output
# ---------------------------------------------------------------------------

def _make_agent_json(findings: list[dict], tmp_path: Path) -> Path:
    """Helper to write a temp JSON file."""
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(findings), encoding="utf-8")
    return path


class TestLoadAgentOutput:
    def test_valid_single_finding(self, tmp_path):
        data = [
            {
                "entry-id": "zkLLM",
                "issue-name": "Test Issue",
                "issue-explanation": "Some explanation",
                "severity": "Critical",
                "category": "Under-constrained Circuit",
                "security-concern": "Proof Forgery (Soundness)",
                "relevant-code": "file.cu:10",
                "paper-reference": "Section 1",
            }
        ]
        path = _make_agent_json(data, tmp_path)
        result = load_agent_output(path)
        assert "zkllm" in result
        assert len(result["zkllm"]) == 1
        f = result["zkllm"][0]
        assert f.issue_name == "Test Issue"
        assert f.severity == "Critical"

    def test_groups_by_entry_id(self, tmp_path):
        data = [
            {
                "entry-id": "zkLLM",
                "issue-name": "Issue A",
                "issue-explanation": "Explanation A",
                "severity": "Critical",
                "category": "Other",
                "security-concern": "Other",
                "relevant-code": "",
                "paper-reference": "-",
            },
            {
                "entry-id": "zkGPT",
                "issue-name": "Issue B",
                "issue-explanation": "Explanation B",
                "severity": "Warning",
                "category": "Other",
                "security-concern": "Other",
                "relevant-code": "",
                "paper-reference": "-",
            },
            {
                "entry-id": "zkLLM",
                "issue-name": "Issue C",
                "issue-explanation": "Explanation C",
                "severity": "Info",
                "category": "Other",
                "security-concern": "Other",
                "relevant-code": "",
                "paper-reference": "-",
            },
        ]
        path = _make_agent_json(data, tmp_path)
        result = load_agent_output(path)
        assert len(result["zkllm"]) == 2
        assert len(result["zkgpt"]) == 1

    def test_missing_required_field_raises(self, tmp_path):
        data = [
            {
                "entry-id": "zkLLM",
                "issue-name": "Incomplete",
                # missing issue-explanation, severity, etc.
            }
        ]
        path = _make_agent_json(data, tmp_path)
        with pytest.raises(ValueError, match="missing required fields"):
            load_agent_output(path)

    def test_invalid_severity_raises(self, tmp_path):
        data = [
            {
                "entry-id": "zkLLM",
                "issue-name": "Bad Severity",
                "issue-explanation": "Explanation",
                "severity": "High",  # invalid
                "category": "Other",
                "security-concern": "Other",
                "relevant-code": "",
                "paper-reference": "-",
            }
        ]
        path = _make_agent_json(data, tmp_path)
        with pytest.raises(ValueError, match="invalid severity"):
            load_agent_output(path)

    def test_invalid_category_raises(self, tmp_path):
        data = [
            {
                "entry-id": "zkLLM",
                "issue-name": "Bad Category",
                "issue-explanation": "Explanation",
                "severity": "Critical",
                "category": "Nonexistent Category",
                "security-concern": "Other",
                "relevant-code": "",
                "paper-reference": "-",
            }
        ]
        path = _make_agent_json(data, tmp_path)
        with pytest.raises(ValueError, match="invalid category"):
            load_agent_output(path)

    def test_invalid_security_concern_raises(self, tmp_path):
        data = [
            {
                "entry-id": "zkLLM",
                "issue-name": "Bad Concern",
                "issue-explanation": "Explanation",
                "severity": "Critical",
                "category": "Other",
                "security-concern": "Made Up Concern",
                "relevant-code": "",
                "paper-reference": "-",
            }
        ]
        path = _make_agent_json(data, tmp_path)
        with pytest.raises(ValueError, match="invalid security-concern"):
            load_agent_output(path)

    def test_not_a_list_raises(self, tmp_path):
        path = tmp_path / "agent.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON array"):
            load_agent_output(path)

    def test_code_refs_parsed_in_agent(self, tmp_path):
        data = [
            {
                "entry-id": "zkLLM",
                "issue-name": "Test",
                "issue-explanation": "Test",
                "severity": "Critical",
                "category": "Other",
                "security-concern": "Other",
                "relevant-code": "file.cu:10-20, other.py:5",
                "paper-reference": "-",
            }
        ]
        path = _make_agent_json(data, tmp_path)
        result = load_agent_output(path)
        f = result["zkllm"][0]
        assert len(f.relevant_code) == 2
        assert f.relevant_code[0] == CodeRef("file.cu", 10, 20)
        assert f.relevant_code[1] == CodeRef("other.py", 5, 5)

    def test_empty_array_returns_empty(self, tmp_path):
        path = _make_agent_json([], tmp_path)
        result = load_agent_output(path)
        assert result == {}
