"""Tests for grader.loader module.

All strings and project names below are synthetic placeholders; none of them
reflect entries in the real benchmark dataset.
"""

import json
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
        refs = parse_code_refs("widget.rs:21-31")
        assert refs == [CodeRef("widget.rs", 21, 31)]

    def test_single_ref_single_line(self):
        refs = parse_code_refs("gadget.cpp:658")
        assert refs == [CodeRef("gadget.cpp", 658, 658)]

    def test_multiple_comma_separated(self):
        refs = parse_code_refs("widget.rs:21-31, helper.py:34-38")
        assert len(refs) == 2
        assert refs[0] == CodeRef("widget.rs", 21, 31)
        assert refs[1] == CodeRef("helper.py", 34, 38)

    def test_file_without_lines(self):
        refs = parse_code_refs("config.yaml")
        assert refs == [CodeRef("config.yaml", None, None)]

    def test_mixed_with_and_without_lines(self):
        refs = parse_code_refs("config.yaml, config.rs:31")
        assert len(refs) == 2
        assert refs[0] == CodeRef("config.yaml", None, None)
        assert refs[1] == CodeRef("config.rs", 31, 31)

    def test_path_with_directories(self):
        refs = parse_code_refs("src/util/widget.rs:36-42")
        assert refs == [CodeRef("src/util/widget.rs", 36, 42)]

    def test_none_returns_empty(self):
        assert parse_code_refs(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_code_refs("") == []

    def test_dash_returns_empty(self):
        assert parse_code_refs("-") == []

    def test_none_string_returns_empty(self):
        assert parse_code_refs("None") == []

    def test_unicode_dash_in_range(self):
        # Real data files sometimes use en-dash (–) in ranges
        refs = parse_code_refs("widget.rs:267–370")
        assert refs == [CodeRef("widget.rs", 267, 370)]

    def test_semicolon_separator(self):
        refs = parse_code_refs("file1.rs:10; file2.rs:20")
        assert len(refs) == 2


# ---------------------------------------------------------------------------
# load_ground_truth (uses synthetic JSON fixture from conftest.py)
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_loads_all_projects(self, fictional_gt_json_path, fictional_gt_rows):
        gt = load_ground_truth(fictional_gt_json_path)
        expected_projects = {row["entry-id"].lower() for row in fictional_gt_rows}
        assert set(gt.keys()) == expected_projects

    def test_total_finding_count(self, fictional_gt_json_path, fictional_gt_rows):
        gt = load_ground_truth(fictional_gt_json_path)
        total = sum(len(v) for v in gt.values())
        assert total == len(fictional_gt_rows)

    def test_finding_has_all_fields(self, fictional_gt_json_path):
        gt = load_ground_truth(fictional_gt_json_path)
        first_project = next(iter(gt.values()))
        f = first_project[0]
        assert isinstance(f, GroundTruthFinding)
        assert f.entry_id
        assert f.issue_id
        assert f.issue_name
        assert f.issue_explanation

    def test_code_refs_parsed(self, fictional_gt_json_path):
        gt = load_ground_truth(fictional_gt_json_path)
        # alpha-01 has two code refs in the fixture
        alpha_01 = next(f for f in gt["alpha"] if f.issue_id == "alpha-01")
        assert len(alpha_01.relevant_code) == 2

    def test_empty_code_refs(self, fictional_gt_json_path):
        gt = load_ground_truth(fictional_gt_json_path)
        # alpha-03 has empty relevant-code in the fixture
        alpha_03 = next(f for f in gt["alpha"] if f.issue_id == "alpha-03")
        assert alpha_03.relevant_code == []

    def test_paper_reference_dash(self, fictional_gt_json_path):
        gt = load_ground_truth(fictional_gt_json_path)
        # alpha-02 has paper_reference = "-" in the fixture
        alpha_02 = next(f for f in gt["alpha"] if f.issue_id == "alpha-02")
        assert alpha_02.paper_reference == "-"


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
                "entry-id": "alpha",
                "issue-name": "Test Issue",
                "issue-explanation": "Some explanation",
                "relevant-code": "file.cu:10",
                "paper-reference": "Section 1",
            }
        ]
        path = _make_agent_json(data, tmp_path)
        result = load_agent_output(path)
        assert "alpha" in result
        assert len(result["alpha"]) == 1
        f = result["alpha"][0]
        assert f.issue_name == "Test Issue"

    def test_groups_by_entry_id(self, tmp_path):
        data = [
            {
                "entry-id": "alpha",
                "issue-name": "Issue A",
                "issue-explanation": "Explanation A",
                "relevant-code": "",
                "paper-reference": "-",
            },
            {
                "entry-id": "beta",
                "issue-name": "Issue B",
                "issue-explanation": "Explanation B",
                "relevant-code": "",
                "paper-reference": "-",
            },
            {
                "entry-id": "alpha",
                "issue-name": "Issue C",
                "issue-explanation": "Explanation C",
                "relevant-code": "",
                "paper-reference": "-",
            },
        ]
        path = _make_agent_json(data, tmp_path)
        result = load_agent_output(path)
        assert len(result["alpha"]) == 2
        assert len(result["beta"]) == 1

    def test_missing_required_field_raises(self, tmp_path):
        data = [
            {
                "entry-id": "alpha",
                "issue-name": "Incomplete",
                # missing issue-explanation, etc.
            }
        ]
        path = _make_agent_json(data, tmp_path)
        with pytest.raises(ValueError, match="missing required fields"):
            load_agent_output(path)

    def test_not_a_list_raises(self, tmp_path):
        path = tmp_path / "agent.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON array"):
            load_agent_output(path)

    def test_code_refs_parsed_in_agent(self, tmp_path):
        data = [
            {
                "entry-id": "alpha",
                "issue-name": "Test",
                "issue-explanation": "Test",
                "relevant-code": "file.cu:10-20, other.py:5",
                "paper-reference": "-",
            }
        ]
        path = _make_agent_json(data, tmp_path)
        result = load_agent_output(path)
        f = result["alpha"][0]
        assert len(f.relevant_code) == 2
        assert f.relevant_code[0] == CodeRef("file.cu", 10, 20)
        assert f.relevant_code[1] == CodeRef("other.py", 5, 5)

    def test_empty_array_returns_empty(self, tmp_path):
        path = _make_agent_json([], tmp_path)
        result = load_agent_output(path)
        assert result == {}


# ---------------------------------------------------------------------------
# load_ground_truth JSON-specific validation
# ---------------------------------------------------------------------------

def _make_gt_json(findings: list[dict], tmp_path: Path) -> Path:
    """Helper to write a temp GT JSON file."""
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(findings), encoding="utf-8")
    return path


class TestLoadGroundTruthJSON:
    def test_not_a_list_raises(self, tmp_path):
        path = tmp_path / "gt.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON array"):
            load_ground_truth(path)

    def test_missing_required_field_raises(self, tmp_path):
        data = [{"entry-id": "alpha", "issue-name": "Incomplete"}]
        path = _make_gt_json(data, tmp_path)
        with pytest.raises(ValueError, match="missing required fields"):
            load_ground_truth(path)

    def test_issue_id_auto_generated(self, tmp_path):
        data = [
            {
                "entry-id": "alpha",
                "issue-name": "No ID",
                "issue-explanation": "Explanation here for the test",
                "relevant-code": "",
                "paper-reference": "-",
            }
        ]
        path = _make_gt_json(data, tmp_path)
        gt = load_ground_truth(path)
        f = gt["alpha"][0]
        assert f.issue_id == "alpha-01"

    def test_explicit_issue_id_preserved(self, tmp_path):
        data = [
            {
                "entry-id": "alpha",
                "issue-id": "zkML-001",
                "issue-name": "With ID",
                "issue-explanation": "Explanation here for the test",
                "relevant-code": "",
                "paper-reference": "-",
            }
        ]
        path = _make_gt_json(data, tmp_path)
        gt = load_ground_truth(path)
        f = gt["alpha"][0]
        assert f.issue_id == "zkML-001"

    def test_empty_array_returns_empty(self, tmp_path):
        path = _make_gt_json([], tmp_path)
        result = load_ground_truth(path)
        assert result == {}

