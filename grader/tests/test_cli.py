"""Tests for grader.cli module."""

import json
from pathlib import Path

import pytest

from grader.cli import _parse_weights, _get_backend, main
from grader.report import DEFAULT_WEIGHTS
from grader.similarity import JaccardSimilarity


TESTS_DIR = Path(__file__).parent
XLSX_PATH = TESTS_DIR.parent.parent / "zkMLDataset.xlsx"
AGENT_JSON_PATH = TESTS_DIR / "test_agent_output.json"


# ---------------------------------------------------------------------------
# _parse_weights
# ---------------------------------------------------------------------------

class TestParseWeights:
    def test_none_returns_defaults(self):
        w = _parse_weights(None)
        assert w == DEFAULT_WEIGHTS

    def test_override_single(self):
        w = _parse_weights("severity=0.5")
        assert w["severity"] == 0.5
        assert w["category"] == DEFAULT_WEIGHTS["category"]

    def test_override_multiple(self):
        w = _parse_weights("severity=0.2, code_location=0.4")
        assert w["severity"] == 0.2
        assert w["code_location"] == 0.4

    def test_invalid_field_raises(self):
        with pytest.raises(ValueError, match="Unknown weight field"):
            _parse_weights("nonexistent=0.5")

    def test_float_parsing(self):
        w = _parse_weights("severity=0.123")
        assert w["severity"] == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# _get_backend
# ---------------------------------------------------------------------------

class TestGetBackend:
    def test_jaccard(self):
        backend = _get_backend("jaccard")
        assert isinstance(backend, JaccardSimilarity)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity backend"):
            _get_backend("nonexistent")


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not XLSX_PATH.exists() or not AGENT_JSON_PATH.exists(),
    reason="test fixtures not found",
)
class TestMainEndToEnd:
    def test_runs_with_json_output(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(XLSX_PATH),
            "--agent-output", str(AGENT_JSON_PATH),
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert "meta" in data
        assert "projects" in data
        assert "overall" in data
        assert data["overall"]["total_gt"] == 30

    def test_runs_with_markdown_output(self, tmp_path):
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(XLSX_PATH),
            "--agent-output", str(AGENT_JSON_PATH),
            "--output-md", out_md,
        ])
        content = Path(out_md).read_text(encoding="utf-8")
        assert "# zkML Benchmark Grading Report" in content

    def test_runs_with_both_outputs(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(XLSX_PATH),
            "--agent-output", str(AGENT_JSON_PATH),
            "--output", out_json,
            "--output-md", out_md,
        ])
        assert Path(out_json).exists()
        assert Path(out_md).exists()

    def test_custom_threshold(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(XLSX_PATH),
            "--agent-output", str(AGENT_JSON_PATH),
            "--threshold", "0.5",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["match_threshold"] == 0.5
        # Higher threshold means fewer matches
        assert data["overall"]["total_matched"] <= 30

    def test_custom_weights(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(XLSX_PATH),
            "--agent-output", str(AGENT_JSON_PATH),
            "--weights", "severity=0.5,code_location=0.1",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["weights"]["severity"] == 0.5
        assert data["meta"]["weights"]["code_location"] == 0.1

    def test_projects_present(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(XLSX_PATH),
            "--agent-output", str(AGENT_JSON_PATH),
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        projects = data["projects"]
        assert "zkllm" in projects
        assert "zkgpt" in projects
        assert "zkml" in projects
        assert "zktorch" in projects

    def test_no_output_still_runs(self, tmp_path, capsys):
        # Should print to stdout without error
        main([
            "--ground-truth", str(XLSX_PATH),
            "--agent-output", str(AGENT_JSON_PATH),
        ])
        captured = capsys.readouterr()
        assert "benchmark_score" in captured.out
