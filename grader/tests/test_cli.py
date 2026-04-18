"""Tests for grader.cli module.

End-to-end tests use the synthetic xlsx + agent JSON fixtures from
conftest.py; no real dataset content is read or asserted against.
"""

import json
from pathlib import Path

import pytest

from grader.cli import _parse_weights, _get_backend, main
from grader.report import DEFAULT_WEIGHTS
from grader.similarity import JaccardSimilarity


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
# End-to-end CLI (uses synthetic fixtures from conftest.py)
# ---------------------------------------------------------------------------

class TestMainEndToEnd:
    def test_runs_with_json_output(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
        fictional_gt_rows,
    ):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert "meta" in data
        assert "projects" in data
        assert "overall" in data
        assert data["overall"]["total_gt"] == len(fictional_gt_rows)

    def test_runs_with_markdown_output(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
    ):
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output-md", out_md,
        ])
        content = Path(out_md).read_text(encoding="utf-8")
        assert "# zkML Benchmark Grading Report" in content

    def test_runs_with_both_outputs(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
    ):
        out_json = str(tmp_path / "report.json")
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output", out_json,
            "--output-md", out_md,
        ])
        assert Path(out_json).exists()
        assert Path(out_md).exists()

    def test_custom_threshold(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
    ):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--threshold", "0.5",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["match_threshold"] == 0.5
        # Higher threshold should not produce more matches than GT size
        assert data["overall"]["total_matched"] <= data["overall"]["total_gt"]

    def test_custom_weights(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
    ):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--weights", "severity=0.5,code_location=0.1",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["weights"]["severity"] == 0.5
        assert data["meta"]["weights"]["code_location"] == 0.1

    def test_projects_present(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
        fictional_gt_rows,
    ):
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        projects = data["projects"]
        expected = {row["entry-id"].lower() for row in fictional_gt_rows}
        assert set(projects.keys()) == expected

    def test_no_output_still_runs(
        self, fictional_xlsx_path, fictional_agent_json_path, capsys,
    ):
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
        ])
        captured = capsys.readouterr()
        assert "benchmark_score" in captured.out
