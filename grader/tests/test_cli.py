"""Tests for grader.cli.

Every end-to-end test uses `_LLM_PROVIDER_OVERRIDE` to inject a
MockLLMProvider — no real API is contacted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import grader.cli as cli_module
from grader.cli import _build_backend, _parse_weights, main
from grader.llm import MockLLMProvider
from grader.report import DEFAULT_WEIGHTS
from grader.similarity import LLMJudgeSimilarity


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate each test from ambient LLM env vars AND from any .env file.

    We patch load_dotenv_if_available to a no-op so the developer's real
    `.env` (if present at the repo root) does not leak keys into tests that
    expect the missing-env error path.
    """
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY", "OPENAI_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    # Prevent .env from being loaded during tests
    monkeypatch.setattr("grader.llm.load_dotenv_if_available", lambda *a, **k: None)


@pytest.fixture
def reset_override():
    """Ensure the test seam is always cleared after each test."""
    yield
    cli_module._LLM_PROVIDER_OVERRIDE = None


def _bulk_responder(score_map: dict[str, float]):
    """Responder that scores each candidate id using score_map."""
    def _responder(system, user, schema):
        ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
        return {
            "judgments": [
                {
                    "gt_id": cid,
                    "match_score": score_map.get(cid, 0.0),
                    "same_root_cause": score_map.get(cid, 0.0) >= 0.7,
                    "reasoning": f"canned for {cid}",
                }
                for cid in ids
            ]
        }
    return _responder


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
# _build_backend
# ---------------------------------------------------------------------------

class TestBuildBackend:
    def test_without_env_raises_value_error(self):
        with pytest.raises(ValueError, match="requires LLM configuration"):
            _build_backend()

    def test_error_message_points_to_env_example(self):
        with pytest.raises(ValueError, match=r"\.env\.example"):
            _build_backend()

    def test_override_returns_llm_judge(self, reset_override):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider([])
        backend = _build_backend()
        assert isinstance(backend, LLMJudgeSimilarity)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestMainEndToEnd:
    def test_runs_with_json_output(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
        fictional_gt_rows, reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({"alpha-01": 0.9, "alpha-02": 0.9, "alpha-03": 0.1,
                             "beta-01": 0.9, "beta-02": 0.1})
        )
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
        assert data["meta"]["similarity_backend"] == "llm-judge"

    def test_runs_with_markdown_output(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
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
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
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
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--threshold", "0.5",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["match_threshold"] == 0.5
        assert data["overall"]["total_matched"] <= data["overall"]["total_gt"]

    def test_custom_weights(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
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
        fictional_gt_rows, reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
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
        self, fictional_xlsx_path, fictional_agent_json_path,
        capsys, reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
        ])
        captured = capsys.readouterr()
        assert "benchmark_score" in captured.out


# ---------------------------------------------------------------------------
# Missing-env path
# ---------------------------------------------------------------------------

class TestMissingEnv:
    def test_running_without_env_produces_helpful_error(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
    ):
        # No override set; env is cleared by the autouse fixture
        with pytest.raises(ValueError, match="requires LLM configuration"):
            main([
                "--ground-truth", str(fictional_xlsx_path),
                "--agent-output", str(fictional_agent_json_path),
            ])


# ---------------------------------------------------------------------------
# CLI help text
# ---------------------------------------------------------------------------

class TestHelpText:
    def test_help_does_not_mention_backend_flag(self, capsys):
        """--backend was removed; it must not appear in --help anymore."""
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "--backend" not in out

    def test_help_mentions_env_example(self, capsys):
        """Help text should point users at .env.example for configuration."""
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert ".env" in out
