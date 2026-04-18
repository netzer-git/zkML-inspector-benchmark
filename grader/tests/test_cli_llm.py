"""Tests for --backend llm-judge CLI wiring.

Uses the `_LLM_PROVIDER_OVERRIDE` test seam to inject a MockLLMProvider so
no real API is ever hit. Also verifies the missing-env error path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import grader.cli as cli_module
from grader.cli import _get_backend, main
from grader.llm import MockLLMProvider
from grader.similarity import JaccardSimilarity, LLMJudgeSimilarity


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate each test from ambient LLM env vars."""
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY", "OPENAI_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def reset_override():
    """Ensure _LLM_PROVIDER_OVERRIDE is always None after each test."""
    yield
    cli_module._LLM_PROVIDER_OVERRIDE = None


def _bulk_responder(score_map: dict[str, float]):
    """Responder that echoes candidates with scores from the map, keyed by candidate id."""
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
# _get_backend
# ---------------------------------------------------------------------------

class TestGetBackendLLM:
    def test_jaccard_still_works(self):
        b = _get_backend("jaccard")
        assert isinstance(b, JaccardSimilarity)

    def test_llm_judge_without_env_raises_value_error(self):
        with pytest.raises(ValueError, match="llm-judge backend requires env"):
            _get_backend("llm-judge")

    def test_llm_judge_error_message_points_to_env_example(self):
        with pytest.raises(ValueError, match=r"\.env\.example"):
            _get_backend("llm-judge")

    def test_llm_judge_uses_override_when_set(self, reset_override):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider([])
        b = _get_backend("llm-judge")
        assert isinstance(b, LLMJudgeSimilarity)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity backend"):
            _get_backend("gemini")

    def test_unknown_backend_mentions_available_options(self):
        try:
            _get_backend("gemini")
        except ValueError as e:
            msg = str(e)
            assert "jaccard" in msg and "llm-judge" in msg


# ---------------------------------------------------------------------------
# End-to-end with mocked LLM
# ---------------------------------------------------------------------------

class TestMainEndToEndWithLLMJudge:
    def test_llm_judge_runs_end_to_end(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
        fictional_gt_rows, reset_override,
    ):
        # Mock LLM returns high match score for every candidate
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({"0": 0.9, "1": 0.9, "2": 0.9})
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--backend", "llm-judge",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["similarity_backend"] == "llm-judge"
        assert data["overall"]["total_gt"] == len(fictional_gt_rows)
        # With high mock scores, total_matched should be at least 1
        assert data["overall"]["total_matched"] >= 1

    def test_llm_judge_missing_env_produces_helpful_error(
        self, tmp_path, fictional_xlsx_path, fictional_agent_json_path,
    ):
        with pytest.raises(ValueError, match="llm-judge backend requires env"):
            main([
                "--ground-truth", str(fictional_xlsx_path),
                "--agent-output", str(fictional_agent_json_path),
                "--backend", "llm-judge",
            ])

    def test_llm_judge_with_env_var_selection(
        self, monkeypatch, tmp_path, fictional_xlsx_path,
        fictional_agent_json_path, reset_override,
    ):
        """Even with env vars set, the override takes precedence for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({"0": 0.5})
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_xlsx_path),
            "--agent-output", str(fictional_agent_json_path),
            "--backend", "llm-judge",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["similarity_backend"] == "llm-judge"


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

class TestBackendHelpText:
    def test_help_mentions_both_backends(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "jaccard" in out
        assert "llm-judge" in out
