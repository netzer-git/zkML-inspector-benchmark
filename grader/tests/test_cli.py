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
                    "match_score": score_map.get(cid, 1),
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
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        fictional_gt_rows, reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({"alpha-01": 5, "alpha-02": 5, "alpha-03": 1,
                             "beta-01": 5, "beta-02": 1})
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
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
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output-md", out_md,
        ])
        content = Path(out_md).read_text(encoding="utf-8")
        assert "# zkML Benchmark Grading Report" in content

    def test_runs_with_both_outputs(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        out_json = str(tmp_path / "report.json")
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output", out_json,
            "--output-md", out_md,
        ])
        assert Path(out_json).exists()
        assert Path(out_md).exists()

    def test_custom_threshold(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--threshold", "5",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["match_threshold"] == 5
        assert data["overall"]["total_matched"] <= data["overall"]["total_gt"]

    def test_custom_weights(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--weights", "severity=0.5,code_location=0.1",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["meta"]["weights"]["severity"] == 0.5
        assert data["meta"]["weights"]["code_location"] == 0.1

    def test_projects_present(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        fictional_gt_rows, reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        projects = data["projects"]
        expected = {row["entry-id"].lower() for row in fictional_gt_rows}
        assert set(projects.keys()) == expected

    def test_no_output_still_runs(
        self, fictional_gt_json_path, fictional_agent_json_path,
        capsys, reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({})
        )
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
        ])
        captured = capsys.readouterr()
        assert "benchmark_score" in captured.out


# ---------------------------------------------------------------------------
# Missing-env path
# ---------------------------------------------------------------------------

class TestMissingEnv:
    def test_running_without_env_produces_helpful_error(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
    ):
        # No override set; env is cleared by the autouse fixture
        with pytest.raises(ValueError, match="requires LLM configuration"):
            main([
                "--ground-truth", str(fictional_gt_json_path),
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


# ---------------------------------------------------------------------------
# Per-project error isolation
# ---------------------------------------------------------------------------

class TestPerProjectErrorIsolation:
    """A single-project API failure must not halt the entire grading run."""

    def test_one_project_fails_others_complete(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        """Mock provider: succeed on 'alpha', fail on 'beta'.

        Verifies:
        - Run completes (no exception bubbles to caller).
        - alpha is scored and appears in the report.
        - beta is absent from project_grades but recorded in meta.failed_projects.
        """
        def responder(system, user, schema):
            ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
            # Candidate IDs start with the project prefix; detect which
            # project this call is for by looking at the first id.
            if ids and ids[0].startswith("beta"):
                raise RuntimeError("simulated API failure for beta")
            return {
                "judgments": [
                    {"gt_id": cid, "match_score": 5, "reasoning": "canned"}
                    for cid in ids
                ]
            }

        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(responder)
        out_json = str(tmp_path / "report.json")

        # Must not raise
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output", out_json,
        ])

        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert "alpha" in data["projects"]
        assert "beta" not in data["projects"]

        failed = data["meta"]["failed_projects"]
        assert len(failed) == 1
        assert failed[0]["project"] == "beta"
        assert failed[0]["error_type"] == "RuntimeError"
        assert "simulated API failure" in failed[0]["error"]

    def test_failures_in_markdown_output(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        """The failed projects section appears in the markdown report."""
        def responder(system, user, schema):
            ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
            if ids and ids[0].startswith("beta"):
                raise RuntimeError("boom")
            return {"judgments": [
                {"gt_id": cid, "match_score": 5, "reasoning": ""}
                for cid in ids
            ]}

        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(responder)
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output-md", out_md,
        ])
        content = Path(out_md).read_text(encoding="utf-8")
        assert "Skipped / failed projects" in content
        assert "Failed during grading" in content
        assert "beta" in content
        assert "RuntimeError" in content

    def test_all_projects_fail_still_produces_report(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        """If every project fails, we still get a (zero-scored) report."""
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            lambda s, u, sc: (_ for _ in ()).throw(RuntimeError("all broken"))
        )
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["projects"] == {}
        assert len(data["meta"]["failed_projects"]) == 2  # alpha + beta
        assert data["overall"]["total_matched"] == 0

    def test_skipped_projects_recorded_in_meta(
        self, tmp_path, fictional_gt_json_path, reset_override,
    ):
        """Agent findings for projects not in GT land in skipped_projects."""
        # Agent has a finding for a project the GT JSON doesn't contain
        agent_path = tmp_path / "agent.json"
        agent_path.write_text(json.dumps([{
            "entry-id": "gamma",  # not in fictional GT
            "issue-name": "Foo",
            "issue-explanation": "Bar",
            "severity": "Warning",
            "category": "Other",
            "security-concern": "Other",
            "relevant-code": "",
            "paper-reference": "-",
        }]))

        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(_bulk_responder({}))
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(agent_path),
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert "gamma" in data["meta"]["skipped_projects"]


# ---------------------------------------------------------------------------
# --entry-id filter
# ---------------------------------------------------------------------------

class TestEntryIdFilter:
    def test_filter_restricts_to_selected_project(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(_bulk_responder({}))
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--entry-id", "alpha",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert set(data["projects"].keys()) == {"alpha"}

    def test_filter_is_case_insensitive(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(_bulk_responder({}))
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--entry-id", "ALPHA",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert set(data["projects"].keys()) == {"alpha"}

    def test_filter_repeatable_across_multiple_entries(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(_bulk_responder({}))
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--entry-id", "alpha",
            "--entry-id", "beta",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert set(data["projects"].keys()) == {"alpha", "beta"}

    def test_unknown_entry_id_produces_empty_report(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        """Filter to a non-existent entry-id → run completes with empty results."""
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(_bulk_responder({}))
        out_json = str(tmp_path / "report.json")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--entry-id", "nonexistent",
            "--output", out_json,
        ])
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        assert data["projects"] == {}
        assert data["overall"]["total_gt"] == 0


# ---------------------------------------------------------------------------
# --judge-trace debug output
# ---------------------------------------------------------------------------

class TestJudgeTrace:
    def test_writes_trace_markdown(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        # Mock: alpha-01 matches perfectly; others don't
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({
                "alpha-01": 5, "alpha-02": 1, "alpha-03": 1,
                "beta-01": 5, "beta-02": 1,
            })
        )
        trace_path = tmp_path / "trace.md"
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--judge-trace", str(trace_path),
        ])
        assert trace_path.exists()
        content = trace_path.read_text(encoding="utf-8")
        assert "# Judge Trace" in content
        assert "Project: `alpha`" in content
        assert "Project: `beta`" in content
        # At least one MATCHED row (alpha-01 scored 0.95)
        assert "**MATCHED**" in content

    def test_trace_contains_candidate_reasoning(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        """Reasoning is now a column in the candidates table, not a bulleted list."""
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({"alpha-01": 5, "alpha-02": 2})
        )
        trace_path = tmp_path / "trace.md"
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--judge-trace", str(trace_path),
        ])
        content = trace_path.read_text(encoding="utf-8")
        # Reasoning column header present in the candidates table
        assert "| Reasoning |" in content
        # Old bulleted list is gone
        assert "Reasoning per candidate:" not in content
        # The mock responder embeds "canned for <gt_id>" in reasoning
        assert "canned for alpha-01" in content

    def test_no_trace_file_without_flag(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(_bulk_responder({}))
        # Pre-list of files in tmp_path before the run
        before = set(tmp_path.iterdir())
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
        ])
        after = set(tmp_path.iterdir())
        # No new file should appear in tmp_path
        assert after == before

    def test_trace_empty_when_filtered_to_nothing(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(_bulk_responder({}))
        trace_path = tmp_path / "trace.md"
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--entry-id", "nonexistent",
            "--judge-trace", str(trace_path),
        ])
        assert trace_path.exists()
        content = trace_path.read_text(encoding="utf-8")
        # When no projects are graded, the trace notes that explicitly.
        assert "No projects graded" in content


# ---------------------------------------------------------------------------
# New behaviors: emoji badges, pair-score breakdown, dup markers
# ---------------------------------------------------------------------------

class TestGradeReportBadges:
    def test_green_badge_for_high_pair_score(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({
                "alpha-01": 5, "alpha-02": 5, "alpha-03": 1,
                "beta-01": 5, "beta-02": 1,
            })
        )
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--output-md", out_md,
        ])
        content = Path(out_md).read_text(encoding="utf-8")
        # Should contain at least one green badge
        assert "\U0001f7e2" in content  # 🟢

    def test_red_badge_for_low_pair_score(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        """When pair scores are low (below 0.4), the red badge should appear."""
        # Force low pair_scores by making the matches clear threshold but
        # agent severity/category/etc differ from GT.
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({"alpha-01": 5})
        )
        # Agent output that matches alpha-01 but has mismatched fields so
        # field scores are low.
        agent_path = tmp_path / "agent.json"
        agent_path.write_text(json.dumps([{
            "entry-id": "alpha",
            "issue-name": "Something",
            "issue-explanation": "Unrelated words entirely here",
            "severity": "Info",       # GT is Critical -> severity score 0
            "category": "Other",      # GT is Under-constrained Circuit -> 0
            "security-concern": "Other",  # GT is Proof Forgery -> 0.1
            "relevant-code": "wrong/path.rs:999",  # no match
            "paper-reference": "-",
        }]))
        out_md = str(tmp_path / "report.md")
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(agent_path),
            "--output-md", out_md,
        ])
        content = Path(out_md).read_text(encoding="utf-8")
        # Low pair_score should trigger red badge
        assert "\U0001f534" in content  # 🔴


class TestJudgeTraceContent:
    def test_trace_does_not_include_pair_score_breakdown(
        self, tmp_path, fictional_gt_json_path, fictional_agent_json_path,
        reset_override,
    ):
        """The trace focuses on the judge's view only. Per-field pair-score
        breakdowns belong in the normal grade report, not the trace."""
        cli_module._LLM_PROVIDER_OVERRIDE = MockLLMProvider(
            _bulk_responder({"alpha-01": 0.95, "alpha-02": 0.95,
                             "alpha-03": 0.1, "beta-01": 0.95, "beta-02": 0.1})
        )
        trace_path = tmp_path / "trace.md"
        main([
            "--ground-truth", str(fictional_gt_json_path),
            "--agent-output", str(fictional_agent_json_path),
            "--judge-trace", str(trace_path),
        ])
        content = trace_path.read_text(encoding="utf-8")
        assert "Pair-score breakdown" not in content

