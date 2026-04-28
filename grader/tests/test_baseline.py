"""Tests for grader.baseline — baseline finding exclusion.

All test data is synthetic — no real dataset content.
Uses MockLLMProvider to avoid any API traffic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from grader.baseline import (
    BaselineFinding,
    filter_baseline,
    load_baseline,
    pair_prefix,
)
from grader.llm import MockLLMProvider
from grader.loader import AgentFinding, CodeRef
from grader.similarity import LLMJudgeSimilarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent(name: str, explanation: str = "default", paper: str = "-") -> AgentFinding:
    return AgentFinding(
        entry_id="test",
        issue_name=name,
        issue_explanation=explanation,
        relevant_code=[],
        paper_reference=paper,
    )


def _baseline(pair: str, bid: str, name: str, explanation: str = "default") -> BaselineFinding:
    return BaselineFinding(
        pair=pair,
        baseline_id=bid,
        issue_name=name,
        issue_explanation=explanation,
    )


def _responder_with_scores(score_map: dict[str, int]):
    """Build a responder that scores each baseline candidate using score_map."""
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
# load_baseline
# ---------------------------------------------------------------------------

class TestLoadBaseline:
    def test_groups_by_pair(self, tmp_path: Path):
        data = [
            {
                "entry-id": "alpha",
                "issue-name": "Issue A",
                "issue-explanation": "Explanation A",
                "relevant-code": "src/foo.rs:10",
                "paper-reference": "Section 1",
            },
            {
                "entry-id": "alpha",
                "issue-name": "Issue B",
                "issue-explanation": "Explanation B",
                "relevant-code": "-",
                "paper-reference": "-",
            },
            {
                "entry-id": "beta",
                "issue-name": "Issue C",
                "issue-explanation": "Explanation C",
                "relevant-code": "main.cpp:5",
                "paper-reference": "Section 2",
            },
        ]
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        result = load_baseline(p)
        assert set(result.keys()) == {"alpha", "beta"}
        assert len(result["alpha"]) == 2
        assert len(result["beta"]) == 1
        assert result["alpha"][0].issue_name == "Issue A"
        assert result["alpha"][0].baseline_id == "alpha-baseline-00"
        assert result["beta"][0].baseline_id == "beta-baseline-02"

    def test_empty_array(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        result = load_baseline(p)
        assert result == {}

    def test_rejects_non_array(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="flat array"):
            load_baseline(p)

    def test_parses_code_refs(self, tmp_path: Path):
        data = [
            {
                "entry-id": "gamma",
                "issue-name": "X",
                "issue-explanation": "Y",
                "relevant-code": "src/foo.rs:10-20, bar.cu:5",
                "paper-reference": "-",
            }
        ]
        p = tmp_path / "refs.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_baseline(p)
        refs = result["gamma"][0].relevant_code
        assert len(refs) == 2
        assert refs[0] == CodeRef("src/foo.rs", 10, 20)
        assert refs[1] == CodeRef("bar.cu", 5, 5)


# ---------------------------------------------------------------------------
# pair_prefix
# ---------------------------------------------------------------------------

class TestPairPrefix:
    def test_simple_pair(self):
        assert pair_prefix("zkllm") == "zkllm"

    def test_condition_entry_id(self):
        assert pair_prefix("zkllm-A-zkLLM-003-rep02") == "zkllm"

    def test_b_max(self):
        assert pair_prefix("zktorch-B-max-rep01") == "zktorch"

    def test_case_insensitive(self):
        assert pair_prefix("ZkGPT-E-clean-rep03") == "zkgpt"


# ---------------------------------------------------------------------------
# filter_baseline
# ---------------------------------------------------------------------------

class TestFilterBaseline:
    def test_excludes_matching_findings(self):
        """Agent finding with high baseline score is excluded."""
        baselines = [_baseline("test", "b-00", "Known gap X")]
        agents = [_agent("Known gap X variant")]

        mock = MockLLMProvider(_responder_with_scores({"b-00": 5}))
        backend = LLMJudgeSimilarity(mock)

        remaining, excluded = filter_baseline(agents, baselines, backend, threshold=4)
        assert len(excluded) == 1
        assert len(remaining) == 0
        assert excluded[0].issue_name == "Known gap X variant"

    def test_keeps_non_matching_findings(self):
        """Agent finding with low baseline score is kept."""
        baselines = [_baseline("test", "b-00", "Totally different issue")]
        agents = [_agent("Unique agent finding")]

        mock = MockLLMProvider(_responder_with_scores({"b-00": 2}))
        backend = LLMJudgeSimilarity(mock)

        remaining, excluded = filter_baseline(agents, baselines, backend, threshold=4)
        assert len(remaining) == 1
        assert len(excluded) == 0

    def test_mixed_findings(self):
        """Two agents: one matches baseline, one doesn't."""
        baselines = [_baseline("test", "b-00", "Known issue")]
        agents = [_agent("Known issue copy"), _agent("Novel finding")]

        call_count = [0]
        def _responder(system, user, schema):
            call_count[0] += 1
            ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
            # First call: agent matches baseline; second call: doesn't
            score = 5 if call_count[0] == 1 else 1
            return {
                "judgments": [
                    {"gt_id": cid, "match_score": score, "reasoning": "canned"}
                    for cid in ids
                ]
            }

        mock = MockLLMProvider(_responder)
        backend = LLMJudgeSimilarity(mock)

        remaining, excluded = filter_baseline(agents, baselines, backend, threshold=4)
        assert len(excluded) == 1
        assert len(remaining) == 1
        assert excluded[0].issue_name == "Known issue copy"
        assert remaining[0].issue_name == "Novel finding"

    def test_empty_baselines_returns_all(self):
        """No baselines → all agent findings pass through."""
        agents = [_agent("Finding A"), _agent("Finding B")]
        remaining, excluded = filter_baseline(
            agents, [], LLMJudgeSimilarity(MockLLMProvider([])), threshold=4,
        )
        assert len(remaining) == 2
        assert len(excluded) == 0

    def test_empty_agents_returns_empty(self):
        """No agent findings → empty result."""
        baselines = [_baseline("test", "b-00", "Known")]
        remaining, excluded = filter_baseline(
            [], baselines, LLMJudgeSimilarity(MockLLMProvider([])), threshold=4,
        )
        assert len(remaining) == 0
        assert len(excluded) == 0

    def test_threshold_boundary(self):
        """Score == threshold should exclude; score == threshold-1 should keep."""
        baselines = [_baseline("test", "b-00", "Borderline")]
        agent_at = _agent("At threshold")
        agent_below = _agent("Below threshold")

        # Agent at threshold: score=4
        mock_at = MockLLMProvider(_responder_with_scores({"b-00": 4}))
        remaining, excluded = filter_baseline(
            [agent_at], baselines, LLMJudgeSimilarity(mock_at), threshold=4,
        )
        assert len(excluded) == 1

        # Agent below threshold: score=3
        mock_below = MockLLMProvider(_responder_with_scores({"b-00": 3}))
        remaining, excluded = filter_baseline(
            [agent_below], baselines, LLMJudgeSimilarity(mock_below), threshold=4,
        )
        assert len(remaining) == 1
        assert len(excluded) == 0
