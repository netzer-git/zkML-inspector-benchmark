"""Matcher integration tests for the LLM-judge backend.

Verifies that when the backend exposes `judge_bulk`, the matcher uses it
instead of the per-pair `score()` loop, and that greedy assignment still
produces sensible matches from the LLM's ranked output.
"""

from __future__ import annotations

import re

from grader.llm import MockLLMProvider
from grader.loader import AgentFinding, GroundTruthFinding
from grader.matcher import match_findings
from grader.similarity import LLMJudgeSimilarity


def _gt(issue_id: str, name: str, explanation: str = "expl") -> GroundTruthFinding:
    return GroundTruthFinding(
        entry_id="test",
        issue_id=issue_id,
        issue_name=name,
        issue_explanation=explanation,
        severity="Critical",
        category="Other",
        security_concern="Other",
        relevant_code=[],
        paper_reference="-",
    )


def _agent(name: str, explanation: str = "expl") -> AgentFinding:
    return AgentFinding(
        entry_id="test",
        issue_name=name,
        issue_explanation=explanation,
        severity="Critical",
        category="Other",
        security_concern="Other",
        relevant_code=[],
        paper_reference="-",
    )


def _responder_from_map(score_map: dict[tuple[str, str], float]):
    """Build a responder that looks up (agent_name_first_word, candidate_id) in score_map.

    The responder parses the user prompt to extract the agent finding text and
    the candidate list, then scores each candidate using the provided map.
    Key: (agent's first word, candidate id).
    """
    def _responder(system, user, schema):
        # AGENT FINDING is the line(s) following "AGENT FINDING:" up to blank
        lines = user.split("\n")
        agent_idx = lines.index("AGENT FINDING:") + 1
        agent_text = lines[agent_idx]
        agent_key = agent_text.split()[0] if agent_text.split() else ""
        candidate_ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
        return {
            "judgments": [
                {
                    "gt_id": cid,
                    "match_score": score_map.get((agent_key, cid), 0.0),
                    "same_root_cause": score_map.get((agent_key, cid), 0.0) >= 0.7,
                    "reasoning": f"{agent_key} vs {cid}",
                }
                for cid in candidate_ids
            ]
        }

    return _responder


class TestMatcherWithLLMJudge:
    def test_basic_match(self):
        provider = MockLLMProvider(_responder_from_map({("agent", "0"): 0.9}))
        judge = LLMJudgeSimilarity(provider)
        gt = [_gt("T-01", "first")]
        agent = [_agent("agent issue")]
        result = match_findings(agent, gt, judge, threshold=0.3)
        assert len(result.matched) == 1
        assert result.matched[0].similarity == 0.9

    def test_judge_bulk_is_used_over_score(self):
        """The matcher should make one bulk call per agent finding."""
        provider = MockLLMProvider(
            _responder_from_map({("agent", "0"): 0.8, ("agent", "1"): 0.1})
        )
        judge = LLMJudgeSimilarity(provider)
        gt = [_gt("T-01", "first"), _gt("T-02", "second")]
        agent = [_agent("agent issue")]
        match_findings(agent, gt, judge, threshold=0.3)
        # One bulk call for the one agent finding
        assert len(provider.calls) == 1

    def test_one_call_per_agent_finding(self):
        provider = MockLLMProvider(
            _responder_from_map({
                ("alpha", "0"): 0.9, ("alpha", "1"): 0.1,
                ("beta", "0"): 0.2, ("beta", "1"): 0.85,
            })
        )
        judge = LLMJudgeSimilarity(provider)
        gt = [_gt("T-01", "first"), _gt("T-02", "second")]
        agent = [_agent("alpha issue"), _agent("beta issue")]
        result = match_findings(agent, gt, judge, threshold=0.3)
        assert len(provider.calls) == 2
        assert len(result.matched) == 2
        # alpha should pair with T-01 (0.9), beta with T-02 (0.85)
        pair_by_agent = {m.agent.issue_name: m.gt.issue_id for m in result.matched}
        assert pair_by_agent["alpha issue"] == "T-01"
        assert pair_by_agent["beta issue"] == "T-02"

    def test_threshold_filter_on_match_score(self):
        provider = MockLLMProvider(
            _responder_from_map({("agent", "0"): 0.25, ("agent", "1"): 0.6})
        )
        judge = LLMJudgeSimilarity(provider)
        gt = [_gt("T-01", "first"), _gt("T-02", "second")]
        agent = [_agent("agent")]
        result = match_findings(agent, gt, judge, threshold=0.5)
        # Only T-02 (score 0.6) clears the 0.5 threshold
        assert len(result.matched) == 1
        assert result.matched[0].gt.issue_id == "T-02"
        assert len(result.missed_gt) == 1

    def test_greedy_assignment_with_conflicting_preferences(self):
        """Both agent findings prefer T-01; greedy gives best-scoring pair first."""
        provider = MockLLMProvider(
            _responder_from_map({
                ("alpha", "0"): 0.95, ("alpha", "1"): 0.5,
                ("beta", "0"): 0.8,  ("beta", "1"): 0.4,
            })
        )
        judge = LLMJudgeSimilarity(provider)
        gt = [_gt("T-01", "first"), _gt("T-02", "second")]
        agent = [_agent("alpha"), _agent("beta")]
        result = match_findings(agent, gt, judge, threshold=0.3)
        # alpha gets T-01 (highest overall), beta falls through to T-02
        pair_by_agent = {m.agent.issue_name: m.gt.issue_id for m in result.matched}
        assert pair_by_agent["alpha"] == "T-01"
        assert pair_by_agent["beta"] == "T-02"

    def test_missing_and_extra_reported(self):
        provider = MockLLMProvider(
            _responder_from_map({
                ("agent1", "0"): 0.9, ("agent1", "1"): 0.1,
                ("agent2", "0"): 0.1, ("agent2", "1"): 0.1,  # no match
            })
        )
        judge = LLMJudgeSimilarity(provider)
        gt = [_gt("T-01", "first"), _gt("T-02", "second")]
        agent = [_agent("agent1"), _agent("agent2")]
        result = match_findings(agent, gt, judge, threshold=0.3)
        assert len(result.matched) == 1
        assert len(result.missed_gt) == 1
        assert result.missed_gt[0].issue_id == "T-02"
        assert len(result.extra_agent) == 1
        assert result.extra_agent[0].issue_name == "agent2"

    def test_empty_agent_list_makes_no_llm_calls(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        gt = [_gt("T-01", "first")]
        result = match_findings([], gt, judge, threshold=0.3)
        assert len(provider.calls) == 0
        assert len(result.missed_gt) == 1

    def test_empty_gt_makes_no_llm_calls(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        agent = [_agent("agent")]
        result = match_findings(agent, [], judge, threshold=0.3)
        assert len(provider.calls) == 0
        assert len(result.extra_agent) == 1


class TestMatcherFallbackPath:
    """Backends without judge_bulk must still work via the per-pair score() loop."""

    def test_jaccard_backend_uses_per_pair_path(self):
        """Using JaccardSimilarity (no judge_bulk) exercises the fallback branch."""
        from grader.similarity import JaccardSimilarity

        jaccard = JaccardSimilarity()
        gt = [_gt("T-01", "widget gadget check", "widget gadget output not validated")]
        agent = [_agent("widget gadget check", "widget gadget output not validated")]
        result = match_findings(agent, gt, jaccard, threshold=0.3)
        assert len(result.matched) == 1
        assert result.matched[0].similarity == 1.0

    def test_backend_without_judge_bulk_attribute(self):
        """A minimal SimilarityBackend that only has score() works via fallback."""
        from grader.similarity import SimilarityBackend

        class PlainBackend(SimilarityBackend):
            def score(self, a: str, b: str) -> float:
                return 1.0 if a == b else 0.0

        gt = [_gt("T-01", "exact match test")]
        agent = [_agent("exact match test")]
        result = match_findings(agent, gt, PlainBackend(), threshold=0.3)
        assert len(result.matched) == 1
