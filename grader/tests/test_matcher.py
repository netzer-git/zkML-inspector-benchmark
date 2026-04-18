"""Tests for grader.matcher.

The matcher uses the LLM judge exclusively. Every test here drives the
matcher through an LLMJudgeSimilarity backed by a MockLLMProvider — no
real API is contacted.
"""

from __future__ import annotations

import re

import pytest

from grader.llm import MockLLMProvider
from grader.loader import AgentFinding, CodeRef, GroundTruthFinding
from grader.matcher import MatchResult, MatchedPair, _build_matching_text, match_findings
from grader.similarity import LLMJudgeSimilarity


def _gt(
    issue_id: str,
    name: str,
    explanation: str = "default explanation",
    severity: str = "Critical",
    paper: str = "-",
    code: list[CodeRef] | None = None,
) -> GroundTruthFinding:
    return GroundTruthFinding(
        entry_id="test",
        issue_id=issue_id,
        issue_name=name,
        issue_explanation=explanation,
        severity=severity,
        category="Other",
        security_concern="Other",
        relevant_code=code or [],
        paper_reference=paper,
    )


def _agent(
    name: str,
    explanation: str = "default explanation",
    severity: str = "Critical",
    paper: str = "-",
    code: list[CodeRef] | None = None,
) -> AgentFinding:
    return AgentFinding(
        entry_id="test",
        issue_name=name,
        issue_explanation=explanation,
        severity=severity,
        category="Other",
        security_concern="Other",
        relevant_code=code or [],
        paper_reference=paper,
    )


def _responder_with_scores(score_map: dict[str, int]):
    """Build a responder that scores each candidate id using score_map.

    Scores are integers on the 1..5 scale; unmapped candidates default to 1.
    """
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
# _build_matching_text
# ---------------------------------------------------------------------------

class TestBuildMatchingText:
    def test_includes_all_three_fields(self):
        text = _build_matching_text("n", "e", "Section 3.1: widget range check")
        assert "n" in text
        assert "e" in text
        assert "Section 3.1" in text

    def test_empty_paper_renders_as_none(self):
        text = _build_matching_text("n", "e", "")
        assert "Paper reference: (none)" in text

    def test_dash_paper_renders_as_none(self):
        text = _build_matching_text("n", "e", "-")
        assert "Paper reference: (none)" in text

    def test_whitespace_collapsed(self):
        text = _build_matching_text("n   a    m  e", "e\n\nx", "  s 1  ")
        assert "n a m e" in text
        assert "e x" in text
        assert "s 1" in text

    def test_does_not_include_code_or_severity(self):
        """Sanity: the builder doesn't take code refs or closed-list fields."""
        text = _build_matching_text("widget bug", "expl", "-")
        assert "widget bug" in text
        # These should not appear unless we pass them in
        assert "Critical" not in text
        assert ".rs" not in text


# ---------------------------------------------------------------------------
# match_findings — empty inputs / edge cases
# ---------------------------------------------------------------------------

class TestMatchFindingsEmpty:
    def test_empty_both(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        result = match_findings([], [], judge)
        assert result.matched == []
        assert result.missed_gt == []
        assert result.extra_agent == []
        assert provider.calls == []

    def test_empty_agent(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        result = match_findings([], [_gt("T-01", "issue")], judge)
        assert len(result.missed_gt) == 1
        assert result.matched == []
        assert provider.calls == []

    def test_empty_gt(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        result = match_findings([_agent("issue")], [], judge)
        assert len(result.extra_agent) == 1
        assert result.matched == []
        assert provider.calls == []


# ---------------------------------------------------------------------------
# match_findings — LLM-driven matching behavior
# ---------------------------------------------------------------------------

class TestMatchFindingsLLM:
    def test_single_perfect_match(self):
        provider = MockLLMProvider(_responder_with_scores({"T-01": 5}))
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("agent")], [_gt("T-01", "gt")], judge, threshold=4
        )
        assert len(result.matched) == 1
        assert result.matched[0].similarity == 5
        assert len(provider.calls) == 1

    def test_one_call_per_agent_finding(self):
        provider = MockLLMProvider(_responder_with_scores(
            {"T-01": 5, "T-02": 1}
        ))
        judge = LLMJudgeSimilarity(provider)
        match_findings(
            [_agent("a"), _agent("b"), _agent("c")],
            [_gt("T-01", "gt1"), _gt("T-02", "gt2")],
            judge,
            threshold=4,
        )
        assert len(provider.calls) == 3  # one per agent finding

    def test_greedy_both_agents_bind_to_best_gt_even_if_same(self):
        """Under N:1 matching, both agents prefer T-01 -> both bind to it.

        The responder gives T-01 score 5 for every agent and T-02 score
        4. Agent 'alpha' processes first and takes T-01 as primary (dup_rank
        0); agent 'beta' also picks T-01 as its best match, landing there as
        a duplicate (dup_rank 1). T-02 is unmatched.
        """
        provider = MockLLMProvider(_responder_with_scores(
            {"T-01": 5, "T-02": 4}
        ))
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("alpha"), _agent("beta")],
            [_gt("T-01", "gt1"), _gt("T-02", "gt2")],
            judge,
            threshold=4,
        )
        assert len(result.matched) == 2
        # Both agents bind to T-01 (their highest-score GT).
        gt_ids = [m.gt.issue_id for m in result.matched]
        assert gt_ids.count("T-01") == 2
        # One is primary (dup_rank=0), one is duplicate (dup_rank=1).
        dup_ranks = sorted(m.dup_rank for m in result.matched)
        assert dup_ranks == [0, 1]
        # T-02 was never matched -> missed.
        assert len(result.missed_gt) == 1
        assert result.missed_gt[0].issue_id == "T-02"

    def test_missed_and_extra_reported(self):
        def responder(system, user, schema):
            ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
            # First agent matches T-01; second agent matches nothing
            agent_line = user.split("AGENT FINDING:")[1].split("\n")[1]
            if agent_line.startswith("agent1"):
                scores = {cid: (5 if cid == "T-01" else 1) for cid in ids}
            else:
                scores = {cid: 1 for cid in ids}
            return {
                "judgments": [
                    {
                        "gt_id": cid,
                        "match_score": scores[cid],
                        "reasoning": "",
                    }
                    for cid in ids
                ]
            }

        provider = MockLLMProvider(responder)
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("agent1"), _agent("agent2")],
            [_gt("T-01", "first"), _gt("T-02", "second")],
            judge,
            threshold=4,
        )
        assert len(result.matched) == 1
        assert result.matched[0].gt.issue_id == "T-01"
        assert len(result.missed_gt) == 1
        assert result.missed_gt[0].issue_id == "T-02"
        assert len(result.extra_agent) == 1

    def test_backend_without_judge_bulk_raises(self):
        from grader.similarity import SimilarityBackend

        class StubBackend(SimilarityBackend):
            def score(self, a: str, b: str) -> float:
                return 0.0

        with pytest.raises(AttributeError, match="judge_bulk"):
            match_findings(
                [_agent("x")], [_gt("T-01", "y")], StubBackend(),
            )


# ---------------------------------------------------------------------------
# Judge text content — verify what the judge sees
# ---------------------------------------------------------------------------

class TestJudgeTextContent:
    def _capture_user_prompt(self, gt_list, agent_list, judge_responses=None):
        """Run the matcher; return the first user prompt sent to the judge."""
        captured: list[str] = []

        def responder(system, user, schema):
            captured.append(user)
            if judge_responses is not None:
                return judge_responses
            # Default: return safe no-op judgments for all candidates
            ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
            return {
                "judgments": [
                    {"gt_id": cid, "match_score": 1, "reasoning": ""}
                    for cid in ids
                ]
            }

        provider = MockLLMProvider(responder)
        judge = LLMJudgeSimilarity(provider)
        match_findings(agent_list, gt_list, judge, threshold=4)
        return captured[0] if captured else ""

    def test_paper_reference_included(self):
        gt = [_gt("T-01", "n", "e", paper='Section 3.1: "the gadget must be range-checked"')]
        prompt = self._capture_user_prompt(gt, [_agent("ag")])
        assert "Section 3.1" in prompt
        assert "range-checked" in prompt

    def test_code_refs_not_shown_to_judge(self):
        """Code refs must NOT appear in the judge prompt."""
        gt = [
            _gt("T-01", "n", "e", paper="-",
                code=[CodeRef("src/secret_file.rs", 42, 42)]),
        ]
        agent = [_agent("ag", code=[CodeRef("src/agent_file.rs", 10, 10)])]
        prompt = self._capture_user_prompt(gt, agent)
        assert "secret_file" not in prompt
        assert "agent_file" not in prompt
        assert ":42" not in prompt

    def test_severity_and_category_not_shown(self):
        """Closed-list fields must NOT appear in the judge prompt."""
        gt = [_gt("T-01", "n", "e", severity="Critical")]
        agent = [_agent("ag", severity="Warning")]
        prompt = self._capture_user_prompt(gt, agent)
        # Severity-value words shouldn't appear unless the name/explanation
        # happens to include them — our synthetic "n", "e" don't.
        assert "Critical" not in prompt
        assert "Warning" not in prompt
        # Category "Other" is the default on both helpers; make sure it's absent
        assert "Other" not in prompt

    def test_empty_paper_renders_as_none(self):
        gt = [_gt("T-01", "n", "e", paper="-")]
        prompt = self._capture_user_prompt(gt, [_agent("ag", paper="")])
        assert "Paper reference: (none)" in prompt


# ---------------------------------------------------------------------------
# Score-threshold gate behavior (1..5 ordinal)
# ---------------------------------------------------------------------------

class TestScoreThreshold:
    def test_score_below_threshold_does_not_match(self):
        provider = MockLLMProvider(lambda s, u, sc: {
            "judgments": [
                {"gt_id": "T-01", "match_score": 3,
                 "reasoning": "related sub-issue but not the same finding"}
            ]
        })
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("agent")], [_gt("T-01", "gt")], judge, threshold=4,
        )
        assert result.matched == []
        assert len(result.missed_gt) == 1
        assert len(result.extra_agent) == 1

    def test_score_at_threshold_matches(self):
        provider = MockLLMProvider(lambda s, u, sc: {
            "judgments": [
                {"gt_id": "T-01", "match_score": 4,
                 "reasoning": "very likely the same finding"}
            ]
        })
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("agent")], [_gt("T-01", "gt")], judge, threshold=4,
        )
        assert len(result.matched) == 1
        assert result.matched[0].similarity == 4

    def test_score_above_threshold_matches(self):
        provider = MockLLMProvider(lambda s, u, sc: {
            "judgments": [
                {"gt_id": "T-01", "match_score": 5,
                 "reasoning": "same root cause"}
            ]
        })
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("agent")], [_gt("T-01", "gt")], judge, threshold=4,
        )
        assert len(result.matched) == 1
        assert result.matched[0].similarity == 5

    def test_custom_higher_threshold(self):
        """threshold=5 only accepts the top of the scale."""
        provider = MockLLMProvider(lambda s, u, sc: {
            "judgments": [
                {"gt_id": "T-01", "match_score": 4, "reasoning": ""}
            ]
        })
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("agent")], [_gt("T-01", "gt")], judge, threshold=5,
        )
        assert result.matched == []


# ---------------------------------------------------------------------------
# Candidate IDs are real issue_ids
# ---------------------------------------------------------------------------

class TestCandidateIds:
    def test_real_issue_ids_appear_in_prompt(self):
        captured: list[str] = []

        def responder(system, user, schema):
            captured.append(user)
            return {"judgments": [
                {"gt_id": "alpha-42", "match_score": 1, "reasoning": ""},
                {"gt_id": "alpha-43", "match_score": 1, "reasoning": ""},
            ]}

        provider = MockLLMProvider(responder)
        judge = LLMJudgeSimilarity(provider)
        match_findings(
            [_agent("ag")],
            [_gt("alpha-42", "first"), _gt("alpha-43", "second")],
            judge,
            threshold=4,
        )
        assert "[alpha-42]" in captured[0]
        assert "[alpha-43]" in captured[0]

    def test_unknown_gt_id_in_response_silently_dropped(self):
        """Defensive: if the LLM returns an id we didn't send, ignore it."""
        provider = MockLLMProvider(lambda s, u, sc: {
            "judgments": [
                {"gt_id": "alpha-01", "match_score": 5, "reasoning": ""},
                {"gt_id": "fictional-id", "match_score": 5, "reasoning": ""},
            ]
        })
        judge = LLMJudgeSimilarity(provider)
        result = match_findings(
            [_agent("ag")], [_gt("alpha-01", "gt")], judge, threshold=4,
        )
        assert len(result.matched) == 1
        assert result.matched[0].gt.issue_id == "alpha-01"


# ---------------------------------------------------------------------------
# MatchResult helpers
# ---------------------------------------------------------------------------

class TestMatchResultExtras:
    def test_extra_by_severity_empty(self):
        result = MatchResult()
        assert result.extra_by_severity == {}

    def test_extra_by_severity_grouped(self):
        result = MatchResult(
            extra_agent=[
                _agent("A", severity="Critical"),
                _agent("B", severity="Warning"),
                _agent("C", severity="Critical"),
                _agent("D", severity="Info"),
            ]
        )
        by_sev = result.extra_by_severity
        assert len(by_sev["Critical"]) == 2
        assert len(by_sev["Warning"]) == 1
        assert len(by_sev["Info"]) == 1

    def test_extra_by_severity_single_type(self):
        result = MatchResult(
            extra_agent=[
                _agent("A", severity="Warning"),
                _agent("B", severity="Warning"),
            ]
        )
        by_sev = result.extra_by_severity
        assert list(by_sev.keys()) == ["Warning"]
        assert len(by_sev["Warning"]) == 2


# ---------------------------------------------------------------------------
# Traces — debug data captured per agent finding for --judge-trace
# ---------------------------------------------------------------------------

class TestTraces:
    def test_trace_one_entry_per_agent_finding_in_order(self):
        provider = MockLLMProvider(_responder_with_scores(
            {"T-01": 5, "T-02": 1}
        ))
        judge = LLMJudgeSimilarity(provider)
        agent = [_agent("a1"), _agent("a2"), _agent("a3")]
        gt = [_gt("T-01", "g1"), _gt("T-02", "g2")]
        result = match_findings(agent, gt, judge, threshold=4)
        assert len(result.traces) == 3
        for i, tr in enumerate(result.traces):
            assert tr.agent_index == i
            assert tr.agent is agent[i]

    def test_trace_candidates_cover_all_gt(self):
        provider = MockLLMProvider(_responder_with_scores(
            {"T-01": 5, "T-02": 3, "T-03": 1}
        ))
        judge = LLMJudgeSimilarity(provider)
        agent = [_agent("a1")]
        gt = [_gt("T-01", "g1"), _gt("T-02", "g2"), _gt("T-03", "g3")]
        result = match_findings(agent, gt, judge, threshold=4)
        assert len(result.traces) == 1
        cand_ids = {r.gt_id for r in result.traces[0].candidates}
        assert cand_ids == {"T-01", "T-02", "T-03"}

    def test_trace_matched_gt_id_set_when_pair_matched(self):
        provider = MockLLMProvider(_responder_with_scores(
            {"T-01": 5}
        ))
        judge = LLMJudgeSimilarity(provider)
        agent = [_agent("a1")]
        gt = [_gt("T-01", "g1")]
        result = match_findings(agent, gt, judge, threshold=4)
        assert result.traces[0].matched_gt_id == "T-01"

    def test_trace_matched_gt_id_none_when_below_threshold(self):
        provider = MockLLMProvider(_responder_with_scores(
            {"T-01": 2}
        ))
        judge = LLMJudgeSimilarity(provider)
        agent = [_agent("a1")]
        gt = [_gt("T-01", "g1")]
        result = match_findings(agent, gt, judge, threshold=4)
        assert result.traces[0].matched_gt_id is None

    def test_empty_agent_produces_no_traces(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        result = match_findings([], [_gt("T-01", "g1")], judge, threshold=4)
        assert result.traces == []

    def test_empty_gt_produces_no_traces(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        result = match_findings([_agent("a1")], [], judge, threshold=4)
        assert result.traces == []

    def test_agent_text_captured_in_trace(self):
        """Trace stores the exact text shown to the judge (includes paper ref)."""
        provider = MockLLMProvider(_responder_with_scores({"T-01": 5}))
        judge = LLMJudgeSimilarity(provider)
        agent = [_agent("named thing", explanation="body",
                        paper="Section 3: quoted claim")]
        gt = [_gt("T-01", "g1")]
        result = match_findings(agent, gt, judge, threshold=4)
        text = result.traces[0].agent_text
        assert "named thing" in text
        assert "body" in text
        assert "Section 3" in text
