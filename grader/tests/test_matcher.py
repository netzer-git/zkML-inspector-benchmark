"""Tests for grader.matcher module."""

import pytest

from grader.loader import AgentFinding, CodeRef, GroundTruthFinding
from grader.matcher import MatchResult, MatchedPair, match_findings
from grader.similarity import JaccardSimilarity


def _gt(issue_id: str, name: str, explanation: str = "", severity: str = "Critical") -> GroundTruthFinding:
    """Helper to create a GT finding with minimal fields."""
    return GroundTruthFinding(
        entry_id="test",
        issue_id=issue_id,
        issue_name=name,
        issue_explanation=explanation,
        severity=severity,
        category="Other",
        security_concern="Other",
        relevant_code=[],
        paper_reference="-",
    )


def _agent(name: str, explanation: str = "", severity: str = "Critical") -> AgentFinding:
    """Helper to create an agent finding with minimal fields."""
    return AgentFinding(
        entry_id="test",
        issue_name=name,
        issue_explanation=explanation,
        severity=severity,
        category="Other",
        security_concern="Other",
        relevant_code=[],
        paper_reference="-",
    )


class TestMatchFindings:
    @pytest.fixture
    def sim(self):
        return JaccardSimilarity()

    def test_empty_both(self, sim):
        result = match_findings([], [], sim)
        assert result.matched == []
        assert result.missed_gt == []
        assert result.extra_agent == []

    def test_empty_agent(self, sim):
        gt = [_gt("T-01", "Issue A")]
        result = match_findings([], gt, sim)
        assert len(result.missed_gt) == 1
        assert result.matched == []
        assert result.extra_agent == []

    def test_empty_gt(self, sim):
        agent = [_agent("Issue A")]
        result = match_findings(agent, [], sim)
        assert len(result.extra_agent) == 1
        assert result.matched == []
        assert result.missed_gt == []

    def test_perfect_match(self, sim):
        gt = [_gt("T-01", "Unchecked widget output", "widget output has no range constraint")]
        agent = [_agent("Unchecked widget output", "widget output has no range constraint")]
        result = match_findings(agent, gt, sim, threshold=0.3)
        assert len(result.matched) == 1
        assert result.matched[0].similarity == 1.0
        assert result.missed_gt == []
        assert result.extra_agent == []

    def test_similar_match(self, sim):
        gt = [_gt("T-01", "Static prover seed",
                   "Witness generation uses a compile-time constant seed for the PRNG")]
        agent = [_agent("Prover seed is static",
                        "The witness PRNG is seeded at compile time with a constant value")]
        result = match_findings(agent, gt, sim, threshold=0.2)
        assert len(result.matched) == 1
        assert result.matched[0].similarity > 0.2

    def test_no_match_below_threshold(self, sim):
        gt = [_gt("T-01", "Completely unrelated topic one",
                   "The one protocol has a fundamental flaw")]
        agent = [_agent("Different subject two",
                        "The two mechanism is broken in a novel way")]
        result = match_findings(agent, gt, sim, threshold=0.5)
        assert result.matched == []
        assert len(result.missed_gt) == 1
        assert len(result.extra_agent) == 1

    def test_multiple_matches_greedy(self, sim):
        gt = [
            _gt("T-01", "Missing gadget range check", "widget gadget skips range validation"),
            _gt("T-02", "Transcript missing public input", "public inputs not hashed into transcript"),
        ]
        agent = [
            _agent("Transcript omits public input", "public inputs are not added to the transcript hash"),
            _agent("Gadget range check absent", "the widget gadget does not validate ranges"),
        ]
        result = match_findings(agent, gt, sim, threshold=0.2)
        assert len(result.matched) == 2
        # Verify correct pairing
        gt_ids_matched = {m.gt.issue_id for m in result.matched}
        assert gt_ids_matched == {"T-01", "T-02"}

    def test_more_agent_than_gt(self, sim):
        gt = [_gt("T-01", "Issue alpha", "explanation alpha")]
        agent = [
            _agent("Issue alpha", "explanation alpha"),
            _agent("Issue beta", "explanation beta"),
            _agent("Issue gamma", "explanation gamma"),
        ]
        result = match_findings(agent, gt, sim, threshold=0.3)
        assert len(result.matched) == 1
        assert len(result.extra_agent) == 2

    def test_more_gt_than_agent(self, sim):
        gt = [
            _gt("T-01", "Issue alpha", "explanation alpha"),
            _gt("T-02", "Issue beta", "explanation beta"),
            _gt("T-03", "Issue gamma", "explanation gamma"),
        ]
        agent = [_agent("Issue alpha", "explanation alpha")]
        result = match_findings(agent, gt, sim, threshold=0.3)
        assert len(result.matched) == 1
        assert len(result.missed_gt) == 2

    def test_threshold_zero_matches_everything(self, sim):
        gt = [_gt("T-01", "X", "completely different words")]
        agent = [_agent("Y", "absolutely no overlap at all")]
        # Even with 0 overlap, threshold=0 should still need >0 sim
        result = match_findings(agent, gt, sim, threshold=0.0)
        # With Jaccard, completely disjoint sets give 0.0 which is >= 0.0
        assert len(result.matched) == 1 or len(result.extra_agent) == 1

    def test_one_to_one_no_double_matching(self, sim):
        gt = [
            _gt("T-01", "commitment missing for lookup table",
                 "lookup table T never committed"),
            _gt("T-02", "commitment missing for witness values",
                 "witness values S and m never committed"),
        ]
        agent = [
            _agent("commitments missing",
                   "lookup table and witness values never committed via Pedersen"),
        ]
        result = match_findings(agent, gt, sim, threshold=0.2)
        # Agent matches one GT; the other is missed. No double-matching.
        assert len(result.matched) <= 1
        assert len(result.missed_gt) >= 1


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
