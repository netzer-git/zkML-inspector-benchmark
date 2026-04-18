"""Tests for LLMJudgeSimilarity.

All tests use MockLLMProvider — no real API is contacted. These cover prompt
construction, structured-output parsing (including edge cases like missing
candidates and out-of-range scores), caching, and the degenerate score()
compat path.
"""

from __future__ import annotations

import pytest

from grader.llm import LLMResponseError, MockLLMProvider
from grader.similarity import (
    JUDGE_SCHEMA,
    JudgeCandidate,
    JudgeResult,
    LLMJudgeSimilarity,
    _build_user_prompt,
    _parse_judge_response,
)


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:
    def test_agent_finding_first(self):
        prompt = _build_user_prompt(
            "test agent finding",
            [JudgeCandidate("0", "first candidate")],
        )
        assert prompt.startswith("AGENT FINDING:\ntest agent finding")

    def test_candidates_numbered_with_id(self):
        prompt = _build_user_prompt(
            "agent",
            [
                JudgeCandidate("x-01", "first"),
                JudgeCandidate("x-02", "second"),
            ],
        )
        assert "[x-01] first" in prompt
        assert "[x-02] second" in prompt

    def test_includes_return_schema_hint(self):
        prompt = _build_user_prompt("agent", [JudgeCandidate("0", "gt")])
        assert "Return JSON" in prompt
        assert "judgments" in prompt


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_parses_valid_response(self):
        candidates = [JudgeCandidate("a", "x"), JudgeCandidate("b", "y")]
        raw = {
            "judgments": [
                {"gt_id": "a", "match_score": 0.9, "same_root_cause": True, "reasoning": "match"},
                {"gt_id": "b", "match_score": 0.1, "same_root_cause": False, "reasoning": "no"},
            ]
        }
        results = _parse_judge_response(raw, candidates)
        assert len(results) == 2
        assert results[0].gt_id == "a"
        assert results[0].match_score == 0.9
        assert results[0].same_root_cause is True
        assert results[1].match_score == 0.1

    def test_preserves_candidate_order(self):
        candidates = [JudgeCandidate("z", ""), JudgeCandidate("a", ""), JudgeCandidate("m", "")]
        raw = {
            "judgments": [
                {"gt_id": "a", "match_score": 0.5, "same_root_cause": False, "reasoning": ""},
                {"gt_id": "m", "match_score": 0.7, "same_root_cause": True, "reasoning": ""},
                {"gt_id": "z", "match_score": 0.2, "same_root_cause": False, "reasoning": ""},
            ]
        }
        results = _parse_judge_response(raw, candidates)
        assert [r.gt_id for r in results] == ["z", "a", "m"]

    def test_missing_candidate_filled_with_zero(self):
        candidates = [JudgeCandidate("a", ""), JudgeCandidate("b", "")]
        raw = {
            "judgments": [
                {"gt_id": "a", "match_score": 0.9, "same_root_cause": True, "reasoning": ""},
            ]
        }
        results = _parse_judge_response(raw, candidates)
        assert results[1].gt_id == "b"
        assert results[1].match_score == 0.0
        assert results[1].same_root_cause is False
        assert "no judgment" in results[1].reasoning

    def test_score_clamped_to_range(self):
        candidates = [JudgeCandidate("a", ""), JudgeCandidate("b", "")]
        raw = {
            "judgments": [
                {"gt_id": "a", "match_score": 1.5, "same_root_cause": True, "reasoning": ""},
                {"gt_id": "b", "match_score": -0.3, "same_root_cause": False, "reasoning": ""},
            ]
        }
        results = _parse_judge_response(raw, candidates)
        assert results[0].match_score == 1.0
        assert results[1].match_score == 0.0

    def test_missing_judgments_key_raises(self):
        with pytest.raises(LLMResponseError, match="judgments"):
            _parse_judge_response({"other": "data"}, [JudgeCandidate("a", "")])

    def test_non_list_judgments_raises(self):
        with pytest.raises(LLMResponseError, match="must be a list"):
            _parse_judge_response({"judgments": "oops"}, [JudgeCandidate("a", "")])

    def test_non_numeric_score_defaults_zero(self):
        candidates = [JudgeCandidate("a", "")]
        raw = {
            "judgments": [
                {"gt_id": "a", "match_score": "high", "same_root_cause": True, "reasoning": ""},
            ]
        }
        results = _parse_judge_response(raw, candidates)
        assert results[0].match_score == 0.0

    def test_reasoning_truncated_to_400(self):
        candidates = [JudgeCandidate("a", "")]
        long = "x" * 500
        raw = {
            "judgments": [
                {"gt_id": "a", "match_score": 0.5, "same_root_cause": False, "reasoning": long},
            ]
        }
        results = _parse_judge_response(raw, candidates)
        assert len(results[0].reasoning) == 400


# ---------------------------------------------------------------------------
# LLMJudgeSimilarity
# ---------------------------------------------------------------------------

def _canned_response(score_map: dict[str, float]):
    """Build a callable responder that echoes candidates with scores from the map."""
    import re

    def _responder(system, user, schema):
        ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
        return {
            "judgments": [
                {
                    "gt_id": gid,
                    "match_score": score_map.get(gid, 0.0),
                    "same_root_cause": score_map.get(gid, 0.0) >= 0.7,
                    "reasoning": f"canned score for {gid}",
                }
                for gid in ids
            ]
        }

    return _responder


class TestLLMJudgeSimilarity:
    def test_judge_bulk_basic(self):
        provider = MockLLMProvider(_canned_response({"0": 0.9, "1": 0.2}))
        judge = LLMJudgeSimilarity(provider)
        candidates = [JudgeCandidate("0", "first"), JudgeCandidate("1", "second")]
        results = judge.judge_bulk("agent", candidates)
        assert len(results) == 2
        assert results[0].match_score == 0.9
        assert results[0].same_root_cause is True
        assert results[1].match_score == 0.2

    def test_judge_bulk_empty_candidates_short_circuits(self):
        provider = MockLLMProvider([])
        judge = LLMJudgeSimilarity(provider)
        results = judge.judge_bulk("agent", [])
        assert results == []
        assert provider.calls == []

    def test_judge_bulk_caches_identical_queries(self):
        provider = MockLLMProvider(_canned_response({"0": 0.9}))
        judge = LLMJudgeSimilarity(provider)
        candidates = [JudgeCandidate("0", "same text")]
        judge.judge_bulk("agent text", candidates)
        judge.judge_bulk("agent text", candidates)
        assert len(provider.calls) == 1  # second call hit the cache

    def test_judge_bulk_different_inputs_do_not_collide(self):
        provider = MockLLMProvider(_canned_response({"0": 0.5}))
        judge = LLMJudgeSimilarity(provider)
        c1 = [JudgeCandidate("0", "text one")]
        c2 = [JudgeCandidate("0", "text two")]
        judge.judge_bulk("agent", c1)
        judge.judge_bulk("agent", c2)
        assert len(provider.calls) == 2

    def test_score_compat_path_returns_match_score(self):
        provider = MockLLMProvider(_canned_response({"_pair": 0.75}))
        judge = LLMJudgeSimilarity(provider)
        assert judge.score("agent text", "gt text") == 0.75

    def test_score_compat_path_one_call_per_invocation(self):
        provider = MockLLMProvider(_canned_response({"_pair": 0.5}))
        judge = LLMJudgeSimilarity(provider)
        judge.score("a", "b")
        assert len(provider.calls) == 1

    def test_last_result_for_returns_structured_judgment(self):
        provider = MockLLMProvider(_canned_response({"0": 0.9, "1": 0.1}))
        judge = LLMJudgeSimilarity(provider)
        candidates = [JudgeCandidate("0", "x"), JudgeCandidate("1", "y")]
        judge.judge_bulk("agent", candidates)
        r = judge.last_result_for("agent", "0")
        assert r is not None
        assert isinstance(r, JudgeResult)
        assert r.match_score == 0.9
        assert r.reasoning

    def test_last_result_for_unknown_returns_none(self):
        judge = LLMJudgeSimilarity(MockLLMProvider([]))
        assert judge.last_result_for("agent", "unknown") is None

    def test_cache_key_ignores_candidate_order(self):
        provider = MockLLMProvider(_canned_response({"a": 0.1, "b": 0.9}))
        judge = LLMJudgeSimilarity(provider)
        c1 = [JudgeCandidate("a", "x"), JudgeCandidate("b", "y")]
        c2 = [JudgeCandidate("b", "y"), JudgeCandidate("a", "x")]
        judge.judge_bulk("agent", c1)
        judge.judge_bulk("agent", c2)
        # Second call should hit the cache because candidate set is the same
        assert len(provider.calls) == 1

    def test_malformed_response_raises(self):
        provider = MockLLMProvider([{"not_judgments": []}])
        judge = LLMJudgeSimilarity(provider)
        with pytest.raises(LLMResponseError):
            judge.judge_bulk("agent", [JudgeCandidate("0", "x")])

    def test_system_prompt_injection(self):
        """Custom system prompt flows through to the provider."""
        provider = MockLLMProvider(_canned_response({"0": 0.5}))
        judge = LLMJudgeSimilarity(provider, system_prompt="custom sys")
        judge.judge_bulk("agent", [JudgeCandidate("0", "x")])
        system_used = provider.calls[0][0]
        assert system_used == "custom sys"

    def test_provider_receives_schema(self):
        seen_schema = {}

        def responder(system, user, schema):
            seen_schema.update(schema)
            return {"judgments": [{"gt_id": "0", "match_score": 0.5,
                                   "same_root_cause": False, "reasoning": "ok"}]}

        provider = MockLLMProvider(responder)
        judge = LLMJudgeSimilarity(provider)
        judge.judge_bulk("agent", [JudgeCandidate("0", "x")])
        # Schema should match JUDGE_SCHEMA top-level structure
        assert seen_schema.get("type") == "object"
        assert "judgments" in seen_schema.get("properties", {})


# ---------------------------------------------------------------------------
# System prompt size budget
# ---------------------------------------------------------------------------

def test_system_prompt_under_token_budget():
    """Rough budget check: system prompt should be under ~250 tokens.

    We approximate tokens with a word count; 1.1-1.3 tokens per word for
    technical English. The plan budget was 250 tokens → allow up to 240 words
    as a conservative proxy.
    """
    from grader.similarity import _DEFAULT_SYSTEM_PROMPT
    word_count = len(_DEFAULT_SYSTEM_PROMPT.split())
    assert word_count < 240, f"system prompt is {word_count} words, over budget"
