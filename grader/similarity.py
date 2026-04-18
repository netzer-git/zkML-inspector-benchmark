"""Similarity interface and LLM-as-judge backend used by the grader.

Production matching uses LLMJudgeSimilarity. The SimilarityBackend ABC
is kept minimal so a single-pair similarity score can be computed for
paper-reference quote matching without forcing it through the LLM when
a cheaper path makes sense in tests.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from grader.llm import LLMProvider, LLMResponseError


class SimilarityBackend(ABC):
    """Abstract base for single-pair text similarity scoring.

    Production matching uses LLMJudgeSimilarity, which additionally exposes
    `judge_bulk()` for bulk ranking. Backends that only need single-pair
    comparison (e.g., for paper-reference quote scoring in tests) implement
    just `score()`.
    """

    @abstractmethod
    def score(self, text_a: str, text_b: str) -> float:
        """Return similarity in [0, 1]. 0 = unrelated, 1 = identical."""


# ---------------------------------------------------------------------------
# LLM-as-judge backend
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeCandidate:
    """One ground-truth finding presented to the judge."""
    gt_id: str
    text: str       # preformatted "<name>: <explanation>"


@dataclass(frozen=True)
class JudgeResult:
    """Structured judgment for a single (agent, gt) pair."""
    gt_id: str
    match_score: float        # [0, 1]
    same_root_cause: bool
    reasoning: str


# Strict JSON schema used by both OpenAI response_format and Anthropic tool_use.
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "gt_id": {"type": "string"},
                    "match_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "same_root_cause": {"type": "boolean"},
                    "reasoning": {"type": "string", "maxLength": 400},
                },
                "required": ["gt_id", "match_score", "same_root_cause", "reasoning"],
            },
        },
    },
    "required": ["judgments"],
}


_DEFAULT_SYSTEM_PROMPT = """\
You are an expert judge for zkML audit findings. For each candidate \
ground-truth finding, decide how closely the auditor's finding describes the \
same root cause in the same ZK circuit / protocol component.

A true match requires the same root cause in the same component. Shared \
keywords alone do not match. "Unconstrained activation witness" and \
"activation output freely chosen by prover" are the same; "unconstrained \
activation" and "unconstrained layer normalization" are different components.

Key failure modes: missing constraint, unconstrained witness, uncommitted \
weights, out-of-order commit, wire disconnect, missing/weak Fiat-Shamir, \
quantization bugs. A soundness gap lets a malicious prover produce a valid \
proof of an incorrect result.

For every candidate return:
- match_score: float in [0, 1], your confidence this is the same finding. \
Guide: ~0.95 clear match, ~0.7 strong overlap with minor differences, ~0.5 related sub-issue, ~0.25 weak \
keyword overlap, ~0.05 unrelated.
- same_root_cause: true only if the same component AND the same failure \
mode. Can be false even at high match_score for adjacent but distinct issues.
- reasoning: one short sentence, under 40 words.

Return exactly one judgment object per candidate, in the order given."""


def _build_user_prompt(agent_text: str, candidates: Sequence[JudgeCandidate]) -> str:
    lines = ["AGENT FINDING:", agent_text.strip(), "", "CANDIDATE GT FINDINGS:"]
    for c in candidates:
        lines.append(f"[{c.gt_id}] {c.text.strip()}")
    lines.append("")
    lines.append(
        'Return JSON: {"judgments": '
        '[{gt_id, match_score, same_root_cause, reasoning}, ...]}'
    )
    return "\n".join(lines)


def _parse_judge_response(
    raw: dict[str, Any], candidates: Sequence[JudgeCandidate]
) -> list[JudgeResult]:
    try:
        items = raw["judgments"]
    except (KeyError, TypeError) as e:
        raise LLMResponseError(f"Missing 'judgments' key in response: {raw!r}") from e
    if not isinstance(items, list):
        raise LLMResponseError(f"'judgments' must be a list, got {type(items).__name__}")

    by_id: dict[str, JudgeResult] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        gt_id = str(item.get("gt_id", ""))
        try:
            score = float(item.get("match_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        reasoning = str(item.get("reasoning", ""))[:400]
        by_id[gt_id] = JudgeResult(
            gt_id=gt_id,
            match_score=score,
            same_root_cause=bool(item.get("same_root_cause", False)),
            reasoning=reasoning,
        )

    out: list[JudgeResult] = []
    for c in candidates:
        if c.gt_id in by_id:
            out.append(by_id[c.gt_id])
        else:
            out.append(JudgeResult(c.gt_id, 0.0, False, "no judgment returned"))
    return out


class LLMJudgeSimilarity(SimilarityBackend):
    """LLM-as-judge similarity backend.

    Primary entry point is `judge_bulk(agent_text, candidates)` — one LLM
    call ranks an agent finding against all candidate GTs. The matcher
    requires this method. `score(a, b)` is a degenerate single-candidate
    variant kept for paper-reference quote matching callers.
    """

    def __init__(self, provider: LLMProvider, system_prompt: str | None = None):
        self.provider = provider
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        # Cache: (agent_text, sorted candidates) hash -> list of JudgeResult
        self._cache: dict[str, list[JudgeResult]] = {}
        # Flat lookup: (agent_text, gt_id) -> JudgeResult for later enrichment
        self._last_results: dict[tuple[str, str], JudgeResult] = {}

    def score(self, text_a: str, text_b: str) -> float:
        """Compatibility path — single-candidate bulk call."""
        results = self.judge_bulk(
            text_a, [JudgeCandidate(gt_id="_pair", text=text_b)]
        )
        if not results:
            return 0.0
        return results[0].match_score

    def judge_bulk(
        self, agent_text: str, candidates: Sequence[JudgeCandidate]
    ) -> list[JudgeResult]:
        if not candidates:
            return []

        key = self._cache_key(agent_text, candidates)
        if key in self._cache:
            return self._cache[key]

        user_prompt = _build_user_prompt(agent_text, candidates)
        raw = self.provider.chat_json(
            system=self.system_prompt,
            user=user_prompt,
            schema=JUDGE_SCHEMA,
            schema_name="zkml_judge",
        )
        results = _parse_judge_response(raw, candidates)
        self._cache[key] = results
        for r in results:
            self._last_results[(agent_text, r.gt_id)] = r
        return results

    def last_result_for(self, agent_text: str, gt_id: str) -> JudgeResult | None:
        """Retrieve the structured judgment for a pair (for later report enrichment)."""
        return self._last_results.get((agent_text, gt_id))

    @staticmethod
    def _cache_key(agent_text: str, candidates: Sequence[JudgeCandidate]) -> str:
        h = hashlib.sha256()
        h.update(agent_text.encode("utf-8"))
        for c in sorted(candidates, key=lambda x: x.gt_id):
            h.update(b"|")
            h.update(c.gt_id.encode("utf-8"))
            h.update(b":")
            h.update(c.text.encode("utf-8"))
        return h.hexdigest()
