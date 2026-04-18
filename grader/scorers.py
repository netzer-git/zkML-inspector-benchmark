"""Per-field scoring functions for matched finding pairs.

Each scorer returns a FieldScore(score, detail) where score is in [0, 1].
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from grader.loader import CodeRef
from grader.similarity import SimilarityBackend


@dataclass
class FieldScore:
    score: float  # [0, 1]
    detail: str


# ---------------------------------------------------------------------------
# 4a. Severity scoring
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"Critical": 2, "Warning": 1, "Info": 0}

# severity_matrix[agent_rank][gt_rank] -> score
_SEVERITY_MATRIX = {
    # Agent=Critical
    (2, 2): 1.0,   # Critical vs Critical
    (2, 1): 0.3,  # Critical vs Warning (over-report by 1)
    (2, 0): 0.1,   # Critical vs Info (over-report by 2)
    # Agent=Warning
    (1, 2): 0.3,   # Warning vs Critical (under-report)
    (1, 1): 1.0,   # Warning vs Warning
    (1, 0): 0.5,  # Warning vs Info (over-report by 1)
    # Agent=Info
    (0, 2): 0.0,   # Info vs Critical (under-report)
    (0, 1): 0.5,   # Info vs Warning (under-report)
    (0, 0): 1.0,   # Info vs Info
}


def score_severity(agent_sev: str, gt_sev: str) -> FieldScore:
    agent_rank = _SEVERITY_RANK.get(agent_sev)
    gt_rank = _SEVERITY_RANK.get(gt_sev)
    if agent_rank is None or gt_rank is None:
        return FieldScore(0.0, f"invalid severity: agent={agent_sev}, gt={gt_sev}")
    score = _SEVERITY_MATRIX[(agent_rank, gt_rank)]
    detail = f"agent={agent_sev}, gt={gt_sev}"
    if score == 1.0:
        detail = f"exact match ({gt_sev})"
    elif agent_rank > gt_rank:
        detail = f"over-reported: agent={agent_sev}, gt={gt_sev}"
    elif agent_rank < gt_rank:
        detail = f"under-reported: agent={agent_sev}, gt={gt_sev}"
    return FieldScore(score, detail)


# ---------------------------------------------------------------------------
# 4b. Category scoring
# ---------------------------------------------------------------------------

_CATEGORY_PROXIMITY: dict[frozenset[str], float] = {
    frozenset({"Under-constrained Circuit", "Witness/Commitment Mismatch"}): 0.4,
    frozenset({"Protocol/Transcript Logic", "Witness/Commitment Mismatch"}): 0.3,
    frozenset({"Engineering/Prototype Gap", "Specification Mismatch"}): 0.3,
}


def score_category(agent_cat: str, gt_cat: str) -> FieldScore:
    if agent_cat == gt_cat:
        return FieldScore(1.0, f"exact match ({gt_cat})")

    pair = frozenset({agent_cat, gt_cat})
    partial = _CATEGORY_PROXIMITY.get(pair, 0.0)
    if partial > 0:
        return FieldScore(partial, f"partial: agent={agent_cat}, gt={gt_cat}")
    return FieldScore(0.0, f"mismatch: agent={agent_cat}, gt={gt_cat}")


# ---------------------------------------------------------------------------
# 4c. Security concern scoring
# ---------------------------------------------------------------------------

_CONCERN_PROXIMITY: dict[frozenset[str], float] = {
    frozenset({"Proof Forgery (Soundness)", "Semantic Subversion (Integrity)"}): 0.3,
    frozenset({"Proof Forgery (Soundness)", "Proof Malleability"}): 0.3,
    frozenset({"Information Leakage (Privacy)", "Governance Bypass"}): 0.2,
}


def score_security_concern(agent_sc: str, gt_sc: str) -> FieldScore:
    if agent_sc == gt_sc:
        return FieldScore(1.0, f"exact match ({gt_sc})")

    # "Other" matching any specific value
    if agent_sc == "Other" or gt_sc == "Other":
        return FieldScore(0.1, f"Other match: agent={agent_sc}, gt={gt_sc}")

    pair = frozenset({agent_sc, gt_sc})
    partial = _CONCERN_PROXIMITY.get(pair, 0.0)
    if partial > 0:
        return FieldScore(partial, f"partial: agent={agent_sc}, gt={gt_sc}")
    return FieldScore(0.0, f"mismatch: agent={agent_sc}, gt={gt_sc}")


# ---------------------------------------------------------------------------
# 4d. Code location scoring
# ---------------------------------------------------------------------------

def _basename(filepath: str) -> str:
    """Extract the basename from a potentially qualified path."""
    return os.path.basename(filepath.replace("\\", "/"))


def _line_distance(agent_ref: CodeRef, gt_ref: CodeRef) -> int | None:
    """Minimum distance between agent line(s) and GT line range. None if no lines."""
    if agent_ref.start_line is None or gt_ref.start_line is None:
        return None
    agent_start = agent_ref.start_line
    agent_end = agent_ref.end_line or agent_start
    gt_start = gt_ref.start_line
    gt_end = gt_ref.end_line or gt_start

    # Check overlap
    if agent_start <= gt_end and agent_end >= gt_start:
        return 0
    # Distance to nearest edge
    return min(abs(agent_start - gt_end), abs(agent_end - gt_start))


def _score_single_code_ref(agent_refs: list[CodeRef], gt_ref: CodeRef) -> float:
    """Find the best score for a single GT code ref against all agent code refs."""
    best = 0.0
    gt_base = _basename(gt_ref.filename)

    for agent_ref in agent_refs:
        agent_base = _basename(agent_ref.filename)
        if agent_base.lower() != gt_base.lower():
            continue
        # Same filename
        dist = _line_distance(agent_ref, gt_ref)
        if dist is None:
            # Same file but no line numbers to compare
            best = max(best, 0.2)
        elif dist == 0:
            best = max(best, 1.0)
        elif dist <= 30:
            best = max(best, 0.7)
        elif dist <= 100:
            best = max(best, 0.4)
        else:
            best = max(best, 0.2)

    return best


def score_code_location(
    agent_refs: list[CodeRef], gt_refs: list[CodeRef]
) -> FieldScore:
    if not gt_refs:
        return FieldScore(1.0, "no code location expected (skip)")

    if not agent_refs:
        return FieldScore(0.0, "agent provided no code references")

    scores = [_score_single_code_ref(agent_refs, gt_ref) for gt_ref in gt_refs]
    avg = sum(scores) / len(scores)
    matched = sum(1 for s in scores if s > 0)
    detail = f"{matched}/{len(gt_refs)} GT refs matched, avg={avg:.2f}"
    return FieldScore(avg, detail)


# ---------------------------------------------------------------------------
# 4e. Paper reference scoring
# ---------------------------------------------------------------------------

_SECTION_PATTERN = re.compile(
    r"(?:Section|Sec\.?)\s+(\d+(?:\.\d+)*)"
    r"|(?:Protocol)\s+(\d+)"
    r"|(?:Theorem)\s+(\d+(?:\.\d+)*)"
    r"|(?:Eq\.?|Equation)\s+\(?(\d+)\)?"
    r"|(?:Example)\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


def _extract_section_ids(text: str) -> list[str]:
    """Extract structured section identifiers from paper reference text."""
    ids: list[str] = []
    for m in _SECTION_PATTERN.finditer(text):
        for i, group_name in enumerate(
            ["Section", "Protocol", "Theorem", "Equation", "Example"], 1
        ):
            if m.group(i):
                ids.append(f"{group_name} {m.group(i)}")
    return ids


def _section_similarity(agent_ids: list[str], gt_ids: list[str]) -> float:
    """Score section ID overlap. Handles exact, parent, and top-level matches."""
    if not gt_ids:
        return 1.0  # No sections expected
    if not agent_ids:
        return 0.0

    best_per_gt = []
    for gt_id in gt_ids:
        gt_parts = gt_id.split()
        gt_type = gt_parts[0]  # "Section", "Protocol", etc.
        gt_num = gt_parts[1] if len(gt_parts) > 1 else ""
        gt_segments = gt_num.split(".")

        best = 0.0
        for agent_id in agent_ids:
            a_parts = agent_id.split()
            a_type = a_parts[0]
            a_num = a_parts[1] if len(a_parts) > 1 else ""
            a_segments = a_num.split(".")

            if a_type != gt_type:
                continue

            if a_num == gt_num:
                best = max(best, 1.0)  # Exact match
            elif gt_segments[: len(a_segments)] == a_segments and len(a_segments) < len(
                gt_segments
            ):
                best = max(best, 0.6)  # Agent gives parent section
            elif a_segments[: len(gt_segments)] == gt_segments and len(gt_segments) < len(
                a_segments
            ):
                best = max(best, 0.6)  # Agent gives child section
            elif a_segments[0] == gt_segments[0]:
                best = max(best, 0.3)  # Same top-level section
        best_per_gt.append(best)

    return sum(best_per_gt) / len(best_per_gt)


def _extract_quotes(text: str) -> str:
    """Extract quoted portions from a paper reference string."""
    quotes = re.findall(r'"([^"]+)"', text)
    if quotes:
        return " ".join(quotes)
    # If no explicit quotes, use the text after the first colon as the "claim"
    parts = text.split(":", 1)
    return parts[1].strip() if len(parts) > 1 else text


def score_paper_reference(
    agent_ref: str, gt_ref: str, similarity: SimilarityBackend
) -> FieldScore:
    if not gt_ref or gt_ref.strip() in ("-", ""):
        return FieldScore(1.0, "no paper reference expected (skip)")

    if not agent_ref or agent_ref.strip() in ("-", ""):
        return FieldScore(0.0, "agent provided no paper reference")

    # Section matching (weight 0.5)
    agent_sections = _extract_section_ids(agent_ref)
    gt_sections = _extract_section_ids(gt_ref)
    section_score = _section_similarity(agent_sections, gt_sections)

    # Quote/claim similarity (weight 0.5)
    agent_quote = _extract_quotes(agent_ref)
    gt_quote = _extract_quotes(gt_ref)
    quote_score = similarity.score(agent_quote, gt_quote)

    combined = 0.5 * section_score + 0.5 * quote_score
    detail = f"section={section_score:.2f}, quote={quote_score:.2f}"
    return FieldScore(combined, detail)
