"""Shared pytest fixtures.

All test data here is synthetic — project names, issue names, file paths,
and paper references are placeholders that do not correspond to any entry
in the real benchmark dataset. This prevents ground-truth content from
leaking into tests (and from there into anything that trains on the tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pytest

from grader.similarity import SimilarityBackend


class WordOverlapSimilarity(SimilarityBackend):
    """Deterministic word-overlap (Jaccard) similarity for tests only.

    Lives in conftest (not production) because scorer tests need a predictable,
    offline similarity for the paper-reference quote sub-score. Production code
    always uses LLMJudgeSimilarity.
    """

    def score(self, text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


@pytest.fixture
def word_overlap_similarity() -> WordOverlapSimilarity:
    """A small word-overlap backend suitable for scoring tests."""
    return WordOverlapSimilarity()


# Synthetic ground-truth findings. Two fictional projects, five findings
# covering all three severities, five categories, four security concerns.
_FICTIONAL_GT_ROWS: list[dict[str, str]] = [
    {
        "entry-id": "alpha",
        "issue-id": "alpha-01",
        "issue-name": "Unchecked gate output",
        "issue-explanation": "Output of the widget gate is written directly to the advice column without a range constraint, so the prover can write any field element.",
        "severity": "Critical",
        "category": "Under-constrained Circuit",
        "security-concern": "Proof Forgery (Soundness)",
        "relevant-code": "src/widget.rs:22-30, src/circuit.rs:88",
        "paper-reference": 'Section 3.1: "Each widget output must be range-checked before use."',
    },
    {
        "entry-id": "alpha",
        "issue-id": "alpha-02",
        "issue-name": "Static prover seed",
        "issue-explanation": "The witness-generation PRNG is seeded with a compile-time constant, making witness entropy predictable.",
        "severity": "Warning",
        "category": "Engineering/Prototype Gap",
        "security-concern": "Semantic Subversion (Integrity)",
        "relevant-code": "src/prover.rs:44",
        "paper-reference": "-",
    },
    {
        "entry-id": "alpha",
        "issue-id": "alpha-03",
        "issue-name": "Accumulator width note",
        "issue-explanation": "Accumulator uses 128 bits, much wider than the paper's stated 64-bit bound. No impact but noted for the record.",
        "severity": "Info",
        "category": "Numerical/Quantization Bug",
        "security-concern": "Other",
        "relevant-code": "",
        "paper-reference": 'Section 4.2: "Accumulator width is bounded by log(n)."',
    },
    {
        "entry-id": "beta",
        "issue-id": "beta-01",
        "issue-name": "Commitment binds wrong tensor",
        "issue-explanation": "The gadget commits to the parameter tensor but the verifier reconstructs the digest from the input tensor, so the two never match.",
        "severity": "Critical",
        "category": "Witness/Commitment Mismatch",
        "security-concern": "Semantic Subversion (Integrity)",
        "relevant-code": "src/gadget.rs:100-120",
        "paper-reference": 'Protocol 1 Step 2: "P commits to parameters before sending them to V."',
    },
    {
        "entry-id": "beta",
        "issue-id": "beta-02",
        "issue-name": "Transcript missing public input",
        "issue-explanation": "Public inputs are never hashed into the transcript, so two proofs with different public inputs can share verifier challenges.",
        "severity": "Warning",
        "category": "Protocol/Transcript Logic",
        "security-concern": "Proof Malleability",
        "relevant-code": "src/transcript.rs:55",
        "paper-reference": "-",
    },
]


# Synthetic agent output. Three matches (one with severity mismatch),
# two missed GT findings (alpha-03, beta-02), one extra/novel finding.
_FICTIONAL_AGENT_OUTPUT: list[dict[str, str]] = [
    {
        # Matches alpha-01 perfectly
        "entry-id": "alpha",
        "issue-name": "Unchecked gate output",
        "issue-explanation": "Widget gate output is written to advice without range check, allowing arbitrary field elements from the prover.",
        "severity": "Critical",
        "category": "Under-constrained Circuit",
        "security-concern": "Proof Forgery (Soundness)",
        "relevant-code": "src/widget.rs:25",
        "paper-reference": 'Section 3.1: "Each widget output must be range-checked before use."',
    },
    {
        # Matches alpha-02 but under-reports severity (Warning -> Info)
        "entry-id": "alpha",
        "issue-name": "Static prover seed",
        "issue-explanation": "The prover PRNG uses a compile-time constant seed.",
        "severity": "Info",
        "category": "Engineering/Prototype Gap",
        "security-concern": "Semantic Subversion (Integrity)",
        "relevant-code": "src/prover.rs:44",
        "paper-reference": "-",
    },
    {
        # Matches beta-01 with slight rewording
        "entry-id": "beta",
        "issue-name": "Commitment bound to wrong tensor",
        "issue-explanation": "The gadget commits to parameters but the verifier reconstructs the digest from the input, so verification cannot succeed against the committed value.",
        "severity": "Critical",
        "category": "Witness/Commitment Mismatch",
        "security-concern": "Semantic Subversion (Integrity)",
        "relevant-code": "src/gadget.rs:105-110",
        "paper-reference": 'Protocol 1 Step 2: "P commits to parameters before sending them to V."',
    },
    {
        # Extra/novel finding: not in GT
        "entry-id": "beta",
        "issue-name": "Redundant constraint in wrapper",
        "issue-explanation": "A defensive constraint is duplicated in the wrapper layer. Benign but could be removed.",
        "severity": "Warning",
        "category": "Engineering/Prototype Gap",
        "security-concern": "Other",
        "relevant-code": "src/wrapper.rs:200",
        "paper-reference": "-",
    },
]


_XLSX_HEADERS = [
    "entry-id", "issue-id", "issue-name", "issue-explanation",
    "severity", "category", "security-concern",
    "relevant-code", "paper-reference",
]


@pytest.fixture(scope="session")
def fictional_xlsx_path(tmp_path_factory) -> Path:
    """A fictional ground-truth xlsx built once per test session."""
    path = tmp_path_factory.mktemp("fixtures") / "fictional_gt.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(_XLSX_HEADERS)
    for row in _FICTIONAL_GT_ROWS:
        ws.append([row[h] for h in _XLSX_HEADERS])
    wb.save(str(path))
    return path


@pytest.fixture(scope="session")
def fictional_agent_json_path(tmp_path_factory) -> Path:
    """A fictional agent output JSON file built once per test session."""
    path = tmp_path_factory.mktemp("fixtures") / "fictional_agent.json"
    path.write_text(json.dumps(_FICTIONAL_AGENT_OUTPUT, indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def fictional_gt_rows() -> list[dict[str, str]]:
    """Raw row data used to build the xlsx — exposed for assertions."""
    return list(_FICTIONAL_GT_ROWS)


@pytest.fixture(scope="session")
def fictional_agent_rows() -> list[dict[str, str]]:
    """Raw agent output — exposed for assertions."""
    return list(_FICTIONAL_AGENT_OUTPUT)
