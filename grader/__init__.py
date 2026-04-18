"""zkML Agents Benchmark Grader.

Compares agent-produced audit findings against a ground truth dataset,
scoring across severity, category, security-concern, relevant-code,
and paper-reference dimensions.
"""

__version__ = "0.1.0"

SEVERITIES = {"Critical", "Warning", "Info"}

CATEGORIES = {
    "Under-constrained Circuit",
    "Protocol/Transcript Logic",
    "Specification Mismatch",
    "Numerical/Quantization Bug",
    "Witness/Commitment Mismatch",
    "Engineering/Prototype Gap",
    "Other",
}

SECURITY_CONCERNS = {
    "Proof Forgery (Soundness)",
    "Information Leakage (Privacy)",
    "Semantic Subversion (Integrity)",
    "Proof Malleability",
    "Denial of Proof (Reliability)",
    "Governance Bypass",
    "Other",
}
