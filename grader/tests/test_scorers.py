"""Tests for grader.scorers module."""

import pytest

from grader.loader import CodeRef
from grader.scorers import (
    score_category,
    score_code_location,
    score_paper_reference,
    score_security_concern,
    score_severity,
    _extract_section_ids,
)


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

class TestScoreSeverity:
    def test_exact_critical(self):
        r = score_severity("Critical", "Critical")
        assert r.score == 1.0

    def test_exact_warning(self):
        r = score_severity("Warning", "Warning")
        assert r.score == 1.0

    def test_exact_info(self):
        r = score_severity("Info", "Info")
        assert r.score == 1.0

    def test_over_report_by_one_critical_vs_warning(self):
        r = score_severity("Critical", "Warning")
        assert r.score == 0.3

    def test_over_report_by_one_warning_vs_info(self):
        r = score_severity("Warning", "Info")
        assert r.score == 0.5

    def test_over_report_by_two(self):
        r = score_severity("Critical", "Info")
        assert r.score == 0.1

    def test_under_report_warning_vs_critical(self):
        r = score_severity("Warning", "Critical")
        assert r.score == 0.3

    def test_under_report_info_vs_critical(self):
        r = score_severity("Info", "Critical")
        assert r.score == 0.0

    def test_under_report_info_vs_warning(self):
        r = score_severity("Info", "Warning")
        assert r.score == 0.5

    def test_invalid_severity(self):
        r = score_severity("High", "Critical")
        assert r.score == 0.0
        assert "invalid" in r.detail

    def test_detail_contains_values(self):
        r = score_severity("Critical", "Warning")
        assert "Critical" in r.detail
        assert "Warning" in r.detail


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

class TestScoreCategory:
    def test_exact_match(self):
        r = score_category("Under-constrained Circuit", "Under-constrained Circuit")
        assert r.score == 1.0

    def test_proximity_uc_wcm(self):
        r = score_category("Under-constrained Circuit", "Witness/Commitment Mismatch")
        assert r.score == 0.4

    def test_proximity_wcm_uc(self):
        # Symmetric
        r = score_category("Witness/Commitment Mismatch", "Under-constrained Circuit")
        assert r.score == 0.4

    def test_proximity_epg_sm(self):
        r = score_category("Engineering/Prototype Gap", "Specification Mismatch")
        assert r.score == 0.3

    def test_proximity_uc_sm(self):
        # "Circuit missing constraints" vs "circuit diverges from paper spec"
        r = score_category("Under-constrained Circuit", "Specification Mismatch")
        assert r.score == 0.4

    def test_proximity_ptl_epg(self):
        # Hardcoded challenges / stub protocols
        r = score_category("Protocol/Transcript Logic", "Engineering/Prototype Gap")
        assert r.score == 0.3

    def test_proximity_uc_epg(self):
        # Empty proof stubs
        r = score_category("Under-constrained Circuit", "Engineering/Prototype Gap")
        assert r.score == 0.2

    def test_proximity_nqb_epg(self):
        r = score_category("Numerical/Quantization Bug", "Engineering/Prototype Gap")
        assert r.score == 0.2

    def test_no_proximity_unrelated(self):
        r = score_category("Protocol/Transcript Logic", "Numerical/Quantization Bug")
        assert r.score == 0.0

    def test_other_vs_specific(self):
        r = score_category("Other", "Under-constrained Circuit")
        assert r.score == 0.0

    def test_exact_other(self):
        r = score_category("Other", "Other")
        assert r.score == 1.0


# ---------------------------------------------------------------------------
# Security concern scoring
# ---------------------------------------------------------------------------

class TestScoreSecurityConcern:
    def test_exact_match(self):
        r = score_security_concern("Proof Forgery (Soundness)", "Proof Forgery (Soundness)")
        assert r.score == 1.0

    def test_proximity_forgery_subversion(self):
        r = score_security_concern(
            "Proof Forgery (Soundness)", "Semantic Subversion (Integrity)"
        )
        assert r.score == 0.3

    def test_proximity_forgery_malleability(self):
        r = score_security_concern("Proof Forgery (Soundness)", "Proof Malleability")
        assert r.score == 0.3

    def test_proximity_leakage_governance(self):
        r = score_security_concern("Information Leakage (Privacy)", "Governance Bypass")
        assert r.score == 0.2

    def test_other_vs_specific(self):
        r = score_security_concern("Other", "Proof Forgery (Soundness)")
        assert r.score == 0.1

    def test_specific_vs_other(self):
        r = score_security_concern("Denial of Proof (Reliability)", "Other")
        assert r.score == 0.1

    def test_no_proximity_unrelated(self):
        r = score_security_concern(
            "Information Leakage (Privacy)", "Denial of Proof (Reliability)"
        )
        assert r.score == 0.0

    def test_symmetry(self):
        a = "Proof Forgery (Soundness)"
        b = "Semantic Subversion (Integrity)"
        assert score_security_concern(a, b).score == score_security_concern(b, a).score


# ---------------------------------------------------------------------------
# Code location scoring
# ---------------------------------------------------------------------------

class TestScoreCodeLocation:
    def test_no_gt_refs_skip(self):
        r = score_code_location([CodeRef("file.rs", 10, 10)], [])
        assert r.score == 1.0
        assert "skip" in r.detail

    def test_no_agent_refs(self):
        r = score_code_location([], [CodeRef("file.rs", 10, 20)])
        assert r.score == 0.0

    def test_exact_overlap(self):
        gt = [CodeRef("verifier.rs", 36, 42)]
        agent = [CodeRef("verifier.rs", 38, 40)]
        r = score_code_location(agent, gt)
        assert r.score == 1.0

    def test_single_line_in_range(self):
        gt = [CodeRef("verifier.rs", 36, 42)]
        agent = [CodeRef("verifier.rs", 39, 39)]
        r = score_code_location(agent, gt)
        assert r.score == 1.0

    def test_off_by_one_line_scores_as_hit(self):
        """Agent pointing at the function-signature line next to GT's body range
        counts as a hit (dist <= 2 -> 1.0)."""
        gt = [CodeRef("zkrelu.cu", 57, 72)]
        agent = [CodeRef("zkrelu.cu", 56, 56)]  # dist = 1
        r = score_code_location(agent, gt)
        assert r.score == 1.0

    def test_off_by_two_lines_scores_as_hit(self):
        gt = [CodeRef("zkrelu.cu", 57, 72)]
        agent = [CodeRef("zkrelu.cu", 74, 74)]  # dist = 2
        r = score_code_location(agent, gt)
        assert r.score == 1.0

    def test_three_lines_away_is_close_not_hit(self):
        """dist=3 falls out of the adjacency window and into the `<=30` band."""
        gt = [CodeRef("zkrelu.cu", 57, 72)]
        agent = [CodeRef("zkrelu.cu", 75, 75)]  # dist = 3
        r = score_code_location(agent, gt)
        assert r.score == 0.7

    def test_within_30_lines(self):
        gt = [CodeRef("file.cu", 50, 60)]
        agent = [CodeRef("file.cu", 80, 80)]
        r = score_code_location(agent, gt)
        assert r.score == 0.7

    def test_within_100_lines(self):
        gt = [CodeRef("file.cu", 50, 60)]
        agent = [CodeRef("file.cu", 150, 150)]
        r = score_code_location(agent, gt)
        assert r.score == 0.4

    def test_same_file_far_away(self):
        gt = [CodeRef("file.cu", 10, 20)]
        agent = [CodeRef("file.cu", 500, 500)]
        r = score_code_location(agent, gt)
        assert r.score == 0.2

    def test_different_file(self):
        gt = [CodeRef("file_a.cu", 10, 20)]
        agent = [CodeRef("file_b.cu", 10, 20)]
        r = score_code_location(agent, gt)
        assert r.score == 0.0

    def test_basename_matching(self):
        # Agent uses short path, GT uses full path
        gt = [CodeRef("src/util/verifier.rs", 36, 42)]
        agent = [CodeRef("verifier.rs", 39, 39)]
        r = score_code_location(agent, gt)
        assert r.score == 1.0

    def test_case_insensitive_filename(self):
        gt = [CodeRef("Verifier.rs", 36, 42)]
        agent = [CodeRef("verifier.rs", 39, 39)]
        r = score_code_location(agent, gt)
        assert r.score == 1.0

    def test_multiple_gt_refs_averaged(self):
        gt = [
            CodeRef("file_a.cu", 10, 20),
            CodeRef("file_b.cu", 30, 40),
        ]
        # Agent only matches file_a
        agent = [CodeRef("file_a.cu", 15, 15)]
        r = score_code_location(agent, gt)
        # file_a: 1.0 (overlap), file_b: 0.0 (no match) -> avg = 0.5
        assert r.score == 0.5

    def test_no_line_numbers_same_file(self):
        gt = [CodeRef("config.yaml", None, None)]
        agent = [CodeRef("config.yaml", None, None)]
        r = score_code_location(agent, gt)
        assert r.score == 0.2  # Same file but no lines to compare

    def test_best_agent_ref_chosen(self):
        gt = [CodeRef("file.cu", 50, 60)]
        agent = [
            CodeRef("file.cu", 200, 200),  # far away -> 0.2
            CodeRef("file.cu", 55, 55),    # overlapping -> 1.0
        ]
        r = score_code_location(agent, gt)
        assert r.score == 1.0  # Should pick the best match


# ---------------------------------------------------------------------------
# Paper reference scoring
# ---------------------------------------------------------------------------

class TestExtractSectionIds:
    def test_section_simple(self):
        ids = _extract_section_ids("Section 6.1.3: some text")
        assert "Section 6.1.3" in ids

    def test_section_top_level(self):
        ids = _extract_section_ids("Section 4 discusses this")
        assert "Section 4" in ids

    def test_protocol(self):
        ids = _extract_section_ids("Protocol 1 Line 2")
        assert "Protocol 1" in ids

    def test_theorem(self):
        ids = _extract_section_ids("Theorem 7.1")
        assert "Theorem 7.1" in ids

    def test_equation(self):
        ids = _extract_section_ids("Eq. (33)")
        assert "Equation 33" in ids

    def test_example(self):
        ids = _extract_section_ids("Example 4.2 (Section 4)")
        assert "Example 4.2" in ids
        assert "Section 4" in ids

    def test_multiple_sections(self):
        text = "Section 2.3: first; Section 4.3: second; Protocol 1 Step (3)"
        ids = _extract_section_ids(text)
        assert "Section 2.3" in ids
        assert "Section 4.3" in ids
        assert "Protocol 1" in ids
        assert "Step 3" in ids

    def test_no_sections(self):
        ids = _extract_section_ids("no sections here at all")
        assert ids == []

    def test_section_symbol(self):
        ids = _extract_section_ids("\u00a74.2: some text")
        assert "Section 4.2" in ids

    def test_figure(self):
        ids = _extract_section_ids("Fig. 5 shows this")
        assert "Figure 5" in ids

    def test_appendix_letter_only(self):
        ids = _extract_section_ids("Appendix A, Protocol 1")
        assert "Appendix A" in ids
        assert "Protocol 1" in ids

    def test_appendix_letter_dot_digit(self):
        ids = _extract_section_ids("See Appendix A.2 for details")
        assert "Appendix A.2" in ids

    def test_definition(self):
        ids = _extract_section_ids("Def. 2.2 and Definition 3")
        assert "Definition 2.2" in ids
        assert "Definition 3" in ids

    def test_step(self):
        ids = _extract_section_ids("Protocol 1 Step (7)")
        assert "Protocol 1" in ids
        assert "Step 7" in ids

    def test_step_no_parens(self):
        ids = _extract_section_ids("Step 4 of the protocol")
        assert "Step 4" in ids

    def test_line_singular(self):
        ids = _extract_section_ids("Protocol 1 Line 2")
        assert "Line 2" in ids

    def test_line_range_normalises_to_start(self):
        # Lines 6-7 and Lines 6–7 (en-dash) both capture as "Line 6"
        ids_hyphen = _extract_section_ids("Lines 6-7")
        assert "Line 6" in ids_hyphen
        ids_endash = _extract_section_ids("Lines 6\u20137")
        assert "Line 6" in ids_endash


class TestScorePaperReference:
    @pytest.fixture
    def sim(self, word_overlap_similarity):
        return word_overlap_similarity

    def test_gt_empty_skip(self, sim):
        r = score_paper_reference("Section 4: something", "-", sim)
        assert r.score == 1.0
        assert "skip" in r.detail

    def test_gt_empty_string_skip(self, sim):
        r = score_paper_reference("anything", "", sim)
        assert r.score == 1.0

    def test_agent_empty(self, sim):
        r = score_paper_reference("-", "Section 4: something", sim)
        assert r.score == 0.0

    def test_agent_empty_string(self, sim):
        r = score_paper_reference("", "Section 4: something", sim)
        assert r.score == 0.0

    def test_exact_section_match(self, sim):
        r = score_paper_reference(
            'Section 6.1.3: "the widget output is range checked"',
            'Section 6.1.3: "The widget output is always range checked before use"',
            sim,
        )
        # Section: 1.0, quote should have decent overlap
        assert r.score >= 0.5

    def test_parent_section_match(self, sim):
        r = score_paper_reference(
            "Section 6.1: overview",
            "Section 6.1.3: specific detail about widget gadgets",
            sim,
        )
        # Section should be 0.6 (parent match)
        assert 0.2 <= r.score <= 0.8

    def test_same_top_level_section(self, sim):
        r = score_paper_reference(
            "Section 6: general discussion",
            "Section 6.1.3: very specific subsection detail",
            sim,
        )
        # Section should be 0.3 (top-level match)
        assert r.score >= 0.1

    def test_completely_different_sections(self, sim):
        r = score_paper_reference(
            "Section 2: background and motivation",
            "Section 8: experimental results and discussion",
            sim,
        )
        # No section overlap, minimal quote overlap
        assert r.score < 0.3

    def test_protocol_match(self, sim):
        r = score_paper_reference(
            "Protocol 1: commit x and y before challenge",
            "Protocol 1 (Lines 6-7): prover must commit x and y before verifier sends challenge",
            sim,
        )
        assert r.score >= 0.5

    def test_multi_anchor_gt_agent_cites_one(self, sim):
        """When the GT lists several paper anchors for one finding, citing
        any *one* valid anchor should score section=1.0 (max, not mean).
        The quote component is scored independently."""
        r = score_paper_reference(
            'Protocol 1: prover commits x and y before challenge',
            'Protocol 1 (Lines 6-7): commit x, y before challenge; '
            'Protocol 2 (Line 22): send auxiliary commitments first',
            sim,
        )
        # Section should match Protocol 1 exactly -> section=1.0 under max.
        # Quote has moderate overlap. Combined >= 0.5.
        assert r.score >= 0.5
        assert "section=1.00" in r.detail
