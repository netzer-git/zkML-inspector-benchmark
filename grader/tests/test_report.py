"""Tests for grader.report module — recall-focused quality gate."""

import json
from pathlib import Path

import pytest

from grader.loader import AgentFinding, CodeRef, GroundTruthFinding
from grader.matcher import MatchResult, MatchedPair
from grader.report import (
    QUALITY_THRESHOLD,
    QUALITY_WEIGHTS,
    PairGrade,
    ProjectGrade,
    _compute_quality,
    build_report,
    grade_pair,
    grade_project,
    write_json_report,
    write_markdown_report,
)
from grader.scorers import FieldScore


def _gt(issue_id: str, name: str,
        code: list[CodeRef] | None = None, paper: str = "-") -> GroundTruthFinding:
    return GroundTruthFinding(
        entry_id="test", issue_id=issue_id, issue_name=name,
        issue_explanation=f"Explanation for {name}",
        relevant_code=code or [], paper_reference=paper,
    )


def _agent(name: str,
           code: list[CodeRef] | None = None, paper: str = "-") -> AgentFinding:
    return AgentFinding(
        entry_id="test", issue_name=name,
        issue_explanation=f"Explanation for {name}",
        relevant_code=code or [], paper_reference=paper,
    )


# ---------------------------------------------------------------------------
# _compute_quality
# ---------------------------------------------------------------------------

class TestComputeQuality:
    def test_perfect_scores(self):
        q = _compute_quality(
            5.0,
            FieldScore(1.0, "perfect"),
            FieldScore(1.0, "perfect"),
        )
        assert q == pytest.approx(1.0)

    def test_minimum_match_no_evidence(self):
        q = _compute_quality(
            4.0,
            FieldScore(0.0, "no refs"),
            FieldScore(0.0, "no paper"),
        )
        # 0.50*(4/5) + 0.30*0 + 0.20*0 = 0.40
        assert q == pytest.approx(0.40)

    def test_threshold_boundary_pass(self):
        # match_score=4, code=0.7, paper=0.0
        # 0.50*(4/5) + 0.30*0.7 + 0.20*0.0 = 0.40 + 0.21 = 0.61
        q = _compute_quality(
            4.0,
            FieldScore(0.7, "nearby"),
            FieldScore(0.0, "no paper"),
        )
        assert q == pytest.approx(0.61)
        assert q >= QUALITY_THRESHOLD

    def test_below_threshold(self):
        # match_score=3, code=0.3, paper=0.0
        # 0.50*(3/5) + 0.30*0.3 + 0.20*0.0 = 0.30 + 0.09 = 0.39
        q = _compute_quality(
            3.0,
            FieldScore(0.3, "far"),
            FieldScore(0.0, "no paper"),
        )
        assert q == pytest.approx(0.39)
        assert q < QUALITY_THRESHOLD

    def test_skipped_fields_use_midpoint(self):
        q = _compute_quality(
            4.0,
            FieldScore(1.0, "no code location expected (skip)"),
            FieldScore(1.0, "skip"),
        )
        # 0.50*(4/5) + 0.30*0.5 + 0.20*0.5 = 0.40 + 0.15 + 0.10 = 0.65
        assert q == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# grade_pair
# ---------------------------------------------------------------------------

class TestGradePair:
    @pytest.fixture
    def sim(self, word_overlap_similarity):
        return word_overlap_similarity

    def test_high_quality_passes(self, sim):
        gt = _gt("T-01", "Issue X", code=[CodeRef("file.rs", 10, 20)])
        agent = _agent("Issue X", code=[CodeRef("file.rs", 15, 15)])
        pair = MatchedPair(agent=agent, gt=gt, similarity=5.0)
        grade = grade_pair(pair, sim)
        assert grade.passed is True
        assert grade.quality > QUALITY_THRESHOLD

    def test_low_quality_fails(self, sim):
        gt = _gt("T-01", "Issue X", code=[CodeRef("file.rs", 10, 20)])
        agent = _agent("Issue X", code=[CodeRef("other.rs", 500, 500)])
        pair = MatchedPair(agent=agent, gt=gt, similarity=2.0)
        grade = grade_pair(pair, sim, quality_threshold=0.55)
        assert grade.passed is False

    def test_code_location_scored(self, sim):
        gt = _gt("T-01", "Issue X", code=[CodeRef("file.rs", 10, 20)])
        agent = _agent("Issue X", code=[CodeRef("file.rs", 15, 15)])
        pair = MatchedPair(agent=agent, gt=gt, similarity=4.0)
        grade = grade_pair(pair, sim)
        assert grade.code_location_score.score == 1.0

    def test_paper_ref_scored(self, sim):
        gt = _gt("T-01", "Issue X", paper="Section 6.1.3: important claim")
        agent = _agent("Issue X", paper="Section 6.1.3: the important claim about ZK")
        pair = MatchedPair(agent=agent, gt=gt, similarity=4.0)
        grade = grade_pair(pair, sim)
        assert grade.paper_reference_score.score > 0.0


# ---------------------------------------------------------------------------
# grade_project
# ---------------------------------------------------------------------------

class TestGradeProject:
    @pytest.fixture
    def sim(self, word_overlap_similarity):
        return word_overlap_similarity

    def test_perfect_project(self, sim):
        gt_list = [_gt("T-01", "Issue A", code=[CodeRef("f.rs", 10, 20)])]
        agent_list = [_agent("Issue A", code=[CodeRef("f.rs", 15, 15)])]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=5.0)],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.recall == 1.0
        assert pg.precision == 1.0
        assert pg.f1 == 1.0

    def test_no_matches(self, sim):
        gt_list = [_gt("T-01", "Issue A"), _gt("T-02", "Issue B")]
        agent_list = [_agent("Unrelated X")]
        match_result = MatchResult(
            matched=[], missed_gt=gt_list, extra_agent=agent_list,
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.recall == 0.0
        assert pg.precision == 0.0
        assert pg.f1 == 0.0
        assert pg.avg_quality == 0.0

    def test_partial_recall(self, sim):
        gt_list = [
            _gt("T-01", "Issue A", code=[CodeRef("f.rs", 10, 20)]),
            _gt("T-02", "Issue B"),
        ]
        agent_list = [_agent("Issue A", code=[CodeRef("f.rs", 15, 15)])]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=5.0)],
            missed_gt=[gt_list[1]], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.recall == 0.5
        assert pg.precision == 1.0

    def test_extra_findings_counted(self, sim):
        gt_list = [_gt("T-01", "Issue A", code=[CodeRef("f.rs", 10, 20)])]
        agent_list = [
            _agent("Issue A", code=[CodeRef("f.rs", 15, 15)]),
            _agent("Extra 1"),
            _agent("Extra 2"),
        ]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=5.0)],
            missed_gt=[], extra_agent=[agent_list[1], agent_list[2]],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.extra_count == 2

    def test_quality_gate_filters_low_matches(self, sim):
        """A match with low quality does not count toward recall."""
        gt_list = [_gt("T-01", "Issue A")]
        agent_list = [_agent("Issue A")]
        # similarity=2.0 with no code/paper => quality = 0.50*(2/5) = 0.20
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=2.0)],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.recall == 0.0  # Quality gate rejects the match

    def test_n_to_1_recall_counts_unique_passed_gts(self, sim):
        """2 agents matched to the same GT -> recall is 1/1 if at least one passes."""
        gt_list = [_gt("T-01", "Issue A")]
        agent_list = [_agent("Agent A1"), _agent("Agent A2")]
        match_result = MatchResult(
            matched=[
                MatchedPair(agent_list[0], gt_list[0], similarity=5.0, dup_rank=0),
                MatchedPair(agent_list[1], gt_list[0], similarity=5.0, dup_rank=1),
            ],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.recall == 1.0


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_overall_metrics(self):
        pg1 = ProjectGrade(
            project="a", recall=1.0, precision=0.5, f1=0.667,
            avg_quality=0.8,
            matches=[PairGrade("A-01", "X", "X", 5.0,
                               FieldScore(1.0, ""), FieldScore(1.0, ""),
                               0.8, True)],
            missed_gt=[], extra_agent=[{"name": "Y"}], extra_count=1,
        )
        pg2 = ProjectGrade(
            project="b", recall=0.5, precision=1.0, f1=0.667,
            avg_quality=0.9,
            matches=[PairGrade("B-01", "Z", "Z", 4.0,
                               FieldScore(0.7, ""), FieldScore(0.5, ""),
                               0.9, True)],
            missed_gt=[{"id": "B-02", "name": "W"}],
            extra_agent=[], extra_count=0,
        )
        report = build_report(
            {"a": pg1, "b": pg2}, threshold=4,
            quality_threshold=0.55, backend_name="jaccard",
        )
        assert report.meta["grader_version"] == "0.1.0"
        assert report.overall["total_passed"] == 2
        assert 0.0 <= report.overall["recall"] <= 1.0

    def test_meta_defaults(self):
        report = build_report(
            {}, threshold=4, quality_threshold=0.55, backend_name="llm-judge",
        )
        assert report.meta["skipped_projects"] == []
        assert report.meta["failed_projects"] == []
        assert report.meta["quality_threshold"] == 0.55

    def test_extra_count_aggregated(self):
        pg1 = ProjectGrade(
            project="a", recall=1.0, precision=1.0, f1=1.0,
            avg_quality=1.0, matches=[], missed_gt=[], extra_agent=[],
            extra_count=3,
        )
        pg2 = ProjectGrade(
            project="b", recall=1.0, precision=1.0, f1=1.0,
            avg_quality=1.0, matches=[], missed_gt=[], extra_agent=[],
            extra_count=2,
        )
        report = build_report(
            {"a": pg1, "b": pg2}, threshold=4,
            quality_threshold=0.55, backend_name="jaccard",
        )
        assert report.overall["total_extra"] == 5


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

class TestWriteJsonReport:
    def test_writes_valid_json(self, tmp_path):
        pg = ProjectGrade(
            project="test", recall=0.8, precision=0.9, f1=0.847,
            avg_quality=0.75,
            matches=[
                PairGrade(
                    gt_id="T-01", gt_name="Issue", agent_name="Issue",
                    match_similarity=4.0,
                    code_location_score=FieldScore(1.0, "exact"),
                    paper_reference_score=FieldScore(0.5, "section match"),
                    quality=0.85, passed=True,
                ),
            ],
            missed_gt=[{"id": "T-02", "name": "Missed"}],
            extra_agent=[{"name": "Extra"}], extra_count=1,
        )
        report = build_report(
            {"test": pg}, threshold=4,
            quality_threshold=0.55, backend_name="jaccard",
        )
        out_path = str(tmp_path / "report.json")
        write_json_report(report, out_path)

        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "meta" in data
        assert "projects" in data
        assert "overall" in data
        assert "test" in data["projects"]
        assert data["projects"]["test"]["recall"] == 0.8


class TestWriteMarkdownReport:
    def test_writes_markdown_file(self, tmp_path):
        pg = ProjectGrade(
            project="test", recall=0.8, precision=0.9, f1=0.847,
            avg_quality=0.75,
            matches=[
                PairGrade(
                    gt_id="T-01", gt_name="Issue A", agent_name="Issue A Found",
                    match_similarity=4.0,
                    code_location_score=FieldScore(0.7, "nearby"),
                    paper_reference_score=FieldScore(0.5, "section match"),
                    quality=0.72, passed=True,
                ),
            ],
            missed_gt=[{"id": "T-02", "name": "Missed Issue"}],
            extra_agent=[{"name": "Novel Finding"}], extra_count=1,
        )
        report = build_report(
            {"test": pg}, threshold=4,
            quality_threshold=0.55, backend_name="jaccard",
        )
        out_path = str(tmp_path / "report.md")
        write_markdown_report(report, out_path)

        content = Path(out_path).read_text(encoding="utf-8")
        assert "# zkML Benchmark Grading Report" in content
        assert "Recall" in content
        assert "test" in content
        assert "Missed Issue" in content
        assert "Novel Finding" in content
