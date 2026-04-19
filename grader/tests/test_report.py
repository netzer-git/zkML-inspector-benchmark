"""Tests for grader.report module."""

import json
import tempfile
from pathlib import Path

import pytest

from grader.loader import AgentFinding, CodeRef, GroundTruthFinding
from grader.matcher import MatchResult, MatchedPair
from grader.report import (
    DEFAULT_WEIGHTS,
    GradeReport,
    PairGrade,
    ProjectGrade,
    build_report,
    grade_pair,
    grade_project,
    write_json_report,
    write_markdown_report,
    _compute_pair_score,
)
from grader.scorers import FieldScore


def _gt(issue_id: str, name: str, severity: str = "Critical",
        category: str = "Other", concern: str = "Other",
        code: list[CodeRef] | None = None, paper: str = "-") -> GroundTruthFinding:
    return GroundTruthFinding(
        entry_id="test", issue_id=issue_id, issue_name=name,
        issue_explanation=f"Explanation for {name}",
        severity=severity, category=category, security_concern=concern,
        relevant_code=code or [], paper_reference=paper,
    )


def _agent(name: str, severity: str = "Critical",
           category: str = "Other", concern: str = "Other",
           code: list[CodeRef] | None = None, paper: str = "-") -> AgentFinding:
    return AgentFinding(
        entry_id="test", issue_name=name,
        issue_explanation=f"Explanation for {name}",
        severity=severity, category=category, security_concern=concern,
        relevant_code=code or [], paper_reference=paper,
    )


# ---------------------------------------------------------------------------
# _compute_pair_score
# ---------------------------------------------------------------------------

class TestComputePairScore:
    def test_all_perfect(self):
        scores = {
            "severity": FieldScore(1.0, ""),
            "category": FieldScore(1.0, ""),
            "security_concern": FieldScore(1.0, ""),
            "code_location": FieldScore(1.0, ""),
            "paper_reference": FieldScore(1.0, ""),
        }
        assert _compute_pair_score(scores, DEFAULT_WEIGHTS) == pytest.approx(1.0)

    def test_all_zero(self):
        scores = {
            "severity": FieldScore(0.0, "mismatch"),
            "category": FieldScore(0.0, "mismatch"),
            "security_concern": FieldScore(0.0, "mismatch"),
            "code_location": FieldScore(0.0, "no refs"),
            "paper_reference": FieldScore(0.0, "no paper"),
        }
        assert _compute_pair_score(scores, DEFAULT_WEIGHTS) == pytest.approx(0.0)

    def test_weight_redistribution_on_skip(self):
        # code_location and paper_reference are skipped
        scores = {
            "severity": FieldScore(1.0, "exact"),
            "category": FieldScore(1.0, "exact"),
            "security_concern": FieldScore(1.0, "exact"),
            "code_location": FieldScore(1.0, "skip"),
            "paper_reference": FieldScore(1.0, "skip"),
        }
        # Only 3 fields active, all perfect -> score should be 1.0
        result = _compute_pair_score(scores, DEFAULT_WEIGHTS)
        assert result == pytest.approx(1.0)

    def test_partial_scores(self):
        scores = {
            "severity": FieldScore(1.0, ""),      # 0.15
            "category": FieldScore(0.0, ""),      # 0.15
            "security_concern": FieldScore(0.3, ""),  # 0.15
            "code_location": FieldScore(0.7, ""),  # 0.30
            "paper_reference": FieldScore(0.5, ""),  # 0.25
        }
        expected = (0.15*1.0 + 0.15*0.0 + 0.15*0.3 + 0.30*0.7 + 0.25*0.5)
        assert _compute_pair_score(scores, DEFAULT_WEIGHTS) == pytest.approx(expected)

    def test_empty_scores(self):
        assert _compute_pair_score({}, DEFAULT_WEIGHTS) == 0.0


# ---------------------------------------------------------------------------
# grade_pair
# ---------------------------------------------------------------------------

class TestGradePair:
    @pytest.fixture
    def sim(self, word_overlap_similarity):
        return word_overlap_similarity

    def test_perfect_match(self, sim):
        gt = _gt("T-01", "Issue X", severity="Critical", category="Other", concern="Other")
        agent = _agent("Issue X", severity="Critical", category="Other", concern="Other")
        pair = MatchedPair(agent=agent, gt=gt, similarity=1.0)
        grade = grade_pair(pair, sim)
        assert grade.gt_id == "T-01"
        assert grade.scores["severity"].score == 1.0
        assert grade.scores["category"].score == 1.0
        assert grade.scores["security_concern"].score == 1.0
        assert grade.pair_score > 0.0

    def test_severity_mismatch_penalized(self, sim):
        gt = _gt("T-01", "Issue X", severity="Critical")
        agent = _agent("Issue X", severity="Warning")
        pair = MatchedPair(agent=agent, gt=gt, similarity=0.8)
        grade = grade_pair(pair, sim)
        # Under-reporting Critical as Warning earns partial credit (0.3).
        assert grade.scores["severity"].score == 0.3

    def test_code_location_scored(self, sim):
        gt = _gt("T-01", "Issue X", code=[CodeRef("file.rs", 10, 20)])
        agent = _agent("Issue X", code=[CodeRef("file.rs", 15, 15)])
        pair = MatchedPair(agent=agent, gt=gt, similarity=0.8)
        grade = grade_pair(pair, sim)
        assert grade.scores["code_location"].score == 1.0

    def test_paper_ref_scored(self, sim):
        gt = _gt("T-01", "Issue X", paper="Section 6.1.3: important claim")
        agent = _agent("Issue X", paper="Section 6.1.3: the important claim about ZK")
        pair = MatchedPair(agent=agent, gt=gt, similarity=0.8)
        grade = grade_pair(pair, sim)
        assert grade.scores["paper_reference"].score > 0.0

    def test_custom_weights(self, sim):
        gt = _gt("T-01", "Issue X")
        agent = _agent("Issue X")
        pair = MatchedPair(agent=agent, gt=gt, similarity=0.8)
        weights = {"severity": 1.0, "category": 0.0, "security_concern": 0.0,
                   "code_location": 0.0, "paper_reference": 0.0}
        grade = grade_pair(pair, sim, weights=weights)
        # All weight on severity (which is 1.0 exact match)
        assert grade.pair_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# grade_project
# ---------------------------------------------------------------------------

class TestGradeProject:
    @pytest.fixture
    def sim(self, word_overlap_similarity):
        return word_overlap_similarity

    def test_perfect_project(self, sim):
        gt_list = [_gt("T-01", "Issue A")]
        agent_list = [_agent("Issue A")]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=1.0)],
            missed_gt=[],
            extra_agent=[],
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
        assert pg.quality == 0.0

    def test_partial_recall(self, sim):
        gt_list = [_gt("T-01", "Issue A"), _gt("T-02", "Issue B")]
        agent_list = [_agent("Issue A")]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=0.9)],
            missed_gt=[gt_list[1]],
            extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.recall == 0.5
        assert pg.precision == 1.0

    def test_extra_findings_by_severity(self, sim):
        gt_list = [_gt("T-01", "Issue A")]
        agent_list = [
            _agent("Issue A"),
            _agent("Extra Crit", severity="Critical"),
            _agent("Extra Warn", severity="Warning"),
        ]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=0.9)],
            missed_gt=[],
            extra_agent=[agent_list[1], agent_list[2]],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.extra_by_severity == {"Critical": 1, "Warning": 1}
        # Severity-weighted precision: matched is the Issue A finding
        # (Critical default = weight 1.0). Extras: Extra Crit = 1.0, Extra Warn
        # = 0.5. total_weight = 1 + 1 + 0.5 = 2.5. precision = 1.0 / 2.5 = 0.4.
        assert pg.precision == pytest.approx(0.4)

    def test_precision_severity_weights(self, sim):
        """Precision weights Critical=1.0, Warning=0.5, Info=0.1."""
        gt_list = [_gt("T-01", "A", severity="Critical")]
        agent_list = [
            _agent("A", severity="Critical"),     # matched, weight 1.0
            _agent("extra info", severity="Info"),  # Info extra, weight 0.1
        ]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=0.9)],
            missed_gt=[],
            extra_agent=[agent_list[1]],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        # weighted_matched=1.0, weighted_total=1.0+0.1=1.1 → 1.0/1.1 ≈ 0.909
        assert pg.precision == pytest.approx(1.0 / 1.1)

    def test_severity_weighted_recall(self, sim):
        gt_list = [
            _gt("T-01", "Critical issue", severity="Critical"),
            _gt("T-02", "Info issue", severity="Info"),
        ]
        agent_list = [_agent("Info issue")]
        # Only the Info issue matched, the Critical one missed
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[1], similarity=0.9)],
            missed_gt=[gt_list[0]],
            extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        # SWR = (3*0 + 2*0 + 1*1) / (3*1 + 2*0 + 1*1) = 1/4 = 0.25
        assert pg.severity_weighted_recall == pytest.approx(0.25)

    def test_composite_formula(self, sim):
        gt_list = [_gt("T-01", "Issue A")]
        agent_list = [_agent("Issue A")]
        match_result = MatchResult(
            matched=[MatchedPair(agent=agent_list[0], gt=gt_list[0], similarity=1.0)],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        expected = 0.4 * pg.f1 + 0.6 * pg.quality
        assert pg.composite == pytest.approx(expected)

    def test_f1_uses_severity_weighted_recall(self, sim):
        """Composite F1 must weight recall by severity so a Critical miss
        hurts more than an Info miss."""
        gt_list = [
            _gt("T-01", "Critical", severity="Critical"),
            _gt("T-02", "Info", severity="Info"),
        ]
        agent_list = [_agent("Info match", severity="Info")]
        match_result = MatchResult(
            matched=[MatchedPair(agent_list[0], gt_list[1], similarity=0.9)],
            missed_gt=[gt_list[0]], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        # Plain recall = 1/2 = 0.5; SWR = 1/4 = 0.25
        assert pg.recall == pytest.approx(0.5)
        assert pg.severity_weighted_recall == pytest.approx(0.25)
        # F1 is computed against SWR, not plain recall.
        expected = (
            2 * pg.precision * pg.severity_weighted_recall
            / (pg.precision + pg.severity_weighted_recall)
        )
        assert pg.f1 == pytest.approx(expected)

    def test_quality_is_severity_weighted(self, sim):
        """Per-GT quality contributions are weighted by GT severity."""
        # Two GTs: Critical gets a low pair score, Info gets a high one.
        # Quality should be dominated by the Critical (weight 3) vs Info (1).
        gt_list = [
            _gt("T-01", "Critical", severity="Critical"),
            _gt("T-02", "Info", severity="Info"),
        ]
        agent_list = [
            _agent("Critical", severity="Warning"),  # severity mismatch -> low
            _agent("Info", severity="Info"),          # exact -> high
        ]
        match_result = MatchResult(
            matched=[
                MatchedPair(agent_list[0], gt_list[0], similarity=0.9),
                MatchedPair(agent_list[1], gt_list[1], similarity=0.9),
            ],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        # The Critical pair has a worse pair_score (severity mismatch drags
        # it down via the severity field); the Info pair is clean. With
        # severity weights 3:1, quality must fall between them but closer
        # to the Critical (lower) value.
        pg_pair_crit = pg.matches[0].pair_score
        pg_pair_info = pg.matches[1].pair_score
        assert pg_pair_crit < pg_pair_info  # sanity
        expected_quality = (3 * pg_pair_crit + 1 * pg_pair_info) / 4
        assert pg.quality == pytest.approx(expected_quality)

    def test_n_to_1_recall_counts_unique_gts(self, sim):
        """2 agents matched to the same GT -> recall is 1/1 (unique GT)."""
        gt_list = [_gt("T-01", "Issue A")]
        agent_list = [_agent("Agent A1"), _agent("Agent A2")]
        match_result = MatchResult(
            matched=[
                MatchedPair(agent_list[0], gt_list[0], similarity=0.95, dup_rank=0),
                MatchedPair(agent_list[1], gt_list[0], similarity=0.90, dup_rank=1),
            ],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, agent_list)
        assert pg.recall == 1.0

    def test_n_to_1_quality_averages_per_gt_group(self, sim):
        """Two agents on the same GT: quality is average of their pair_scores."""
        gt_list = [_gt("T-01", "A"), _gt("T-02", "B")]
        a1 = _agent("A1")
        a2 = _agent("A2-dup")
        a3 = _agent("B-match")
        # Construct a match_result: agents 1,2 both bound to T-01 (split),
        # agent 3 bound to T-02 alone.
        match_result = MatchResult(
            matched=[
                MatchedPair(a1, gt_list[0], similarity=0.9, dup_rank=0),
                MatchedPair(a2, gt_list[0], similarity=0.6, dup_rank=1),
                MatchedPair(a3, gt_list[1], similarity=0.8, dup_rank=0),
            ],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, [a1, a2, a3])
        # Each pair has a pair_score computed from field scores. For a
        # minimal _agent/_gt pair these will all be equal (same defaults).
        # Quality = mean of [mean(T-01 pairs), T-02 pair]. When the two
        # T-01 pairs have identical field scores, their avg equals either
        # of them. So quality = the common pair_score.
        # Verify shape: quality is a single-number average across GT groups.
        assert 0 <= pg.quality <= 1.0
        # Both agents on T-01 count as matched (no extras).
        assert pg.extra_by_severity == {}

    def test_n_to_1_duplicates_not_extras(self, sim):
        """A duplicate match (N:1) does NOT count as an extra."""
        gt_list = [_gt("T-01", "A")]
        a1 = _agent("match")
        a2 = _agent("dup")
        match_result = MatchResult(
            matched=[
                MatchedPair(a1, gt_list[0], similarity=0.9, dup_rank=0),
                MatchedPair(a2, gt_list[0], similarity=0.8, dup_rank=1),
            ],
            missed_gt=[], extra_agent=[],
        )
        pg = grade_project("test", match_result, sim, gt_list, [a1, a2])
        # Both agents are matched; no extras; precision reflects both as covered.
        assert pg.extra_by_severity == {}


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_overall_metrics(self):
        pg1 = ProjectGrade(
            project="a", recall=1.0, precision=0.5, f1=0.667,
            severity_weighted_recall=1.0, quality=0.8, composite=0.747,
            matches=[PairGrade("A-01", "X", "X", 1.0, {}, 0.8)],
            missed_gt=[], extra_agent=[{"name": "Y", "severity": "Info"}],
            extra_by_severity={"Info": 1},
        )
        pg2 = ProjectGrade(
            project="b", recall=0.5, precision=1.0, f1=0.667,
            severity_weighted_recall=0.5, quality=0.9, composite=0.807,
            matches=[PairGrade("B-01", "Z", "Z", 0.9, {}, 0.9)],
            missed_gt=[{"id": "B-02", "name": "W", "severity": "Warning"}],
            extra_agent=[], extra_by_severity={},
        )
        report = build_report(
            {"a": pg1, "b": pg2}, threshold=0.3,
            weights=DEFAULT_WEIGHTS, backend_name="jaccard",
        )
        assert report.meta["grader_version"] == "0.1.0"
        assert report.overall["total_matched"] == 2
        assert report.overall["total_gt"] == 3  # 1 + 2
        assert report.overall["total_agent"] == 3  # 2 + 1
        assert 0.0 <= report.overall["benchmark_score"] <= 1.0

    def test_extra_by_severity_aggregated(self):
        pg1 = ProjectGrade(
            project="a", recall=1.0, precision=1.0, f1=1.0,
            severity_weighted_recall=1.0, quality=1.0, composite=1.0,
            matches=[], missed_gt=[], extra_agent=[],
            extra_by_severity={"Critical": 2, "Warning": 1},
        )
        pg2 = ProjectGrade(
            project="b", recall=1.0, precision=1.0, f1=1.0,
            severity_weighted_recall=1.0, quality=1.0, composite=1.0,
            matches=[], missed_gt=[], extra_agent=[],
            extra_by_severity={"Critical": 1, "Info": 3},
        )
        report = build_report(
            {"a": pg1, "b": pg2}, threshold=0.3,
            weights=DEFAULT_WEIGHTS, backend_name="jaccard",
        )
        extra = report.overall["extra_by_severity"]
        assert extra["Critical"] == 3
        assert extra["Warning"] == 1
        assert extra["Info"] == 3

    def test_meta_defaults_empty_failed_and_skipped(self):
        """When build_report is called without the optional lists, meta has
        skipped_projects=[] and failed_projects=[]."""
        report = build_report(
            {}, threshold=0.3, weights=DEFAULT_WEIGHTS, backend_name="llm-judge",
        )
        assert report.meta["skipped_projects"] == []
        assert report.meta["failed_projects"] == []

    def test_meta_records_failed_and_skipped_projects(self):
        report = build_report(
            {}, threshold=0.3, weights=DEFAULT_WEIGHTS, backend_name="llm-judge",
            skipped_projects=["unknown_project"],
            failed_projects=[
                {"project": "broken", "error_type": "RuntimeError", "error": "boom"},
            ],
        )
        assert report.meta["skipped_projects"] == ["unknown_project"]
        assert len(report.meta["failed_projects"]) == 1
        assert report.meta["failed_projects"][0]["project"] == "broken"
        assert report.meta["failed_projects"][0]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

class TestWriteJsonReport:
    def test_writes_valid_json(self, tmp_path):
        pg = ProjectGrade(
            project="test", recall=0.8, precision=0.9, f1=0.847,
            severity_weighted_recall=0.85, quality=0.75, composite=0.789,
            matches=[
                PairGrade(
                    gt_id="T-01", gt_name="Issue", agent_name="Issue",
                    match_similarity=0.9,
                    scores={"severity": FieldScore(1.0, "exact match (Critical)")},
                    pair_score=0.85,
                ),
            ],
            missed_gt=[{"id": "T-02", "name": "Missed", "severity": "Warning"}],
            extra_agent=[{"name": "Extra", "severity": "Info"}],
            extra_by_severity={"Info": 1},
        )
        report = build_report(
            {"test": pg}, threshold=0.3, weights=DEFAULT_WEIGHTS, backend_name="jaccard",
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

    def test_json_roundtrip_types(self, tmp_path):
        pg = ProjectGrade(
            project="x", recall=0.0, precision=0.0, f1=0.0,
            severity_weighted_recall=0.0, quality=0.0, composite=0.0,
            matches=[], missed_gt=[], extra_agent=[], extra_by_severity={},
        )
        report = build_report(
            {"x": pg}, threshold=0.3, weights=DEFAULT_WEIGHTS, backend_name="jaccard",
        )
        out_path = str(tmp_path / "report.json")
        write_json_report(report, out_path)
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        # All numeric values should be JSON-serializable (no numpy types)
        assert isinstance(data["overall"]["benchmark_score"], float)


class TestWriteMarkdownReport:
    def test_writes_markdown_file(self, tmp_path):
        pg = ProjectGrade(
            project="test", recall=0.8, precision=0.9, f1=0.847,
            severity_weighted_recall=0.85, quality=0.75, composite=0.789,
            matches=[
                PairGrade(
                    gt_id="T-01", gt_name="Issue A", agent_name="Issue A Found",
                    match_similarity=0.9,
                    scores={
                        "severity": FieldScore(1.0, "exact"),
                        "category": FieldScore(1.0, "exact"),
                        "security_concern": FieldScore(0.3, "partial"),
                        "code_location": FieldScore(0.7, "nearby"),
                        "paper_reference": FieldScore(0.5, "section match"),
                    },
                    pair_score=0.72,
                ),
            ],
            missed_gt=[{"id": "T-02", "name": "Missed Issue", "severity": "Critical"}],
            extra_agent=[{"name": "Novel Finding", "severity": "Info"}],
            extra_by_severity={"Info": 1},
        )
        report = build_report(
            {"test": pg}, threshold=0.3, weights=DEFAULT_WEIGHTS, backend_name="jaccard",
        )
        out_path = str(tmp_path / "report.md")
        write_markdown_report(report, out_path)

        content = Path(out_path).read_text(encoding="utf-8")
        assert "# zkML Benchmark Grading Report" in content
        assert "Benchmark Score" in content
        assert "test" in content
        assert "Missed Issue" in content
        assert "Novel Finding" in content
        assert "Info: 1" in content
