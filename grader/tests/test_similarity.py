"""Tests for grader.similarity module."""

import pytest

from grader.similarity import JaccardSimilarity, normalize_text


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_removes_punctuation(self):
        assert normalize_text("file.rs:10-15, other.cu:3") == "file rs 10 15 other cu 3"

    def test_collapses_whitespace(self):
        assert normalize_text("  lots   of   spaces  ") == "lots of spaces"

    def test_strips_accents(self):
        result = normalize_text("naïve café")
        # NFKD decomposition separates accented chars; the tokens still work for similarity
        assert "nai" in result
        assert "cafe" in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_unicode_math_symbols(self):
        # Should strip mathematical symbols like ⟦ ⟧ ∈
        result = normalize_text("⟦S⟧ ∈ [0, 1]")
        assert "s" in result
        assert "0" in result


class TestJaccardSimilarity:
    @pytest.fixture
    def sim(self):
        return JaccardSimilarity()

    def test_identical_texts(self, sim):
        assert sim.score("hello world", "hello world") == 1.0

    def test_identical_ignoring_case(self, sim):
        assert sim.score("Hello World", "hello world") == 1.0

    def test_completely_disjoint(self, sim):
        assert sim.score("the quick brown fox", "alpha beta gamma delta") == 0.0

    def test_partial_overlap(self, sim):
        score = sim.score("unchecked widget output", "widget output unchecked")
        # Overlap: {unchecked, widget, output} out of union: {unchecked, widget, output}
        assert score == 1.0  # fully identical token sets, different order
        score = sim.score("unchecked widget output", "widget output missing bound")
        # {unchecked,widget,output} ∩ {widget,output,missing,bound} = {widget,output} = 2
        # union = 5 -> 0.4
        assert 0.3 <= score <= 0.5

    def test_empty_first(self, sim):
        assert sim.score("", "hello world") == 0.0

    def test_empty_second(self, sim):
        assert sim.score("hello world", "") == 0.0

    def test_both_empty(self, sim):
        assert sim.score("", "") == 0.0

    def test_single_word_match(self, sim):
        assert sim.score("commitment", "commitment") == 1.0

    def test_single_word_no_match(self, sim):
        assert sim.score("commitment", "soundness") == 0.0

    def test_symmetry(self, sim):
        a = "transcript missing widget"
        b = "No transcript for the widget"
        assert sim.score(a, b) == sim.score(b, a)

    def test_superset_not_perfect(self, sim):
        # "a b c" vs "a b" -> intersection=2, union=3 -> 0.667
        score = sim.score("alpha beta gamma", "alpha beta")
        assert 0.5 < score < 1.0

    def test_score_range(self, sim):
        score = sim.score(
            "The first gadget never validates its input",
            "Missing validation on the first gadget input",
        )
        assert 0.0 <= score <= 1.0
