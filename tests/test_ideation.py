"""Tests for sciantist.ideation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sciantist.ideation import (
    _slugify,
    _strip_markdown_fence,
    _tail_text,
    _truncate_for_model,
    _sha256_text,
    _relevance_score,
    _parse_expert_markdown,
    _set_ideation_prompt_ideas_count,
    _extend_ideation_prompt_with_expert,
    _extract_idea_payload_list,
    _format_websearch_ideas_for_ideation_context,
    _iter_candidate_files,
    _read_text_if_exists,
    _build_tried_ideas_summary,
    _load_file_summary_cache,
    _save_file_summary_cache,
    _extract_json_object,
)


class TestTailText:
    """Test _tail_text function."""

    def test_under_max_returns_same(self) -> None:
        text = "short text"
        result = _tail_text(text, 8000)
        assert result == text

    def test_over_max_returns_tail(self) -> None:
        text = "a" * 10000
        result = _tail_text(text, 100)
        assert len(result) == 100
        assert result.startswith("a" * 50)

    def test_exactly_max_returns_same(self) -> None:
        text = "x" * 100
        result = _tail_text(text, 100)
        assert result == text


class TestStripMarkdownFence:
    """Test _strip_markdown_fence function."""

    def test_no_fence_returns_same(self) -> None:
        text = "plain text"
        result = _strip_markdown_fence(text)
        assert result == text

    def test_fence_stripped(self) -> None:
        text = "```json\n{\"key\": \"value\"}\n```"
        result = _strip_markdown_fence(text)
        assert result == '{"key": "value"}'

    def test_partial_fence_start(self) -> None:
        text = "```\ncontent\n"
        result = _strip_markdown_fence(text)
        assert result == "content"

    def test_partial_fence_end(self) -> None:
        text = "content\n```"
        result = _strip_markdown_fence(text)
        assert result == "content\n```"


class TestSlugify:
    """Test _slugify function."""

    def test_lowercase(self) -> None:
        result = _slugify("Test Idea")
        assert result == "test-idea"

    def test_special_chars_replaced(self) -> None:
        result = _slugify("Test@#$Idea")
        assert result == "test-idea"

    def test_trims_to_60_chars(self) -> None:
        result = _slugify("x" * 100)
        assert len(result) == 60

    def test_empty_string_defaults(self) -> None:
        result = _slugify("   ")
        assert result == "experiment"

    def test_dashes_become_single(self) -> None:
        result = _slugify("test---idea")
        assert result == "test-idea"


class TestTruncateForModel:
    """Test _truncate_for_model function."""

    def test_under_max_returns_same(self) -> None:
        text = "short content"
        result = _truncate_for_model(text, 100)
        assert result == text

    def test_over_max_truncates(self) -> None:
        text = "x" * 200
        result = _truncate_for_model(text, 100)
        assert "... [truncated] ..." in result

    def test_result_under_max_chars(self) -> None:
        text = "x" * 200
        result = _truncate_for_model(text, 100)
        assert "..." in result


class TestSha256Text:
    """Test _sha256_text function."""

    def test_same_input_same_hash(self) -> None:
        hash1 = _sha256_text("test content")
        hash2 = _sha256_text("test content")
        assert hash1 == hash2

    def test_different_input_different_hash(self) -> None:
        hash1 = _sha256_text("content a")
        hash2 = _sha256_text("content b")
        assert hash1 != hash2

    def test_returns_hex_string(self) -> None:
        result = _sha256_text("test")
        assert all(c in "0123456789abcdef" for c in result)


class TestRelevanceScore:
    """Test _relevance_score function."""

    def test_train_keyword_high_score(self) -> None:
        assert _relevance_score("train.py") > 0
        assert _relevance_score("trainer.py") > _relevance_score("script.py")

    def test_config_high_score(self) -> None:
        assert _relevance_score("config.yaml") > _relevance_score("README.md")

    def test_test_file_negative_score(self) -> None:
        assert _relevance_score("tests/test_a.py") < 0

    def test_py_suffix_bonus(self) -> None:
        assert _relevance_score("model.py") > _relevance_score("model.txt")


class TestParseExpertMarkdown:
    """Test _parse_expert_markdown function."""

    def test_empty_returns_empty_list(self) -> None:
        result = _parse_expert_markdown("")
        assert result == []

    def test_single_expert(self) -> None:
        text = "# Test Expert\n\nThis is a description of the expert."
        result = _parse_expert_markdown(text)
        assert len(result) == 1
        assert result[0][0] == "Test Expert"
        assert "description" in result[0][1]

    def test_multiple_experts(self) -> None:
        text = "# Expert One\n\nDescription one\n\n# Expert Two\n\nDescription two"
        result = _parse_expert_markdown(text)
        assert len(result) == 2
        assert result[0][0] == "Expert One"
        assert result[1][0] == "Expert Two"

    def test_expert_without_description(self) -> None:
        text = "# JustATitle\n\n"
        result = _parse_expert_markdown(text)
        assert result[0][1] == "(no description provided)"


class TestSetIdeationPromptIdeasCount:
    """Test _set_ideation_prompt_ideas_count function."""

    def test_replaces_count(self) -> None:
        prompt = "Generate exactly 5 different idea(s)."
        result = _set_ideation_prompt_ideas_count(prompt, 10)
        assert "Generate exactly 10 different idea(s)" in result

    def test_preserves_rest_of_prompt(self) -> None:
        prompt = "Generate exactly 5 different idea(s). Do something else."
        result = _set_ideation_prompt_ideas_count(prompt, 3)
        assert "Do something else" in result

    def test_min_count_of_one(self) -> None:
        prompt = "Generate exactly 5 different idea(s)."
        result = _set_ideation_prompt_ideas_count(prompt, 0)
        assert "Generate exactly 1 different idea(s)" in result


class TestExtendIdeationPromptWithExpert:
    """Test _extend_ideation_prompt_with_expert function."""

    def test_appends_expert_block(self) -> None:
        prompt = "Original prompt content."
        result = _extend_ideation_prompt_with_expert(prompt, "Test Expert", "A test expert description.")
        assert "Original prompt content" in result
        assert "Test Expert" in result
        assert "A test expert description" in result


class TestExtractIdeaPayloadList:
    """Test _extract_idea_payload_list function."""

    def test_list_of_ideas(self) -> None:
        payload = {"ideas": [{"idea_title": "Idea 1"}, {"idea_title": "Idea 2"}]}
        result = _extract_idea_payload_list(payload)
        assert len(result) == 2

    def test_empty_ideas_returns_as_single(self) -> None:
        payload = {"other_key": "value"}
        result = _extract_idea_payload_list(payload)
        assert result == [payload]

    def test_non_dict_in_ideas_filtered(self) -> None:
        payload = {"ideas": [{"idea_title": "Valid"}, "not a dict", None]}
        result = _extract_idea_payload_list(payload)
        assert len(result) == 1
        assert result[0]["idea_title"] == "Valid"


class TestFormatWebsearchIdeasForIdeationContext:
    """Test _format_websearch_ideas_for_ideation_context function."""

    def test_empty_payload_returns_single_item(self) -> None:
        result = _format_websearch_ideas_for_ideation_context({})
        assert "idea_1" in result

    def test_single_idea(self) -> None:
        payload = {
            "ideas": [{
                "idea_title": "Test Idea",
                "idea_details": "Details here",
                "rough_outline": "The outline",
                "why_it_is_distinct": "It is different",
                "online_evidence": ["paper1", "repo1"],
            }]
        }
        result = _format_websearch_ideas_for_ideation_context(payload)
        assert "Test Idea" in result
        assert "Details here" in result

    def test_max_items_limit(self) -> None:
        ideas = [{"idea_title": f"Idea {i}", "idea_details": "d", "rough_outline": "o", "why_it_is_distinct": "w", "online_evidence": []} for i in range(20)]
        payload = {"ideas": ideas}
        result = _format_websearch_ideas_for_ideation_context(payload, max_items=5)
        assert "Idea 0" in result
        assert "Idea 4" in result
        assert "Idea 5" not in result


class TestIterCandidateFiles:
    """Test _iter_candidate_files function."""

    def test_empty_when_no_matches(self, tmp_path: Path) -> None:
        result = _iter_candidate_files(tmp_path, [])
        assert result == []

    def test_finds_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").touch()
        result = _iter_candidate_files(tmp_path, [], allowed_suffixes=[".py"])
        assert len(result) == 1

    def test_excludes_by_deny_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").touch()
        (tmp_path / "venv").mkdir()
        (tmp_path / "venv" / "module.py").touch()
        result = _iter_candidate_files(tmp_path, [r"(^|/)venv(/|$)"], allowed_suffixes=[".py"])
        assert all("venv" not in f for f in result)

    def test_excludes_aider_only_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").touch()
        result = _iter_candidate_files(tmp_path, [], aider_only_patterns=[r"(^|/)tests?(/|$)"], allowed_suffixes=[".py"])
        assert all("tests" not in f for f in result)

    def test_includes_aider_only_when_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").touch()
        result = _iter_candidate_files(tmp_path, [], aider_only_patterns=[r"(^|/)tests?(/|$)"], include_aider_only=True, allowed_suffixes=[".py"])
        assert any("tests" in f for f in result)


class TestReadTextIfExists:
    """Test _read_text_if_exists function."""

    def test_missing_file_returns_empty_string(self) -> None:
        result = _read_text_if_exists(Path("/nonexistent/file.txt"))
        assert result == ""

    def test_reads_file_content(self, tmp_path: Path) -> None:
        path = tmp_path / "test.txt"
        path.write_text("Hello, World!")
        result = _read_text_if_exists(path)
        assert result == "Hello, World!"


class TestBuildTriedIdeasSummary:
    """Test _build_tried_ideas_summary function."""

    def test_empty_leaderboard(self) -> None:
        result = _build_tried_ideas_summary([])
        assert result == "(no prior leaderboard entries)"

    def test_no_finished_entries(self) -> None:
        result = _build_tried_ideas_summary([{"status": "running", "unified_metric": 0.9}])
        assert result == "(no finished leaderboard entries)"

    def test_sorted_by_metric_delta(self) -> None:
        entries = [
            {"status": "finished", "idea_title": "Low", "metric_delta": 0.1},
            {"status": "finished", "idea_title": "High", "metric_delta": 0.9},
            {"status": "finished", "idea_title": "Mid", "metric_delta": 0.5},
        ]
        result = _build_tried_ideas_summary(entries)
        assert "High" in result.split("\n\n")[0]

    def test_limit_enforced(self) -> None:
        entries = [{"status": "finished", "idea_title": f"Idea {i}", "metric_delta": i} for i in range(50)]
        result = _build_tried_ideas_summary(entries, limit=10)
        assert result.count("idea_title") == 10


class TestLoadFileSummaryCache:
    """Test _load_file_summary_cache function."""

    def test_missing_file_returns_default(self) -> None:
        result = _load_file_summary_cache(Path("/nonexistent/cache.json"))
        assert result["version"] == 1
        assert result["files"] == {}

    def test_invalid_json_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("not valid json")
        result = _load_file_summary_cache(path)
        assert result["version"] == 1

    def test_non_dict_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text('["list", "not", "dict"]')
        result = _load_file_summary_cache(path)
        assert result["version"] == 1

    def test_valid_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text('{"version": 2, "files": {"test.py": {"sha256": "abc"}}}')
        result = _load_file_summary_cache(path)
        assert result["version"] == 2
        assert "test.py" in result["files"]


class TestSaveFileSummaryCache:
    """Test _save_file_summary_cache function."""

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "cache.json"
        cache = {"version": 1, "files": {}}
        _save_file_summary_cache(path, cache)
        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache = {"version": 1, "files": {"test.py": {"sha256": "abc123", "summary_block": "summary"}}}
        _save_file_summary_cache(path, cache)
        loaded = _load_file_summary_cache(path)
        assert loaded["files"]["test.py"]["sha256"] == "abc123"


class TestExtractJsonObject:
    """Test _extract_json_object function."""

    def test_valid_json(self) -> None:
        result = _extract_json_object('{"key": "value", "num": 42}')
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_fenced_json(self) -> None:
        result = _extract_json_object('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"

    def test_json_with_extra_text(self) -> None:
        result = _extract_json_object('some text {"key": "value"} more text')
        assert result["key"] == "value"

    def test_nested_braces(self) -> None:
        result = _extract_json_object('{"outer": {"inner": "value"}}')
        assert result["outer"]["inner"] == "value"

    def test_invalid_returns_empty_dict(self) -> None:
        result = _extract_json_object("not json at all {}}")
        assert result == {}