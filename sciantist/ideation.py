"""Ideation helpers: file discovery, prompt construction, and MCP query parsing."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from llmclient.minimax_client import MiniMaxMCPClient
from sciantist.config import ExperimentOutcome, LoopConfig
from sciantist.state import _file_lock, _resolve_output_dir


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _vprint(config: LoopConfig, message: str) -> None:
    """Print verbose lines when verbose mode is enabled."""
    if config.verbose:
        logger.debug(message)


def _tail_text(text: str, max_chars: int = 8000) -> str:
    """Return trailing text for concise logging/prompts."""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _strip_markdown_fence(text: str) -> str:
    """Strip optional fenced markdown wrapper from model output."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _slugify(text: str) -> str:
    """Create a git-safe short slug from arbitrary text."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return normalized[:60] if normalized else "experiment"


def _build_tried_ideas_summary(leaderboard_entries: list[dict[str, Any]], limit: int = 40) -> str:
    """Build compact tried-ideas context for ideation prompts."""
    if not leaderboard_entries:
        return "(no prior leaderboard entries)"

    def _metric_delta_value(entry: dict[str, Any]) -> float:
        raw = entry.get("metric_delta", -200.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return -200.0

    finished_entries = [
        entry for entry in leaderboard_entries if str(entry.get("status", "")).strip().lower() == "finished"
    ]
    if not finished_entries:
        return "(no finished leaderboard entries)"

    sorted_entries = sorted(finished_entries, key=_metric_delta_value, reverse=True)
    rows: list[str] = []
    for entry in sorted_entries[:limit]:
        metric_delta = _metric_delta_value(entry)
        rows.append(
            "\n".join(
                [
                    f"- idea_title: {entry.get('idea_title', '')}",
                    f"  improvement: {metric_delta}",
                ]
            )
        )
    return "\n\n".join(rows)


def _iter_candidate_files(
    root: Path,
    deny_patterns: list[str],
    aider_only_patterns: list[str] | None = None,
    include_aider_only: bool = False,
    allowed_suffixes: list[str] | None = None,
) -> list[str]:
    """Collect candidate files, with optional inclusion of aider-only matches."""
    effective_allowed_suffixes = set(allowed_suffixes or [".py", ".yaml", ".yml"])
    effective_aider_only_patterns = aider_only_patterns or []
    files: list[str] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in effective_allowed_suffixes:
            continue
        rel = file_path.relative_to(root).as_posix()
        if any(re.search(pattern, rel) for pattern in deny_patterns):
            continue
        if not include_aider_only and any(re.search(pattern, rel) for pattern in effective_aider_only_patterns):
            continue
        files.append(str(file_path))
    return files


def _read_text_if_exists(path: Path) -> str:
    """Read a UTF-8 text file if it exists, else return an empty string."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_codebase_glimpse(
    repo_root: Path,
    deny_patterns: list[str],
    aider_only_patterns: list[str] | None = None,
    limit: int = 120,
) -> str:
    """Return a compact file listing for prompt grounding."""
    effective_aider_only_patterns = aider_only_patterns or []
    paths: list[str] = []
    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(repo_root).as_posix()
        if rel.startswith(".git/"):
            continue
        if any(re.search(pattern, rel) for pattern in deny_patterns):
            continue
        if any(re.search(pattern, rel) for pattern in effective_aider_only_patterns):
            continue
        paths.append(rel)
    paths.sort()
    trimmed = paths[:limit]
    suffix = "\n..." if len(paths) > limit else ""
    return "\n".join(trimmed) + suffix


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from model output, with fenced and partial recovery."""

    def _strip_fence(raw_text: str) -> str:
        stripped = raw_text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if not lines:
            return stripped
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _iter_braced_segments(raw_text: str) -> list[str]:
        segments: list[str] = []
        text_len = len(raw_text)
        start = 0
        while start < text_len:
            open_idx = raw_text.find("{", start)
            if open_idx == -1:
                break

            depth = 0
            in_string = False
            quote_char = ""
            escaped = False
            end_idx: int | None = None

            for idx in range(open_idx, text_len):
                char = raw_text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote_char:
                        in_string = False
                    continue

                if char in {'"', "'"}:
                    in_string = True
                    quote_char = char
                    continue
                if char == "{":
                    depth += 1
                    continue
                if char == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = idx + 1
                        break

            if end_idx is None:
                break
            segments.append(raw_text[open_idx:end_idx])
            start = open_idx + 1
        return segments

    def _parse_candidate(raw_text: str) -> dict[str, Any] | None:
        stripped = raw_text.strip()
        if not stripped:
            return None

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped, idx)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        for segment in _iter_braced_segments(stripped):
            try:
                parsed = ast.literal_eval(segment)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                return parsed

        return None

    candidates = [text, _strip_fence(text)]
    for candidate in candidates:
        payload = _parse_candidate(candidate)
        if payload is not None:
            return payload

    raise RuntimeError("Ideation model did not return a parseable JSON object.")


def _truncate_for_model(content: str, max_chars: int) -> str:
    """Trim large file contents while preserving both prefix and suffix context."""
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    head = content[:half]
    tail = content[-half:]
    return f"{head}\n\n... [truncated] ...\n\n{tail}"


def _sha256_text(content: str) -> str:
    """Return SHA-256 hex digest for text content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _load_file_summary_cache(path: Path) -> dict[str, Any]:
    """Load summary cache from disk or return initialized cache object."""
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "files": {}}

    if not isinstance(payload, dict):
        return {"version": 1, "files": {}}
    files_map = payload.get("files")
    if not isinstance(files_map, dict):
        payload["files"] = {}
    payload.setdefault("version", 1)
    return payload


def _save_file_summary_cache(path: Path, cache: dict[str, Any]) -> None:
    """Persist summary cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _initialize_file_summary_cache_if_needed(config: LoopConfig, output_root: Path) -> None:
    """Seed summary cache for new runs from a configured initial cache file."""
    cache_path = output_root / config.file_summary_cache_name
    if cache_path.exists():
        return

    init_cache_raw = config.init_cache_summary_file.strip()
    if not init_cache_raw:
        return

    init_cache_path = Path(init_cache_raw).expanduser()
    if not init_cache_path.is_absolute():
        init_cache_path = (Path.cwd() / init_cache_path).resolve()
    if not init_cache_path.exists():
        _vprint(config, f"Initial summary cache file not found: {init_cache_path}")
        return

    initial_cache = _load_file_summary_cache(init_cache_path)
    _save_file_summary_cache(cache_path, initial_cache)
    _vprint(config, f"Initialized file summary cache from {init_cache_path}")


def _relevance_score(rel_path: str) -> int:
    """Heuristic relevance score for training/config related files."""
    lower = rel_path.lower()
    score = 0
    keywords = [
        "train",
        "trainer",
        "model",
        "config",
        "optim",
        "loss",
        "data",
        "dataset",
        "experiment",
        "engine",
        "main",
    ]
    for keyword in keywords:
        if keyword in lower:
            score += 3

    if lower.endswith(".py"):
        score += 2
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        score += 1
    if "/tests/" in lower or lower.startswith("tests/"):
        score -= 10
    return score


def _make_minimax_client(config: LoopConfig) -> MiniMaxMCPClient:
    """Construct a MiniMax MCP client from loop config."""
    api_key = config.openai_api_key or os.getenv(config.openai_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {config.openai_api_key_env}")
    return MiniMaxMCPClient(
        base_url=config.openai_api_base,
        api_key=api_key,
        model_name=config.model_name,
        mcp_base_dir=config.target_repo,
        wandb_entity=config.wandb_entity,
    )


def _summarize_file_for_prompt(config: LoopConfig, client: MiniMaxMCPClient, rel_path: str, content: str) -> str:
    """Generate an LLM summary for a single file for ideation grounding."""
    summary_prompt = (
        "Summarize this code file for experiment ideation. "
        "Return only JSON with schema: "
        '{"summary":"...", "main_points":["...", "..."]}. '
        "Keep summary concise and focused on trainable behavior and tunable knobs.\n\n"
        f"Path: {rel_path}\n"
        f"Content:\n{content}\n"
    )
    raw = client.make_query(
        query=summary_prompt,
        system_prompt=(
            "You are a strict code summarizer. Return valid JSON only, no markdown, no prose outside JSON."
        ),
    )
    payload = _extract_json_object(raw)
    summary = str(payload.get("summary", "")).strip()
    points = payload.get("main_points", [])
    if isinstance(points, list):
        point_items = [str(item).strip() for item in points if str(item).strip()]
    else:
        point_items = []

    if not summary:
        summary = "(no summary produced)"
    if not point_items:
        point_items = ["(no main points produced)"]

    rendered_points = "\n".join(f"- {point}" for point in point_items[:8])
    return f'<file_summary path="{rel_path}">\nsummary: {summary}\nmain_points:\n{rendered_points}\n</file_summary>'


def _build_file_summaries_for_prompt(config: LoopConfig, repo_root: Path, files_to_edit: list[str]) -> str:
    """Summarize relevant editable files with LLM and return tagged summary blocks."""
    if not config.include_file_summaries:
        return ""
    if not files_to_edit:
        return ""

    ranked_files: list[tuple[int, str, Path]] = []
    for file_str in files_to_edit:
        path = Path(file_str)
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            rel = path.name
        ranked_files.append((_relevance_score(rel), rel, path))

    ranked_files.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = ranked_files[: max(1, config.max_summary_files)]
    _vprint(config, f"Preparing LLM summaries for {len(selected)} relevant files.")

    output_root = _resolve_output_dir(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / config.file_summary_cache_name
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

    with _file_lock(lock_path):
        _initialize_file_summary_cache_if_needed(config, output_root)

    client = _make_minimax_client(config)
    blocks: list[str] = []
    for _, rel_path, file_path in selected:
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as error:
            _vprint(config, f"Skipping summary for {rel_path}: {error}")
            continue

        digest = _sha256_text(text)
        with _file_lock(lock_path):
            cache = _load_file_summary_cache(cache_path)
            cache_files = cache.get("files")
            if not isinstance(cache_files, dict):
                cache_files = {}
                cache["files"] = cache_files
            cached_entry = cache_files.get(rel_path)
            if isinstance(cached_entry, dict):
                cached_digest = cached_entry.get("sha256")
                cached_block = cached_entry.get("summary_block")
                if cached_digest == digest and isinstance(cached_block, str) and cached_block.strip():
                    _vprint(config, f"Summary cache hit for {rel_path}")
                    blocks.append(cached_block)
                    continue

        truncated = _truncate_for_model(text, config.max_file_summary_chars)
        _vprint(
            config,
            f"Summarizing file {rel_path} (original_chars={len(text)}, sent_chars={len(truncated)})",
        )
        try:
            block = _summarize_file_for_prompt(config, client, rel_path, truncated)
            with _file_lock(lock_path):
                cache = _load_file_summary_cache(cache_path)
                cache_files = cache.get("files")
                if not isinstance(cache_files, dict):
                    cache_files = {}
                    cache["files"] = cache_files

                # Another worker may have summarized and written this file while we were querying the model.
                cached_entry = cache_files.get(rel_path)
                if isinstance(cached_entry, dict):
                    cached_digest = cached_entry.get("sha256")
                    cached_block = cached_entry.get("summary_block")
                    if cached_digest == digest and isinstance(cached_block, str) and cached_block.strip():
                        blocks.append(cached_block)
                        continue

                cache_files[rel_path] = {
                    "sha256": digest,
                    "summary_block": block,
                    "updated_at": _utc_now_iso(),
                }
                _save_file_summary_cache(cache_path, cache)
            blocks.append(block)
        except Exception as error:
            _vprint(config, f"Summary generation failed for {rel_path}: {error}")

    _vprint(config, f"File summary cache synchronized via lock at {cache_path}")

    return "\n\n".join(blocks)


def _build_ideation_prompt(
    seed_idea: str,
    user_priority_prompt: str,
    codebase_glimpse: str,
    experiments_history: str,
    file_summaries: str,
    improvement_brief: str,
    prior_stage_ideas: str,
    memory_notes: str,
    ideas_count: int,
    websearch_ideas: str = "",
) -> str:
    """Build the user prompt for websearch-grounded idea generation."""
    ideas_count = max(1, int(ideas_count))
    summary_section = file_summaries if file_summaries.strip() else "(no file summaries available)"
    improvement_section = (
        improvement_brief.strip() if improvement_brief.strip() else "(no improvement brief available)"
    )
    prior_stage_section = (
        prior_stage_ideas.strip() if prior_stage_ideas.strip() else "(no prior stage ideas available)"
    )
    memory_section = memory_notes.strip() if memory_notes.strip() else "(no memory.md notes available)"
    websearch_section = (
        websearch_ideas.strip() if websearch_ideas.strip() else "(no dedicated websearch-idea shortlist available)"
    )
    user_priority_section = (
        user_priority_prompt.strip() if user_priority_prompt.strip() else "(no user-priority prompt configured)"
    )
    return (
        "You are an autonomous ML scientist improving an existing training codebase. "
        "Use web research if needed and produce high-value experimental different SoTA ideas that have shown to improve performance or efficiency. "
        "Honor these constraints: Modify training scripts - this is the only files you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc. and Config parameters, avoid validation/evaluation harness, "
        ", prefer simple changes when gains are similar.\n\n"
        "Treat the user-priority ideas section below as high-priority guidance. "
        "Incorporate it directly into the generated ideas whenever feasible.\n\n"
        f"Generate exactly {ideas_count} different idea(s).\n"
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "ideas": [\n'
        "    {\n"
        '      "idea_title": "short title",\n'
        '      "idea_branch_name": "short_branch_slug",\n'
        '      "rough_outline": "concise technical outline",\n'
        '      "aider_plan_prompt": "detailed prompt for aider planning",\n'
        '      "aider_impl_prompt": "implementation prompt for aider"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Seed input idea:\n{seed_idea}\n\n"
        f"Codebase file glimpse:\n{codebase_glimpse}\n\n"
        f"Relevant file summaries (LLM-generated):\n{summary_section}\n\n"
        f"Long-term experiment memory from memory.md:\n{memory_section}\n\n"
        f"Dedicated websearch-backed idea shortlist:\n{websearch_section}\n\n"
        f"Improvement brief from prior experiment summaries + websearch:\n{improvement_section}\n\n"
        f"Stage-level improvement ideas from previous stage(s):\n{prior_stage_section}\n\n"
        f"User-priority ideas/instructions (take precedence):\n{user_priority_section}\n\n"
        # f"Past experiments from experiments.md:\n{history}\n"
    )


def _build_websearch_idea_prompt(
    seed_idea: str,
    user_priority_prompt: str,
    codebase_glimpse: str,
    experiments_history: str,
    file_summaries: str,
    improvement_brief: str,
    prior_stage_ideas: str,
    memory_notes: str,
    ideas_count: int = 10,
    expert_str: str = "",
) -> str:
    """Build a dedicated websearch prompt that returns distinct online-backed ideas."""
    ideas_count = max(1, int(ideas_count))
    history = experiments_history[-12000:] if experiments_history else "(no prior experiments yet)"
    summary_section = file_summaries if file_summaries.strip() else "(no file summaries available)"
    improvement_section = (
        improvement_brief.strip() if improvement_brief.strip() else "(no improvement brief available)"
    )
    prior_stage_section = (
        prior_stage_ideas.strip() if prior_stage_ideas.strip() else "(no prior stage ideas available)"
    )
    memory_section = memory_notes.strip() if memory_notes.strip() else "(no memory.md notes available)"
    user_priority_section = (
        user_priority_prompt.strip() if user_priority_prompt.strip() else "(no user-priority prompt configured)"
    )

    if not expert_str:
        expert_str = (
            "You are an autonomous ML scientist doing web-research-first ideation for an existing training codebase. "
        )

    return (
        f"{expert_str}"
        "Search the web for strong practical ideas (papers, repos, docs, benchmark reports, engineering writeups) "
        "that are likely to improve this codebase.\n\n"
        "Generate ideas that are mutually distinct, and avoid near-duplicates, also look at the memory and avoid duplicates, but in case there are promising directions, you may explore similar ideas if they have distinct evidence and distinctness notes.\n\n"
        f"Return exactly {ideas_count} distinct ideas.\n"
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "ideas": [\n'
        "    {\n"
        '      "idea_title": "short title",\n'
        '      "idea_branch_name": "short_branch_slug",\n'
        '      "idea_details": "description of the idea in 3-4 sentences in more detail",\n'
        '      "rough_outline": "concise technical proposal tailored to this problem / codebase",\n'
        '      "online_evidence": ["paper/repo/doc/writeup pointers"],\n'
        '      "why_it_is_distinct": "one sentence describing how this differs from the other proposed ideas"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Seed input idea:\n{seed_idea}\n\n"
        f"Codebase file glimpse:\n{codebase_glimpse}\n\n"
        f"Relevant file summaries (LLM-generated):\n{summary_section}\n\n"
        f"Long-term experiment memory from memory.md:\n{memory_section}\n\n"
        f"Improvement brief from prior experiment summaries + websearch:\n{improvement_section}\n\n"
        f"Stage-level improvement ideas from previous stage(s):\n{prior_stage_section}\n\n"
        f"User-priority ideas/instructions (take precedence):\n{user_priority_section}\n\n"
        f"Past experiments from experiments.md:\n{history}\n"
    )


def _format_websearch_ideas_for_ideation_context(payload: dict[str, Any], max_items: int = 10) -> str:
    """Format websearch idea payload into compact context text for main ideation."""
    items = _extract_idea_payload_list(payload)
    if not items:
        return "(no websearch ideas returned)"

    rows: list[str] = []
    for index, item in enumerate(items[: max(1, int(max_items))], start=1):
        title = str(item.get("idea_title", "")).strip() or f"idea_{index}"
        details = str(item.get("idea_details", "")).strip() or "(no details provided)"
        outline = str(item.get("rough_outline", "")).strip() or "(no outline provided)"
        distinct_note = str(item.get("why_it_is_distinct", "")).strip() or "(distinctness note missing)"
        online_evidence = item.get("online_evidence")
        if isinstance(online_evidence, list):
            evidence_items = [str(entry).strip() for entry in online_evidence if str(entry).strip()]
        else:
            evidence_items = []

        evidence_text = "; ".join(evidence_items[:3]) if evidence_items else "(no online evidence pointers)"
        rows.append(
            "\n".join(
                [
                    f"{index}. {title}",
                    f"   details: {details}",
                    f"   outline: {outline}",
                    f"   distinctness: {distinct_note}",
                    f"   evidence: {evidence_text}",
                ]
            )
        )

    return "\n\n".join(rows)


def _set_ideation_prompt_ideas_count(ideation_prompt: str, ideas_count: int) -> str:
    """Update ideation prompt text so it requests an exact idea count."""
    count = max(1, int(ideas_count))
    return re.sub(
        r"Generate exactly\s+\d+\s+different idea\(s\)\.",
        f"Generate exactly {count} different idea(s).",
        ideation_prompt,
        count=1,
    )


def _extend_ideation_prompt_with_expert(
    ideation_prompt: str,
    expert_name: str,
    expert_description: str,
) -> str:
    """Append expert role instructions to the base ideation prompt."""
    return (
        f"{ideation_prompt}\n\n"
        "Use this additional expert persona for this idea only. "
        "Bias the proposal toward this expert perspective while keeping it practical for this codebase.\n"
        f"Expert: {expert_name}\n"
        f"Expert description:\n{expert_description}\n"
    )


def _parse_expert_markdown(experts_markdown: str) -> list[tuple[str, str]]:
    """Parse top-level '# Heading' expert blocks from markdown text."""
    lines = experts_markdown.splitlines()
    heading_indices: list[int] = []
    for index, line in enumerate(lines):
        if line.startswith("# "):
            heading_indices.append(index)

    specs: list[tuple[str, str]] = []
    for offset, heading_index in enumerate(heading_indices):
        end_index = heading_indices[offset + 1] if offset + 1 < len(heading_indices) else len(lines)
        heading = lines[heading_index][2:].strip()
        description = "\n".join(lines[heading_index + 1 : end_index]).strip()
        if not heading:
            continue
        if not description:
            description = "(no description provided)"
        specs.append((heading, description))
    return specs


def _load_expert_specs(experts_path: Path) -> list[tuple[str, str]]:
    """Load expert specs from expert.md-style markdown file."""
    raw = _read_text_if_exists(experts_path)
    if not raw.strip():
        return []
    return _parse_expert_markdown(raw)


def _extract_idea_payload_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract a normalized list of idea payload objects from model JSON output."""
    ideas_obj = payload.get("ideas")
    if isinstance(ideas_obj, list):
        idea_items = [item for item in ideas_obj if isinstance(item, dict)]
        if idea_items:
            return idea_items
    return [payload]


def _build_improvement_brief_with_websearch(
    config: LoopConfig, experiments_history: str, seed_idea: str, memory_notes: str
) -> str:
    """Use past summaries + websearch to build a prioritized improvement brief for next ideas."""
    history = experiments_history[-24000:] if experiments_history else "(no prior experiments yet)"
    if history.strip() == "(no prior experiments yet)":
        return "(no prior experiments yet)"

    client = _make_minimax_client(config)
    prompt = (
        "Analyze the experiment summaries and metrics below, then use websearch to propose better next ideas. "
        "Focus on feasible changes using common libraries already likely available in ML repos (torch, torchvision, transformers). "
        "Prioritize architecture, optimizer, hyperparameters, batch size, and model size changes.\n\n"
        "Return only JSON with schema:\n"
        "{\n"
        '  "top_improvements": [\n'
        "    {\n"
        '      "title": "short name",\n'
        '      "why": "evidence-based rationale from past runs",\n'
        '      "concrete_changes": ["exact parameter or module adjustments"],\n'
        '      "expected_tradeoffs": "metric/vram/stability",\n'
        '      "search_backing": ["short citations or repo/doc pointers"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Seed goal:\n{seed_idea}\n\n"
        f"Memory notes:\n{memory_notes}\n\n"
        f"Past experiment summaries:\n{history}\n"
    )
    try:
        raw = client.make_query_with_alltools(
            query=prompt,
            system_prompt=(
                "You are a strict ML research planner. Return only valid JSON. "
                "Ground suggestions in prior run outcomes and web findings."
            ),
        )
        parsed = _extract_json_object(raw)
        return json.dumps(parsed, indent=2, ensure_ascii=True)
    except Exception as error:
        return f"(improvement brief generation failed: {error})"


def _build_stage_improvement_ideas_with_websearch(
    config: LoopConfig,
    stage_index: int,
    seed_idea: str,
    stage_baseline_metric: float | None,
    merged_feature_branches: list[str],
    skipped_conflict_feature_branches: list[str],
    stage_outcomes: list[ExperimentOutcome],
    expert_name: str | None = None,
    expert_description: str | None = None,
) -> str:
    """Generate stage-level next-step ideas from all candidate summaries via LLM + websearch."""
    if not stage_outcomes:
        return "(no stage outcomes available for stage improvement synthesis)"

    def _metric_history_summary(histories: dict[str, list[float]] | None) -> str:
        if not isinstance(histories, dict) or not histories:
            return "(no metric histories)"

        summary_rows: list[str] = []
        for metric_name, values in histories.items():
            if not isinstance(values, list) or not values:
                continue
            summary_rows.append(
                (
                    f"{metric_name}: steps={len(values)}, first={values[0]:.6f}, "
                    f"last={values[-1]:.6f}, best={max(values):.6f}"
                )
            )
        if not summary_rows:
            return "(no metric histories)"
        return "; ".join(summary_rows)

    compact_rows: list[str] = []
    for outcome in stage_outcomes:
        compact_rows.append(
            "\n".join(
                [
                    f"- idea_title: {outcome.idea_title}",
                    f"  status: {outcome.status}",
                    f"  kept: {outcome.kept}",
                    f"  unified_metric: {outcome.unified_metric}",
                    f"  baseline_metric: {outcome.baseline_metric}",
                    f"  metric_delta: {outcome.metric_delta}",
                    (
                        "  metric_histories: "
                        f"{_tail_text(json.dumps(outcome.metric_histories or {}, ensure_ascii=True), max_chars=5000)}"
                    ),
                    f"  metric_history_trends: {_metric_history_summary(outcome.metric_histories)}",
                    f"  avg_gpu_util: {outcome.avg_gpu_util}",
                    f"  avg_gpu_memory_pct: {outcome.avg_gpu_memory}",
                    f"  error_log_excerpt: {outcome.err_log_excerpt or '(no stderr excerpt)'}",
                    f"  summary: {_tail_text(outcome.summary, max_chars=2500)}",
                ]
            )
        )

    stage_payload = "\n\n".join(compact_rows)
    stage_payload = _truncate_for_model(stage_payload, max_chars=28000)
    expert_block = ""
    if expert_name and expert_description:
        expert_block = (
            "\nUse this additional expert persona while proposing improvements. "
            "Bias recommendations toward this perspective while keeping them practical for this codebase.\n"
            f"Expert: {expert_name}\n"
            f"Expert description:\n{expert_description}\n"
        )
    client = _make_minimax_client(config)
    prompt = (
        "You are analyzing a completed autonomous ML experimentation stage. "
        "Use the provided candidate summaries plus websearch to propose concrete, prioritized next-stage ideas.\n\n"
        "Must explicitly consider and compare:\n"
        "- architecture changes\n"
        "- optimizer changes\n"
        "- hyperparameter schedules\n"
        "- batch size / gradient accumulation\n"
        "- model size and capacity trade-offs\n"
        "Also search for SotA or strong common-repo approaches in torch / torchvision / transformers ecosystems.\n\n"
        "Return only JSON with schema:\n"
        "{\n"
        '  "stage_takeaways": ["..."],\n'
        '  "top_improvements": [\n'
        "    {\n"
        '      "title": "short actionable title",\n'
        '      "category": "architecture|optimizer|hyperparameters|batch|model_size|other",\n'
        '      "why_now": "grounded in this stage outcomes",\n'
        '      "proposed_changes": ["specific config/code edits"],\n'
        '      "risk_or_tradeoff": "compute/stability/memory",\n'
        '      "web_backing": ["repo/doc/paper pointers"]\n'
        "    }\n"
        "  ],\n"
        '  "next_stage_prompt_hints": ["phrases to include in next ideation"]\n'
        "}\n\n"
        f"Stage index: {stage_index}\n"
        f"Seed goal: {seed_idea}\n"
        f"Stage baseline metric: {stage_baseline_metric}\n"
        f"Merged branches: {merged_feature_branches}\n"
        f"Conflict-skipped branches: {skipped_conflict_feature_branches}\n\n"
        f"Candidate outcomes and summaries:\n{stage_payload}\n"
        f"{expert_block}"
    )
    try:
        raw = client.make_query_with_alltools(
            query=prompt,
            system_prompt=(
                f"You are a strict ML stage planner. Return valid JSON only and ground suggestions in the provided data plus web findings. \n{expert_block}"
            ),
        )
        parsed = _extract_json_object(raw)
        return json.dumps(parsed, indent=2, ensure_ascii=True)
    except Exception as error:
        return f"(stage improvement synthesis failed: {error})"


def _update_stage_memory_with_llm(
    config: LoopConfig,
    memory_path: Path,
    stage_index: int,
    stage_summary_text: str,
    stage_outcomes: list[ExperimentOutcome],
    expert_name: Optional[str] = None,
    expert_description: Optional[str] = None,
) -> None:
    """Update output-dir memory.md using stage summary and per-candidate outcomes."""
    existing_memory = _read_text_if_exists(memory_path).strip()
    existing_memory_section = existing_memory if existing_memory else "(memory.md is currently empty)"

    expert_block = ""
    if expert_name and expert_description:
        expert_block = (
            "\nUse this additional expert persona creating the memory. "
            f"Expert: {expert_name}\n"
            f"Expert description:\n{expert_description}\n"
        )

    outcome_rows: list[str] = []
    for outcome in stage_outcomes:
        outcome_rows.append(
            "\n".join(
                [
                    f"- idea_title: {outcome.idea_title}",
                    f"  status: {outcome.status}",
                    f"  kept: {outcome.kept}",
                    f"  unified_metric: {outcome.unified_metric}",
                    f"  baseline_metric: {outcome.baseline_metric}",
                    f"  metric_delta: {outcome.metric_delta}",
                    f"  avg_gpu_util: {outcome.avg_gpu_util}",
                    f"  avg_gpu_memory_pct: {outcome.avg_gpu_memory}",
                    f"  summary: {_tail_text(outcome.summary, max_chars=2500)}",
                ]
            )
        )
    outcomes_section = "\n\n".join(outcome_rows) if outcome_rows else "(no stage outcomes available)"

    prompt = (
        "Update the experiment memory document using the new stage evidence. "
        "The memory should help future ideation focus on promising directions and avoid repeated dead ends.\n\n"
        "Return markdown only (no code fences) with these sections:\n"
        "# Experiment Memory\n"
        "## What We Tried\n"
        "## What Looks Promising\n"
        "## What Looks Less Promising\n"
        "## Open Hypotheses To Try Next\n"
        "## Operational Guardrails\n"
        "## Short Next-Stage Prompt Hints\n\n"
        "Constraints:\n"
        "- Keep content concise and specific.\n"
        "- Ground statements in evidence from stage outcomes.\n"
        "- Preserve still-valid prior memory, but remove contradictions when new evidence disproves old notes.\n"
        "- Mention uncertain items explicitly as hypotheses, not facts.\n\n"
        f"Stage index: {stage_index}\n"
        f"Stage summary:\n{stage_summary_text}\n\n"
        f"Stage run summaries:\n{outcomes_section}\n\n"
        f"Existing memory.md:\n{existing_memory_section}\n"
        f"{expert_block}"
    )

    try:
        client = _make_minimax_client(config)
        response = client.make_query(
            query=prompt,
            system_prompt=(
                "You are a precise ML experimentation memory manager. "
                "Return markdown only and keep it practical for future ideation."
                f"{expert_block}"
            ),
        )
        updated_memory = _strip_markdown_fence(response)
        if not updated_memory:
            _vprint(config, f"Memory update for stage {stage_index} returned empty content; keeping prior memory.md")
            return
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(f"{updated_memory}\n", encoding="utf-8")
        _vprint(config, f"Updated experiment memory at {memory_path}")
    except Exception as error:
        _vprint(config, f"Failed to update memory.md for stage {stage_index}: {error}")


def _query_ideation_json(config: LoopConfig, prompt: str, expert_str: str = "") -> dict[str, Any]:
    """Run ideation through MCP websearch client and parse JSON output."""
    client = _make_minimax_client(config)
    _vprint(config, f"Running ideation with model={config.model_name} base={config.openai_api_base}")
    _vprint(config, f"Ideation prompt preview:\n{_tail_text(prompt, max_chars=6000)}")

    if expert_str:
        system_prompt = f"You are a \n{expert_str}\nReturn only strict JSON and avoid markdown wrappers."
    else:
        system_prompt = "You are a precise research engineer. Return only strict JSON and avoid markdown wrappers."

    response = client.make_query_with_alltools(
        query=prompt,
        system_prompt=system_prompt,
    )
    _vprint(config, f"Raw ideation response:\n{_tail_text(response, max_chars=6000)}")
    return _extract_json_object(response)


def _query_stage_ideation_payloads(
    config: LoopConfig,
    ideation_prompt: str,
    ideas_per_stage: int,
    expert_specs: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Return stage idea payloads using batched prompting by default."""
    if ideas_per_stage < 1:
        return []

    parsed_payloads: list[dict[str, Any]] = []
    specs = list(expert_specs or [])

    # First M ideas are sampled one-by-one with explicit expert context.
    expert_count = min(len(specs), ideas_per_stage)
    for expert_index in range(expert_count):
        expert_name, expert_description = specs[expert_index]
        single_prompt = _set_ideation_prompt_ideas_count(ideation_prompt, 1)
        expert_prompt = _extend_ideation_prompt_with_expert(single_prompt, expert_name, expert_description)
        raw_payload = _query_ideation_json(config, expert_prompt)
        parsed_payloads.append(_extract_idea_payload_list(raw_payload)[0])

    remaining_ideas = ideas_per_stage - len(parsed_payloads)
    if remaining_ideas <= 0:
        return parsed_payloads[:ideas_per_stage]

    # In expert mode, generate all leftover ideas in one normal batched run.
    if specs:
        remaining_prompt = _set_ideation_prompt_ideas_count(ideation_prompt, remaining_ideas)
        payload = _query_ideation_json(config, remaining_prompt)
        batch_payloads = _extract_idea_payload_list(payload)
        if not batch_payloads:
            batch_payloads = [payload]
        if len(batch_payloads) < remaining_ideas:
            _vprint(
                config,
                (
                    f"Expert mode batched ideation returned {len(batch_payloads)} ideas; "
                    f"backfilling to {remaining_ideas} using single prompts."
                ),
            )
            single_prompt = _set_ideation_prompt_ideas_count(ideation_prompt, 1)
            while len(batch_payloads) < remaining_ideas:
                single_payload = _query_ideation_json(config, single_prompt)
                batch_payloads.append(_extract_idea_payload_list(single_payload)[0])
        parsed_payloads.extend(batch_payloads[:remaining_ideas])
        return parsed_payloads[:ideas_per_stage]

    if config.stage_multi_ideation_prompts:
        for _ in range(ideas_per_stage):
            raw_payload = _query_ideation_json(config, ideation_prompt)
            parsed_payloads.append(_extract_idea_payload_list(raw_payload)[0])
        return parsed_payloads

    payload = _query_ideation_json(config, ideation_prompt)
    parsed_payloads = _extract_idea_payload_list(payload)

    if not parsed_payloads:
        return [payload]

    if len(parsed_payloads) < ideas_per_stage:
        _vprint(
            config,
            (
                f"Batched ideation returned {len(parsed_payloads)} ideas; "
                f"backfilling to {ideas_per_stage} using single prompts."
            ),
        )
        while len(parsed_payloads) < ideas_per_stage:
            parsed_payloads.append(_query_ideation_json(config, ideation_prompt))

    return parsed_payloads[:ideas_per_stage]


def _normalize_idea_payload(idea_payload: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Return normalized idea fields from a raw ideation payload."""

    if "ideas" in idea_payload:
        first_idea = (
            idea_payload["ideas"][0] if isinstance(idea_payload["ideas"], list) and idea_payload["ideas"] else {}
        )
        return _normalize_idea_payload(first_idea)

    idea_title = str(idea_payload.get("idea_title", "autonomous_experiment"))
    idea_branch_name = str(idea_payload.get("idea_branch_name", _slugify(idea_title)))
    idea_outline = str(idea_payload.get("rough_outline", ""))
    aider_plan_prompt = str(idea_payload.get("aider_plan_prompt", "Create a technical implementation plan."))
    aider_impl_prompt = str(idea_payload.get("aider_impl_prompt", "Implement the approved plan now."))
    return idea_title, idea_branch_name, idea_outline, aider_plan_prompt, aider_impl_prompt
