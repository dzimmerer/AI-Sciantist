"""Aider integration helpers for planning and implementation steps."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from sciantist.config import LoopConfig
from sciantist.repo_ops import _commit_if_dirty


def _tail_text(text: str, max_chars: int = 8000) -> str:
    """Return the trailing slice of text for concise logging/prompts."""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _find_traceback_excerpt(log_text: str) -> str:
    """Extract a Python traceback excerpt from a log blob."""
    marker = "Traceback (most recent call last):"
    idx = log_text.rfind(marker)
    if idx == -1:
        return _tail_text(log_text, max_chars=3000)
    return _tail_text(log_text[idx:], max_chars=5000)


def _vprint(config: LoopConfig, message: str) -> None:
    """Print verbose aider logs when verbose mode is enabled."""
    if config.verbose:
        logger.debug(message)


def _load_aider() -> tuple[Any, Any, Any]:
    """Import aider symbols lazily with a clear runtime error."""
    try:
        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model
    except ImportError as error:
        raise RuntimeError("aider-chat is required. Install with: uv sync --group aider") from error
    return Coder, InputOutput, Model


def _configure_llm_env(config: LoopConfig, api_key: str) -> None:
    """Set provider-specific environment variables for the configured model."""
    aider_model = (config.aider_model or "").strip().lower()
    if aider_model.startswith("openai/"):
        os.environ["OPENAI_API_BASE"] = config.openai_api_base.rstrip("/") + "/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        return
    if aider_model.startswith("anthropic/"):
        os.environ["ANTHROPIC_API_KEY"] = api_key
        # Allow custom base URL overrides for Anthropic-compatible gateways.
        os.environ["ANTHROPIC_BASE_URL"] = config.openai_api_base.rstrip("/")
        return
    raise RuntimeError(
        f"Unsupported aider model provider in '{config.aider_model}'. Expected prefix openai/ or anthropic/."
    )


def _detect_repo_root(files_to_edit: list[str], fallback_repo: str) -> Path:
    """Infer git repo root from edited files, falling back to configured repo."""
    fallback = Path(fallback_repo)
    for file_path in files_to_edit:
        candidate = Path(file_path)
        start_dir = candidate if candidate.is_dir() else candidate.parent
        probe = subprocess.run(
            ["git", "-C", str(start_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            resolved = probe.stdout.strip()
            if resolved:
                return Path(resolved)
    return fallback


def _run_with_aider(
    config: LoopConfig,
    files_to_edit: list[str],
    plan_prompt: str,
    impl_prompt: str,
    commit_message: str | None,
) -> None:
    """Execute planning and implementation via aider."""
    Coder, InputOutput, Model = _load_aider()
    io = InputOutput(yes=True)
    model = Model(config.aider_model)
    _vprint(
        config,
        (
            f"Launching aider with model={config.aider_model} over {len(files_to_edit)} files. "
            f"Plan prompt chars={len(plan_prompt)}, impl prompt chars={len(impl_prompt)}"
        ),
    )
    _vprint(config, f"Aider plan prompt:\n{_tail_text(plan_prompt, max_chars=8000)}")
    _vprint(config, f"Aider implementation prompt:\n{_tail_text(impl_prompt, max_chars=8000)}")
    coder = Coder.create(
        main_model=model,
        # editor_model=model,
        edit_format="diff",
        # auto_accept_architect=True,
        io=io,
        fnames=files_to_edit,
        suggest_shell_commands=False,
        auto_lint=True,
        auto_commits=False,
        test_cmd="uv run pytest",
        auto_test=True,
    )
    plan_response = coder.run(
        "Plan and implement the following task: "
        + plan_prompt
        + "\n\n In particular:\n"
        + impl_prompt
        + "\n\n CRITICAL: You must generate the complete code for EVERY file mentioned in the plan. Do not use placeholders like 'rest of code here' or skip files. Output the full required changes."
    )
    if isinstance(plan_response, str):
        _vprint(config, f"Aider response:\n{_tail_text(plan_response, max_chars=12000)}")
    else:
        _vprint(config, f"Aider response:\n{plan_response}")

    if commit_message:
        commit_response = coder.run(f"/commit {commit_message}")
        if isinstance(commit_response, str):
            _vprint(config, f"Aider commit response:\n{_tail_text(commit_response, max_chars=12000)}")


def _run_with_opencode(
    config: LoopConfig,
    files_to_edit: list[str],
    plan_prompt: str,
    impl_prompt: str,
    commit_message: str | None,
) -> None:
    """Execute planning and implementation via OpenCode CLI."""
    _vprint(
        config,
        (
            f"Launching OpenCode CLI over {len(files_to_edit)} files. "
            f"Plan prompt chars={len(plan_prompt)}, impl prompt chars={len(impl_prompt)}"
        ),
    )
    _vprint(config, f"OpenCode plan prompt:\n{_tail_text(plan_prompt, max_chars=8000)}")
    _vprint(config, f"OpenCode implementation prompt:\n{_tail_text(impl_prompt, max_chars=8000)}")

    files_context = ", ".join(files_to_edit)
    full_prompt = (
        f"You must strictly limit your edits to the following files: {files_context}\n\n"
        "Plan and implement the following task:\n"
        f"{plan_prompt}\n\n"
        "In particular:\n"
        f"{impl_prompt}\n\n"
        "CRITICAL: You must generate the complete code for EVERY file mentioned in the plan. "
        "Do not use placeholders like 'rest of code here' or skip files. Output the full required changes.\n\n"
        "AUTOMATIC TESTING: After making your changes, you must run `uv run pytest` to test the codebase. "
        "If the tests fail, automatically debug and fix the issues."
    )

    repo_root = _detect_repo_root(files_to_edit, config.target_repo)
    max_test_fix_attempts = max(0, int(config.max_fix_attempts))
    total_opencode_runs = max_test_fix_attempts + 1
    opencode_prompt = full_prompt
    tests_passed = False

    for run_index in range(total_opencode_runs):
        _vprint(
            config,
            f"OpenCode run {run_index + 1}/{total_opencode_runs} in {repo_root}.",
        )
        cmd = ["opencode", "run", opencode_prompt]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(repo_root),
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "'opencode' command not found. Ensure OpenCode CLI is installed and on PATH."
            ) from error

        run_stdout = result.stdout or ""
        run_stderr = result.stderr or ""
        if result.returncode == 0:
            _vprint(config, f"OpenCode response:\n{_tail_text(run_stdout, max_chars=12000)}")
        else:
            _vprint(
                config,
                (
                    f"OpenCode exited with error code {result.returncode}:\n{_tail_text(run_stderr, max_chars=4000)}"
                    f"\n\nStdout:\n{_tail_text(run_stdout, max_chars=12000)}"
                ),
            )
            raise RuntimeError(f"OpenCode run failed with exit code {result.returncode}")

        test_cmd = ["uv", "run", "pytest"]
        test_result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(repo_root),
        )
        test_stdout = test_result.stdout or ""
        test_stderr = test_result.stderr or ""
        if test_result.returncode == 0:
            _vprint(config, "OpenCode changes passed `uv run pytest`.")
            tests_passed = True
            break

        _vprint(
            config,
            (
                f"Tests failed with exit code {test_result.returncode}:\n{_tail_text(test_stderr, max_chars=4000)}"
                f"\n\nStdout:\n{_tail_text(test_stdout, max_chars=12000)}"
            ),
        )
        if run_index >= max_test_fix_attempts:
            break

        opencode_prompt = (
            f"{full_prompt}\n\n"
            "TEST FAILURE FEEDBACK:\n"
            f"Attempt {run_index + 1} produced failing tests.\n"
            "Use the original task requirements above, then fix only what is required to make tests pass.\n"
            "You must run `uv run pytest` after your edits and ensure all tests pass before finishing.\n\n"
            f"Pytest stderr:\n{_tail_text(test_stderr, max_chars=6000)}\n\n"
            f"Pytest stdout:\n{_tail_text(test_stdout, max_chars=12000)}"
        )

    if not tests_passed:
        raise RuntimeError(f"OpenCode completed, but tests still failed after {total_opencode_runs} run(s).")

    if commit_message:
        add_cmd = ["git", "add", "--", *files_to_edit]
        add_result = subprocess.run(
            add_cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(repo_root),
        )
        if add_result.returncode != 0:
            raise RuntimeError(f"Failed to stage files for commit: {_tail_text(add_result.stderr, max_chars=3000)}")

        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(repo_root),
        )
        commit_stdout = commit_result.stdout or ""
        commit_stderr = commit_result.stderr or ""
        commit_output = f"{commit_stdout}\n{commit_stderr}".lower()
        if commit_result.returncode == 0:
            _vprint(config, f"OpenCode commit response:\n{_tail_text(commit_stdout, max_chars=12000)}")
        elif "nothing to commit" in commit_output or "working tree clean" in commit_output:
            # No-op commit is acceptable when OpenCode made no file changes.
            _vprint(config, "OpenCode made no changes to commit; continuing without creating a commit.")
        else:
            _vprint(
                config,
                (
                    f"OpenCode commit failed with exit code {commit_result.returncode}:\n"
                    f"{_tail_text(commit_stderr, max_chars=4000)}\n\n"
                    f"Stdout:\n{_tail_text(commit_stdout, max_chars=12000)}"
                ),
            )
            raise RuntimeError("OpenCode completed but git commit failed")


def _run_aider_plan_and_impl(
    config: LoopConfig,
    files_to_edit: list[str],
    plan_prompt: str,
    impl_prompt: str,
    commit_message: str | None = None,
) -> None:
    """Execute planning and implementation in non-interactive mode."""
    api_key = config.openai_api_key or os.getenv(config.openai_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {config.openai_api_key_env}")

    _configure_llm_env(config, api_key)

    coder_backend = (config.coder_backend or "aider").strip().lower()
    if coder_backend == "aider":
        _run_with_aider(config, files_to_edit, plan_prompt, impl_prompt, commit_message)
        return
    if coder_backend == "opencode":
        _run_with_opencode(config, files_to_edit, plan_prompt, impl_prompt, commit_message)
        return
    raise RuntimeError(f"Unsupported coder backend: {config.coder_backend!r}. Expected 'aider' or 'opencode'.")


def _try_fix_crash_with_aider(
    config: LoopConfig,
    files_to_edit: list[str],
    idea_title: str,
    err_log: str,
    attempt: int,
) -> str | None:
    """Ask aider to fix crash based on traceback and commit the patch."""
    traceback_excerpt = _find_traceback_excerpt(err_log)
    plan_prompt = (
        "Create a concise plan to fix this training crash without touching validation/evaluation code. "
        "Focus on minimal robust changes.\n\n"
        f"Experiment: {idea_title}\n"
        f"Crash traceback/log:\n{traceback_excerpt}\n"
        f"Error log:\n{err_log[:6000]} \n"
    )
    impl_prompt = (
        "Implement the crash fix now. Respect existing style and avoid unrelated refactors. Do not add dependencies."
    )
    _run_aider_plan_and_impl(config, files_to_edit, plan_prompt, impl_prompt)
    return _commit_if_dirty(config.target_repo, f"fix: crash repair attempt {attempt}")


def _resolve_merge_conflicts_with_aider(
    config: LoopConfig,
    candidate_repo: Path,
    conflict_paths: list[str],
    feature_branch: str,
    merge_error_detail: str,
) -> None:
    """Resolve merge-conflicted files in a worktree using aider."""
    if not conflict_paths:
        raise RuntimeError(f"Merge failed before conflict resolution for {feature_branch}: {merge_error_detail}")

    files_to_edit = [str(candidate_repo / rel_path) for rel_path in conflict_paths]
    plan_prompt = (
        "Resolve this git merge conflict between base branch and feature branch. "
        "Preserve working behavior and keep the feature branch intent when conflicts occur.\n\n"
        f"Base branch: {config.branch_name}\n"
        f"Feature branch: {feature_branch}\n"
        f"Conflicted files: {', '.join(conflict_paths)}"
    )
    impl_prompt = (
        "Resolve all merge markers and produce a clean buildable merge result. "
        "Prefer incoming feature-branch logic for directly conflicting hunks unless it clearly breaks the base flow."
    )
    _run_aider_plan_and_impl(
        config,
        files_to_edit,
        plan_prompt,
        impl_prompt,
        commit_message=f"resolve merge conflict: {feature_branch}",
    )
