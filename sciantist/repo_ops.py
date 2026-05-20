"""Git and worktree operations used by the autonomous loop."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from loguru import logger

from sciantist.config import LoopConfig
from sciantist.state import _file_lock


def _run_command(command: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    """Run a command and return its captured output."""
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)


def _git(repo: str, *args: str) -> str:
    """Run a git command in a repository and return stdout."""
    result = _run_command(["git", *args], cwd=repo)
    return result.stdout.strip()


def _git_try(repo: str, *args: str) -> tuple[bool, str]:
    """Run a git command and capture success plus output/error."""
    try:
        output = _git(repo, *args)
        return True, output
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else str(error)
        return False, stderr


def _commit_tree_id(repo_path: str | Path, commit: str) -> str:
    """Return tree object id for a commit, or empty string when unavailable."""
    commit_ref = str(commit).strip()
    if not commit_ref:
        return ""
    ok, output = _git_try(str(repo_path), "rev-parse", f"{commit_ref}^{{tree}}")
    if not ok:
        return ""
    return output.strip()


def _push_to_origin(repo: str, ref: str | None = None) -> tuple[bool, str]:
    """Push a branch or refspec to origin. Returns (success, output)."""
    if ref is None:
        push_cmd = ["git", "push", "origin", "--all"]
    else:
        push_cmd = ["git", "push", "origin", ref]
    try:
        result = _run_command(push_cmd, cwd=repo)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else str(error)
        return False, stderr


def _ensure_repo_ready(config: LoopConfig) -> None:
    """Validate repository state and move to the experiment branch."""
    repo_path = Path(config.target_repo)
    if not repo_path.exists():
        raise RuntimeError(f"Target repository does not exist: {config.target_repo}")

    success, _ = _git_try(config.target_repo, "rev-parse", "--is-inside-work-tree")
    if not success:
        raise RuntimeError(f"Not a git repository: {config.target_repo}")

    dirty = _git(config.target_repo, "status", "--porcelain")
    if dirty and not config.allow_dirty_repo:
        raise RuntimeError("Target repository has uncommitted changes. Set --allow-dirty to continue.")

    start_out_branch = config.start_out_branch.strip()
    if not start_out_branch:
        raise RuntimeError("start_out_branch must not be empty.")

    start_out_exists, _ = _git_try(config.target_repo, "rev-parse", "--verify", start_out_branch)
    if start_out_exists:
        _git(config.target_repo, "checkout", start_out_branch)
    else:
        _git(config.target_repo, "checkout", "-b", start_out_branch)

    if config.branch_name == start_out_branch:
        return

    branch_exists, _ = _git_try(config.target_repo, "rev-parse", "--verify", config.branch_name)
    if branch_exists:
        _git(config.target_repo, "checkout", config.branch_name)
    else:
        _git(config.target_repo, "checkout", "-b", config.branch_name, start_out_branch)


def _resolve_train_command_for_repo(train_command: str, source_repo: str, target_repo: str) -> str:
    """Remap first command token from source repo root to target repo root.

    Rules:
    - If first token is relative, prefix it with target_repo.
    - If first token is absolute and starts with source_repo, replace that prefix with target_repo.
    - Otherwise keep first token unchanged.
    """
    source_root = str(Path(source_repo).expanduser().resolve())
    target_root = str(Path(target_repo).expanduser().resolve())

    try:
        tokens = shlex.split(train_command, posix=True)
    except ValueError:
        return train_command
    if not tokens:
        return train_command

    first_token = tokens[0]
    first_path = Path(first_token).expanduser()
    remapped_first = first_token

    if not first_path.is_absolute():
        remapped_first = str(Path(target_root) / first_path)
    else:
        first_abs = str(first_path.resolve())
        if first_abs == source_root:
            remapped_first = target_root
        else:
            prefix = source_root + "/"
            if first_abs.startswith(prefix):
                suffix = first_abs[len(prefix) :]
                remapped_first = str(Path(target_root) / suffix)

    tokens[0] = remapped_first
    return shlex.join(tokens)


def _candidate_worktree_path(output_root: Path, stage_index: int, candidate_index: int) -> Path:
    """Return deterministic worktree path for a stage candidate."""
    return output_root / "worktrees" / f"stage_{stage_index:04d}" / f"candidate_{candidate_index:02d}"


def _create_candidate_worktree(
    main_repo: str,
    worktree_path: Path,
    feature_branch: str,
    baseline_commit: str,
) -> Path:
    """Create an isolated worktree and branch for a candidate."""
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    branch_exists, _ = _git_try(main_repo, "rev-parse", "--verify", feature_branch)
    if branch_exists:
        _run_command(
            [
                "git",
                "worktree",
                "add",
                "--force",
                str(worktree_path),
                feature_branch,
            ],
            cwd=main_repo,
        )
    else:
        _run_command(
            [
                "git",
                "worktree",
                "add",
                "--force",
                "-b",
                feature_branch,
                str(worktree_path),
                baseline_commit,
            ],
            cwd=main_repo,
        )
    return worktree_path


def _cleanup_candidate_worktree(main_repo: str, worktree_path: Path) -> None:
    """Remove a candidate worktree and prune stale entries."""
    _git_try(main_repo, "worktree", "remove", "--force", str(worktree_path))
    _git_try(main_repo, "worktree", "prune")


def _extract_diff_summary(repo: str, baseline_commit: str, trial_commit: str) -> str:
    """Summarize changed files and line stats between two commits."""
    success, files_output = _git_try(repo, "diff", "--name-only", baseline_commit, trial_commit)
    changed_files = files_output.splitlines() if success and files_output else []

    success, stat_output = _git_try(repo, "diff", "--shortstat", baseline_commit, trial_commit)
    shortstat = stat_output if success else ""
    files_rendered = ", ".join(changed_files[:12]) if changed_files else "(no changed files)"
    if len(changed_files) > 12:
        files_rendered += ", ..."
    return f"Files: {files_rendered}. Stats: {shortstat or 'n/a'}."


def _extract_diff_patch(repo: str, baseline_commit: str, trial_commit: str, max_chars: int = 24000) -> str:
    """Return a compact diff patch between two commits for model analysis."""
    if baseline_commit == trial_commit:
        return "(no code changes)"

    success, patch = _git_try(repo, "diff", "--no-color", "--unified=0", baseline_commit, trial_commit)
    if not success:
        return f"(failed to load diff patch: {patch})"
    if not patch.strip():
        return "(no textual diff available)"
    if len(patch) <= max_chars:
        return patch
    half = max_chars // 2
    return f"{patch[:half]}\n\n... [truncated] ...\n\n{patch[-half:]}"


def _commit_all(repo: str, message: str) -> str:
    """Stage all changes and create a commit, returning its hash."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _commit_if_dirty(repo: str, message: str) -> str | None:
    """Commit all changes if the worktree is dirty, else return None."""
    dirty = _git(repo, "status", "--porcelain")
    if not dirty:
        return None
    return _commit_all(repo, message)


def _sanitize_branch_token(value: str) -> str:
    """Sanitize arbitrary text into a git-branch-safe token."""
    token = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not token:
        token = "idea"
    return token[:40]


def _make_feature_branch_name(idea_branch_name: str, suffix: str = "") -> str:
    """Create feature branch name using feat/branchname_yymmdd_hhmmss[_suffix]."""
    token = _sanitize_branch_token(idea_branch_name)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    suffix_token = _sanitize_branch_token(suffix) if suffix else ""
    if suffix_token:
        return f"feat/{token}_{stamp}_{suffix_token}"
    return f"feat/{token}_{stamp}"


def _checkout_feature_branch(repo: str, base_branch: str, feature_branch: str, base_commit: str) -> None:
    """Create and checkout a fresh feature branch from a base commit."""
    _git(repo, "checkout", base_branch)
    _git(repo, "checkout", "-b", feature_branch, base_commit)


def _merge_feature_into_base(repo: str, base_branch: str, feature_branch: str) -> str:
    """Merge feature branch back into base branch and return new HEAD."""
    _git(repo, "checkout", base_branch)
    ff_ok, _ = _git_try(repo, "merge", "--ff-only", feature_branch)
    if not ff_ok:
        _git(repo, "merge", "--no-ff", "-m", f"merge: {feature_branch}", feature_branch)
    return _git(repo, "rev-parse", "HEAD")


def _attempt_merge_on_base_branch(
    config: LoopConfig,
    lock_path: Path,
    feature_branch: str,
) -> tuple[bool, str]:
    """Acquire global merge lock, attempt merge, and reset on failure."""
    with _file_lock(lock_path):
        _git(config.target_repo, "checkout", config.branch_name)
        pre_merge_commit = _git(config.target_repo, "rev-parse", "HEAD")

        ff_ok, ff_output = _git_try(config.target_repo, "merge", "--ff-only", feature_branch)
        if ff_ok:
            merged_commit = _git(config.target_repo, "rev-parse", "HEAD")
            return True, merged_commit

        noff_ok, noff_output = _git_try(
            config.target_repo,
            "merge",
            "--no-ff",
            "-m",
            f"merge: {feature_branch}",
            feature_branch,
        )
        if noff_ok:
            merged_commit = _git(config.target_repo, "rev-parse", "HEAD")
            return True, merged_commit

        _git_try(config.target_repo, "merge", "--abort")
        _git_try(config.target_repo, "reset", "--hard", pre_merge_commit)
        logger.warning(f"Merge failed for {feature_branch}: {noff_output or ff_output}")
        return False, ""
