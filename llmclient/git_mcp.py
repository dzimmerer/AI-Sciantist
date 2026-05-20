"""MCP server exposing read-only Git inspection tools."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

REPO_DIR = Path(os.getenv("GIT_MCP_REPO_DIR", ".")).resolve()

mcp = FastMCP("Git Repository Explorer")


def _normalize_repo_path(repo_relative_path: str) -> str:
    """Return a safe repository-relative path for Git commands."""
    normalized = repo_relative_path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("repo_relative_path must not be empty")

    candidate = (REPO_DIR / normalized).resolve()
    if not candidate.is_relative_to(REPO_DIR):
        raise ValueError("Path traversal outside repository is not allowed")
    return candidate.relative_to(REPO_DIR).as_posix()


def _run_git(args: list[str], max_chars: int = 120000, timeout_seconds: int = 60) -> dict[str, Any]:
    """Run a git command in REPO_DIR and return structured output."""
    command = ["git", "--no-pager", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, timeout_seconds),
            check=False,
        )
    except FileNotFoundError as error:
        return {"error": f"Git executable not found: {error}"}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout_seconds} seconds", "command": command}

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if len(stdout) > max_chars:
        stdout = stdout[-max_chars:]
    if len(stderr) > max_chars:
        stderr = stderr[-max_chars:]

    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _ensure_repo() -> dict[str, Any] | None:
    """Validate REPO_DIR points to a Git work tree."""
    probe = _run_git(["rev-parse", "--is-inside-work-tree"], max_chars=4000)
    if probe.get("error"):
        return probe
    if probe.get("exit_code") != 0:
        return {
            "error": f"Path is not a Git repository: {REPO_DIR}",
            "details": probe,
        }
    return None


@mcp.tool()
def git_repo_info() -> dict[str, Any]:
    """Return repository metadata including root path and current branch."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error

    top_level = _run_git(["rev-parse", "--show-toplevel"], max_chars=4000)
    branch = _run_git(["branch", "--show-current"], max_chars=4000)
    head = _run_git(["rev-parse", "HEAD"], max_chars=4000)

    return {
        "repo_dir": str(REPO_DIR),
        "top_level": (top_level.get("stdout") or "").strip(),
        "current_branch": (branch.get("stdout") or "").strip(),
        "head": (head.get("stdout") or "").strip(),
    }


@mcp.tool()
def git_status(short: bool = True) -> dict[str, Any]:
    """Get git status output."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    args = ["status", "--short"] if short else ["status"]
    return _run_git(args)


@mcp.tool()
def git_log(max_count: int = 20, ref: str = "HEAD") -> dict[str, Any]:
    """Get commit history in one-line format."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    normalized_count = max(1, min(max_count, 200))
    return _run_git(["log", "--oneline", f"--max-count={normalized_count}", ref])


@mcp.tool()
def git_log_patch(max_count: int = 5, ref: str = "HEAD") -> dict[str, Any]:
    """Get commit history including patches, equivalent to git log -p."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    normalized_count = max(1, min(max_count, 50))
    return _run_git(["log", "-p", f"--max-count={normalized_count}", ref])


@mcp.tool()
def git_diff(revision_range: str = "HEAD~1..HEAD") -> dict[str, Any]:
    """Get git diff for a revision range."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    normalized_range = revision_range.strip() or "HEAD~1..HEAD"
    return _run_git(["diff", normalized_range])


@mcp.tool()
def git_show(revision: str = "HEAD") -> dict[str, Any]:
    """Show a commit or object details, equivalent to git show <revision>."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    normalized_revision = revision.strip() or "HEAD"
    return _run_git(["show", normalized_revision])


@mcp.tool()
def git_show_file(repo_relative_path: str, revision: str = "HEAD") -> dict[str, Any]:
    """Read a file at a specific revision/branch, equivalent to git show <rev>:<path>."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    try:
        safe_rel_path = _normalize_repo_path(repo_relative_path)
    except Exception as exc:
        return {"error": str(exc)}

    normalized_revision = revision.strip() or "HEAD"
    return _run_git(["show", f"{normalized_revision}:{safe_rel_path}"])


@mcp.tool()
def git_list_branches(all_branches: bool = True) -> dict[str, Any]:
    """List local branches, and optionally remote branches."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    args = ["branch", "-a"] if all_branches else ["branch"]
    return _run_git(args)


@mcp.tool()
def git_list_tags(max_count: int = 200) -> dict[str, Any]:
    """List tags sorted by creation date."""
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    normalized_count = max(1, min(max_count, 2000))
    return _run_git(["tag", "--sort=-creatordate", "--format=%(refname:short)", f"--count={normalized_count}"])


if __name__ == "__main__":
    mcp.run()
