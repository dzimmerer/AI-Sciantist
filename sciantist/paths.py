"""Path and run-id helper utilities for worker orchestration."""

from __future__ import annotations

import uuid
from pathlib import Path

from sciantist.config import LoopConfig
from sciantist.state import _build_lock_path


def _build_worker_run_id(worker_id: str, run_index: int) -> str:
    """Create unique run identifiers for idempotent persistence."""
    return f"{worker_id}-r{run_index:06d}-{uuid.uuid4().hex[:8]}"


def _build_worker_worktree_path(output_root: Path, worker_id: str, run_index: int) -> Path:
    """Return deterministic path for one worker iteration worktree."""
    return output_root / "worktrees" / worker_id / f"run_{run_index:06d}"


def _build_worker_output_dir(output_root: Path, worker_id: str, run_index: int) -> Path:
    """Return output directory for one worker iteration artifacts."""
    return output_root / "workers" / worker_id / f"run_{run_index:06d}"


def _build_worker_memory_path(output_root: Path, worker_id: str) -> Path:
    """Return worker-scoped memory path used for worker-local ideation context."""
    return output_root / "workers" / worker_id / "memory.md"


def _build_leaderboard_lock_path(output_root: Path, config: LoopConfig) -> Path:
    """Return dedicated lock path for leaderboard and shared ledger/state updates."""
    return _build_lock_path(output_root, config.lock_file_name)


def _build_repo_lock_path(output_root: Path) -> Path:
    """Return dedicated lock path for serialized repo mutations on base branch."""
    return output_root / "repo.lock"


def _build_memory_lock_path(output_root: Path) -> Path:
    """Return dedicated lock path for memory.md updates."""
    return output_root / "memory.lock"
